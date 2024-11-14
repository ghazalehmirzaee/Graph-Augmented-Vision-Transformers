import os
import sys
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import cv2
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def print_status(message):
    """Simple status printing function"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


class PatchEmbed(nn.Module):
    """Split image into patches and embed them"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=14,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0.):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=drop_rate)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.pos_embed, std=.02)
        torch.nn.init.normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class VisionTransformerGradCAM:
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, input_tensor, target_category):
        # Register hooks
        handle_activation = self.model.blocks[-1].register_forward_hook(self.save_activation)
        handle_gradient = self.model.blocks[-1].register_full_backward_hook(self.save_gradient)

        # Forward pass
        model_output = self.model(input_tensor)

        # Target for backprop
        if target_category is None:
            target_category = model_output.argmax(dim=1)

        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(model_output)
        one_hot[0][target_category] = 1
        model_output.backward(gradient=one_hot, retain_graph=True)

        # Get gradients and activations
        gradients = self.gradients.detach().cpu()
        activations = self.activations.detach().cpu()

        # Remove hooks
        handle_activation.remove()
        handle_gradient.remove()

        # Weight gradients
        weights = gradients.mean(dim=2).mean(dim=2)  # Global average pooling

        # Generate CAM
        batch_size, n_tokens, _ = activations.shape
        cam = torch.zeros(batch_size, 1, int(np.sqrt(n_tokens - 1)), int(np.sqrt(n_tokens - 1)))

        # Exclude CLS token
        activations = activations[:, 1:, :]
        weights = weights.view(batch_size, -1, 1)
        cam_features = activations.view(batch_size, int(np.sqrt(n_tokens - 1)), int(np.sqrt(n_tokens - 1)), -1)
        cam_features = cam_features.permute(0, 3, 1, 2)

        # Compute weighted sum
        for i in range(batch_size):
            for j in range(weights.shape[1]):
                cam[:, 0, :, :] += weights[i, j] * cam_features[i, j, :, :]

        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)

        # Normalize
        cam = cam.view(224, 224).numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam


def get_images_with_multiple_boxes(csv_path, min_boxes=2, max_boxes=3):
    df = pd.read_csv(csv_path)
    counts = df['Image Index'].value_counts()
    valid_images = counts[(counts >= min_boxes) & (counts <= max_boxes)].index

    image_info = {}
    for img in valid_images:
        data = df[df['Image Index'] == img]
        image_info[img] = {
            'bboxes': [],
            'labels': []
        }
        for _, row in data.iterrows():
            bbox = [
                float(row['Bbox_x']),
                float(row['Bbox_y']),
                float(row['Bbox_x'] + row['Bbox_w']),
                float(row['Bbox_y'] + row['Bbox_h'])
            ]
            image_info[img]['bboxes'].append(bbox)
            image_info[img]['labels'].append(row['Finding Label'])

    return image_info


def process_image(image_path, model, bboxes, labels, transform, output_dir):
    try:
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)

        # Get predictions and generate GradCAM
        grad_cam = VisionTransformerGradCAM(model)

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))

        # Original image with boxes
        img_array = np.array(image)
        img_with_boxes = Image.fromarray(img_array.copy())
        draw = ImageDraw.Draw(img_with_boxes)

        colors = plt.cm.rainbow(np.linspace(0, 1, len(labels)))
        for bbox, label, color in zip(bboxes, labels, colors):
            color_rgb = tuple(int(c * 255) for c in color[:3])
            draw.rectangle(bbox, outline=color_rgb, width=3)
            draw.text((bbox[0], bbox[1] - 15), label, fill=color_rgb)

        axes[0, 0].imshow(img_with_boxes)
        axes[0, 0].set_title('Original with Bounding Boxes')
        axes[0, 0].axis('off')

        # Original image
        axes[0, 1].imshow(img_array)
        axes[0, 1].set_title('Original Image')
        axes[0, 1].axis('off')

        # GradCAM for ground truth
        disease_names = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
            'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]

        combined_cam = np.zeros((224, 224))
        for label in labels:
            if label in disease_names:
                class_idx = disease_names.index(label)
                cam = grad_cam(input_tensor, class_idx)
                combined_cam = np.maximum(combined_cam, cam)

        # Overlay GradCAM
        cam_image = np.uint8(255 * combined_cam)
        cam_heatmap = cv2.applyColorMap(cam_image, cv2.COLORMAP_JET)
        cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(
            cv2.resize(img_array, (224, 224)),
            0.6,
            cam_heatmap,
            0.4,
            0
        )

        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title('GradCAM Visualization')
        axes[1, 0].axis('off')

        # Predictions
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.sigmoid(outputs).squeeze().numpy()

        pred_text = "Predictions:\n"
        for i, (prob, disease) in enumerate(zip(probs, disease_names)):
            if prob > 0.5:
                pred_text += f"{disease}: {prob:.3f}\n"

        axes[1, 1].text(0.1, 0.5, pred_text, fontsize=12)
        axes[1, 1].axis('off')

        # Save figure
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'gradcam_{os.path.basename(image_path)}')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return True

    except Exception as e:
        print_status(f"Error processing {image_path}: {str(e)}")
        return False


def safe_load_checkpoint(checkpoint_path, model):
    """Safely load checkpoint with proper error handling"""
    try:
        print_status("Loading checkpoint...")
        # Load with pickle support
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print_status("Found model state dict in checkpoint")
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print_status("Loading full checkpoint")
            model.load_state_dict(checkpoint)

        print_status("Checkpoint loaded successfully")
        return True
    except Exception as e:
        print_status(f"Error loading checkpoint: {str(e)}")
        try:
            # Try alternative loading method
            print_status("Attempting alternative loading method...")
            torch.serialization.add_safe_globals([('numpy._core.multiarray', 'scalar')])
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print_status("Alternative loading successful")
            return True
        except Exception as e2:
            print_status(f"Alternative loading failed: {str(e2)}")
            return False


def main():
    # Create output directory
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    print_status(f"Created output directory: {output_dir}")

    # Paths
    bbox_csv = '/users/gm00051/ChestX-ray14/labels/BBox_List_2017.csv'
    image_dir = '/users/gm00051/ChestX-ray14/images'
    # Use best_model.pt instead of checkpoint_epoch_82_auc_0.7225.pt
    checkpoint = '/users/gm00051/projects/cvpr/baseline/Graph-Augmented-Vision-Transformers/scripts/checkpoints/best_model.pt'

    print_status("Loading model...")

    # Initialize model
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        drop_rate=0.0
    )

    # Load checkpoint with new safe loading function
    if not safe_load_checkpoint(checkpoint, model):
        print_status("Failed to load checkpoint. Exiting...")
        return

    model.eval()
    print_status("Model loaded successfully")

    # Define transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Get images with multiple boxes
    print_status("Finding images with multiple boxes...")
    try:
        image_info = get_images_with_multiple_boxes(bbox_csv)
        print_status(f"Found {len(image_info)} images with multiple boxes")
    except Exception as e:
        print_status(f"Error reading bbox CSV: {str(e)}")
        return

    # Process each image
    successful = 0
    total = len(image_info)

    for idx, (img_name, info) in enumerate(image_info.items(), 1):
        print_status(f"Processing image {idx}/{total}: {img_name}")

        image_path = os.path.join(image_dir, img_name)
        if not os.path.exists(image_path):
            print_status(f"Image not found: {image_path}")
            continue

        try:
            success = process_image(
                image_path=image_path,
                model=model,
                bboxes=info['bboxes'],
                labels=info['labels'],
                transform=transform,
                output_dir=output_dir
            )

            if success:
                successful += 1
                print_status(f"Successfully processed {img_name}")
            else:
                print_status(f"Failed to process {img_name}")

        except Exception as e:
            print_status(f"Error processing {img_name}: {str(e)}")
            continue

        # Save progress periodically
        if idx % 10 == 0:
            print_status(f"Progress: {idx}/{total} images processed ({successful} successful)")

    print_status(f"Processing complete! Successfully processed {successful}/{total} images")
    print_status(f"Results saved in {output_dir}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print_status(f"Fatal error: {str(e)}")
