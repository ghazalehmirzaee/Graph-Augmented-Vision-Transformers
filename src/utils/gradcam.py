import os
import sys
import traceback

import numpy as np
import pandas as pd
import cv2
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import pickle
import io
import types


def print_status(message):
    """Simple status printing function"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


class CustomUnpickler(pickle.Unpickler):
    """Custom unpickler to handle numpy._core"""

    def find_class(self, module, name):
        # Handle numpy._core.multiarray
        if module == 'numpy._core.multiarray':
            if name == 'scalar':
                return float
            module = 'numpy'

        # Handle any other numpy._core references
        if module.startswith('numpy._core'):
            module = 'numpy'

        return super().find_class(module, name)


def load_tensor_from_buffer(buffer):
    """Load tensor data from buffer with custom unpickling"""
    try:
        return CustomUnpickler(buffer).load()
    except:
        return None


def fix_state_dict(state_dict):
    """Fix any numpy or scalar values in state dict"""
    fixed_dict = {}
    for key, value in state_dict.items():
        if isinstance(value, np.ndarray):
            fixed_dict[key] = torch.from_numpy(value)
        elif isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
            fixed_dict[key] = torch.tensor(float(value))
        elif isinstance(value, torch.Tensor):
            fixed_dict[key] = value
        else:
            try:
                fixed_dict[key] = torch.tensor(value)
            except:
                fixed_dict[key] = value
    return fixed_dict


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


import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from datetime import datetime
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')


def print_status(message):
    """Simple status printing function"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


class VisionTransformerGradCAM:
    def __init__(self, model, layer_name='blocks.11.attn'):
        self.model = model
        self.layer_name = layer_name
        self.feature_maps = []
        self.gradients = []

        # Register hooks
        for name, module in self.model.named_modules():
            if self.layer_name in name:
                module.register_forward_hook(self.save_features)
                module.register_full_backward_hook(self.save_gradients)

    def save_features(self, module, input, output):
        self.feature_maps.append(output)

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0])

    def generate_attention_maps(self, input_tensor):
        """Generate attention maps for each layer"""
        attention_maps = []
        x = self.model.patch_embed(input_tensor)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.model.pos_embed

        for block in self.model.blocks:
            x = block.norm1(x)
            qkv = block.attn.qkv(x).reshape(x.shape[0], x.shape[1], 3, block.attn.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * block.attn.scale
            attn = attn.softmax(dim=-1)
            attention_maps.append(attn.detach().cpu().numpy())

        return attention_maps

    def __call__(self, input_tensor, target_category=None):
        self.feature_maps = []
        self.gradients = []

        # Forward pass
        model_output = self.model(input_tensor)
        if target_category is None:
            target_category = model_output.argmax(dim=1)

        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(model_output)
        one_hot[0][target_category] = 1
        model_output.backward(gradient=one_hot, retain_graph=True)

        if not self.gradients or not self.feature_maps:
            print_status("Error: Gradients or feature maps were not captured.")
            return None

        # Get gradients and feature maps
        gradients = self.gradients[-1][0]  # Use the last saved gradient
        feature_maps = self.feature_maps[-1][0]  # Use the last saved feature map

        # Remove CLS token
        gradients = gradients[1:]
        feature_maps = feature_maps[1:]

        # Calculate importance weights
        weights = gradients.mean(dim=0)  # [hidden_dim]

        # Compute weighted sum of feature maps
        cam = (weights @ feature_maps.T).reshape(int(feature_maps.shape[0] ** 0.5), -1)
        cam = F.relu(cam)  # Apply ReLU

        # Normalize between 0 and 1
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.detach().cpu().numpy()


def process_image(image_path, model, bboxes, labels, transform, output_dir, ground_truth_labels=None):
    try:
        # Load and preprocess image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Create figure
        plt.figure(figsize=(20, 8))

        # Plot original image with boxes
        plt.subplot(1, 2, 1)
        img_with_boxes = img.copy()
        colors = plt.cm.rainbow(np.linspace(0, 1, len(labels)))

        for bbox, label, color in zip(bboxes, labels, colors):
            color_rgb = tuple(int(c * 255) for c in color[:3])
            cv2.rectangle(
                img_with_boxes,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                color_rgb,
                2
            )

            # Add label with background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.rectangle(
                img_with_boxes,
                (int(bbox[0]), int(bbox[1] - text_size[1] - 4)),
                (int(bbox[0] + text_size[0]), int(bbox[1])),
                color_rgb,
                -1
            )
            cv2.putText(
                img_with_boxes,
                label,
                (int(bbox[0]), int(bbox[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )

        plt.imshow(img_with_boxes)
        plt.title("Original with Ground Truth", fontsize=12)
        plt.axis('off')

        # Generate GradCAM
        plt.subplot(1, 2, 2)
        input_tensor = transform(Image.fromarray(img)).unsqueeze(0)
        grad_cam = VisionTransformerGradCAM(model)

        # Get predictions
        with torch.no_grad():
            predictions = torch.sigmoid(model(input_tensor)).squeeze()

        disease_names = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
            'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]

        combined_cam = np.zeros((224, 224))
        pred_text = "Predictions:\n"

        for idx, (prob, disease) in enumerate(zip(predictions, disease_names)):
            if prob > 0.5:  # Only for predicted diseases
                cam = grad_cam(input_tensor, idx)
                if cam is None:
                    print_status(f"Skipping {disease} for image {image_path} due to missing Grad-CAM output.")
                    continue

                # Resize and weight the CAM by prediction confidence
                cam_resized = cv2.resize(cam, (224, 224))
                cam_resized = cam_resized * float(prob)
                combined_cam = np.maximum(combined_cam, cam_resized)
                pred_text += f"{disease}: {prob:.3f}\n"

        # Overlay GradCAM heatmap
        if np.any(combined_cam):
            heatmap = cv2.applyColorMap(np.uint8(255 * combined_cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            img_resized = cv2.resize(img, (224, 224))
            alpha = combined_cam[..., np.newaxis]
            overlay = img_resized * (1 - alpha * 0.7) + heatmap * (alpha * 0.7)
            overlay = overlay.astype(np.uint8)

            plt.imshow(overlay)
            plt.title("GradCAM (Predicted Diseases)", fontsize=12)
            plt.axis('off')

            # Display predictions and ground truth labels
            plt.text(1.05, 0.5, pred_text, transform=plt.gca().transAxes,
                     fontsize=10, verticalalignment='center')
            if ground_truth_labels:
                ground_truth_text = "Ground Truth:\n" + "\n".join(ground_truth_labels)
                plt.text(1.05, 0.1, ground_truth_text, transform=plt.gca().transAxes,
                         fontsize=10, verticalalignment='center', color='red')

            # Save visualization
            plt.tight_layout()
            save_path = os.path.join(output_dir, f'analysis_{os.path.basename(image_path)}')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        return True

    except Exception as e:
        print_status(f"Error processing image: {str(e)}")
        traceback.print_exc()
        return False


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


def safe_load_checkpoint(checkpoint_path, model):
    """Safely load checkpoint using torch.load"""
    print_status("Loading checkpoint...")

    try:
        # Attempt to load the checkpoint file using torch.load
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract state_dict if it exists, otherwise assume the checkpoint itself is the state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Resize position embeddings if necessary
        if 'pos_embed' in state_dict and hasattr(model, 'pos_embed'):
            old_shape = model.pos_embed.shape
            new_shape = state_dict['pos_embed'].shape
            if len(new_shape) == 3 and len(old_shape) == 3 and old_shape[1] != new_shape[1]:
                print_status(f"Resizing pos_embed from {new_shape} to {old_shape}")
                state_dict['pos_embed'] = torch.nn.functional.interpolate(
                    state_dict['pos_embed'].permute(0, 2, 1).unsqueeze(-1), size=(old_shape[1], 1), mode='bilinear', align_corners=False
                ).squeeze(-1).permute(0, 2, 1)

        # Load the state dictionary into the model
        model.load_state_dict(state_dict, strict=False)
        print_status("Successfully loaded checkpoint")
        return True

    except Exception as e:
        # Handle any errors during loading
        print_status(f"Failed to load checkpoint: {str(e)}")
        return False

def debug_tensor_shape(tensor, name):
    print_status(f"{name} shape: {tensor.shape}")


def main():
    # Create output directory
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    print_status(f"Created output directory: {output_dir}")

    # Paths
    bbox_csv = '/users/gm00051/ChestX-ray14/labels/BBox_List_2017.csv'
    image_dir = '/users/gm00051/ChestX-ray14/images'
    checkpoint = '/users/gm00051/projects/cvpr/baseline/Graph-Augmented-Vision-Transformers/scripts/checkpoints/best_model.pt'

    print_status("Loading model...")

    # Initialize and load model
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

    if not safe_load_checkpoint(checkpoint, model):
        print_status("Failed to load checkpoint. Exiting...")
        return

    model.eval()
    print_status("Model ready for inference")

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    try:
        print_status("Reading bounding box data...")
        image_info = get_images_with_multiple_boxes(bbox_csv)
        print_status(f"Found {len(image_info)} images with multiple boxes")

        # Process images
        all_images = list(image_info.keys())
        selected_images = np.random.choice(all_images, size=20, replace=False)

        successful = 0
        for idx, img_name in enumerate(selected_images, 1):
            print_status(f"Processing image {idx}/20: {img_name}")

            image_path = os.path.join(image_dir, img_name)
            if not os.path.exists(image_path):
                print_status(f"Image not found: {image_path}")
                continue

            success = process_image(
                image_path=image_path,
                model=model,
                bboxes=image_info[img_name]['bboxes'],
                labels=image_info[img_name]['labels'],
                transform=transform,
                output_dir=output_dir
            )

            if success:
                successful += 1

            if idx % 5 == 0:
                print_status(f"Progress: {idx}/20 images processed ({successful} successful)")

        print_status(f"Processing complete! Successfully processed {successful}/20 images")
        print_status(f"Results saved in {output_dir}")

    except Exception as e:
        print_status(f"Error during processing: {str(e)}")


if __name__ == '__main__':
    main()

