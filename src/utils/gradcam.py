import os
import sys
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
import warnings
warnings.filterwarnings('ignore')

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
    """Process a single image"""
    try:
        # Convert image to RGB
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img)

        # Get input tensor
        input_tensor = transform(image).unsqueeze(0)

        # Initialize GradCAM
        grad_cam = VisionTransformerGradCAM(model)

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))

        # Original image with boxes
        img_array = np.array(image)
        img_with_boxes = img_array.copy()

        # Draw boxes using OpenCV instead of PIL
        colors = plt.cm.rainbow(np.linspace(0, 1, len(labels)))
        for bbox, label, color in zip(bboxes, labels, colors):
            color_rgb = tuple(int(c * 255) for c in color[:3])
            cv2.rectangle(
                img_with_boxes,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                color_rgb,
                3
            )
            cv2.putText(
                img_with_boxes,
                label,
                (int(bbox[0]), int(bbox[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color_rgb,
                2
            )

        axes[0, 0].imshow(img_with_boxes)
        axes[0, 0].set_title('Original with Bounding Boxes')
        axes[0, 0].axis('off')

        # Original image
        axes[0, 1].imshow(img_array)
        axes[0, 1].set_title('Original Image')
        axes[0, 1].axis('off')

        # GradCAM visualization
        combined_cam = np.zeros((224, 224))
        disease_names = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
            'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]

        for label in labels:
            if label in disease_names:
                class_idx = disease_names.index(label)
                cam = grad_cam(input_tensor, class_idx)
                combined_cam = np.maximum(combined_cam, cam)

        # Overlay GradCAM
        resized_img = cv2.resize(img_array, (224, 224))
        cam_image = np.uint8(255 * combined_cam)
        heatmap = cv2.applyColorMap(cam_image, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(resized_img, 0.6, heatmap, 0.4, 0)

        axes[1, 0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('GradCAM Visualization')
        axes[1, 0].axis('off')

        # Model predictions
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.sigmoid(outputs).squeeze().numpy()

        pred_text = "Predictions:\n"
        for i, (prob, disease) in enumerate(zip(probs, disease_names)):
            if prob > 0.5:
                pred_text += f"{disease}: {prob:.3f}\n"

        axes[1, 1].text(0.1, 0.5, pred_text, fontsize=12)
        axes[1, 1].axis('off')

        # Save visualization
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'gradcam_{os.path.basename(image_path)}')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return True

    except Exception as e:
        print_status(f"Error processing image: {str(e)}")
        return False


def extract_state_dict_tensors(checkpoint_path):
    """Extract only the tensor data from checkpoint"""
    with open(checkpoint_path, 'rb') as f:
        # Skip potential metadata
        try:
            magic_number = torch.serialization._utils._get_magic_number(f)
            if magic_number == b'PK\x03\x04':
                f.seek(0)
        except:
            f.seek(0)

        # Try unpickling with a custom class loader
        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # Convert old numpy scalar type to tensor directly
                if module == 'numpy._core.multiarray' and name == 'scalar':
                    return lambda x, dtype: torch.tensor(x)
                return super().find_class(module, name)

        try:
            data = CustomUnpickler(f).load()
            if isinstance(data, dict) and 'model_state_dict' in data:
                return data['model_state_dict']
            return data
        except Exception as e:
            print_status(f"Custom unpickling failed: {str(e)}")
            return None


def safe_load_checkpoint(checkpoint_path, model):
    """Safely load checkpoint with proper error handling"""
    print_status("Loading checkpoint...")

    try:
        # Method 1: Direct tensor dictionary loading
        print_status("Attempting direct tensor loading...")
        state_dict = extract_state_dict_tensors(checkpoint_path)
        if state_dict is not None:
            model.load_state_dict(state_dict)
            print_status("Successfully loaded checkpoint using direct tensor loading")
            return True
    except Exception as e:
        print_status(f"Direct tensor loading failed: {str(e)}")

    try:
        # Method 2: Manual state dict reconstruction
        print_status("Attempting manual state dict reconstruction...")
        with open(checkpoint_path, 'rb') as f:
            checkpoint = torch.load(f, map_location='cpu')

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = {}
            for key, value in checkpoint['model_state_dict'].items():
                if isinstance(value, torch.Tensor):
                    state_dict[key] = value
                else:
                    # Convert non-tensor values to tensors
                    try:
                        state_dict[key] = torch.tensor(value)
                    except:
                        print_status(f"Skipping conversion of {key}")
                        state_dict[key] = value

            model.load_state_dict(state_dict)
            print_status("Successfully loaded checkpoint using manual reconstruction")
            return True
    except Exception as e:
        print_status(f"Manual reconstruction failed: {str(e)}")

    try:
        # Method 3: Load only tensor data
        print_status("Attempting tensor-only loading...")
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict)
        print_status("Successfully loaded checkpoint using tensor-only loading")
        return True
    except Exception as e:
        print_status(f"Tensor-only loading failed: {str(e)}")

    print_status("All loading methods failed")
    return False


def load_raw_state_dict(path):
    """Load raw state dict from file"""
    try:
        # Try loading with torch's built-in loader
        return torch.load(path, map_location='cpu', weights_only=True)
    except:
        pass

    try:
        # Try loading with custom unpickler
        with open(path, 'rb') as f:
            unpickler = CustomUnpickler(f)
            data = unpickler.load()
            if isinstance(data, dict) and 'model_state_dict' in data:
                return data['model_state_dict']
            return data
    except:
        pass

    return None


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Handle numpy._core.multiarray.scalar
        if module == 'numpy._core.multiarray' and name == 'scalar':
            return torch.tensor
        if module == 'numpy' and name == 'dtype':
            import numpy
            return numpy.dtype
        return super().find_class(module, name)


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

    # Try loading with debug information
    print_status("Attempting to load checkpoint...")
    try:
        # First, try to examine the checkpoint
        with open(checkpoint, 'rb') as f:
            print_status("Successfully opened checkpoint file")

        # Try different loading methods
        if not safe_load_checkpoint(checkpoint, model):
            print_status("All loading methods failed. Trying raw state dict loading...")
            state_dict = load_raw_state_dict(checkpoint)
            if state_dict is not None:
                print_status("Successfully loaded raw state dict")
                model.load_state_dict(state_dict)
            else:
                print_status("Failed to load checkpoint. Exiting...")
                return
    except Exception as e:
        print_status(f"Error during checkpoint loading: {str(e)}")
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

    # Process images
    try:
        print_status("Reading bounding box data...")
        image_info = get_images_with_multiple_boxes(bbox_csv)
        print_status(f"Found {len(image_info)} images with multiple boxes")

        successful = 0
        total = len(image_info)

        for idx, (img_name, info) in enumerate(image_info.items(), 1):
            print_status(f"Processing image {idx}/{total}: {img_name}")

            image_path = os.path.join(image_dir, img_name)
            if not os.path.exists(image_path):
                print_status(f"Image not found: {image_path}")
                continue

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

            if idx % 10 == 0:
                print_status(f"Progress: {idx}/{total} images processed ({successful} successful)")

        print_status(f"Processing complete! Successfully processed {successful}/{total} images")
        print_status(f"Results saved in {output_dir}")

    except Exception as e:
        print_status(f"Error during processing: {str(e)}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print_status(f"Fatal error: {str(e)}")

