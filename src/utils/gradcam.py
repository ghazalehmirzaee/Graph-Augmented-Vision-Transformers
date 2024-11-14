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
        """Generate CAM for the specified target category with improved visualization"""
        B = input_tensor.size(0)  # Batch size

        # Register hooks
        handle_activation = self.model.blocks[-1].register_forward_hook(self.save_activation)
        handle_gradient = self.model.blocks[-1].register_full_backward_hook(self.save_gradient)

        try:
            # Forward pass
            model_output = self.model(input_tensor)

            if target_category is None:
                target_category = model_output.argmax(dim=1)

            # Zero gradients
            self.model.zero_grad()

            # Backward pass for target category
            one_hot = torch.zeros_like(model_output)
            one_hot[0][target_category] = 1
            model_output.backward(gradient=one_hot, retain_graph=True)

            # Get gradients and activations
            gradients = self.gradients[:, 1:, :]  # Remove CLS token
            activations = self.activations[:, 1:, :]  # Remove CLS token

            # Calculate attention weights
            weights = torch.mean(gradients, dim=-1)  # Global average pooling

            # Reshape activations and weights
            n_patches = int(np.sqrt(activations.shape[1]))
            activations = activations.reshape(B, n_patches, n_patches, -1)
            weights = weights.reshape(B, n_patches, n_patches)

            # Apply weights to activation maps
            cam = torch.zeros(B, n_patches, n_patches, device=activations.device)
            for b in range(B):
                for i in range(n_patches):
                    for j in range(n_patches):
                        cam[b, i, j] = torch.sum(weights[b, i, j] * activations[b, i, j])

            # Apply ReLU and normalize
            cam = F.relu(cam)

            # Interpolate to input resolution
            cam = F.interpolate(
                cam.unsqueeze(1),
                size=(224, 224),
                mode='bicubic',
                align_corners=False
            ).squeeze(1)

            # Normalize and enhance contrast
            cam = cam.detach().cpu().numpy()
            cam = cam[0]  # Take first batch element

            # Enhance contrast using histogram equalization
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cam = (cam * 255).astype(np.uint8)
            cam = cv2.equalizeHist(cam)
            cam = cam.astype(float) / 255.0

            # Apply gamma correction to enhance high activation areas
            gamma = 0.5
            cam = np.power(cam, gamma)

            return cam

        finally:
            handle_activation.remove()
            handle_gradient.remove()


def process_image(image_path, model, bboxes, labels, transform, output_dir):
    """Process a single image with improved visualization"""
    try:
        # Read and preprocess image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_size = img.shape[:2]

        # Transform for model input
        image_tensor = transform(Image.fromarray(img)).unsqueeze(0)

        # Initialize visualization
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))
        fig.suptitle(f'Analysis Results for {os.path.basename(image_path)}', fontsize=16)

        # 1. Original image with bounding boxes
        for bbox, label, color in zip(bboxes, labels, plt.cm.rainbow(np.linspace(0, 1, len(labels)))):
            color_rgb = tuple(int(c * 255) for c in color[:3])
            cv2.rectangle(
                img,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                color_rgb,
                3
            )
            # Add label with background for better visibility
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(
                img,
                (int(bbox[0]), int(bbox[1] - label_size[1] - 10)),
                (int(bbox[0] + label_size[0]), int(bbox[1])),
                color_rgb,
                -1
            )
            cv2.putText(
                img,
                label,
                (int(bbox[0]), int(bbox[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        axes[0, 0].imshow(img)
        axes[0, 0].set_title('Original with Ground Truth Boxes')
        axes[0, 0].axis('off')

        # 2. Original image
        axes[0, 1].imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Original Image')
        axes[0, 1].axis('off')

        # 3. Combined GradCAM visualization
        grad_cam = VisionTransformerGradCAM(model)
        combined_cam = np.zeros((224, 224))

        # Get model predictions
        with torch.no_grad():
            predictions = torch.sigmoid(model(image_tensor)).squeeze().cpu().numpy()

        disease_names = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
            'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]

        # Generate GradCAM for ground truth labels
        for label in labels:
            if label in disease_names:
                class_idx = disease_names.index(label)
                cam = grad_cam(image_tensor, class_idx)
                combined_cam = np.maximum(combined_cam, cam)

        # Create heatmap overlay
        heatmap = cv2.applyColorMap(np.uint8(255 * combined_cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Overlay with original image
        img_resized = cv2.resize(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB), (224, 224))
        overlay = cv2.addWeighted(img_resized, 0.7, heatmap, 0.3, 0)

        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title('GradCAM Visualization')
        axes[1, 0].axis('off')

        # 4. Predictions text
        axes[1, 1].axis('off')
        pred_text = "Model Predictions:\n\n"
        pred_text += "Ground Truth:\n"
        for label in labels:
            pred_text += f"- {label}\n"

        pred_text += "\nPredicted (conf > 0.5):\n"
        for i, (prob, disease) in enumerate(zip(predictions, disease_names)):
            if prob > 0.5:
                pred_text += f"- {disease}: {prob:.3f}\n"

        axes[1, 1].text(0.1, 0.5, pred_text, fontsize=12, transform=axes[1, 1].transAxes,
                        verticalalignment='center')

        # Save the figure
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'analysis_{os.path.basename(image_path)}')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return True

    except Exception as e:
        print_status(f"Error processing image: {str(e)}")
        import traceback
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



# def safe_load_checkpoint(checkpoint_path, model):
#     """Safely load checkpoint with numpy fixes"""
#     print_status("Loading checkpoint...")
#
#     try:
#         # First attempt: Load with custom unpickler
#         with open(checkpoint_path, 'rb') as f:
#             checkpoint = load_tensor_from_buffer(f)
#
#         if checkpoint is not None:
#             if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
#                 state_dict = checkpoint['model_state_dict']
#             else:
#                 state_dict = checkpoint
#
#             # Fix any numpy values
#             state_dict = fix_state_dict(state_dict)
#
#             # Load state dict
#             model.load_state_dict(state_dict)
#             model.load_state_dict(state_dict, strict=False)
#

#             print_status("Successfully loaded checkpoint")
#             return True
#
#     except Exception as e:
#         print_status(f"First attempt failed: {str(e)}")
#
#     try:
#         # Second attempt: Load with memory mapping
#         print_status("Attempting memory-mapped loading...")
#         state_dict = {}
#
#         with open(checkpoint_path, 'rb') as f:
#             # Skip magic number and protocol
#             f.seek(2)
#
#             try:
#                 data = torch.load(
#                     f,
#                     map_location='cpu',
#                     pickle_module=pickle,
#                     weights_only=True
#                 )
#
#                 if isinstance(data, dict) and 'model_state_dict' in data:
#                     state_dict = data['model_state_dict']
#                 else:
#                     state_dict = data
#
#                 model.load_state_dict(state_dict)
#                 print_status("Successfully loaded checkpoint using memory mapping")
#                 return True
#
#             except Exception as inner_e:
#                 print_status(f"Memory-mapped loading failed: {str(inner_e)}")
#
#     except Exception as e:
#         print_status(f"Second attempt failed: {str(e)}")
#
#     try:
#         # Third attempt: Raw file reading
#         print_status("Attempting raw file reading...")
#         with open(checkpoint_path, 'rb') as f:
#             content = f.read()
#
#         # Try to load from memory buffer
#         buffer = io.BytesIO(content)
#         data = load_tensor_from_buffer(buffer)
#
#         if data is not None:
#             if isinstance(data, dict) and 'model_state_dict' in data:
#                 state_dict = data['model_state_dict']
#             else:
#                 state_dict = data
#
#             state_dict = fix_state_dict(state_dict)
#             model.load_state_dict(state_dict)
#             print_status("Successfully loaded checkpoint using raw reading")
#             return True
#
#     except Exception as e:
#         print_status(f"Third attempt failed: {str(e)}")
#
#     print_status("All loading attempts failed")
#     return False

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

    # Try loading checkpoint
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

