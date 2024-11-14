import os
import sys
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import cv2
from datetime import datetime

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# Import matplotlib with Agg backend
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import torch (CPU only)
import torch
import torchvision.transforms as transforms


def get_device():
    """Get appropriate device while handling CUDA errors gracefully"""
    return torch.device('cpu')  # Force CPU usage


def print_status(message):
    """Simple status printing function"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


class VisionTransformer(torch.nn.Module):
    """Vision Transformer model definition"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=14,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0.):
        super().__init__()
        self.patch_embed = torch.nn.Conv2d(in_chans, embed_dim,
                                           kernel_size=patch_size, stride=patch_size)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2 + 1, embed_dim))
        self.pos_drop = torch.nn.Dropout(p=drop_rate)

        # Transformer blocks
        self.blocks = torch.nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate)
            for _ in range(depth)
        ])

        self.norm = torch.nn.LayerNorm(embed_dim)
        self.head = torch.nn.Linear(embed_dim, num_classes)


class TransformerBlock(torch.nn.Module):
    """Transformer block definition"""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0.):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, qkv_bias, drop)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden_dim=int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MultiHeadAttention(torch.nn.Module):
    """Multi-head attention mechanism"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = torch.nn.Dropout(drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(torch.nn.Module):
    """MLP module"""

    def __init__(self, in_features, hidden_dim, drop=0.):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features, hidden_dim)
        self.act = torch.nn.GELU()
        self.drop1 = torch.nn.Dropout(drop)
        self.fc2 = torch.nn.Linear(hidden_dim, in_features)
        self.drop2 = torch.nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class VisionTransformerGradCAM:
    """Class to handle GradCAM for Vision Transformer"""

    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None

    def get_activations(self, module, input, output):
        self.activations = output

    def get_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, input_tensor, target_category):
        # Register hooks
        handle_activation = self.model.blocks[-1].register_forward_hook(self.get_activations)
        handle_gradient = self.model.blocks[-1].register_full_backward_hook(self.get_gradients)

        # Forward pass
        output = self.model(input_tensor)

        if target_category is None:
            target_category = output.argmax(dim=1)

        # Zero gradients
        self.model.zero_grad()

        # Backward pass
        output[0, target_category].backward()

        # Get weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Generate CAM
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.nn.functional.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()

        # Normalize
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min())

        # Remove hooks
        handle_activation.remove()
        handle_gradient.remove()

        return cam


def get_images_with_multiple_boxes(csv_path, min_boxes=2, max_boxes=3):
    """Find images with multiple bounding boxes"""
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
                row['Bbox_x'],
                row['Bbox_y'],
                row['Bbox_x'] + row['Bbox_w'],
                row['Bbox_y'] + row['Bbox_h']
            ]
            image_info[img]['bboxes'].append(bbox)
            image_info[img]['labels'].append(row['Finding Label'])

    return image_info


def process_image(image_path, model, bboxes, labels, transform, output_dir):
    """Process a single image"""
    try:
        device = get_device()

        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Initialize GradCAM
        grad_cam = VisionTransformerGradCAM(model)

        # Get predictions
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()

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
        visualization = cv2.addWeighted(
            cv2.resize(img_array, (224, 224)) / 255.,
            0.6,
            cv2.applyColorMap(np.uint8(255 * combined_cam), cv2.COLORMAP_JET) / 255.,
            0.4,
            0
        )

        axes[1, 0].imshow(visualization)
        axes[1, 0].set_title('GradCAM - Ground Truth')
        axes[1, 0].axis('off')

        # Save predictions text
        pred_text = "Predictions:\n"
        for i, prob in enumerate(probs):
            if prob > 0.5:
                pred_text += f"{disease_names[i]}: {prob:.3f}\n"

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


def main():
    # Create output directory
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    print_status(f"Created output directory: {output_dir}")

    # Paths
    bbox_csv = '/users/gm00051/ChestX-ray14/labels/BBox_List_2017.csv'
    image_dir = '/users/gm00051/ChestX-ray14/images'
    checkpoint = '/users/gm00051/projects/cvpr/baseline/Graph-Augmented-Vision-Transformers/scripts/checkpoints/checkpoint_epoch_82_auc_0.7225.pt'

    print_status("Loading model...")
    device = get_device()
    print_status(f"Using device: {device}")

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

    checkpoint_data = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint_data['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print_status("Finding images with multiple boxes...")
    image_info = get_images_with_multiple_boxes(bbox_csv)

    print_status(f"Processing {len(image_info)} images...")
    for img_name, info in image_info.items():
        image_path = os.path.join(image_dir, img_name)
        print_status(f"Processing {img_name}")

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
            print_status(f"Successfully processed {img_name}")

    print_status("Processing complete!")


if __name__ == '__main__':
    main()

    