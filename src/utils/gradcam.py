import sys
import os
from datetime import datetime

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Import the rest of the packages
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import cv2

# Import torch first
import torch
import torchvision.transforms as transforms

# Import matplotlib last
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from src.models.vit import VisionTransformer


def print_status(message):
    """Simple status printing function"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    # Paths
    bbox_csv = '/users/gm00051/ChestX-ray14/labels/BBox_List_2017.csv'
    image_dir = '/users/gm00051/ChestX-ray14/images'
    checkpoint = '/users/gm00051/projects/cvpr/baseline/Graph-Augmented-Vision-Transformers/scripts/checkpoints/checkpoint_epoch_82_auc_0.7225.pt'

    print_status("Loading model...")
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

    checkpoint_data = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint_data['model_state_dict'])
    model = model.eval()

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

    