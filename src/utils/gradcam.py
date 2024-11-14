
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
import warnings

warnings.filterwarnings('ignore')


def print_status(message):
    """Simple status printing function"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


class VisionTransformerGradCAM:
    """GradCAM implementation for Vision Transformer"""

    def __init__(self, model):
        self.model = model
        self.feature_maps = None
        self.gradients = None

    def save_feature_maps(self, module, input, output):
        self.feature_maps = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def get_cam(self, input_tensor, target_category):
        """
        Generate class activation map for target category
        """
        # Register hooks on last transformer block
        handle_features = self.model.blocks[-1].register_forward_hook(self.save_feature_maps)
        handle_gradient = self.model.blocks[-1].register_full_backward_hook(self.save_gradients)

        try:
            # Forward pass
            model_output = self.model(input_tensor)
            if target_category is None:
                target_category = torch.argmax(model_output)

            # Zero gradients
            self.model.zero_grad()

            # Backward pass for target category
            one_hot = torch.zeros_like(model_output)
            one_hot[0, target_category] = 1
            model_output.backward(gradient=one_hot, retain_graph=True)

            # Get feature maps and gradients (excluding CLS token)
            feature_maps = self.feature_maps[0, 1:].detach()  # [196, 768]
            gradients = self.gradients[0, 1:].detach()  # [196, 768]

            # Calculate importance weights
            weights = torch.mean(gradients, dim=0)  # [768]

            # Reshape feature maps to spatial dimensions
            feature_maps = feature_maps.reshape(14, 14, -1)  # [14, 14, 768]

            # Generate CAM
            cam = torch.zeros(14, 14, device=feature_maps.device)
            for i in range(14):
                for j in range(14):
                    cam[i, j] = torch.sum(feature_maps[i, j] * weights)

            # Apply ReLU
            cam = F.relu(cam)

            # Interpolate to image size
            cam = cam.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            cam = F.interpolate(
                cam,
                size=(224, 224),
                mode='bicubic',
                align_corners=False
            )

            # Normalize
            cam = cam.squeeze().cpu().numpy()
            if cam.max() != cam.min():
                cam = (cam - cam.min()) / (cam.max() - cam.min())

            return cam

        finally:
            # Remove hooks
            handle_features.remove()
            handle_gradient.remove()


def get_images_with_multiple_boxes(csv_path, min_boxes=2, max_boxes=3):
    """Get images that have multiple bounding boxes"""
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


def generate_visualization(image_path, model, bboxes, labels, transform, output_path):
    """Generate side-by-side visualization with bounding boxes and GradCAM"""
    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create figure
    plt.figure(figsize=(20, 8))

    # Plot original image with bounding boxes
    plt.subplot(1, 2, 1)
    plt.title("Original with Ground Truth Boxes", fontsize=12)

    # Draw bounding boxes
    img_with_boxes = img.copy()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(labels)))

    for bbox, label, color in zip(bboxes, labels, colors):
        color_rgb = tuple(int(c * 255) for c in color[:3])
        # Draw box
        cv2.rectangle(
            img_with_boxes,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            color_rgb,
            2
        )
        # Add label with background
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
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
            (int(bbox[0]), int(bbox[1] - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

    plt.imshow(img_with_boxes)
    plt.axis('off')

    # Plot GradCAM visualization
    plt.subplot(1, 2, 2)
    plt.title("GradCAM Visualization", fontsize=12)

    # Get model predictions
    input_tensor = transform(Image.fromarray(img)).unsqueeze(0)
    model.eval()
    gradcam = VisionTransformerGradCAM(model)

    with torch.no_grad():
        predictions = torch.sigmoid(model(input_tensor))

    # Generate combined GradCAM for all predicted classes
    disease_names = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
        'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
        'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
    ]

    combined_cam = np.zeros((224, 224))
    predicted_diseases = []

    for idx, (prob, disease) in enumerate(zip(predictions[0], disease_names)):
        if prob > 0.5:  # Only for predicted diseases
            cam = gradcam.get_cam(input_tensor, idx)
            combined_cam = np.maximum(combined_cam, cam)
            predicted_diseases.append(f"{disease}: {prob:.3f}")

    # Create heatmap overlay
    img_resized = cv2.resize(img, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * combined_cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay heatmap
    overlay = cv2.addWeighted(img_resized, 0.7, heatmap, 0.3, 0)
    plt.imshow(overlay)
    plt.axis('off')

    # Add predictions text
    if predicted_diseases:
        pred_text = "Predictions:\n" + "\n".join(predicted_diseases)
        plt.text(1.05, 0.5, pred_text, transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='center')

    # Save and close
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # Setup
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    print_status(f"Created output directory: {output_dir}")

    # Paths
    bbox_csv = '/users/gm00051/ChestX-ray14/labels/BBox_List_2017.csv'
    image_dir = '/users/gm00051/ChestX-ray14/images'
    checkpoint = '/users/gm00051/projects/cvpr/baseline/Graph-Augmented-Vision-Transformers/scripts/checkpoints/best_model.pt'

    # Initialize model (using your original VisionTransformer class)
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

    # Load checkpoint
    print_status("Loading model checkpoint...")
    checkpoint = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print_status("Model loaded successfully")

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Get images with multiple boxes
    print_status("Reading bounding box data...")
    image_info = get_images_with_multiple_boxes(bbox_csv)
    print_status(f"Found {len(image_info)} images with multiple boxes")

    # Process random 20 images
    selected_images = np.random.choice(list(image_info.keys()), size=20, replace=False)

    for idx, img_name in enumerate(selected_images, 1):
        print_status(f"Processing image {idx}/20: {img_name}")

        image_path = os.path.join(image_dir, img_name)
        output_path = os.path.join(output_dir, f'analysis_{img_name}')

        try:
            generate_visualization(
                image_path=image_path,
                model=model,
                bboxes=image_info[img_name]['bboxes'],
                labels=image_info[img_name]['labels'],
                transform=transform,
                output_path=output_path
            )
        except Exception as e:
            print_status(f"Error processing {img_name}: {str(e)}")
            continue

    print_status("Processing complete!")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print_status(f"Fatal error: {str(e)}")
