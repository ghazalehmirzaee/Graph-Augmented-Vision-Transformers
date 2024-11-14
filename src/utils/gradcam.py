import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
from src.models.vit import VisionTransformer
from collections import defaultdict
import logging


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def get_images_with_multiple_boxes(csv_path, min_boxes=2, max_boxes=3):
    """Find images that have between min_boxes and max_boxes bounding boxes"""
    df = pd.read_csv(csv_path)

    # Count bounding boxes per image
    bbox_counts = df['Image Index'].value_counts()

    # Get images with desired number of boxes
    valid_images = bbox_counts[
        (bbox_counts >= min_boxes) &
        (bbox_counts <= max_boxes)
        ].index.tolist()

    # Create a dictionary of image information
    image_info = defaultdict(list)
    for img in valid_images:
        img_data = df[df['Image Index'] == img]
        bboxes = []
        labels = []
        for _, row in img_data.iterrows():
            bbox = [
                row['Bbox_x'],
                row['Bbox_y'],
                row['Bbox_x'] + row['Bbox_w'],
                row['Bbox_y'] + row['Bbox_h']
            ]
            bboxes.append(bbox)
            labels.append(row['Finding Label'])
        image_info[img] = {'bboxes': bboxes, 'labels': labels}

    return image_info


class VisionTransformerWrapper(torch.nn.Module):
    """Wrapper class for ViT to enable GradCAM visualization"""

    def __init__(self, vit_model):
        super().__init__()
        self.vit = vit_model

    def forward(self, x):
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.vit.pos_embed
        x = self.vit.pos_drop(x)

        for i, block in enumerate(self.vit.blocks):
            if i < len(self.vit.blocks) - 1:
                x = block(x)
            else:
                self.activations = x
                x = block(x)

        x = self.vit.norm(x)
        x = x[:, 0]
        x = self.vit.head(x)
        return x


def load_model(checkpoint_path):
    """Load the pretrained model"""
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

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    return VisionTransformerWrapper(model)


def process_single_image(image_path, model, bboxes, labels, transform, output_dir, logger):
    """Process a single image and generate visualizations"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        input_tensor = transform(image)
        input_tensor = input_tensor.to(device)

        # Get model predictions
        with torch.no_grad():
            output = model(input_tensor.unsqueeze(0))
            predictions = torch.sigmoid(output).squeeze().cpu().numpy()

        # Disease names
        disease_names = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
            'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))

        # 1. Original image with ground truth boxes
        original_image = np.array(image)
        img_with_boxes = original_image.copy()
        draw = ImageDraw.Draw(Image.fromarray(img_with_boxes))

        # Draw bounding boxes with different colors
        colors = plt.cm.rainbow(np.linspace(0, 1, len(labels)))
        for bbox, label, color in zip(bboxes, labels, colors):
            x1, y1, x2, y2 = bbox
            color_rgb = tuple(int(c * 255) for c in color[:3])
            draw.rectangle([(x1, y1), (x2, y2)], outline=color_rgb, width=3)
            draw.text((x1, y1 - 15), label, fill=color_rgb)

        axes[0, 0].imshow(img_with_boxes)
        axes[0, 0].set_title('Ground Truth Bounding Boxes')
        axes[0, 0].axis('off')

        # 2. Original image
        axes[0, 1].imshow(original_image)
        axes[0, 1].set_title('Original Image')
        axes[0, 1].axis('off')

        # 3. GradCAM for ground truth diseases
        grad_cam = GradCAM(model=model, target_layers=[model.vit.blocks[-1]])
        combined_cam_gt = np.zeros((224, 224))

        for label in labels:
            if label in disease_names:
                class_idx = disease_names.index(label)
                cam = grad_cam(input_tensor=input_tensor.unsqueeze(0),
                               target_category=class_idx)[0]
                combined_cam_gt = np.maximum(combined_cam_gt, cam)

        visualization_gt = show_cam_on_image(
            cv2.resize(original_image, (224, 224)) / 255.,
            combined_cam_gt,
            use_rgb=True
        )
        axes[1, 0].imshow(visualization_gt)
        axes[1, 0].set_title('GradCAM - Ground Truth Diseases')
        axes[1, 0].axis('off')

        # 4. GradCAM for predicted diseases
        combined_cam_pred = np.zeros((224, 224))
        pred_diseases = []

        for i, prob in enumerate(predictions):
            if prob > 0.5:
                pred_diseases.append((disease_names[i], prob))
                cam = grad_cam(input_tensor=input_tensor.unsqueeze(0),
                               target_category=i)[0]
                combined_cam_pred = np.maximum(combined_cam_pred, cam)

        visualization_pred = show_cam_on_image(
            cv2.resize(original_image, (224, 224)) / 255.,
            combined_cam_pred,
            use_rgb=True
        )
        axes[1, 1].imshow(visualization_pred)
        axes[1, 1].set_title('GradCAM - Predicted Diseases')
        axes[1, 1].axis('off')

        # Add text information
        plt.figtext(0.02, 0.98, f'Ground Truth: {", ".join(labels)}', fontsize=12)
        plt.figtext(0.02, 0.96, 'Predictions:', fontsize=12)
        for i, (disease, prob) in enumerate(pred_diseases):
            plt.figtext(0.02, 0.94 - i * 0.02, f'{disease}: {prob:.3f}', fontsize=10)

        plt.tight_layout()

        # Save the visualization
        image_name = os.path.basename(image_path)
        save_path = os.path.join(output_dir, f'visualization_{image_name}')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Successfully processed {image_name}")
        return True

    except Exception as e:
        logger.error(f"Error processing {os.path.basename(image_path)}: {str(e)}")
        return False


def main():
    # Setup logging
    logger = setup_logging()

    # Create output directory
    output_dir = 'outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Paths
    checkpoint_path = '/users/gm00051/projects/cvpr/baseline/Graph-Augmented-Vision-Transformers/scripts/checkpoints/checkpoint_epoch_82_auc_0.7225.pt'
    bbox_csv_path = '/users/gm00051/ChestX-ray14/labels/BBox_List_2017.csv'
    images_dir = '/users/gm00051/ChestX-ray14/images'

    # Load model
    logger.info("Loading model...")
    model = load_model(checkpoint_path)

    # Define transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Get images with multiple bounding boxes
    logger.info("Finding images with multiple bounding boxes...")
    image_info = get_images_with_multiple_boxes(bbox_csv_path)
    logger.info(f"Found {len(image_info)} images with 2-3 bounding boxes")

    # Process each image
    successful = 0
    for image_filename, info in image_info.items():
        logger.info(f"Processing {image_filename}...")
        image_path = os.path.join(images_dir, image_filename)

        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            continue

        success = process_single_image(
            image_path=image_path,
            model=model,
            bboxes=info['bboxes'],
            labels=info['labels'],
            transform=transform,
            output_dir=output_dir,
            logger=logger
        )

        if success:
            successful += 1

    logger.info(f"Processing complete. Successfully processed {successful}/{len(image_info)} images")
    logger.info(f"Results saved in {output_dir}")


if __name__ == '__main__':
    main()

    