# src/data/dataset.py
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import numpy as np
from torchvision import transforms
import logging

logger = logging.getLogger(__name__)


class ChestXrayDataset(Dataset):
    """NIH ChestX-ray14 dataset"""

    def __init__(self, image_dir, label_file, transform=None):
        """
        Args:
            image_dir (str): Directory with chest X-ray images
            label_file (str): Path to label file
            transform (callable, optional): Transform to be applied on images
        """
        self.image_dir = image_dir

        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        # Read label file
        try:
            self.labels_df = pd.read_csv(label_file, delimiter=' ', header=None)
            self.image_paths = self.labels_df.iloc[:, 0].values
            self.labels = self.labels_df.iloc[:, 1:15].values.astype(np.float32)
        except Exception as e:
            logger.error(f"Error reading label file: {str(e)}")
            raise

        # Disease names
        self.disease_names = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
            'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]

        # Calculate class weights for balanced loss
        self.class_weights = self._calculate_class_weights()

        logger.info(f"Loaded {len(self.image_paths)} images")
        self._log_class_distribution()

    def _calculate_class_weights(self):
        """Calculate class weights based on label distribution"""
        pos_counts = np.sum(self.labels, axis=0)
        neg_counts = len(self.labels) - pos_counts
        weights = neg_counts / pos_counts
        return torch.FloatTensor(weights)

    def _log_class_distribution(self):
        """Log the distribution of classes in the dataset"""
        pos_counts = np.sum(self.labels, axis=0)
        for disease, count in zip(self.disease_names, pos_counts):
            logger.info(f"{disease}: {count} positive samples "
                        f"({count / len(self.labels) * 100:.2f}%)")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            # Load image
            img_path = os.path.join(self.image_dir, self.image_paths[idx])
            image = Image.open(img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            labels = torch.FloatTensor(self.labels[idx])
            return image, labels

        except Exception as e:
            logger.error(f"Error loading image {self.image_paths[idx]}: {str(e)}")
            raise

