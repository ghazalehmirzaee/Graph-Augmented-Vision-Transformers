# src/data/transforms.py
from torchvision import transforms
import torch
import random
import numpy as np


class ChestXrayTransforms:
    """Custom transformations for chest X-ray images"""

    @staticmethod
    def get_train_transforms(config):
        return transforms.Compose([
            transforms.Resize((config['data']['image_size'],
                               config['data']['image_size'])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=(-5, 5),
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                fillcolor=0
            ),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @staticmethod
    def get_val_transforms(config):
        return transforms.Compose([
            transforms.Resize((config['data']['image_size'],
                               config['data']['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

