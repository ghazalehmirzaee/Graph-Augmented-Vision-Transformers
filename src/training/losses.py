# src/training/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class HybridLoss(nn.Module):
    """Combination of BCE and Focal Loss with class weights"""

    def __init__(self, class_weights=None, alpha=1, gamma=2):
        super().__init__()
        self.class_weights = class_weights
        self.focal = FocalLoss(alpha, gamma)

    def forward(self, inputs, targets):
        # Binary Cross Entropy with class weights
        if self.class_weights is not None:
            bce_loss = F.binary_cross_entropy_with_logits(
                inputs, targets,
                pos_weight=self.class_weights,
                reduction='mean'
            )
        else:
            bce_loss = F.binary_cross_entropy_with_logits(
                inputs, targets,
                reduction='mean'
            )

        # Focal Loss
        focal_loss = self.focal(inputs, targets)

        # Combine losses
        return 0.5 * bce_loss + 0.5 * focal_loss

