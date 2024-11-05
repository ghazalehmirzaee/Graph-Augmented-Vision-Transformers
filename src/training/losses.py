# src/training/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = (1 - pt) ** self.gamma * bce_loss

        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss

        return focal_loss.mean()


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=1, gamma_neg=4, clip=0.05):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip

    def forward(self, inputs, targets):
        # Clipping probabilities
        xs_pos = torch.sigmoid(inputs)
        xs_neg = 1 - xs_pos

        # Clipping for numerical stability
        xs_pos = (xs_pos + self.clip).clamp(max=1)
        xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Asymmetric Focusing
        los_pos = targets * torch.log(xs_pos) * (1 - xs_pos).pow(self.gamma_pos)
        los_neg = (1 - targets) * torch.log(xs_neg) * xs_pos.pow(self.gamma_neg)

        loss = -(los_pos + los_neg).mean()
        return loss


class DynamicWeightedLoss(nn.Module):
    def __init__(self, num_classes, class_weights=None):
        super().__init__()
        self.weighted_bce = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        self.focal = FocalLoss(gamma=2)
        self.asl = AsymmetricLoss()

        # Initialize dynamic weights
        self.lambda1 = nn.Parameter(torch.tensor(1.0))
        self.lambda2 = nn.Parameter(torch.tensor(1.0))
        self.lambda3 = nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs, targets):
        # Normalize weights using softmax
        lambdas = F.softmax(
            torch.stack([self.lambda1, self.lambda2, self.lambda3]),
            dim=0
        )

        # Calculate individual losses
        wbce_loss = self.weighted_bce(inputs, targets)
        focal_loss = self.focal(inputs, targets)
        asl_loss = self.asl(inputs, targets)

        # Combine losses with dynamic weights
        total_loss = (
                lambdas[0] * wbce_loss +
                lambdas[1] * focal_loss +
                lambdas[2] * asl_loss
        )

        return total_loss

