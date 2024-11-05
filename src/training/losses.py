# src/training/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicWeightedLoss(nn.Module):
    def __init__(self, num_classes, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights

        # Initialize loss weights
        self.lambda_wbce = nn.Parameter(torch.tensor(1.0))
        self.lambda_focal = nn.Parameter(torch.tensor(1.0))
        self.lambda_asl = nn.Parameter(torch.tensor(1.0))

        self.register_buffer('pos_weight', class_weights)
        self.gamma = 2.0  # Focal Loss gamma

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C) tensor of logits
            targets: (B, C) tensor of binary targets
        Returns:
            total_loss: scalar tensor
            loss_dict: dictionary of individual losses
        """
        # Get loss weights through softmax
        weights = F.softmax(torch.stack([
            self.lambda_wbce,
            self.lambda_focal,
            self.lambda_asl
        ]), dim=0)

        # Weighted BCE Loss
        wbce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='mean'
        )

        # Focal Loss
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_loss = -(1 - pt) ** self.gamma * F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        ).mean()

        # Asymmetric Loss
        gamma_pos, gamma_neg = 1, 4
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1 - xs_pos
        asymmetric_loss = (
                targets * -torch.log(xs_pos.clamp(min=1e-8)) * ((1 - xs_pos) ** gamma_pos) +
                (1 - targets) * -torch.log(xs_neg.clamp(min=1e-8)) * (xs_pos ** gamma_neg)
        ).mean()

        # Combine losses
        total_loss = (
                weights[0] * wbce_loss +
                weights[1] * focal_loss +
                weights[2] * asymmetric_loss
        )

        loss_dict = {
            'wbce': wbce_loss,
            'focal': focal_loss,
            'asl': asymmetric_loss
        }

        return total_loss, loss_dict

    def get_loss_weights(self):
        """Get current loss weights"""
        return F.softmax(torch.stack([
            self.lambda_wbce,
            self.lambda_focal,
            self.lambda_asl
        ]), dim=0).detach().cpu().numpy()

    