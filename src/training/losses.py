# src/training/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicWeightedLoss(nn.Module):
    def __init__(self, num_classes, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights

        # Initialize loss weights as learnable parameters
        self.lambda_wbce = nn.Parameter(torch.tensor(1.0))
        self.lambda_focal = nn.Parameter(torch.tensor(1.0))
        self.lambda_asl = nn.Parameter(torch.tensor(1.0))

        # Register class weights as buffer
        if class_weights is not None:
            self.register_buffer('pos_weight', class_weights)
        else:
            self.register_buffer('pos_weight', torch.ones(num_classes))

        self.gamma = 2.0  # Focal Loss gamma

    def forward(self, logits, targets):
        # Get normalized loss weights
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
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = ((1 - pt) ** self.gamma * bce_loss).mean()

        # Asymmetric Loss
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1 - xs_pos
        gamma_pos, gamma_neg = 1, 4

        los_pos = targets * torch.log(torch.clamp(xs_pos, min=1e-8)) * (1 - xs_pos).pow(gamma_pos)
        los_neg = (1 - targets) * torch.log(torch.clamp(xs_neg, min=1e-8)) * xs_pos.pow(gamma_neg)
        asl_loss = -(los_pos + los_neg).mean()

        # Combine losses
        total_loss = (
                weights[0] * wbce_loss +
                weights[1] * focal_loss +
                weights[2] * asl_loss
        )

        loss_dict = {
            'wbce': wbce_loss.detach(),
            'focal': focal_loss.detach(),
            'asl': asl_loss.detach()
        }

        return total_loss, loss_dict

    def get_loss_weights(self):
        weights = F.softmax(torch.stack([
            self.lambda_wbce,
            self.lambda_focal,
            self.lambda_asl
        ]), dim=0)
        return weights.detach().cpu().numpy()

