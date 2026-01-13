"""
Focal Loss implementation for addressing class imbalance.

Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017)
FL(pt) = -α(1-pt)^γ * log(pt)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    Args:
        alpha: Weighting factor in [0, 1] to balance positive/negative examples,
               or a list of weights for each class
        gamma: Focusing parameter for modulating loss (default: 2.0)
        reduction: Specifies the reduction to apply to the output
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

        # Handle alpha (class weights)
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = alpha
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) logits
            targets: (N,) class indices

        Returns:
            Focal loss value
        """
        # Get probabilities
        p = F.softmax(inputs, dim=1)

        # Get class probabilities
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Calculate focal term: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Calculate focal loss
        focal_loss = focal_weight * ce_loss

        # Apply alpha if provided
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                # Move alpha to same device as inputs
                alpha = self.alpha.to(inputs.device)
                alpha_t = alpha.gather(0, targets)
                focal_loss = alpha_t * focal_loss
            else:
                focal_loss = self.alpha * focal_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def compute_class_weights(class_counts, mode="inverse", smooth=1.0):
    """
    Compute class weights for focal loss alpha parameter.

    Args:
        class_counts: dict or list of class counts
        mode: 'inverse' or 'balanced'
        smooth: smoothing factor to avoid division by zero

    Returns:
        Tensor of class weights
    """
    if isinstance(class_counts, dict):
        # Convert dict to list
        num_classes = max(class_counts.keys()) + 1
        counts = [class_counts.get(i, 0) for i in range(num_classes)]
    else:
        counts = list(class_counts)

    counts = torch.tensor(counts, dtype=torch.float32)

    if mode == "inverse":
        # Inverse frequency: weight = 1 / (count + smooth)
        weights = 1.0 / (counts + smooth)
    elif mode == "balanced":
        # Balanced: weight = total / (num_classes * count)
        total = counts.sum()
        num_classes = len(counts)
        weights = total / (num_classes * (counts + smooth))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Normalize weights to sum to num_classes (keeps loss magnitude similar)
    weights = weights * len(counts) / weights.sum()

    return weights


if __name__ == "__main__":
    # Test focal loss
    print("Testing Focal Loss...")

    # Example class distribution (from KIOCMIL)
    class_counts = {0: 55, 2: 248, 4: 376, 6: 206, 8: 137}

    # Compute weights
    weights = compute_class_weights(class_counts, mode="inverse")
    print(f"\nClass weights (inverse): {weights}")

    # Create focal loss
    focal_loss = FocalLoss(alpha=weights.tolist(), gamma=2.0)

    # Test with dummy data
    batch_size = 8
    num_classes = 5
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))

    loss = focal_loss(logits, targets)
    print(f"\nTest loss: {loss.item():.4f}")

    # Compare with CE loss
    ce_loss = F.cross_entropy(logits, targets)
    print(f"CE loss: {ce_loss.item():.4f}")
    print(f"Focal/CE ratio: {loss.item() / ce_loss.item():.2f}")
