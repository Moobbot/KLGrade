"""
Focal Loss implementation for DETR with Class-Balanced weighting.

References:
- Focal Loss: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
- Class-Balanced Loss: Cui et al., "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Sigmoid focal loss for binary classification.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        inputs: Predicted logits (before sigmoid), shape [N, C]
        targets: Ground truth labels (0 or 1), shape [N, C]
        alpha: Weighting factor for positive class (default 0.25)
        gamma: Focusing parameter (default 2.0)
        reduction: 'none' | 'mean' | 'sum'
        
    Returns:
        Focal loss value
    """
    # Apply sigmoid to get probabilities
    p = torch.sigmoid(inputs)
    
    # Calculate focal loss
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma
    
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        focal_loss = alpha_t * focal_weight * ce_loss
    else:
        focal_loss = focal_weight * ce_loss
    
    if reduction == "mean":
        return focal_loss.mean()
    elif reduction == "sum":
        return focal_loss.sum()
    else:
        return focal_loss


def calculate_cb_weights(
    class_counts: Dict[int, int],
    beta: float = 0.9999,
    normalize: bool = True
) -> np.ndarray:
    """
    Calculate Class-Balanced weights based on effective number of samples.
    
    Formula: w_i = (1 - β) / (1 - β^{n_i})
    where n_i is the number of samples in class i
    
    Args:
        class_counts: Dictionary mapping class_id to sample count
        beta: Balance parameter (default 0.9999)
        normalize: Whether to normalize weights to sum to num_classes
        
    Returns:
        Array of weights per class (same order as sorted class_ids)
    """
    num_classes = len(class_counts)
    class_ids = sorted(class_counts.keys())
    
    # Calculate CB weights
    weights = np.zeros(num_classes)
    for idx, class_id in enumerate(class_ids):
        n_i = class_counts[class_id]
        if n_i == 0:
            weights[idx] = 0.0
        else:
            # CB weight formula
            weights[idx] = (1 - beta) / (1 - beta ** n_i)
    
    # Normalize weights
    if normalize and weights.sum() > 0:
        weights = weights * num_classes / weights.sum()
    
    return weights


def calculate_cb_weights_from_coco(
    coco_json_path: str,
    beta: float = 0.9999,
    normalize: bool = True
) -> Dict[str, any]:
    """
    Calculate Class-Balanced weights from COCO JSON file.
    
    Args:
        coco_json_path: Path to COCO format JSON annotation file
        beta: Balance parameter
        normalize: Whether to normalize weights
        
    Returns:
        Dictionary with:
        - 'weights': numpy array of CB weights
        - 'class_counts': dict of class counts
        - 'class_names': list of class names
    """
    import json
    from collections import Counter
    
    # Load COCO JSON
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Count annotations per category
    category_counts = Counter([ann['category_id'] for ann in coco_data['annotations']])
    
    # Get class names in order
    categories = sorted(coco_data['categories'], key=lambda x: x['id'])
    class_names = [cat['name'] for cat in categories]
    
    # Calculate weights
    weights = calculate_cb_weights(category_counts, beta, normalize)
    
    return {
        'weights': weights,
        'class_counts': dict(category_counts),
        'class_names': class_names
    }


def focal_loss_for_detr(
    class_logits: torch.Tensor,
    target_classes: torch.Tensor,
    alpha_weights: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    num_classes: int = None
) -> torch.Tensor:
    """
    Focal loss adapted for DETR's classification heads.
    
    Args:
        class_logits: Predicted logits, shape [batch_size, num_queries, num_classes + 1]
        target_classes: Target class indices, shape [batch_size, num_queries]
        alpha_weights: Per-class alpha weights (CB weights), shape [num_classes]
        gamma: Focusing parameter
        num_classes: Number of object classes (excluding background)
        
    Returns:
        Focal loss value
    """
    if num_classes is None:
        num_classes = class_logits.shape[-1] - 1  # Exclude no-object class
    
    # Get probabilities
    probs = torch.softmax(class_logits, dim=-1)
    
    # Create one-hot encoding
    # target_classes includes no-object class (id = num_classes)
    target_one_hot = F.one_hot(target_classes, num_classes=num_classes + 1).float()
    
    # Calculate cross entropy
    ce_loss = F.cross_entropy(
        class_logits.view(-1, num_classes + 1),
        target_classes.view(-1),
        reduction='none'
    )
    
    # Get probability of true class
    p_t = (probs * target_one_hot).sum(dim=-1).view(-1)
    
    # Focal weight
    focal_weight = (1 - p_t) ** gamma
    
    # Apply per-class alpha if provided
    if alpha_weights is not None:
        # Map target classes to alpha values
        alpha_t = torch.ones_like(target_classes, dtype=torch.float32)
        for class_id in range(num_classes):
            mask = target_classes == class_id
            alpha_t[mask] = alpha_weights[class_id]
        alpha_t = alpha_t.view(-1)
    else:
        alpha_t = 1.0
    
    # Compute focal loss
    focal_loss = alpha_t * focal_weight * ce_loss
    
    return focal_loss.mean()


# Example usage and testing
if __name__ == "__main__":
    # Test CB weight calculation
    class_counts = {
        0: 75,    # KL0 (minority)
        1: 582,   # KL1
        2: 929,   # KL2 (majority)
        3: 396,   # KL3
        4: 225    # KL4
    }
    
    weights = calculate_cb_weights(class_counts, beta=0.9999)
    
    print("Class-Balanced Weights:")
    for class_id, count in class_counts.items():
        print(f"  Class {class_id} (n={count:3d}): weight = {weights[class_id]:.4f}")
    
    print(f"\nWeight ratio (max/min): {weights.max() / weights.min():.2f}")
    
    # Test focal loss
    batch_size, num_queries, num_classes = 2, 100, 5
    class_logits = torch.randn(batch_size, num_queries, num_classes + 1)
    target_classes = torch.randint(0, num_classes + 1, (batch_size, num_queries))
    
    loss = focal_loss_for_detr(
        class_logits,
        target_classes,
        alpha_weights=torch.tensor(weights),
        gamma=2.0,
        num_classes=num_classes
    )
    
    print(f"\nTest Focal Loss: {loss.item():.4f}")
