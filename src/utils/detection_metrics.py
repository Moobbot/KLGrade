"""
Utility functions for computing detection metrics (IoU, mAP, etc.)
"""

import torch
import numpy as np
from typing import List, Tuple, Dict


def box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate IoU between two boxes in [x1, y1, x2, y2] format.

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        IoU score
    """
    # Intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    # Union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def compute_iou_metrics(
    pred_boxes: List[np.ndarray], gt_boxes: List[np.ndarray], iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute IoU-based detection metrics.

    Args:
        pred_boxes: List of predicted boxes, each [N, 4] in [x1, y1, x2, y2] format
        gt_boxes: List of ground truth boxes, each [M, 4]
        iou_threshold: IoU threshold for considering a detection as correct

    Returns:
        Dictionary with metrics: mean_iou, precision, recall, f1
    """
    all_ious = []
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for pred, gt in zip(pred_boxes, gt_boxes):
        if len(pred) == 0 and len(gt) == 0:
            continue

        if len(gt) == 0:
            # All predictions are false positives
            false_positives += len(pred)
            continue

        if len(pred) == 0:
            # All ground truths are false negatives
            false_negatives += len(gt)
            continue

        # Compute IoU matrix
        iou_matrix = np.zeros((len(pred), len(gt)))
        for i, p_box in enumerate(pred):
            for j, g_box in enumerate(gt):
                iou_matrix[i, j] = box_iou(p_box, g_box)

        # Match predictions to ground truths (greedy matching)
        matched_gt = set()
        for i in range(len(pred)):
            max_iou_idx = np.argmax(iou_matrix[i])
            max_iou = iou_matrix[i, max_iou_idx]

            if max_iou >= iou_threshold and max_iou_idx not in matched_gt:
                true_positives += 1
                matched_gt.add(max_iou_idx)
                all_ious.append(max_iou)
            else:
                false_positives += 1

        # Unmatched ground truths are false negatives
        false_negatives += len(gt) - len(matched_gt)

    # Calculate metrics
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    mean_iou = np.mean(all_ious) if all_ious else 0.0

    return {
        "mean_iou": float(mean_iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positives": int(true_positives),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
    }


def normalize_boxes(boxes: torch.Tensor, img_width: int, img_height: int) -> np.ndarray:
    """
    Convert normalized boxes [cx, cy, w, h] to pixel coordinates [x1, y1, x2, y2].

    Args:
        boxes: Tensor of shape [N, 4] with normalized [cx, cy, w, h]
        img_width: Image width
        img_height: Image height

    Returns:
        Array of shape [N, 4] with pixel [x1, y1, x2, y2]
    """
    boxes = boxes.cpu().numpy()
    result = np.zeros_like(boxes)

    # Convert from [cx, cy, w, h] to [x1, y1, x2, y2]
    result[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * img_width  # x1
    result[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * img_height  # y1
    result[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * img_width  # x2
    result[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * img_height  # y2

    return result


def filter_boxes_by_confidence(
    boxes: torch.Tensor, confs: torch.Tensor, threshold: float = 0.5
) -> np.ndarray:
    """
    Filter boxes by confidence threshold.

    Args:
        boxes: Tensor of shape [N, 4]
        confs: Tensor of shape [N] or [N, C]
        threshold: Confidence threshold

    Returns:
        Filtered boxes as numpy array
    """
    if confs.dim() == 2:
        # Multi-class confidence, take max
        confs = confs.max(dim=1)[0]

    mask = confs >= threshold
    filtered_boxes = boxes[mask]

    return filtered_boxes.cpu().numpy()
