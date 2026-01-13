"""
GIoU (Generalized Intersection over Union) Loss

Improves upon standard IoU loss by also considering the smallest enclosing box.
Better handles objects of different scales.
"""

import torch
import torch.nn as nn
from torch import Tensor


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """
    Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format

    Args:
        boxes: [N, 4] in (cx, cy, w, h) format

    Returns:
        boxes_xyxy: [N, 4] in (x1, y1, x2, y2) format
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_area(boxes: Tensor) -> Tensor:
    """
    Compute area of boxes

    Args:
        boxes: [N, 4] in (x1, y1, x2, y2) format

    Returns:
        area: [N]
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Compute IoU between two sets of boxes

    Args:
        boxes1: [N, 4] in (x1, y1, x2, y2) format
        boxes2: [M, 4] in (x1, y1, x2, y2) format

    Returns:
        iou: [N, M]
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # Intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    # Union
    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Compute Generalized IoU between two sets of boxes

    GIoU = IoU - |C \ (A ∪ B)| / |C|
    where C is the smallest enclosing box

    Args:
        boxes1: [N, 4] in (x1, y1, x2, y2) format
        boxes2: [M, 4] in (x1, y1, x2, y2) format

    Returns:
        giou: [N, M]
    """
    # Regular IoU
    iou = box_iou(boxes1, boxes2)

    # Area of boxes
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # Smallest enclosing box
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    area_c = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    # Union area
    inter = iou * (area1[:, None] + area2 - iou * (area1[:, None] + area2))
    union = area1[:, None] + area2 - inter

    # GIoU
    giou = iou - (area_c - union) / area_c

    return giou


class GIoULoss(nn.Module):
    """
    Generalized IoU Loss

    Loss = 1 - GIoU
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred_boxes: Tensor, target_boxes: Tensor) -> Tensor:
        """
        Args:
            pred_boxes: Predicted boxes [N, 4] in (cx, cy, w, h) format
            target_boxes: Target boxes [N, 4] in (cx, cy, w, h) format

        Returns:
            loss: Scalar loss value
        """
        # Convert to xyxy format
        pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)

        # Compute GIoU
        giou = torch.diag(generalized_box_iou(pred_boxes_xyxy, target_boxes_xyxy))

        # Loss = 1 - GIoU
        loss = 1 - giou

        return loss.mean()
