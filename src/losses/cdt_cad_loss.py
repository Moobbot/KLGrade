"""
CDT-CAD Loss Functions

Implements:
- Hungarian Matcher for optimal assignment
- Composite Loss (Classification + L1 + GIoU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple

from .giou_loss import GIoULoss, box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """
    Hungarian Matcher for optimal bipartite matching between predictions and ground truth

    Finds the optimal one-to-one assignment that minimizes the total matching cost.
    """

    def __init__(
        self, cost_class: float = 2.0, cost_bbox: float = 5.0, cost_giou: float = 2.0
    ):
        """
        Args:
            cost_class: Weight for classification cost
            cost_bbox: Weight for L1 bbox cost
            cost_giou: Weight for GIoU cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(
        self, outputs: Dict[str, Tensor], targets: List[Dict[str, Tensor]]
    ) -> List[Tuple[Tensor, Tensor]]:
        """
        Args:
            outputs: Model outputs
                - 'pred_logits': [B, num_queries, num_classes+1]
                - 'pred_boxes': [B, num_queries, 4]
            targets: List of target dicts (one per image)
                - 'labels': [num_objects]
                - 'boxes': [num_objects, 4] in normalized (cx, cy, w, h) format

        Returns:
            indices: List of (pred_idx, target_idx) tuples for each image
        """
        B, num_queries = outputs["pred_logits"].shape[:2]

        # Flatten predictions
        out_prob = (
            outputs["pred_logits"].flatten(0, 1).softmax(-1)
        )  # [B*num_queries, num_classes+1]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [B*num_queries, 4]

        # Flatten targets
        tgt_ids = torch.cat([t["labels"] for t in targets])
        tgt_bbox = torch.cat([t["boxes"] for t in targets])

        # Compute classification cost (negative log-likelihood)
        cost_class = -out_prob[:, tgt_ids]

        # Compute L1 cost
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute GIoU cost
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )

        # Final cost matrix
        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        C = C.view(B, num_queries, -1).cpu()

        # Perform Hungarian matching for each image
        sizes = [len(t["labels"]) for t in targets]
        indices = []

        for i, (c, _size) in enumerate(zip(C.split(sizes, -1), sizes)):
            # c: [num_queries, num_objects_in_image]
            pred_idx, tgt_idx = linear_sum_assignment(c[i])
            indices.append(
                (
                    torch.as_tensor(pred_idx, dtype=torch.int64),
                    torch.as_tensor(tgt_idx, dtype=torch.int64),
                )
            )

        return indices


class CDTCADLoss(nn.Module):
    """
    Composite loss for CDT-CAD training

    Combines:
    - Classification loss (Cross Entropy or Focal Loss)
    - L1 bbox regression loss
    - GIoU loss
    """

    def __init__(
        self,
        num_classes: int,
        weight_class: float = 2.0,
        weight_bbox: float = 5.0,
        weight_giou: float = 2.0,
        eos_coef: float = 0.1,  # Weight for "no object" class
    ):
        """
        Args:
            num_classes: Number of object classes (excluding "no object")
            weight_class: Weight for classification loss
            weight_bbox: Weight for L1 loss
            weight_giou: Weight for GIoU loss
            eos_coef: Weight for "no object" class in classification
        """
        super().__init__()

        self.num_classes = num_classes
        self.weight_class = weight_class
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou

        self.matcher = HungarianMatcher(
            cost_class=weight_class, cost_bbox=weight_bbox, cost_giou=weight_giou
        )

        self.giou_loss = GIoULoss()

        # Class weights (lower weight for "no object" class)
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def forward(
        self, outputs: Dict[str, Tensor], targets: List[Dict[str, Tensor]]
    ) -> Dict[str, Tensor]:
        """
        Args:
            outputs: Model outputs
                - 'pred_logits': [B, num_queries, num_classes+1]
                - 'pred_boxes': [B, num_queries, 4]
            targets: List of target dicts (one per image)
                - 'labels': [num_objects]
                - 'boxes': [num_objects, 4] in normalized (cx, cy, w, h) format

        Returns:
            losses: Dictionary of losses
                - 'loss_ce': Classification loss
                - 'loss_bbox': L1 bbox loss
                - 'loss_giou': GIoU loss
                - 'loss': Total loss
        """
        # Get optimal matching
        indices = self.matcher(outputs, targets)

        # Number of target objects
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=outputs["pred_logits"].device
        )
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # ===== Classification Loss =====
        loss_ce = self._loss_labels(outputs, targets, indices, num_boxes)

        # ===== Bbox Losses =====
        loss_bbox, loss_giou = self._loss_boxes(outputs, targets, indices, num_boxes)

        # ===== Total Loss =====
        losses = {
            "loss_ce": loss_ce,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
            "loss": self.weight_class * loss_ce
            + self.weight_bbox * loss_bbox
            + self.weight_giou * loss_giou,
        }

        return losses

    def _loss_labels(
        self,
        outputs: Dict[str, Tensor],
        targets: List[Dict[str, Tensor]],
        indices: List[Tuple[Tensor, Tensor]],
        num_boxes: float,
    ) -> Tensor:
        """Classification loss (Cross Entropy)"""
        pred_logits = outputs["pred_logits"]  # [B, num_queries, num_classes+1]

        # Build target classes
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )

        target_classes = torch.full(
            pred_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=pred_logits.device,
        )  # Initialize all as "no object"

        target_classes[idx] = target_classes_o

        # Cross entropy loss
        loss_ce = F.cross_entropy(
            pred_logits.transpose(1, 2), target_classes, weight=self.empty_weight
        )

        return loss_ce

    def _loss_boxes(
        self,
        outputs: Dict[str, Tensor],
        targets: List[Dict[str, Tensor]],
        indices: List[Tuple[Tensor, Tensor]],
        num_boxes: float,
    ) -> Tuple[Tensor, Tensor]:
        """Bbox regression losses (L1 + GIoU)"""
        idx = self._get_src_permutation_idx(indices)

        pred_boxes = outputs["pred_boxes"][idx]  # [num_matched, 4]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        # L1 loss
        loss_bbox = F.l1_loss(pred_boxes, target_boxes, reduction="none")
        loss_bbox = loss_bbox.sum() / num_boxes

        # GIoU loss
        loss_giou = self.giou_loss(pred_boxes, target_boxes)

        return loss_bbox, loss_giou

    def _get_src_permutation_idx(
        self, indices: List[Tuple[Tensor, Tensor]]
    ) -> Tuple[Tensor, Tensor]:
        """Get permutation indices for matched predictions"""
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
