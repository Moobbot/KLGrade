"""
Training Script for KIOCMIL with Detection (End-to-End)

REAL Implementation using Hungarian Matching for Detection Loss.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import sys
from tqdm import tqdm
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F

# Add project root to path
# File is at: scripts/training/end_to_end/train_end_to_end.py
# Root is at: ../../../
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.models.kiocmil_with_detection import KiocmilWithDetection
from src.datasets.kiocmil_dataset_end_to_end import (
    KiocmilDatasetEndToEnd,
    collate_end_to_end,
)
from src.datasets.kiocmil_transforms_v2 import get_photometric_transforms


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network."""

    def __init__(
        self, cost_class: float = 1, cost_bbox: float = 5, cost_giou: float = 2
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets, head_type="knee"):
        """
        Params:
            outputs: dict containing 'pred_boxes' (B, N, 4) and 'pred_logits' (B, N, num_classes)
            targets: dict containing 'boxes' (list of Tensors) and 'labels' (list of Tensors)
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
        """
        bs, num_queries = outputs["pred_boxes"].shape[:2]

        # Use the correct head predictions
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        out_prob = outputs["pred_logits"].flatten(
            0, 1
        )  # [batch_size * num_queries, num_classes]

        # Also concat the target labels and boxes
        if head_type == "knee":
            tgt_ids = torch.cat([v["knee_labels"] for v in targets])
            tgt_bbox = torch.cat([v["knee_boxes"] for v in targets])
        else:
            tgt_ids = torch.cat([v["lesion_classes"] for v in targets])
            tgt_bbox = torch.cat([v["lesion_boxes"] for v in targets])

        if len(tgt_bbox) == 0:
            return None

        # Compute the classification cost.
        # out_prob is (B*N, C), tgt_ids is (M,)
        # We want probability of target class
        # cost_class = -prob[target_class]
        # Gather the probability of the TARGET class for each query
        # Note: out_prob here is assumed to be sigmoid probabilities
        # We negate it because matching minimizes cost

        # If out_prob is (Batch*Queries, NumClasses) and tgt_ids is integers
        # We extract column `tgt_ids` from `out_prob`
        # But this requires repeating out_prob for each target?
        # Matrix form: Cost(i, j) = -out_prob[i, tgt_ids[j]]
        out_prob_selector = out_prob[:, tgt_ids]
        cost_class = -out_prob_selector

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        # bbox_iou needs x1y1x2y2
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )

        # Final cost matrix
        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [
            len(v["knee_boxes"]) if head_type == "knee" else len(v["lesion_boxes"])
            for v in targets
        ]
        indices = [
            linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
        ]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]


class SetCriterion(nn.Module):
    """This class computes the loss for DETR."""

    def __init__(self, num_classes, matcher, weight_dict, losses):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses

    def loss_labels(self, outputs, targets, indices, num_boxes, head_type="knee"):
        """Classification loss (NLL)"""
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]
        # src_logits is (B, N, C) - these are Sigmoid Probabilities in our simplified head

        idx = self._get_src_permutation_idx(indices)

        if head_type == "knee":
            target_classes_o = torch.cat(
                [t["knee_labels"][J] for t, (_, J) in zip(targets, indices)]
            ).to(src_logits.device)
            # Knee head only has 1 class (idx 0), but target_labels generated as 0
        else:
            target_classes_o = torch.cat(
                [t["lesion_classes"][J] for t, (_, J) in zip(targets, indices)]
            ).to(src_logits.device)

        target_classes = torch.full(
            src_logits.shape[:2],
            fill_value=0,
            dtype=torch.float32,
            device=src_logits.device,
        )
        # However, for 100% background, we want target 0.
        # But here we have multi-class for lesion.
        # Let's use BCE.

        target_classes_onehot = torch.zeros_like(src_logits)

        # Determine positive indices
        # idx is (Batch_idx, Query_idx)
        # We set target class prob to 1 at these indices
        if len(target_classes_o) > 0:
            # If knee (1 class), just set index 0 to 1
            if head_type == "knee":
                target_classes_onehot[idx] = 1.0
            else:
                # For lesion, set specific class index
                # idx is tuple (b, q)
                # target_classes_o contains class IDs
                target_classes_onehot[idx[0], idx[1], target_classes_o] = 1.0

        loss_ce = F.binary_cross_entropy(src_logits, target_classes_onehot)
        # Weighting: simple BCE averages over all N queries.
        # Background is dominant. Maybe Focal Loss better?
        # For simplicity, just use BCE.
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, head_type="knee"):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]

        if head_type == "knee":
            target_boxes = torch.cat(
                [t["knee_boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
            )
        else:
            target_boxes = torch.cat(
                [t["lesion_boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
            )

        if len(target_boxes) == 0:
            return {
                "loss_bbox": torch.tensor(0.0).to(src_boxes.device),
                "loss_giou": torch.tensor(0.0).to(src_boxes.device),
            }

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "boxes": self.loss_boxes,
        }
        assert loss in loss_map, f"do not know {loss}"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, head_type="knee"):
        """This performs the loss computation."""
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets, head_type)
        indices = self.matcher(outputs, targets, head_type)
        if indices is None:  # No targets
            return {
                "loss_ce": torch.tensor(0.0).to(outputs["pred_boxes"].device),
                "loss_bbox": torch.tensor(0.0).to(outputs["pred_boxes"].device),
                "loss_giou": torch.tensor(0.0).to(outputs["pred_boxes"].device),
            }

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        if head_type == "knee":
            num_boxes = sum(len(t["knee_boxes"]) for t in targets)
        else:
            num_boxes = sum(len(t["lesion_boxes"]) for t in targets)

        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        num_boxes = torch.clamp(num_boxes / 1, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(
                    loss, outputs, targets, indices, num_boxes, head_type=head_type
                )
            )

        return losses


class EndToEndTrainer:
    """Trainer for end-to-end KIOCMIL with detection."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device="cuda",
        learning_rate=1e-3,
        save_dir="runs/end_to_end",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Matcher & Criterion
        self.matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
        weight_dict = {"loss_ce": 1, "loss_bbox": 5, "loss_giou": 2}
        losses = ["labels", "boxes"]
        self.criterion = SetCriterion(
            10, self.matcher, weight_dict, losses
        )  # num_classes not really used in init

        # Optimizer
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=1e-4,
        )

        self.best_loss = float("inf")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        # Warmup for detection: Skip classification module for first 5 epochs
        # This prevents OOM due to garbage detections generating too many patches
        warmup = epoch <= 5

        for images, targets in pbar:
            images = images.to(self.device)
            # Targets to device
            targets_gpu = []
            for t in targets:
                t_gpu = {}
                t_gpu["knee_boxes"] = t["knee_boxes"].to(self.device)
                t_gpu["knee_labels"] = torch.zeros(
                    len(t["knee_boxes"]), dtype=torch.long, device=self.device
                )  # Class 0 always
                t_gpu["lesion_boxes"] = t["lesion_boxes"].to(self.device)
                t_gpu["lesion_classes"] = t["lesion_classes"].to(self.device)
                t_gpu["label"] = t["label"].to(self.device)
                targets_gpu.append(t_gpu)

            # Forward
            outputs = self.model(images, skip_classification=warmup)

            # 1. Detection Losses
            # Knee
            knee_out = {
                "pred_boxes": outputs["knee_boxes"],
                "pred_logits": outputs["knee_confs"],
            }
            loss_dict_knee = self.criterion(knee_out, targets_gpu, head_type="knee")
            loss_knee = sum(
                loss_dict_knee[k] * self.criterion.weight_dict[k]
                for k in loss_dict_knee.keys()
            )

            # Lesion
            lesion_out = {
                "pred_boxes": outputs["lesion_boxes"],
                "pred_logits": outputs["lesion_confs"],
            }
            loss_dict_lesion = self.criterion(
                lesion_out, targets_gpu, head_type="lesion"
            )
            loss_lesion = sum(
                loss_dict_lesion[k] * self.criterion.weight_dict[k]
                for k in loss_dict_lesion.keys()
            )

            # 2. Classification Loss (KIOCMIL)
            loss_cls = torch.tensor(0.0, device=self.device)

            if not warmup:
                # We need to map batch items correctly. Model outputs logits with batch order preserved.
                global_labels = torch.stack([t["label"] for t in targets_gpu])

                # Simple CE loss
                loss_cls = F.cross_entropy(outputs["logits_10"], global_labels)

            # Total Loss
            loss = loss_knee + loss_lesion + loss_cls

            if loss.requires_grad:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
                self.optimizer.step()
            else:
                # No gradients to compute (e.g. empty batch in warmup)
                pass

            total_loss += loss.item()
            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.2f}",
                    "Kn": f"{loss_knee.item():.2f}",
                    "Les": f"{loss_lesion.item():.2f}",
                    "Cls": f"{loss_cls.item():.2f}",
                }
            )
        return total_loss / len(self.train_loader)

    def save_checkpoint(self, epoch, loss):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
        }
        torch.save(checkpoint, self.save_dir / "latest.pt")
        if loss < self.best_loss:
            self.best_loss = loss
            torch.save(checkpoint, self.save_dir / "best.pt")

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0

        # Validation loop
        # We don't skip classification in validation usually,
        # but if model is very early, loss might be high.
        # Let's assume validation is full forward pass.

        pbar = tqdm(self.val_loader, desc=f"Val Epoch {epoch}")

        for images, targets in pbar:
            images = images.to(self.device)
            # Targets to device
            targets_gpu = []
            for t in targets:
                t_gpu = {}
                t_gpu["knee_boxes"] = t["knee_boxes"].to(self.device)
                t_gpu["knee_labels"] = torch.zeros(
                    len(t["knee_boxes"]), dtype=torch.long, device=self.device
                )
                t_gpu["lesion_boxes"] = t["lesion_boxes"].to(self.device)
                t_gpu["lesion_classes"] = t["lesion_classes"].to(self.device)
                t_gpu["label"] = t["label"].to(self.device)
                targets_gpu.append(t_gpu)

            # Forward
            outputs = self.model(images)  # No skip in validation

            # 1. Detection Losses
            knee_out = {
                "pred_boxes": outputs["knee_boxes"],
                "pred_logits": outputs["knee_confs"],
            }
            loss_dict_knee = self.criterion(knee_out, targets_gpu, head_type="knee")
            loss_knee = sum(
                loss_dict_knee[k] * self.criterion.weight_dict[k]
                for k in loss_dict_knee.keys()
            )

            lesion_out = {
                "pred_boxes": outputs["lesion_boxes"],
                "pred_logits": outputs["lesion_confs"],
            }
            loss_dict_lesion = self.criterion(
                lesion_out, targets_gpu, head_type="lesion"
            )
            loss_lesion = sum(
                loss_dict_lesion[k] * self.criterion.weight_dict[k]
                for k in loss_dict_lesion.keys()
            )

            # 2. Classification Loss
            if "logits_10" in outputs:
                global_labels = torch.stack([t["label"] for t in targets_gpu])
                loss_cls = F.cross_entropy(outputs["logits_10"], global_labels)
            else:
                # Should not happen if skip_classification=False
                loss_cls = torch.tensor(0.0, device=self.device)

            loss = loss_knee + loss_lesion + loss_cls
            total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self, num_epochs, patience=10):
        print(f"Starting training for {num_epochs} epochs with patience {patience}...")
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)

            print(
                f"Epoch {epoch} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}"
            )

            # Checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": val_loss,
            }
            torch.save(checkpoint, self.save_dir / "latest.pt")

            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(checkpoint, self.save_dir / "best.pt")
                print(f"✅ New best model saved (Val Loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"⚠️ No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("🛑 Early stopping triggered.")
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="yolo11l")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--save-dir", type=str, default="runs/end_to_end/full_training")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--patience", type=int, default=15)

    # Train Dataset args
    parser.add_argument("--train-img-dir", required=True)
    parser.add_argument("--train-knee-label-dir", required=True)
    parser.add_argument("--train-lesion-label-dir", required=True)
    parser.add_argument("--train-split-file", required=True)

    # Val Dataset args
    parser.add_argument("--val-img-dir", required=True)
    parser.add_argument("--val-knee-label-dir", required=True)
    parser.add_argument("--val-lesion-label-dir", required=True)
    parser.add_argument("--val-split-file", required=True)

    args = parser.parse_args()

    # Model
    model = KiocmilWithDetection(
        backbone_name=args.backbone,
        num_classes=args.num_classes,
        pretrained_kiocmil=None,  # Train everything
        freeze_kiocmil=False,
    )

    # Train Dataset
    train_dataset = KiocmilDatasetEndToEnd(
        img_dir=args.train_img_dir,
        knee_label_dir=args.train_knee_label_dir,
        lesion_label_dir=args.train_lesion_label_dir,
        split_file=args.train_split_file,
    )

    # Val Dataset
    val_dataset = KiocmilDatasetEndToEnd(
        img_dir=args.val_img_dir,
        knee_label_dir=args.val_knee_label_dir,
        lesion_label_dir=args.val_lesion_label_dir,
        split_file=args.val_split_file,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_end_to_end,
        num_workers=4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_end_to_end,
        num_workers=4,
    )

    trainer = EndToEndTrainer(
        model,
        train_loader,
        val_loader,
        save_dir=args.save_dir,
    )

    trainer.train(args.epochs, patience=args.patience)


if __name__ == "__main__":
    main()
