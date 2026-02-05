"""
Training Script for KIOCMIL with Detection (End-to-End)

This script trains the end-to-end model that combines:
1. Knee detection
2. Lesion detection
3. KL grade classification

Training is done in 2 phases:
- Phase 1: Train detection heads only (KIOCMIL frozen)
- Phase 2: Fine-tune entire model end-to-end
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.kiocmil_with_detection import KiocmilWithDetection
from src.datasets.kiocmil_dataset_v3 import KiocmilDatasetV3, collate_kiocmil_v3
from src.datasets.kiocmil_transforms_v2 import get_photometric_transforms


class DetectionLoss(nn.Module):
    """
    Loss function for detection head.

    Combines:
    - Bounding box regression loss (IoU or smooth L1)
    - Confidence loss (BCE)
    """

    def __init__(self, bbox_weight=1.0, conf_weight=1.0):
        super().__init__()
        self.bbox_weight = bbox_weight
        self.conf_weight = conf_weight
        self.bce_loss = nn.BCELoss()
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(self, pred_boxes, pred_confs, target_boxes, target_confs):
        """
        Args:
            pred_boxes: (B, N, 4) predicted boxes
            pred_confs: (B, N, C) predicted confidences
            target_boxes: (B, M, 4) target boxes
            target_confs: (B, M, C) target confidences
        """
        # For simplicity, use smooth L1 for boxes and BCE for confidence
        # In production, use proper matching (Hungarian algorithm)

        # Dummy implementation - needs proper box matching
        bbox_loss = self.smooth_l1(pred_boxes.mean(), target_boxes.mean())
        conf_loss = self.bce_loss(pred_confs.mean(), target_confs.mean())

        total_loss = self.bbox_weight * bbox_loss + self.conf_weight * conf_loss

        return total_loss, {
            "bbox_loss": bbox_loss.item(),
            "conf_loss": conf_loss.item(),
        }


class EndToEndTrainer:
    """Trainer for end-to-end KIOCMIL with detection."""

    def __init__(
        self,
        model: KiocmilWithDetection,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
        learning_rate: float = 1e-3,
        save_dir: str = "runs/end_to_end",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Loss functions
        self.detection_loss = DetectionLoss()
        self.classification_loss = nn.CrossEntropyLoss()

        # Optimizer (only for unfrozen parameters)
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
        )

        # Metrics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")

    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()

        total_loss = 0
        total_det_loss = 0
        total_cls_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch_data in enumerate(pbar):
            # Note: batch_data is from KiocmilDatasetV3
            # It contains: {'knees': [...], 'label': ...}

            # For now, create dummy images since dataset doesn't provide them
            # In production, modify dataset to return images
            # Using 256x256 to reduce memory usage (optimized from 640x640)
            B = len(batch_data)
            dummy_images = torch.randn(B, 3, 256, 256, device=self.device)

            # Forward pass
            outputs = self.model(dummy_images)

            # Get labels
            labels = torch.tensor(
                [item["label"] for item in batch_data], device=self.device
            )

            # Compute losses
            # Detection loss (dummy for now - needs proper targets)
            det_loss = torch.tensor(0.0, device=self.device)

            # Classification loss
            cls_loss = self.classification_loss(outputs["logits_10"], labels)

            # Total loss
            loss = det_loss + cls_loss

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            total_det_loss += det_loss.item()
            total_cls_loss += cls_loss.item()

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "cls": f"{cls_loss.item():.4f}",
                }
            )

        avg_loss = total_loss / len(self.train_loader)
        avg_det_loss = total_det_loss / len(self.train_loader)
        avg_cls_loss = total_cls_loss / len(self.train_loader)

        return {
            "loss": avg_loss,
            "det_loss": avg_det_loss,
            "cls_loss": avg_cls_loss,
        }

    def validate(self, epoch: int):
        """Validate model."""
        self.model.eval()

        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch_data in tqdm(self.val_loader, desc=f"Val {epoch}"):
                # Create dummy images (256x256 for memory efficiency)
                B = len(batch_data)
                dummy_images = torch.randn(B, 3, 256, 256, device=self.device)

                # Forward
                outputs = self.model(dummy_images)

                # Labels
                labels = torch.tensor(
                    [item["label"] for item in batch_data], device=self.device
                )

                # Loss
                cls_loss = self.classification_loss(outputs["logits_10"], labels)
                total_loss += cls_loss.item()

                # Accuracy
                preds = torch.argmax(outputs["logits_10"], dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += B

        avg_loss = total_loss / len(self.val_loader)
        accuracy = total_correct / total_samples

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
        }

    def save_checkpoint(self, epoch: int, metrics: dict):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }

        # Save latest
        torch.save(checkpoint, self.save_dir / "latest.pt")

        # Save best
        if metrics["val_loss"] < self.best_val_loss:
            self.best_val_loss = metrics["val_loss"]
            torch.save(checkpoint, self.save_dir / "best.pt")
            print(f"✅ Saved best model (val_loss: {metrics['val_loss']:.4f})")

    def train(self, num_epochs: int):
        """Train for multiple epochs."""
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"Save directory: {self.save_dir}")
        print(f"{'='*60}\n")

        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 60)

            # Train
            train_metrics = self.train_epoch(epoch)
            print(
                f"Train - Loss: {train_metrics['loss']:.4f}, "
                f"Cls: {train_metrics['cls_loss']:.4f}"
            )

            # Validate
            val_metrics = self.validate(epoch)
            print(
                f"Val   - Loss: {val_metrics['loss']:.4f}, "
                f"Acc: {val_metrics['accuracy']:.4f}"
            )

            # Save
            metrics = {
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
            self.save_checkpoint(epoch, metrics)

            # Save metrics history
            self.train_losses.append(train_metrics["loss"])
            self.val_losses.append(val_metrics["loss"])

        # Save final metrics
        with open(self.save_dir / "metrics.json", "w") as f:
            json.dump(
                {
                    "train_losses": self.train_losses,
                    "val_losses": self.val_losses,
                },
                f,
                indent=2,
            )

        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Train End-to-End Model")

    # Model args
    parser.add_argument(
        "--backbone", type=str, default="yolo11l", help="Backbone model"
    )
    parser.add_argument("--num-classes", type=int, default=10, help="Number of classes")
    parser.add_argument(
        "--pretrained-kiocmil",
        type=str,
        default="runs/kiocmil_cada/cada_10class_balanced/best_acc_model.pt",
        help="Pretrained KIOCMIL checkpoint",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train from scratch without pretrained KIOCMIL weights",
    )
    parser.add_argument(
        "--freeze-kiocmil",
        action="store_true",
        default=True,
        help="Freeze KIOCMIL weights (Phase 1)",
    )

    # Data args
    parser.add_argument(
        "--train-img-dir",
        type=str,
        default="datasets/dataset_v0/images",
        help="Training images directory",
    )
    parser.add_argument(
        "--train-knee-label-dir",
        type=str,
        default="datasets/dataset_v0/labels",
        help="Training knee labels directory",
    )
    parser.add_argument(
        "--train-lesion-label-dir",
        type=str,
        default="datasets/dataset_v0/labels",
        help="Training lesion labels directory",
    )
    parser.add_argument(
        "--train-split-file",
        type=str,
        default="datasets/dataset_v0/train.txt",
        help="Training split file",
    )
    parser.add_argument(
        "--val-split-file",
        type=str,
        default="datasets/dataset_v0/val.txt",
        help="Validation split file",
    )

    # Training args
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="runs/end_to_end/test_5epochs",
        help="Save directory",
    )

    args = parser.parse_args()

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model
    print("\nInitializing model...")
    model = KiocmilWithDetection(
        backbone_name=args.backbone,
        num_classes=args.num_classes,
        pretrained_kiocmil=None if args.no_pretrained else args.pretrained_kiocmil,
        freeze_kiocmil=args.freeze_kiocmil,
    )
    print(f"✅ Model initialized")

    # Data
    print("\nLoading datasets...")
    transform = get_photometric_transforms(level="light", use_clahe=True)

    train_dataset = KiocmilDatasetV3(
        img_dir=args.train_img_dir,
        knee_label_dir=args.train_knee_label_dir,
        lesion_label_dir=args.train_lesion_label_dir,
        split_file=args.train_split_file,
        geometric_transform=None,
        photometric_transform=transform,
        ctx_size=(384, 384),
        patch_size=(224, 224),
    )

    val_dataset = KiocmilDatasetV3(
        img_dir=args.train_img_dir,
        knee_label_dir=args.train_knee_label_dir,
        lesion_label_dir=args.train_lesion_label_dir,
        split_file=args.val_split_file,
        geometric_transform=None,
        photometric_transform=transform,
        ctx_size=(384, 384),
        patch_size=(224, 224),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_kiocmil_v3,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_kiocmil_v3,
    )

    print(f"✅ Train dataset: {len(train_dataset)} samples")
    print(f"✅ Val dataset: {len(val_dataset)} samples")

    # Trainer
    trainer = EndToEndTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.lr,
        save_dir=args.save_dir,
    )

    # Train
    trainer.train(num_epochs=args.epochs)


if __name__ == "__main__":
    main()
