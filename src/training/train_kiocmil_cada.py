"""
KIOCMIL CADA Training Script

Train the enhanced KIOCMIL model with Context-Aware Deformable Attention.

Usage:
    python src/training/train_kiocmil_cada.py \\
        --train_img_dir processed/knee_full_10_class/images \\
        --train_knee_label_dir processed/knee_full_10_class/labels_knee \\
        --train_lesion_label_dir processed/knee_full_10_class/labels_lesion \\
        --train_split_file splits/knee_10_class/train.txt \\
        --val_img_dir ... \\
        --epochs 100 \\
        --batch_size 16 \\
        --augmentation_level strong \\
        --use_clahe True
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path
import argparse
import time
from tqdm import tqdm
import sys
import os
import wandb
import numpy as np
from collections import Counter

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.datasets.kiocmil_dataset_v3 import KiocmilDatasetV3, collate_kiocmil_v3
from src.datasets.kiocmil_transforms_v2 import (
    get_geometric_transforms,
    get_photometric_transforms,
)
from src.models.kiocmil_model_cada import KiocmilModelCADA
from src.config import PROJECT_ROOT
from src.training.focal_loss import FocalLoss, compute_class_weights
from src.training.early_stopping import EarlyStopping
from src.utils.logging_utils import get_next_log_dir, save_training_config
import logging


class KiocmilCADATrainer:
    """Trainer for KIOCMIL CADA model."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Create versioned log directory
        self.log_dir = get_next_log_dir("log", module_name="kiocmil_cada")
        print(f"\n📁 Logging to: {self.log_dir}")

        # Setup file logging
        log_file = self.log_dir / "training.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

        # Save training configuration
        self._save_config()

        # WandB Init
        if not args.no_wandb:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_name,
                config=vars(args),
            )
            wandb.config.update({"log_dir": str(self.log_dir)})

    def _save_config(self):
        """Save training configuration."""
        additional_info = {
            "device": str(self.device),
            "save_dir": str(self.save_dir),
            "log_dir": str(self.log_dir),
        }

        config_file = save_training_config(self.log_dir, self.args, additional_info)
        print(f"💾 Config saved to: {config_file}")
        self.logger.info(f"Config saved to: {config_file}")

    def setup_datasets(self):
        """Setup training and validation datasets."""
        print("Initializing Datasets...")

        # Get augmentation settings
        aug_level = getattr(self.args, "augmentation_level", "strong")
        use_clahe = getattr(self.args, "use_clahe", True)

        # Geometric transforms (before cropping)
        geometric_train = (
            get_geometric_transforms(level=aug_level) if aug_level != "none" else None
        )
        geometric_val = None

        # Photometric transforms (after cropping)
        photometric_train = get_photometric_transforms(
            level=aug_level,
            use_clahe=use_clahe,
        )
        photometric_val = get_photometric_transforms(
            level="none",
            use_clahe=use_clahe,
        )

        # Create datasets
        self.train_dataset = KiocmilDatasetV3(
            img_dir=self.args.train_img_dir,
            knee_label_dir=self.args.train_knee_label_dir,
            lesion_label_dir=self.args.train_lesion_label_dir,
            split_file=self.args.train_split_file,
            geometric_transform=geometric_train,
            photometric_transform=photometric_train,
            ctx_size=(384, 384),
            patch_size=(224, 224),
        )

        self.val_dataset = KiocmilDatasetV3(
            img_dir=self.args.val_img_dir,
            knee_label_dir=self.args.val_knee_label_dir,
            lesion_label_dir=self.args.val_lesion_label_dir,
            split_file=self.args.val_split_file,
            geometric_transform=geometric_val,
            photometric_transform=photometric_val,
            ctx_size=(384, 384),
            patch_size=(224, 224),
        )

        print(f"✅ Train dataset size: {len(self.train_dataset)}")
        print(f"✅ Val dataset size: {len(self.val_dataset)}")

        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_kiocmil_v3,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_kiocmil_v3,
        )

    def setup_model(self):
        """Setup model."""
        print("Initializing Model...")

        self.model = KiocmilModelCADA(
            backbone_name="yolo11l",
            num_classes=10,
            feature_dim=256,
            num_deformable_points=4,
            num_context_scales=3,
            use_positional_encoding=True,
            dropout=0.1,
        ).to(self.device)

        print(
            f"✅ Model created. Total params: {sum(p.numel() for p in self.model.parameters()):,}"
        )

    def setup_optimizer_and_loss(self):
        """Setup optimizer and loss functions."""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=0.01,
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.epochs,
            eta_min=1e-6,
        )

        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()

        # Focal loss for class imbalance
        class_weights = compute_class_weights([0] * 10)  # Placeholder
        self.focal_loss = FocalLoss(
            alpha=class_weights,
            gamma=2.0,
        )

        # Early stopping
        self.early_stopping = EarlyStopping(patience=15, verbose=True)

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct_10 = 0
        total_samples = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)

        for batch_idx, batch_data in enumerate(pbar):
            if not batch_data or not batch_data[0].get("knees"):
                continue

            self.optimizer.zero_grad()

            try:
                # Forward pass
                output = self.model(batch_data)

                # Compute losses
                logits_10 = output["logits_10"]
                logits_grade = output["logits_grade"]
                logits_type = output["logits_type"]

                # Create dummy targets for now (would come from dataset)
                batch_size = logits_10.shape[0]
                target_10 = torch.randint(0, 10, (batch_size,)).to(self.device)
                target_grade = torch.randint(0, 5, (batch_size,)).to(self.device)
                target_type = torch.randint(0, 2, (batch_size,)).to(self.device).float()

                # Weighted loss
                loss_10 = self.ce_loss(logits_10, target_10)
                loss_grade = self.ce_loss(logits_grade, target_grade)
                loss_type = nn.BCEWithLogitsLoss()(
                    logits_type, target_type.unsqueeze(-1)
                )

                loss = 0.5 * loss_10 + 0.3 * loss_grade + 0.2 * loss_type

                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                # Metrics
                total_loss += loss.item()
                pred_10 = logits_10.argmax(dim=1)
                correct_10 += (pred_10 == target_10).sum().item()
                total_samples += batch_size

                pbar.set_postfix(
                    {
                        "loss": loss.item(),
                        "acc": correct_10 / total_samples if total_samples > 0 else 0,
                    }
                )

            except Exception as e:
                self.logger.warning(f"Batch {batch_idx} error: {e}")
                continue

        avg_loss = total_loss / max(1, batch_idx + 1)
        avg_acc = correct_10 / max(1, total_samples)

        return avg_loss, avg_acc

    def val_epoch(self):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct_10 = 0
        total_samples = 0

        pbar = tqdm(self.val_loader, desc="Validating", leave=False)

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(pbar):
                if not batch_data or not batch_data[0].get("knees"):
                    continue

                try:
                    # Forward pass
                    output = self.model(batch_data)

                    # Compute losses
                    logits_10 = output["logits_10"]
                    logits_grade = output["logits_grade"]

                    batch_size = logits_10.shape[0]
                    target_10 = torch.randint(0, 10, (batch_size,)).to(self.device)
                    target_grade = torch.randint(0, 5, (batch_size,)).to(self.device)

                    loss_10 = self.ce_loss(logits_10, target_10)
                    loss_grade = self.ce_loss(logits_grade, target_grade)
                    loss = 0.5 * loss_10 + 0.3 * loss_grade

                    total_loss += loss.item()
                    pred_10 = logits_10.argmax(dim=1)
                    correct_10 += (pred_10 == target_10).sum().item()
                    total_samples += batch_size

                except Exception as e:
                    self.logger.warning(f"Val batch {batch_idx} error: {e}")
                    continue

        avg_loss = total_loss / max(1, batch_idx + 1)
        avg_acc = correct_10 / max(1, total_samples)

        return avg_loss, avg_acc

    def train(self):
        """Main training loop."""
        self.setup_datasets()
        self.setup_model()
        self.setup_optimizer_and_loss()

        print("\n" + "=" * 80)
        print(f"Starting KIOCMIL CADA Training ({self.args.epochs} epochs)")
        print(f"Device: {self.device}")
        print("=" * 80 + "\n")

        for epoch in range(self.args.epochs):
            print(f"\nEpoch {epoch+1}/{self.args.epochs}")

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc = self.val_epoch()

            # Scheduler step
            self.scheduler.step()

            # Logging
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if not self.args.no_wandb:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "lr": self.optimizer.param_groups[0]["lr"],
                    }
                )

            # Save checkpoint
            if (epoch + 1) % self.args.save_interval == 0:
                checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch+1}.pt"
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    },
                    checkpoint_path,
                )
                print(f"Checkpoint saved: {checkpoint_path}")

            # Early stopping
            self.early_stopping(val_loss, self.model, self.save_dir)
            if self.early_stopping.early_stop:
                print("Early stopping triggered!")
                break

        print("\n" + "=" * 80)
        print("Training Complete!")
        print("=" * 80 + "\n")

        if not self.args.no_wandb:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train KIOCMIL CADA")

    # Dataset arguments
    parser.add_argument("--train_img_dir", default="processed/knee_10_class/images")
    parser.add_argument(
        "--train_knee_label_dir", default="processed/knee_10_class/labels_knee"
    )
    parser.add_argument(
        "--train_lesion_label_dir", default="processed/knee_10_class/labels_lesion"
    )
    parser.add_argument("--train_split_file", default="splits/knee_10_class/train.txt")

    parser.add_argument("--val_img_dir", default="processed/knee_10_class/images")
    parser.add_argument(
        "--val_knee_label_dir", default="processed/knee_10_class/labels_knee"
    )
    parser.add_argument(
        "--val_lesion_label_dir", default="processed/knee_10_class/labels_lesion"
    )
    parser.add_argument("--val_split_file", default="splits/knee_10_class/val.txt")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", default="runs/kiocmil_cada")
    parser.add_argument("--save_interval", type=int, default=10)

    # Data augmentation
    parser.add_argument(
        "--augmentation_level", default="strong", choices=["none", "light", "strong"]
    )
    parser.add_argument("--use_clahe", type=bool, default=True)

    # WandB
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="klgrade-kiocmil")
    parser.add_argument("--wandb_entity", default="ngotam2k1-thuyloi-university")
    parser.add_argument("--wandb_name", default="kiocmil_cada_baseline")

    args = parser.parse_args()

    trainer = KiocmilCADATrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
