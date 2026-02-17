#!/usr/bin/env python3
"""
Unified DETR Training Script

This script trains a DETR model using the unified `DETRTrainer` class from `src.training`.
It supports both standard training and balanced training (using RepeatFactorSampler and Focal Loss).

Usage:
    # Standard training
    python scripts/training/train_detr.py --model facebook/detr-resnet-50 --epochs 50

    # Balanced training (RFS + Focal Loss)
    python scripts/training/train_detr.py --balanced --focal-loss
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.detr_trainer import DETRTrainer


def main():
    parser = argparse.ArgumentParser(description="Train DETR on KLGrade dataset")

    # Data arguments
    parser.add_argument(
        "--img_dir", type=str, default="processed/knee/images", help="Image directory"
    )
    parser.add_argument(
        "--label_dir", type=str, default="processed/knee/labels", help="Label directory"
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        default="splits",
        help="Split directory (containing train.txt/val.txt)",
    )

    # Class configuration
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        choices=[4, 5, 8, 10],
        help="Number of classes (overrides use_labels_10_class)",
    )
    parser.add_argument(
        "--use_labels_10_class",
        action="store_true",
        help="Use 10-class dataset (Legacy flag)",
    )

    # Training configuration
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/detr-resnet-50",
        help="HuggingFace model name",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument(
        "--output", type=str, default="runs/detr", help="Output directory"
    )

    # Advanced training options
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Use RepeatFactorSampler for class balancing",
    )
    parser.add_argument(
        "--focal-loss",
        action="store_true",
        help="Use Focal Loss instead of standard Cross Entropy",
    )

    args = parser.parse_args()

    print(f"Initializing DETR Trainer...")
    trainer = DETRTrainer(
        img_dir=args.img_dir,
        label_dir=args.label_dir,
        output_dir=args.output,
        model_name=args.model,
        num_classes=args.num_classes,
        use_labels_10_class=args.use_labels_10_class,
        split_dir=args.split_dir,
        batch_size=args.batch,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device,
        use_balanced_sampler=args.balanced,
        use_focal_loss=args.focal_loss,
    )

    trainer.prepare_data()
    trainer.setup_model()
    trainer.train()


if __name__ == "__main__":
    main()
