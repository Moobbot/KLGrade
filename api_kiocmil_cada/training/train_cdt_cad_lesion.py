"""
CDT-CAD Lesion Detector Training Script

Train CDT-CAD model for KL grade detection on cropped knee images.
Uses context-aware deformable transformers for improved detection.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml
import cv2
import numpy as np
from tqdm import tqdm
import json
from typing import Dict, List, Tuple

from src.models.cdt_cad.cdt_cad_model import CDTCAD
from src.losses.cdt_cad_loss import CDTCADLoss


class LesionDetectionDataset(Dataset):
    """
    Dataset for CDT-CAD lesion detection.

    Loads images and lesion bounding boxes in DETR format.
    """

    def __init__(
        self, img_dir: Path, label_dir: Path, split_file: Path, img_size: int = 640
    ):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size

        # Load image list from split file
        with open(split_file, "r") as f:
            self.image_paths = [line.strip() for line in f if line.strip()]

        print(f"Loaded {len(self.image_paths)} images from {split_file}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Cannot read image: {img_path}")

        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        # Resize image
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = image.astype(np.float32) / 255.0

        # Normalize (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std

        # Convert to tensor (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        # Load labels
        img_name = Path(img_path).stem
        label_path = self.label_dir / f"{img_name}.txt"

        boxes = []
        labels = []

        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])

                        # Labels are already KL grades (0-4)
                        boxes.append([cx, cy, w, h])
                        labels.append(cls)

        # Convert to tensors
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            # Empty image (no lesions)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,  # Normalized (cx, cy, w, h)
            "labels": labels,
            "image_id": idx,
            "orig_size": torch.tensor([orig_h, orig_w]),
        }

        return image, target


def collate_fn(batch):
    """Custom collate function for variable number of objects."""
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_ce = 0
    total_bbox = 0
    total_giou = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for images, targets in pbar:
        images = images.to(device)
        targets = [
            {k: v.to(device) for k, v in t.items() if isinstance(v, torch.Tensor)}
            for t in targets
        ]

        # Forward pass
        outputs = model(images)

        # Compute loss
        losses = criterion(outputs, targets)
        loss = losses["loss"]

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)

        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        total_ce += losses["loss_ce"].item()
        total_bbox += losses["loss_bbox"].item()
        total_giou += losses["loss_giou"].item()

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "ce": f"{losses['loss_ce'].item():.4f}",
                "bbox": f"{losses['loss_bbox'].item():.4f}",
                "giou": f"{losses['loss_giou'].item():.4f}",
            }
        )

    n = len(dataloader)
    return {
        "loss": total_loss / n,
        "loss_ce": total_ce / n,
        "loss_bbox": total_bbox / n,
        "loss_giou": total_giou / n,
    }


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_ce = 0
    total_bbox = 0
    total_giou = 0

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            targets = [
                {k: v.to(device) for k, v in t.items() if isinstance(v, torch.Tensor)}
                for t in targets
            ]

            outputs = model(images)
            losses = criterion(outputs, targets)

            total_loss += losses["loss"].item()
            total_ce += losses["loss_ce"].item()
            total_bbox += losses["loss_bbox"].item()
            total_giou += losses["loss_giou"].item()

    n = len(dataloader)
    return {
        "loss": total_loss / n,
        "loss_ce": total_ce / n,
        "loss_bbox": total_bbox / n,
        "loss_giou": total_giou / n,
    }


def main():
    parser = argparse.ArgumentParser(description="Train CDT-CAD Lesion Detector")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument(
        "--num-queries", type=int, default=100, help="Number of object queries"
    )
    parser.add_argument("--img-size", type=int, default=640, help="Input image size")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument(
        "--num-encoder-layers", type=int, default=6, help="Number of encoder layers"
    )
    parser.add_argument(
        "--num-decoder-layers", type=int, default=6, help="Number of decoder layers"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="runs/cdt_cad/lesion_detector",
        help="Save directory",
    )

    args = parser.parse_args()

    # Load dataset config
    with open(args.data, "r") as f:
        data_config = yaml.safe_load(f)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets
    train_dataset = LesionDetectionDataset(
        img_dir=Path(data_config["path"]) / "images",
        label_dir=Path(data_config["path"]) / "labels",
        split_file=Path(data_config["train"]),
        img_size=args.img_size,
    )

    val_dataset = LesionDetectionDataset(
        img_dir=Path(data_config["path"]) / "images",
        label_dir=Path(data_config["path"]) / "labels",
        split_file=Path(data_config["val"]),
        img_size=args.img_size,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    # Create model
    print("Creating CDT-CAD model...")
    model = CDTCAD(
        num_classes=5,  # KL0-KL4
        num_queries=args.num_queries,
        hidden_dim=args.hidden_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        pretrained_backbone=True,
    ).to(device)

    # Create loss
    criterion = CDTCADLoss(
        num_classes=5,
        weight_class=2.0,
        weight_bbox=5.0,
        weight_giou=2.0,
        eos_coef=0.1,
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_loss = float("inf")
    history = []

    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*50}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Print metrics
        print(
            f"\nTrain Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}"
        )
        print(
            f"Train CE: {train_metrics['loss_ce']:.4f} | Val CE: {val_metrics['loss_ce']:.4f}"
        )
        print(
            f"Train BBox: {train_metrics['loss_bbox']:.4f} | Val BBox: {val_metrics['loss_bbox']:.4f}"
        )
        print(
            f"Train GIoU: {train_metrics['loss_giou']:.4f} | Val GIoU: {val_metrics['loss_giou']:.4f}"
        )

        # Save history
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
        )

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_metrics["loss"],
                },
                save_dir / "best.pt",
            )
            print(f"✅ Saved best model (val_loss: {best_val_loss:.4f})")

        # Save checkpoint
        if epoch % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                save_dir / f"epoch_{epoch}.pt",
            )

    # Save final model
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        save_dir / "last.pt",
    )

    # Save training history
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*50}")
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {save_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
