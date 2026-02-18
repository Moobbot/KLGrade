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
from torch.cuda.amp import autocast, GradScaler
import yaml
import cv2
import numpy as np
from tqdm import tqdm
import json
from typing import Dict, List, Tuple
import wandb

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
            # Split files contain image stems (no extensions)
            self.image_stems = [line.strip() for line in f if line.strip()]

        print(f"Loaded {len(self.image_stems)} images from {split_file}")

    def __len__(self):
        return len(self.image_stems)

    def __getitem__(self, idx):
        img_stem = self.image_stems[idx]
        
        # Try different extensions
        img_path = None
        for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
            path = self.img_dir / f"{img_stem}{ext}"
            if path.exists():
                img_path = path
                break
        
        if img_path is None:
             raise ValueError(f"Image not found: {img_stem} (checked extensions: .png, .jpg, .jpeg)")

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")

        # Convert to RGB
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, scaler=None, accumulation_steps=1):
    """Train for one epoch with mixed precision support."""
    model.train()
    total_loss = 0
    total_ce = 0
    total_bbox = 0
    total_giou = 0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for i, (images, targets) in enumerate(pbar):
        images = images.to(device)
        targets = [
            {k: v.to(device) for k, v in t.items() if isinstance(v, torch.Tensor)}
            for t in targets
        ]

        # Mixed precision forward pass
        if scaler is not None:
            with autocast():
                outputs = model(images)
                losses = criterion(outputs, targets)
                loss = losses["loss"] / accumulation_steps
        else:
            outputs = model(images)
            losses = criterion(outputs, targets)
            loss = losses["loss"] / accumulation_steps

        # Calculate accuracy
        pred_logits = outputs["pred_logits"]  # [B, num_queries, num_classes+1]
        pred_classes = pred_logits.argmax(dim=-1)  # [B, num_queries]
        
        for b_idx, target in enumerate(targets):
            if len(target["labels"]) > 0:
                # Get predicted class for boxes with highest confidence
                pred_scores = pred_logits[b_idx].softmax(dim=-1)[:, :-1].max(dim=-1)[0]
                top_pred_idx = pred_scores.argmax()
                pred_class = pred_classes[b_idx, top_pred_idx]
                
                # Compare with ground truth (take first label as image-level label)
                gt_class = target["labels"][0]
                if pred_class == gt_class:
                    total_correct += 1
                total_samples += 1

        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation
        if (i + 1) % accumulation_steps == 0:
            if scaler is not None:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()
            optimizer.zero_grad()

        # Update metrics
        total_loss += loss.item()
        total_ce += losses["loss_ce"].item()
        total_bbox += losses["loss_bbox"].item()
        total_giou += losses["loss_giou"].item()

        current_acc = total_correct / total_samples if total_samples > 0 else 0
        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "ce": f"{losses['loss_ce'].item():.4f}",
                "acc": f"{current_acc:.4f}",
            }
        )

    n = len(dataloader)
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    return {
        "loss": total_loss / n,
        "loss_ce": total_ce / n,
        "loss_bbox": total_bbox / n,
        "loss_giou": total_giou / n,
        "accuracy": accuracy,
    }


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_ce = 0
    total_bbox = 0
    total_giou = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            targets = [
                {k: v.to(device) for k, v in t.items() if isinstance(v, torch.Tensor)}
                for t in targets
            ]

            outputs = model(images)
            losses = criterion(outputs, targets)

            # Calculate accuracy
            pred_logits = outputs["pred_logits"]
            pred_classes = pred_logits.argmax(dim=-1)
            
            for b_idx, target in enumerate(targets):
                if len(target["labels"]) > 0:
                    pred_scores = pred_logits[b_idx].softmax(dim=-1)[:, :-1].max(dim=-1)[0]
                    top_pred_idx = pred_scores.argmax()
                    pred_class = pred_classes[b_idx, top_pred_idx]
                    
                    gt_class = target["labels"][0]
                    if pred_class == gt_class:
                        total_correct += 1
                    total_samples += 1

            total_loss += losses["loss"].item()
            total_ce += losses["loss_ce"].item()
            total_bbox += losses["loss_bbox"].item()
            total_giou += losses["loss_giou"].item()

    n = len(dataloader)
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    return {
        "loss": total_loss / n,
        "loss_ce": total_ce / n,
        "loss_bbox": total_bbox / n,
        "loss_giou": total_giou / n,
        "accuracy": accuracy,
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
    parser.add_argument(
        "--num-classes", type=int, default=5, help="Number of classes (5 for KL0-KL4, 10 for KL0-a/b to KL4-a/b)"
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
    parser.add_argument(
        "--use-amp",
        action="store_true",
        default=True,
        help="Use automatic mixed precision training",
    )
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (effective batch size = batch * accumulation_steps)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Number of epochs with no improvement after which training will be stopped",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="cdt-cad-training",
        help="WandB project name",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="WandB run name (auto-generated if not specified)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging",
    )
    parser.add_argument(
        "--weight-class", type=float, default=2.0, help="Weight for classification loss"
    )
    parser.add_argument(
        "--weight-bbox", type=float, default=5.0, help="Weight for bbox L1 loss"
    )
    parser.add_argument(
        "--weight-giou", type=float, default=2.0, help="Weight for GIoU loss"
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

    # Create dataloaders with optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Load dataset config
    with open(args.data, "r") as f:
        data_config = yaml.safe_load(f)
    
    print(f"Number of classes: {args.num_classes}")

    # Create model
    print("Creating CDT-CAD model...")
    model = CDTCAD(
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        hidden_dim=args.hidden_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        pretrained_backbone=True,
    ).to(device)

    # Create loss
    criterion = CDTCADLoss(
        num_classes=args.num_classes,
        weight_class=args.weight_class,
        weight_bbox=args.weight_bbox,
        weight_giou=args.weight_giou,
        eos_coef=0.1,
    ).to(device)  # Move to device

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Mixed precision scaler
    scaler = GradScaler() if args.use_amp else None
    if args.use_amp:
        print("✅ Using Automatic Mixed Precision (AMP) training")
    
    # Gradient accumulation
    if args.accumulation_steps > 1:
        print(f"✅ Using gradient accumulation: {args.accumulation_steps} steps")
        print(f"   Effective batch size: {args.batch * args.accumulation_steps}")
    
    # Early stopping
    print(f"✅ Early stopping patience: {args.early_stopping_patience} epochs")
    
    # Initialize WandB
    if not args.no_wandb:
        wandb_run_name = args.wandb_run_name or f"cdt_cad_{args.num_classes}class_b{args.batch}_img{args.img_size}_dim{args.hidden_dim}"
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            config={
                "num_classes": args.num_classes,
                "batch_size": args.batch,
                "accumulation_steps": args.accumulation_steps,
                "effective_batch_size": args.batch * args.accumulation_steps,
                "img_size": args.img_size,
                "hidden_dim": args.hidden_dim,
                "num_encoder_layers": args.num_encoder_layers,
                "num_decoder_layers": args.num_decoder_layers,
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "early_stopping_patience": args.early_stopping_patience,
                "use_amp": args.use_amp,
            }
        )
        print(f"✅ WandB logging enabled: {wandb_run_name}")
    else:
        print("⚠️  WandB logging disabled")

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Training history
    history = []
    best_val_loss = float("inf")
    epochs_no_improve = 0

    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*50}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            scaler=scaler, accumulation_steps=args.accumulation_steps
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
            f"Train Acc: {train_metrics['accuracy']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}"
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
        
        # Log to WandB
        if not args.no_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_metrics["loss"],
                "train/accuracy": train_metrics["accuracy"],
                "train/ce": train_metrics["loss_ce"],
                "train/bbox": train_metrics["loss_bbox"],
                "train/giou": train_metrics["loss_giou"],
                "val/loss": val_metrics["loss"],
                "val/accuracy": val_metrics["accuracy"],
                "val/ce": val_metrics["loss_ce"],
                "val/bbox": val_metrics["loss_bbox"],
                "val/giou": val_metrics["loss_giou"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            })

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                },
                save_dir / "best.pt",
            )
            print(f"✅ Saved best model (val_loss: {val_metrics['loss']:.4f}, val_acc: {val_metrics['accuracy']:.4f})")
            
            # Log best model to WandB
            if not args.no_wandb:
                wandb.run.summary["best_val_loss"] = best_val_loss
                wandb.run.summary["best_val_accuracy"] = val_metrics["accuracy"]
                wandb.run.summary["best_epoch"] = epoch
        else:
            epochs_no_improve += 1
            print(f"⚠️  No improvement for {epochs_no_improve} epoch(s)")
            
            # Early stopping check
            if epochs_no_improve >= args.early_stopping_patience:
                print(f"\n🛑 Early stopping triggered after {epoch} epochs")
                print(f"   Best val_loss: {best_val_loss:.4f}")
                break

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

    print("="*50)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {save_dir}")
    print("="*50)
    
    # Finish WandB run
    if not args.no_wandb:
        # Save best model artifact
        artifact = wandb.Artifact(
            name=f"cdt_cad_{args.num_classes}class_best",
            type="model",
            description=f"Best CDT-CAD model ({args.num_classes} classes)"
        )
        artifact.add_file(str(save_dir / "best.pt"))
        wandb.log_artifact(artifact)
        
        wandb.finish()
        print("✅ WandB run finished")


if __name__ == "__main__":
    main()
