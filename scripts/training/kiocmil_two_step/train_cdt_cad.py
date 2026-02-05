"""
Training script for CDT-CAD model

Usage:
    python scripts/training/train_cdt_cad.py --config configs/cdt_cad_baseline.yaml --data processed/yolo11_labels.yaml
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
from tqdm import tqdm
import wandb
from datetime import datetime
import os
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.cdt_cad import CDTCAD
from src.losses.cdt_cad_loss import CDTCADLoss
from src.datasets.cdt_cad_dataset import CDTCADDataset, collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Train CDT-CAD model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--data", type=str, required=True, help="Path to data YAML file"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Number of epochs (overrides config)"
    )
    parser.add_argument(
        "--batch", type=int, default=None, help="Batch size (overrides config)"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument(
        "--output", type=str, default="runs/cdt_cad", help="Output directory"
    )
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument(
        "--subset", type=int, default=None, help="Use subset of data (for testing)"
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_data_config(data_path: str) -> dict:
    """Load data configuration"""
    with open(data_path, "r") as f:
        data_config = yaml.safe_load(f)
    return data_config


def create_datasets(data_config: dict, config: dict, subset: int = None):
    """Create train and validation datasets"""
    # Get paths
    img_dir = data_config["train"]
    val_img_dir = data_config["val"]

    # Determine label directory
    if "labels" in data_config:
        label_dir = data_config["labels"]
        val_label_dir = data_config.get("val_labels", label_dir)
    else:
        # Assume labels are in parallel directory
        label_dir = str(Path(img_dir).parent / "labels")
        val_label_dir = str(Path(val_img_dir).parent / "labels")

    # Split files
    splits_dir = Path("processed/splits")
    train_split = splits_dir / "train.txt" if splits_dir.exists() else None
    val_split = splits_dir / "val.txt" if splits_dir.exists() else None

    # Create datasets
    train_dataset = CDTCADDataset(
        img_dir=img_dir,
        label_dir=label_dir,
        split_file=train_split,
        image_size=(800, 800),
        augment=True,
        clahe=True,
        num_classes=len(data_config["names"]),
    )

    val_dataset = CDTCADDataset(
        img_dir=val_img_dir,
        label_dir=val_label_dir,
        split_file=val_split,
        image_size=(800, 800),
        augment=False,
        clahe=False,
        num_classes=len(data_config["names"]),
    )

    # Subset for testing
    if subset:
        train_dataset.image_names = train_dataset.image_names[:subset]
        val_dataset.image_names = val_dataset.image_names[
            : min(subset // 5, len(val_dataset.image_names))
        ]

    return train_dataset, val_dataset


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    clip_max_norm: float = 0.1,
):
    """Train for one epoch"""
    model.train()
    criterion.train()

    total_loss = 0
    total_loss_ce = 0
    total_loss_bbox = 0
    total_loss_giou = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for images, targets in pbar:
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward
        outputs = model(images)

        # Compute loss
        loss_dict = criterion(outputs, targets)

        loss = loss_dict["loss"]

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)

        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        total_loss_ce += loss_dict["loss_ce"].item()
        total_loss_bbox += loss_dict["loss_bbox"].item()
        total_loss_giou += loss_dict["loss_giou"].item()

        # Update progress bar
        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "ce": f"{loss_dict['loss_ce'].item():.4f}",
                "bbox": f"{loss_dict['loss_bbox'].item():.4f}",
                "giou": f"{loss_dict['loss_giou'].item():.4f}",
            }
        )

    # Average metrics
    metrics = {
        "train/loss": total_loss / len(dataloader),
        "train/loss_ce": total_loss_ce / len(dataloader),
        "train/loss_bbox": total_loss_bbox / len(dataloader),
        "train/loss_giou": total_loss_giou / len(dataloader),
    }

    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
):
    """Validation"""
    model.eval()
    criterion.eval()

    total_loss = 0
    total_loss_ce = 0
    total_loss_bbox = 0
    total_loss_giou = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")

    for images, targets in pbar:
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward
        outputs = model(images)

        # Compute loss
        loss_dict = criterion(outputs, targets)

        # Update metrics
        total_loss += loss_dict["loss"].item()
        total_loss_ce += loss_dict["loss_ce"].item()
        total_loss_bbox += loss_dict["loss_bbox"].item()
        total_loss_giou += loss_dict["loss_giou"].item()

        pbar.set_postfix({"loss": f"{loss_dict['loss'].item():.4f}"})

    # Average metrics
    metrics = {
        "val/loss": total_loss / len(dataloader),
        "val/loss_ce": total_loss_ce / len(dataloader),
        "val/loss_bbox": total_loss_bbox / len(dataloader),
        "val/loss_giou": total_loss_giou / len(dataloader),
    }

    return metrics


def main():
    args = parse_args()

    # Load configs
    config = load_config(args.config)
    data_config = load_data_config(args.data)

    # Override config with command line args
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch:
        config["training"]["batch_size"] = args.batch

    # Setup output directory
    if args.name:
        exp_name = args.name
    else:
        exp_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    output_dir = Path(args.output) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configs
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    with open(output_dir / "data_config.yaml", "w") as f:
        yaml.dump(data_config, f)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # W&B
    if args.wandb:
        wandb.init(
            project="KLGrade-CDT-CAD", name=exp_name, config={**config, **data_config}
        )

    # Create datasets
    print("Loading datasets...")
    train_dataset, val_dataset = create_datasets(data_config, config, args.subset)
    print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Create model
    print("Creating model...")
    model = CDTCAD(
        num_classes=config["model"]["num_classes"],
        num_queries=config["model"].get("num_queries", 100),
        hidden_dim=config["model"].get("hidden_dim", 256),
        num_encoder_layers=config["model"].get("num_encoder_layers", 6),
        num_decoder_layers=config["model"].get("num_decoder_layers", 6),
        num_feature_levels=config["model"].get("num_feature_levels", 4),
        n_heads=config["model"].get("n_heads", 8),
        dim_feedforward=config["model"].get("dim_feedforward", 1024),
        dropout=config["model"].get("dropout", 0.1),
        n_points=config["model"].get("n_points", 4),
        dilation_rates=config["feature_extractor"].get("dilation_rates", [1, 2, 4, 8]),
        num_iterations=config["feature_extractor"].get("num_iterations", 3),
        wavelet=config["feature_extractor"].get("wavelet_type", "haar"),
        pretrained_backbone=True,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Create criterion
    criterion = CDTCADLoss(
        num_classes=config["model"]["num_classes"],
        weight_class=config["loss_weights"].get("class_loss", 2.0),
        weight_bbox=config["loss_weights"].get("bbox_loss", 5.0),
        weight_giou=config["loss_weights"].get("giou_loss", 2.0),
    ).to(device)

    # Optimizer
    param_dicts = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": config["training"]["learning_rate"] * 0.1,
        },
    ]

    optimizer = torch.optim.AdamW(
        param_dicts,
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 1e-4),
    )

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    print(f"\nStarting training for {config['training']['epochs']} epochs...")

    for epoch in range(start_epoch, config["training"]["epochs"]):
        # Train
        train_metrics = train_one_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            device,
            epoch,
            clip_max_norm=config["training"].get("clip_max_norm", 0.1),
        )

        # Validate
        val_metrics = validate(model, criterion, val_loader, device, epoch)

        # Update learning rate
        lr_scheduler.step()

        # Log metrics
        metrics = {
            **train_metrics,
            **val_metrics,
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
        }

        if args.wandb:
            wandb.log(metrics)

        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_metrics['train/loss']:.4f}")
        print(f"  Val Loss: {val_metrics['val/loss']:.4f}")

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "best_val_loss": best_val_loss,
        }

        torch.save(checkpoint, output_dir / "last.pt")

        # Save best model
        if val_metrics["val/loss"] < best_val_loss:
            best_val_loss = val_metrics["val/loss"]
            torch.save(checkpoint, output_dir / "best.pt")
            print(f"  New best model saved! (val_loss: {best_val_loss:.4f})")

        # Save periodic checkpoints
        if (epoch + 1) % 50 == 0:
            torch.save(checkpoint, output_dir / f"epoch_{epoch}.pt")

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Models saved to: {output_dir}")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
