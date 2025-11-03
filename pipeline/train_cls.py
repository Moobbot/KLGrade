"""
Train a KL-grade classifier from cropped knee patches.

Steps:
  1) Run pipeline/preprocess_crops.py to generate processed/classification/labels.csv
  2) Split CSV into train/val using splits (optional) or random split
  3) Train ResNet with class-imbalance handling (WeightedRandomSampler or class weights)
  4) Optional MixUp/CutMix
  5) Save best model
"""

import os
import sys
import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import CLASSES, BATCH_SIZE, EPOCHS, IMG_SIZE
from pipeline.dataset_cls import ClassificationDataset
from pipeline.augmentations import (
    get_train_transforms,
    get_val_transforms,
    build_weighted_sampler,
    build_weighted_sampler_from_labels,
    mixup_data,
    cutmix_data,
)
from pipeline.model_cls import build_model
from pipeline.utils.split_utils import (
    build_subset_indices_from_splits,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train KL-grade classifier")
    p.add_argument(
        "--csv", default=os.path.join("processed", "classification", "labels.csv")
    )
    p.add_argument(
        "--backbone",
        default="resnet50",
        choices=["cnn_basic", "resnet50", "resnet101", "efficientnet_b0"],
    )
    p.add_argument("--batch", type=int, default=BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument(
        "--size",
        type=int,
        default=None,
        help="Input resize (defaults to config.IMG_SIZE if not set)",
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument(
        "--sampler", action="store_true", help="Use WeightedRandomSampler for imbalance"
    )
    p.add_argument(
        "--class_weights", action="store_true", help="Use class-weighted CE loss"
    )
    p.add_argument(
        "--mixup", type=float, default=0.0, help="MixUp alpha (0 to disable)"
    )
    p.add_argument(
        "--cutmix", type=float, default=0.0, help="CutMix alpha (0 to disable)"
    )
    p.add_argument(
        "--train_split",
        type=str,
        default=None,
        help="Path to train.txt (list of original image filenames)",
    )
    p.add_argument(
        "--val_split_file",
        type=str,
        default=None,
        help="Path to val.txt (list of original image filenames)",
    )
    p.add_argument(
        "--splits_dir",
        type=str,
        default=None,
        help="Directory containing train.txt/val.txt (required unless files are provided)",
    )
    p.add_argument("--save", default=os.path.join("models", "cls_resnet50.pt"))
    return p.parse_args()


def compute_class_weights(csv_path: str):
    import pandas as pd

    df = pd.read_csv(csv_path)
    counts = df["label"].value_counts().to_dict()
    num_classes = len(CLASSES)
    weights = torch.zeros(num_classes, dtype=torch.float32)
    total = len(df)
    for c in range(num_classes):
        n = counts.get(c, 1)
        weights[c] = total / (num_classes * n)
    return weights


def train_one_epoch(
    model, loader, criterion, optimizer, device, mixup_alpha=0.0, cutmix_alpha=0.0
):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        target = y

        if mixup_alpha > 0:
            x, (y_a, y_b), lam = mixup_data(x, y, mixup_alpha)
            target = (y_a, y_b, lam)
        elif cutmix_alpha > 0:
            x, (y_a, y_b), lam = cutmix_data(x, y, cutmix_alpha)
            target = (y_a, y_b, lam)

        optimizer.zero_grad()
        logits = model(x)
        if isinstance(target, tuple):
            y_a, y_b, lam = target
            loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
        else:
            loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return running_loss / total, correct / total


def main():
    args = parse_args()

    # Datasets
    size = args.size or IMG_SIZE
    train_full = ClassificationDataset(args.csv, transforms=get_train_transforms(size))
    val_full = ClassificationDataset(args.csv, transforms=get_val_transforms(size))

    # Use provided splits if available
    train_file = args.train_split
    val_file = args.val_split_file
    if args.splits_dir and not (train_file and val_file):
        train_file = os.path.join(args.splits_dir, "train.txt")
        val_file = os.path.join(args.splits_dir, "val.txt")

    if (
        train_file
        and os.path.exists(train_file)
        and val_file
        and os.path.exists(val_file)
    ):
        df = train_full.df
        train_indices, val_indices, _ = build_subset_indices_from_splits(
            df["path"], train_file, val_file
        )
        if not train_indices or not val_indices:
            raise SystemExit(
                "No indices matched the provided split files. Check filename stems and crop naming."
            )
        train_ds = Subset(train_full, train_indices)
        val_ds = Subset(val_full, val_indices)
        print(f"Using external splits: train {len(train_ds)} | val {len(val_ds)}")
    else:
        raise SystemExit(
            "Please provide --splits_dir or both --train_split and --val_split_file for external splits."
        )

    # Loaders
    if args.sampler:
        # Build weights on the actual training subset to avoid index mismatches
        if isinstance(train_ds, Subset):
            base_df = train_full.df
            labels_subset = [int(base_df.iloc[i]["label"]) for i in train_ds.indices]
            sampler = build_weighted_sampler_from_labels(labels_subset)
        else:
            sampler = build_weighted_sampler(args.csv)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch, sampler=sampler, num_workers=2
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch, shuffle=True, num_workers=2
        )
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.backbone).to(device)

    # Loss
    if args.class_weights:
        w = compute_class_weights(args.csv).to(device)
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()

    # Optim
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args.mixup, args.cutmix
        )
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch}/{args.epochs} | Train loss {train_loss:.4f} acc {train_acc:.4f} | Val loss {val_loss:.4f} acc {val_acc:.4f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.save)
            print(f"Saved best model -> {args.save}")


if __name__ == "__main__":
    main()
