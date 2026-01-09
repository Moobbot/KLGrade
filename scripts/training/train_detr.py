"""
Training example for DETR (Detection Transformer) using CocoDataset.

This script demonstrates:
1. Converting YOLO labels to COCO format
2. Loading dataset with CocoDataset and DETR processor
3. Setting up DETR model from HuggingFace
4. Training loop with custom collate function
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.datasets import (
    CocoDataset,
    create_coco_json,
    detr_collate_fn_dynamic_padding,
)
from src.datasets.transforms import get_detr_processor
from src.config import CLASSES, CLASSES_10_CLASS
import torch
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor
from tqdm import tqdm
import argparse
import wandb


def prepare_coco_annotations(
    label_dir: str, 
    img_dir: str, 
    output_dir: str, 
    use_labels_new: bool = False,
    split_dir: str = "splits",
    num_classes: int = None
):
    """
    Convert YOLO labels to COCO JSON format.

    Args:
        label_dir: Base directory for labels
        img_dir: Directory containing images
        output_dir: Output directory for COCO JSON files
        use_labels_new: Use labels_new (10 classes) instead of labels (5 classes)
        split_dir: Directory containing split files (train.txt, val.txt)
        num_classes: Number of classes (overrides use_labels_new if provided)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine class names and label subdirectory
    if num_classes is not None:
        # Explicit class count provided - map to class configuration
        from src.config import CLASSES, CLASSES_10_CLASS, CLASSES_4_CLASS, CLASSES_8_CLASS
        class_map = {
            5: CLASSES,
            10: CLASSES_10_CLASS,
            4: CLASSES_4_CLASS,
            8: CLASSES_8_CLASS,
        }
        class_names = class_map.get(num_classes, CLASSES)
        label_subdir = "labels"
        suffix = f"_{num_classes}class"
    elif use_labels_new:
        class_names = CLASSES_10_CLASS
        label_subdir = "labels_new"
        suffix = "_new"
    else:
        class_names = CLASSES
        label_subdir = "labels"
        suffix = ""

    # Only modify label_dir if using subdirectory logic
    if num_classes is None:
        actual_label_dir = Path(label_dir).parent / label_subdir
    else:
        actual_label_dir = Path(label_dir)

    print(f"Converting YOLO labels to COCO format...")
    print(f"  Label dir: {actual_label_dir}")
    print(f"  Classes: {len(class_names)}")

    # Create annotations for train split
    train_json = create_coco_json(
        yolo_label_dir=str(actual_label_dir),
        img_dir=img_dir,
        output_path=str(output_path / f"annotations_train{suffix}.json"),
        class_names=class_names,
        split_file=str(Path(split_dir) / "train.txt"),
    )

    # Create annotations for val split
    val_json = create_coco_json(
        yolo_label_dir=str(actual_label_dir),
        img_dir=img_dir,
        output_path=str(output_path / f"annotations_val{suffix}.json"),
        class_names=class_names,
        split_file=str(Path(split_dir) / "val.txt"),
    )

    return train_json, val_json


def train_detr(
    img_dir: str = "processed/knee/images",
    label_dir: str = "processed/knee/labels",
    use_labels_new: bool = False,
    split_dir: str = "splits",
    num_classes: int = None,
    model_name: str = "facebook/detr-resnet-50",
    epochs: int = 50,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "runs/detr",
):
    """
    Train DETR model on KLGrade dataset.

    Args:
        img_dir: Directory containing images
        label_dir: Base directory for labels
        use_labels_new: Use labels_new (10 classes) instead of labels (5 classes)
        split_dir: Directory containing split files (train.txt, val.txt)
        num_classes: Number of classes (4, 5, 8, or 10) - overrides use_labels_new if provided
        model_name: HuggingFace DETR model name
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        device: Device to train on ('cuda' or 'cpu')
        output_dir: Directory to save checkpoints and results
    """

    print("=" * 60)
    print("DETR Training on KLGrade Dataset")
    print("=" * 60)

    # Determine class configuration
    from src.config import CLASSES, CLASSES_10_CLASS, CLASSES_4_CLASS, CLASSES_8_CLASS
    
    if num_classes is not None:
        class_map = {
            5: CLASSES,
            10: CLASSES_10_CLASS,
            4: CLASSES_4_CLASS,
            8: CLASSES_8_CLASS,
        }
        class_names = class_map.get(num_classes, CLASSES)
        num_classes_actual = len(class_names)
        label_suffix = f"_{num_classes}class"
    elif use_labels_new:
        class_names = CLASSES_10_CLASS
        num_classes_actual = len(CLASSES_10_CLASS)
        label_suffix = "_new"
    else:
        class_names = CLASSES
        num_classes_actual = len(CLASSES)
        label_suffix = ""

    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Classes: {num_classes_actual}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs: {epochs}")
    print(f"  Device: {device}")

    # Initialize WandB
    run_name = f"DETR-{num_classes_actual}class-{Path(output_dir).name}"
    wandb.init(
        entity="ngotam2k1-thuyloi-university",
        project="KLGrade-Knee-OA",
        name=run_name,
        config={
            "model": model_name,
            "num_classes": num_classes_actual,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "device": device,
            "img_dir": img_dir,
            "label_dir": label_dir,
            "architecture": "DETR",
        },
    )
    print(f"\n✅ WandB initialized: {run_name}")

    # Step 1: Prepare COCO annotations
    print("\n" + "=" * 60)
    print("Step 1: Preparing COCO annotations")
    print("=" * 60)

    coco_dir = "processed/coco"
    train_json, val_json = prepare_coco_annotations(
        label_dir=label_dir,
        img_dir=img_dir,
        output_dir=coco_dir,
        use_labels_new=use_labels_new,
        split_dir=split_dir,
        num_classes=num_classes,
    )

    # Step 2: Load DETR processor and model
    print("\n" + "=" * 60)
    print("Step 2: Loading DETR model and processor")
    print("=" * 60)

    processor = get_detr_processor(model_name=model_name)
    print(f"✅ Loaded processor: {type(processor).__name__}")

    # Load pre-trained DETR model and modify for our number of classes
    model = DetrForObjectDetection.from_pretrained(
        model_name,
        num_labels=num_classes_actual,
        ignore_mismatched_sizes=True,  # Allow different number of classes
    )
    model.to(device)
    print(f"✅ Loaded model: {model_name}")
    print(
        f"   Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
    )

    # Step 3: Create datasets
    print("\n" + "=" * 60)
    print("Step 3: Creating datasets")
    print("=" * 60)

    train_dataset = CocoDataset(
        coco_json_path=train_json, img_dir=img_dir, processor=processor
    )

    val_dataset = CocoDataset(
        coco_json_path=val_json, img_dir=img_dir, processor=processor
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=detr_collate_fn_dynamic_padding,
        num_workers=0,  # Set to 0 for Windows, increase on Linux
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=detr_collate_fn_dynamic_padding,
        num_workers=0,
    )

    print(f"✅ Train loader: {len(train_loader)} batches")
    print(f"✅ Val loader: {len(val_loader)} batches")

    # Step 4: Setup optimizer and scheduler
    print("\n" + "=" * 60)
    print("Step 4: Setting up optimizer")
    print("=" * 60)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Step 5: Training loop
    print("\n" + "=" * 60)
    print("Step 5: Training")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in train_pbar:
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

            # Forward pass
            outputs = model(
                pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels
            )

            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for batch in val_pbar:
                pixel_values = batch["pixel_values"].to(device)
                pixel_mask = batch["pixel_mask"].to(device)
                labels = [
                    {k: v.to(device) for k, v in t.items()} for t in batch["labels"]
                ]

                outputs = model(
                    pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels
                )

                val_loss += outputs.loss.item()
                val_pbar.set_postfix({"loss": f"{outputs.loss.item():.4f}"})

        avg_val_loss = val_loss / len(val_loader)

        # Update scheduler
        scheduler.step()

        # Log metrics to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": avg_train_loss,
            "val/loss": avg_val_loss,
            "train/lr": scheduler.get_last_lr()[0],
        })

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = output_path / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                },
                checkpoint_path,
            )
            print(f"  ✅ Saved best model (val_loss: {avg_val_loss:.4f})")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_path / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                },
                checkpoint_path,
            )
            print(f"  💾 Saved checkpoint: {checkpoint_path}")

    # Finish WandB run
    wandb.finish()

    print("\n" + "=" * 60)
    print("✅ Training completed!")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"   Models saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DETR on KLGrade dataset")
    parser.add_argument("--img_dir", type=str, default="processed/knee/images")
    parser.add_argument("--label_dir", type=str, default="processed/knee/labels")
    parser.add_argument("--use_labels_new", action="store_true")
    parser.add_argument("--split_dir", type=str, default="splits")
    parser.add_argument("--num_classes", type=int, default=None, choices=[4, 5, 8, 10],
                       help="Number of classes (overrides use_labels_new)")
    parser.add_argument("--model", type=str, default="facebook/detr-resnet-50")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=str, default="runs/detr")

    args = parser.parse_args()

    train_detr(
        img_dir=args.img_dir,
        label_dir=args.label_dir,
        use_labels_new=args.use_labels_new,
        split_dir=args.split_dir,
        num_classes=args.num_classes,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        device=args.device,
        output_dir=args.output,
    )
