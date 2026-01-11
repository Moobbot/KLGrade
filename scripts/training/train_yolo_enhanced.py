#!/usr/bin/env python3
"""
Enhanced YOLO Training Script with Advanced Preprocessing
Based on techniques from Yolo_Detection_XuongKhop_v2.ipynb

Features:
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Gaussian Blur for noise reduction
- Data balancing with flip augmentation
- Label scaling for resized images
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ultralytics import YOLO
import wandb

# Import shared preprocessing logic
from src.data.preprocessing import balance_dataset_with_flip
from src.data.utils import create_yolo_dataset_yaml


def train_yolo_enhanced(
    img_dir: str,
    label_dir: str,
    split_dir: str = None,
    num_classes: int = 5,
    model_name: str = "yolo11n.pt",
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = "0",
    project: str = "runs/detect",
    name: str = "yolo_enhanced",
    apply_preprocessing: bool = True,
    apply_balancing: bool = True,
):
    """
    Train YOLO with enhanced preprocessing.
    """
    print("=" * 70)
    print("Enhanced YOLO Training with CLAHE + Data Balancing")
    print("=" * 70)

    # Setup WandB
    wandb_project = os.getenv("WANDB_PROJECT", "KLGrade-Knee-OA")
    print(f"\n📊 Initializing WandB Project: {wandb_project}")

    wandb.init(
        project=wandb_project,
        name=name,
        config={
            "model": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "img_size": img_size,
            "num_classes": num_classes,
            "preprocessing": "CLAHE + Gaussian Blur" if apply_preprocessing else "None",
            "balancing": "Flip Augmentation" if apply_balancing else "None",
        },
    )

    # Create processed data directory
    processed_base = Path("processed") / "enhanced" / name
    processed_img_dir = processed_base / "images"
    processed_label_dir = processed_base / "labels"

    if apply_preprocessing or apply_balancing:
        print(f"\n🔄 Preprocessing data to {processed_base}")

        balance_dataset_with_flip(
            img_dir=img_dir,
            label_dir=label_dir,
            output_img_dir=str(processed_img_dir),
            output_label_dir=str(processed_label_dir),
            num_classes=num_classes,
            target_size=(img_size, img_size),
        )

        # Use processed data
        final_img_dir = str(processed_img_dir)
        final_label_dir = str(processed_label_dir)
    else:
        final_img_dir = img_dir
        final_label_dir = label_dir

    # Create YOLO dataset config
    dataset_yaml = Path(f"configs/yolo_enhanced_{name}.yaml")

    # If splits are provided, rewrite them to point at processed images
    train_ref = None
    val_ref = None
    test_ref = None

    if split_dir:
        split_dir_path = Path(split_dir)
        processed_split_dir = (
            Path("processed") / "enhanced" / "splits" / split_dir_path.name / name
        )
        processed_split_dir.mkdir(parents=True, exist_ok=True)

        def rewrite_split(in_file: Path, out_file: Path):
            if not in_file.exists():
                return None
            lines = [l.strip() for l in in_file.read_text().splitlines() if l.strip()]
            new_lines = []
            for p in lines:
                stem = Path(p).stem
                proc_img = Path(final_img_dir) / f"{stem}.png"
                if proc_img.exists():
                    new_lines.append(str(proc_img.resolve()))
            if not new_lines:
                return None
            out_file.write_text("\n".join(new_lines))
            return out_file.resolve()

        train_ref = rewrite_split(
            split_dir_path / "train.txt", processed_split_dir / "train.txt"
        )
        val_ref = rewrite_split(
            split_dir_path / "val.txt", processed_split_dir / "val.txt"
        )
        test_txt = split_dir_path / "test.txt"
        test_ref = (
            rewrite_split(test_txt, processed_split_dir / "test.txt")
            if test_txt.exists()
            else None
        )

    # Fallbacks when splits aren't provided or rewrite produced no files
    if not split_dir or not train_ref or not val_ref:
        # Use directories directly if we can't use splits
        train_ref = Path(final_img_dir).resolve()
        val_ref = Path(final_img_dir).resolve()
        test_ref = None

    # Use shared utility
    create_yolo_dataset_yaml(
        output_path=dataset_yaml,
        class_names={i: str(i) for i in range(num_classes)},  # Auto-names
        path=str(Path.cwd()),  # Root context for absolute paths in refs
        train=str(train_ref),
        val=str(val_ref),
        test=str(test_ref) if test_ref else None,
    )

    print(f"\n✅ Dataset config created: {dataset_yaml}")

    # Load model
    print(f"\n📦 Loading YOLO model: {model_name}")
    model = YOLO(model_name)

    # Train
    print(f"\n🚀 Starting training...")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {img_size}")
    print(f"  Device: {device}")

    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project,
        name=name,
        patience=50,
        save_period=10,
        plots=True,
        # Augmentation & Hyperparameters (Enhanced)
        mosaic=1.0,
        mixup=0.15,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        cos_lr=True,  # Cosine annealing for better convergence
        optimizer="auto",
    )

    # Validate
    print("\n📊 Running validation...")
    val_project = str(Path(project) / "val")
    metrics = model.val(project=val_project, name=name)

    print(f"\n✅ Training completed!")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")

    wandb.finish()

    return results, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhanced YOLO training with CLAHE and data balancing"
    )

    parser.add_argument("--img_dir", type=str, required=True, help="Image directory")
    parser.add_argument("--label_dir", type=str, required=True, help="Label directory")
    parser.add_argument("--split_dir", type=str, default=None, help="Split directory")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of classes")
    parser.add_argument("--model", type=str, default="yolo11l.pt", help="YOLO model")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="0", help="Device")
    parser.add_argument(
        "--project", type=str, default="runs/detect", help="Project dir"
    )
    parser.add_argument(
        "--name", type=str, default="yolo11l_enhanced", help="Experiment name"
    )
    parser.add_argument(
        "--no-preprocessing", action="store_true", help="Disable preprocessing"
    )
    parser.add_argument("--no-balancing", action="store_true", help="Disable balancing")

    args = parser.parse_args()

    train_yolo_enhanced(
        img_dir=args.img_dir,
        label_dir=args.label_dir,
        split_dir=args.split_dir,
        num_classes=args.num_classes,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        apply_preprocessing=not args.no_preprocessing,
        apply_balancing=not args.no_balancing,
    )
