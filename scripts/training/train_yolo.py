"""
Training example for YOLO11 (Ultralytics) using YoloDataset.

This script demonstrates:
1. Loading dataset with YoloDataset
2. Setting up YOLO11 model from Ultralytics
3. Training configuration
4. Training loop with validation
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.datasets import (
    YoloDataset,
    get_default_train_transform,
    get_default_val_transform,
)
from src.config import CLASSES, CLASSES_10_CLASS, CLASSES_FILTERED
from ultralytics import YOLO
import torch


def train_yolo11(
    img_dir: str = "dataset/dataset_v1/images",
    label_dir: str = "dataset/dataset_v1/labels",
    use_labels_new: bool = False,
    use_filtered: bool = False,
    model_name: str = "yolo11n.pt",
    epochs: int = 100,
    img_size: int = 640,
    batch_size: int = 16,
    device: str = "0",
    project: str = "runs/detect",
    name: str = "yolo11_klgrade",
):
    """
    Train YOLO11 model on KLGrade dataset.

    Args:
        img_dir: Directory containing images
        label_dir: Base directory for labels
        use_labels_new: Use labels_new (10 classes) instead of labels (5 classes)
        model_name: YOLO11 model variant (yolo11n.pt, yolo11s.pt, yolo11m.pt, etc.)
        epochs: Number of training epochs
        img_size: Input image size
        batch_size: Batch size for training
        device: GPU device (e.g., "0" or "cpu")
        project: Project directory for saving results
        name: Experiment name
    """

    print("=" * 60)
    print("YOLO11 Training on KLGrade Dataset")
    print("=" * 60)

    # Determine class names and number of classes
    if use_filtered:
        class_names = CLASSES_FILTERED
        num_classes = len(CLASSES_FILTERED)
        label_subdir = "labels"  # Filtered dataset uses 'labels' folder
    elif use_labels_new:
        class_names = CLASSES_10_CLASS
        num_classes = len(CLASSES_10_CLASS)
        label_subdir = "labels_new"
    else:
        class_names = CLASSES
        num_classes = len(CLASSES)
        label_subdir = "labels"

    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Classes: {num_classes} ({label_subdir})")
    print(f"  Image size: {img_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Device: {device}")

    # Create dataset YAML for YOLO (Ultralytics format)
    dataset_yaml_path = Path(f"processed/yolo11_{label_subdir}.yaml")

    # Get absolute paths
    img_dir_abs = Path(img_dir).absolute()
    label_dir_abs = (Path(label_dir).parent / label_subdir).absolute()

    yaml_content = f"""# KLGrade Dataset for YOLO11
# Generated automatically

path: {img_dir_abs.parent}  # Dataset root
train: images  # Train images (relative to 'path')
val: images    # Val images (relative to 'path')

# Classes
names:
"""

    # Add class names
    for class_id, class_name in sorted(class_names.items()):
        yaml_content += f"  {class_id}: {class_name}\n"

    # Save YAML file
    with open(dataset_yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"\n✅ Dataset YAML created: {dataset_yaml_path}")

    # Note: YOLO uses a different approach - it reads labels from a labels/ directory
    # parallel to images/ directory. We need to ensure our structure matches.
    print(
        "\n⚠️  Important: YOLO11 expects labels in a 'labels/' directory parallel to 'images/'"
    )
    print(f"   Make sure your labels are in: {img_dir_abs.parent / label_subdir}")
    print(f"   Or create symlinks if needed.")

    # Load YOLO11 model
    print(f"\n📦 Loading YOLO11 model: {model_name}")
    model = YOLO(model_name)

    # Train the model
    print("\n🚀 Starting training...")
    results = model.train(
        data=str(dataset_yaml_path),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        # Additional training arguments
        patience=50,  # Early stopping patience
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        plots=True,  # Save training plots
        # Augmentation (YOLO has built-in augmentations)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
    )

    print("\n✅ Training completed!")
    print(f"   Results saved to: {project}/{name}")

    # Validate the model
    print("\n📊 Running validation...")
    metrics = model.val()

    print(f"\n   mAP50: {metrics.box.map50:.4f}")
    print(f"   mAP50-95: {metrics.box.map:.4f}")

    # Export model (optional - requires onnx package)
    try:
        print("\n💾 Exporting model to ONNX...")
        model.export(format="onnx")
        print("✅ ONNX export successful!")
    except ImportError:
        print("⚠️  ONNX export skipped (install 'onnx' package to enable)")
    except Exception as e:
        print(f"⚠️  ONNX export failed: {e}")

    print("\n" + "=" * 60)
    print("Training pipeline completed successfully!")
    print("=" * 60)


def quick_test():
    """Quick test with fewer epochs for debugging."""
    print("Running quick test (5 epochs)...\n")

    train_yolo11(
        img_dir="processed/knee/images",
        label_dir="processed/knee/labels",
        use_labels_new=False,
        model_name="yolo11n.pt",
        epochs=5,
        img_size=640,
        batch_size=4,
        device="0",
        name="yolo11_test",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train YOLO11 on KLGrade dataset")
    parser.add_argument("--img_dir", type=str, default="dataset/dataset_v1/images")
    parser.add_argument("--label_dir", type=str, default="dataset/dataset_v1/labels")
    parser.add_argument(
        "--use_labels_new", action="store_true", help="Use labels_new (10 classes)"
    )
    parser.add_argument(
        "--use_filtered", action="store_true", help="Use filtered dataset (7 classes)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="YOLO11 model (yolo11n.pt, yolo11s.pt, yolo11m.pt, etc.)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=str, default="runs/detect")
    parser.add_argument("--name", type=str, default="yolo11_klgrade")
    parser.add_argument("--test", action="store_true", help="Run quick test (5 epochs)")

    args = parser.parse_args()

    if args.test:
        quick_test()
    else:
        train_yolo11(
            img_dir=args.img_dir,
            label_dir=args.label_dir,
            use_labels_new=args.use_labels_new,
            use_filtered=args.use_filtered,
            model_name=args.model,
            epochs=args.epochs,
            img_size=args.img_size,
            batch_size=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
        )
