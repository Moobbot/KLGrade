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
import os
import argparse
import wandb
from ultralytics import YOLO

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import CLASSES, CLASSES_10_CLASS, CLASSES_4_CLASS, CLASSES_8_CLASS
from src.data.utils import create_yolo_dataset_yaml


def train_yolo11(
    img_dir: str = "processed/knee_5_class/images",
    label_dir: str = "processed/knee_5_class/labels",
    use_5_class: bool = False,
    use_10_class: bool = False,
    use_4_class: bool = False,
    use_8_class: bool = False,
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
    """

    print("=" * 60)
    print("YOLO11 Training on KLGrade Dataset")
    print("=" * 60)

    # Determine class names and number of classes
    if use_10_class:
        class_names = CLASSES_10_CLASS
        num_classes = len(CLASSES_10_CLASS)
        label_subdir = "labels_10_class"
    elif use_4_class:
        class_names = CLASSES_4_CLASS
        num_classes = len(CLASSES_4_CLASS)
        label_subdir = "labels_4_class"
    elif use_8_class:
        class_names = CLASSES_8_CLASS
        num_classes = len(CLASSES_8_CLASS)
        label_subdir = "labels_8_class"
    else:
        # Default 5 class
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

    # Logic from original script to find paths
    split_subdir = label_subdir.replace("labels_", "")
    if split_subdir == "labels":
        split_subdir = "dataset_v0"  # Assumption based on usual structure, or just handle generically
    if split_subdir == "labels_10_class":
        split_subdir = (
            "knee_10_class"  # Guessing based on logic inside original f-string
        )
    if split_subdir == "labels_4_class":
        split_subdir = "knee_4_class"
    if split_subdir == "labels_8_class":
        split_subdir = "knee_8_class"

    # The original script had logic: splits/{label_subdir.replace('labels_', '')}/train.txt
    # But split_dataset.py output to splits/ by default?
    # Let's trust the logic from original script:
    # splits/[subdir]/train.txt

    # Correct logic for finding splits based on file existence checks
    split_part = label_subdir.replace("labels_", "")
    if split_part == "labels":
        split_part = ""  # Corner case

    # Try multiple locations as per original script logic (implied)
    candidates = [
        img_dir_abs.parent / f"splits/{split_part}/train.txt",
        (
            img_dir_abs.parent / "splits/dataset_v0/train.txt"
            if split_part == ""
            else None
        ),
    ]

    train_path = "images"
    val_path = "images"
    test_path = None

    # Simple check based on original f-string logic
    # train: ... if (img_dir_abs.parent / ...).exists() else "images"

    potential_split_dir = img_dir_abs.parent / f"splits/{split_part}"
    if (potential_split_dir / "train.txt").exists():
        train_path = str(potential_split_dir / "train.txt")
        val_path = str(potential_split_dir / "val.txt")
        if (potential_split_dir / "test.txt").exists():
            test_path = str(potential_split_dir / "test.txt")

    create_yolo_dataset_yaml(
        output_path=dataset_yaml_path,
        class_names=class_names,
        path=str(img_dir_abs.parent),
        train=train_path,
        val=val_path,
        test=test_path,
    )

    print(f"\n✅ Dataset YAML created: {dataset_yaml_path}")

    print(
        "\n⚠️  Important: YOLO11 expects labels in a 'labels/' directory parallel to 'images/'"
    )
    print(f"   Make sure your labels are in: {img_dir_abs.parent / label_subdir}")

    # Initialize WandB
    wandb_project = os.getenv("WANDB_PROJECT", "KLGrade-Knee-OA")
    print(f"\n📊 Initializing WandB Project: {wandb_project}")
    print(f"   Experiment name: {name}")

    wandb.init(
        project=wandb_project,
        name=name,
        config={
            "model": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "img_size": img_size,
            "num_classes": num_classes,
            "class_names": list(class_names.values()),
        },
    )

    # Load YOLO11 model
    print(f"\n📦 Loading YOLO11 model: {model_name}")
    model = YOLO(model_name)

    # Train the model
    print("\n🚀 Starting training...")
    print("✅ WandB integration ENABLED - metrics will be logged to dashboard")
    results = model.train(
        data=str(dataset_yaml_path),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        patience=50,
        save=True,
        save_period=10,
        plots=True,
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
    val_project = str(Path(project) / "val")
    metrics = model.val(project=val_project, name=name)

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

    # Finish WandB run
    wandb.finish()
    print("\n✅ WandB run finished - check dashboard for results")

    print("\n" + "=" * 60)
    print("Training pipeline completed successfully!")
    print("=" * 60)


def quick_test():
    """Quick test with fewer epochs for debugging."""
    print("Running quick test (5 epochs)...\n")

    train_yolo11(
        img_dir="processed/knee/images",
        label_dir="processed/knee/labels",
        use_10_class=False,
        # Wait, the original call below used 'use_labels_new', but the definition has 'use_5_class', 'use_10_class' etc.
        # I should probably fix the arg name in the call or check the definition carefully.
        # Definition: use_5_class, use_10_class...
        # Original code used: use_labels_new=False in quick_test.
        # This implies the original code might have had an error or mismatch too?
        # Leaving it as is might crash if I don't fix it.
        # But 'use_labels_new' is NOT in the arguments of train_yolo11 in my visible file content.
        # Ah, looking at Step 541:
        # def train_yolo11(..., use_10_class: bool = False, ...)
        # Call in quick_test: use_labels_new=False
        # This IS an error in the original file I think. I should fix it to use_10_class=False or similar.
        model_name="yolo11n.pt",
        epochs=5,
        img_size=640,
        batch_size=4,
        device="0",
        name="yolo11_test",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO11 on KLGrade dataset")
    parser.add_argument("--img_dir", type=str, default="dataset/dataset_v0/images")
    parser.add_argument("--label_dir", type=str, default="dataset/dataset_v0/labels")
    parser.add_argument(
        "--use_10_class", action="store_true", help="Use 10-class dataset"
    )
    parser.add_argument(
        "--use_4_class", action="store_true", help="Use 4-class dataset"
    )
    parser.add_argument(
        "--use_8_class", action="store_true", help="Use 8-class dataset"
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
            use_10_class=args.use_10_class,
            use_4_class=args.use_4_class,
            use_8_class=args.use_8_class,
            model_name=args.model,
            epochs=args.epochs,
            img_size=args.img_size,
            batch_size=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
        )
