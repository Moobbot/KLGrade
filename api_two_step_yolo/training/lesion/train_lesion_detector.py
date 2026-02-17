"""
Train YOLO11 Lesion Detector (Osteophytes & Joint Space Narrowing).
"""

import argparse
from ultralytics import YOLO
from pathlib import Path
import os
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))


def train_lesion_detector(
    data_yaml: str,
    epochs: int = 50,
    imgsz: int = 640,
    batch_size: int = 16,
    device: str = "0",
    project: str = "runs/detect",
    name: str = "train_lesion_detector",
    wandb_project: str = "klgrade-lesion-detection",
):
    """
    Train YOLO11 model for lesion detection.
    Classes: 0: Osteophytes, 1: Joint Space Narrowing

    Args:
        data_yaml: Path to dataset.yaml
        epochs: Number of training epochs
        imgsz: Image size
        batch_size: Batch size
        device: Device ID
        project: Project directory
        name: Run name
    """
    # Initialize YOLO11 Large model (pretrained)
    print("Loading YOLO11l model...")
    model = YOLO("yolo11l.pt")

    # Train
    print(f"Starting training for {epochs} epochs...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        plots=True,
        save=True,
        val=True,
        patience=20,
        # Augmentation settings for small objects
        degrees=10.0,
        translate=0.2,
        scale=0.6,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
    )

    print("Training complete!")
    print(f"Best model saved at: {project}/{name}/weights/best.pt")

    # Validation
    metrics = model.val()
    print(f"mAP@50: {metrics.box.map50}")
    print(f"mAP@50-95: {metrics.box.map}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Lesion Detector")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="Device ID")
    parser.add_argument(
        "--name",
        type=str,
        default="train_lesion_detector",
        help="Run name for saving model",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="klgrade-lesion-detection",
        help="WandB project name",
    )

    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Error: Dataset not found at {args.data}")
        sys.exit(1)

    train_lesion_detector(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        device=args.device,
        name=args.name,
        wandb_project=args.wandb_project,
    )
