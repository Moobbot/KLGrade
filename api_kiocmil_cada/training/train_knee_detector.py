"""
Train YOLO11 Knee Detector for KIOCMIL-CADA Pipeline.
"""

import argparse
from ultralytics import YOLO
from pathlib import Path
import os
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))


def train_knee_detector(
    data_yaml: str,
    epochs: int = 50,
    imgsz: int = 640,
    batch_size: int = 16,
    device: str = "0",
    project: str = "runs/detect",
    name: str = "train_knee_detector",
):
    """
    Train YOLO11 model for knee detection.

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
        patience=20,  # Early stopping
    )

    print("Training complete!")
    print(f"Best model saved at: {project}/{name}/weights/best.pt")

    # Validation
    metrics = model.val()
    print(f"mAP@50: {metrics.box.map50}")
    print(f"mAP@50-95: {metrics.box.map}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Knee Detector")
    parser.add_argument(
        "--data",
        type=str,
        default="processed/knee_detection/dataset.yaml",
        help="Path to dataset.yaml",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="Device ID")

    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Error: Dataset not found at {args.data}")
        sys.exit(1)

    train_knee_detector(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        device=args.device,
    )
