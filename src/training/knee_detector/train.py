"""
Train YOLO11 Knee Detector for KIOCMIL-CADA Pipeline.
"""

import argparse
from ultralytics import YOLO
from pathlib import Path
import os
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))


def train_knee_detector(
    model_name: str = "yolo11l.pt",
    data_yaml: str = "processed/knee_detection/dataset.yaml",
    epochs: int = 50,
    imgsz: int = 640,
    batch_size: int = 16,
    device: str = "0",
    project: str = "runs/detect",
    name: str = "train_knee_detector",
    patience: int = 20,
    save: bool = True,
    save_period: int = 10,
    cache: bool = False,
    workers: int = 8,
    pretrained: bool = True,
    optimizer: str = "auto",
    verbose: bool = True,
    plots: bool = True,
    val: bool = True,
):
    """
    Train YOLO11 model for knee detection.

    Args:
        model_name: Name/Path of the YOLO model
        data_yaml: Path to dataset.yaml
        epochs: Number of training epochs
        imgsz: Image size
        batch_size: Batch size
        device: Device ID
        project: Project directory
        name: Run name
        patience: Early stopping patience
        save: Save checkpoints
        save_period: Save checkpoint every x epochs
        cache: Cache images for faster training
        workers: Number of dataloader workers
        pretrained: Use pretrained weights
        optimizer: Optimizer (auto, sgd, adam, adamw)
        verbose: Print verbose output
        plots: Save plots
        val: Validate during training
    """
    # Initialize YOLO model
    print(f"Loading YOLO model: {model_name}...")
    model = YOLO(model_name)

    # Train
    print(f"Starting training for {epochs} epochs...")
    print(f"Project: {project}, Name: {name}, Patience: {patience}")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        exist_ok=False,  # Don't overwrite, error if exists (handled by timestamp naming in bash)
        plots=plots,
        save=save,
        val=val,
        patience=patience,
        save_period=save_period,
        cache=cache,
        workers=workers,
        pretrained=pretrained,
        optimizer=optimizer,
        verbose=verbose,
    )

    print("Training complete!")
    best_model_path = Path(project) / name / "weights" / "best.pt"
    print(f"Best model saved at: {best_model_path}")

    # Validation
    try:
        metrics = model.val()
        print(f"mAP@50: {metrics.box.map50}")
        print(f"mAP@50-95: {metrics.box.map}")
    except Exception as e:
        print(f"Validation step failed: {e}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Knee Detector")
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11l.pt",
        help="Model name or path (e.g., yolo11n.pt, yolo11l.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="processed/knee_detection/dataset.yaml",
        help="Path to dataset.yaml",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="0", help="Device ID")
    parser.add_argument(
        "--project", type=str, default="runs/detect", help="Project directory"
    )
    parser.add_argument(
        "--name", type=str, default="train_knee_detector", help="Run name"
    )
    parser.add_argument(
        "--patience", type=int, default=20, help="Early stopping patience"
    )
    # Boolean flags and other args
    parser.add_argument(
        "--save", type=str, default="True", help="Save checkpoints (True/False)"
    )
    parser.add_argument("--save_period", type=int, default=10, help="Save period")
    parser.add_argument(
        "--cache", type=str, default="False", help="Cache images (True/False/ram/disk)"
    )
    parser.add_argument("--workers", type=int, default=8, help="Number of workers")
    parser.add_argument(
        "--pretrained", type=str, default="True", help="Use pretrained weights"
    )
    parser.add_argument("--optimizer", type=str, default="auto", help="Optimizer")
    parser.add_argument("--verbose", type=str, default="True", help="Verbose output")
    parser.add_argument("--plots", type=str, default="True", help="Save plots")
    parser.add_argument(
        "--val", type=str, default="True", help="Validate during training"
    )

    args = parser.parse_args()

    # Helper to convert string bools
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            # For cache, it can be 'ram' or 'disk' which are truthy strings
            return v

    if not os.path.exists(args.data):
        if not Path(args.data).exists():
            print(f"Warning: Dataset path maybe incorrect: {args.data}")

    train_knee_detector(
        model_name=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        save=str2bool(args.save),
        save_period=args.save_period,
        cache=str2bool(args.cache),
        workers=args.workers,
        pretrained=str2bool(args.pretrained),
        optimizer=args.optimizer,
        verbose=str2bool(args.verbose),
        plots=str2bool(args.plots),
        val=str2bool(args.val),
    )
