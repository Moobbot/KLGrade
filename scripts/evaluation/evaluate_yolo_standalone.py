#!/usr/bin/env python3
"""
Standalone YOLO Evaluation Script

This script allows for easy evaluation of trained YOLO models on specific dataset splits.
It is designed to be independent of the full training pipeline.

Usage:
    python scripts/evaluation/evaluate_yolo_standalone.py \\
        --model runs/detect/train/weights/best.pt \\
        --data configs/yolo_5_class_baseline.yaml \\
        --split test
"""

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO

# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.yaml_config import create_yolo_config


def evaluate_model(
    model_path: str,
    data_path: str = None,
    dataset_dir: str = None,
    nc: int = None,
    class_names: str = None,
    split: str = "test",
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = "0",
    project: str = "runs/evaluate",
    name: str = "eval",
):
    """
    Evaluate a YOLO model on a specific split.
    """
    print("=" * 60)
    print(f"YOLO Standalone Evaluation")
    print("=" * 60)

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return False

    # Handle dataset configuration
    if dataset_dir:
        if not nc or not class_names:
            print("❌ Error: --dataset-dir requires --nc and --class-names")
            return False

        dataset_dir = Path(dataset_dir)
        names_list = [n.strip() for n in class_names.split(",")]

        if len(names_list) != nc:
            print(
                f"❌ Error: nc ({nc}) does not match number of class names ({len(names_list)})"
            )
            return False

        project_path = Path(project) / name
        data_path = create_yolo_config(
            dataset_dir=dataset_dir,
            nc=nc,
            names=names_list,
            output_path=project_path / "temp_eval_config.yaml",
        )
    elif not data_path:
        print(
            "❌ Error: Must provide either --data or (--dataset-dir, --nc, --class-names)"
        )
        return False

    print(f"Configuration:")
    print(f"  Model: {model_path}")
    print(f"  Data: {data_path}")
    print(f"  Split: {split}")
    print(f"  Device: {device}")

    try:
        # Load model
        print(f"\nLoading model...")
        model = YOLO(str(model_path))

        # Run validation
        print(f"\nRunning evaluation on {split} set...")
        results = model.val(
            data=str(data_path),
            split=split,
            batch=batch_size,
            imgsz=imgsz,
            device=device,
            project=project,
            name=name,
            exist_ok=True,
            plots=True,
            save=True,
            verbose=True,
        )

        print("\n" + "=" * 60)
        print("✅ Evaluation Complete")
        print("=" * 60)

        # Print key metrics
        print(f"\nKey Metrics on {split}:")
        print(f"  mAP50-95: {results.box.map:.4f}")
        print(f"  mAP50:    {results.box.map50:.4f}")
        print(f"  Precision: {results.box.mp:.4f}")
        print(f"  Recall:    {results.box.mr:.4f}")

        print(f"\nResults saved to: {results.save_dir}")

        # Save metrics to JSON
        metrics = {
            "map50_95": results.box.map,
            "map50": results.box.map50,
            "precision": results.box.mp,
            "recall": results.box.mr,
        }

        import json

        metrics_file = Path(results.save_dir) / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Metrics saved to: {metrics_file}")

        return True

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone YOLO Evaluation")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to YOLO model weights (.pt)"
    )

    # Option 1: Existing YAML
    parser.add_argument("--data", type=str, help="Path to data config (.yaml)")

    # Option 2: CLI Config
    parser.add_argument(
        "--dataset-dir",
        type=str,
        help="Directory containing train.txt/val.txt/test.txt",
    )
    parser.add_argument("--nc", type=int, help="Number of classes")
    parser.add_argument(
        "--class-names",
        type=str,
        help="Comma-separated class names (e.g. 'cat,dog,car')",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Split to evaluate",
    )
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="0", help="Device (0, 1, cpu)")
    parser.add_argument(
        "--project", type=str, default="runs/evaluate", help="Output project directory"
    )
    parser.add_argument("--name", type=str, default="eval", help="Experiment name")

    args = parser.parse_args()

    success = evaluate_model(
        model_path=args.model,
        data_path=args.data,
        dataset_dir=args.dataset_dir,
        nc=args.nc,
        class_names=args.class_names,
        split=args.split,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
    )

    sys.exit(0 if success else 1)
