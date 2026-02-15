"""
Evaluate trained Lesion Detector on Test set.
Classes: 0: Osteophytes, 1: Joint Space Narrowing
"""

from ultralytics import YOLO
import argparse
import sys
from pathlib import Path


def evaluate_lesion_detector(model_path: str, data_yaml: str):
    print(f"Loading model from {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Running evaluation on Test split...")
    metrics = model.val(
        data=data_yaml,
        split="test",
        project="runs/detect",
        name="val_lesion_detector",
        plots=True,
    )

    print("\nEvaluation Results:")
    print(f"mAP@50:    {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f} (Proxy for Mean IoU performance)")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")

    # Class-wise metrics if available
    # metrics.box.maps is array of maps per class
    if len(metrics.box.maps) >= 2:
        print("\nClass-wise mAP@50-95:")
        print(f"  Osteophytes (0): {metrics.box.maps[0]:.4f}")
        print(f"  Joint Space (1): {metrics.box.maps[1]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to best.pt")
    parser.add_argument(
        "--data",
        default="datasets/processed/lesion_detection/dataset.yaml",
        help="Path to dataset.yaml",
    )
    args = parser.parse_args()

    evaluate_lesion_detector(args.model, args.data)
