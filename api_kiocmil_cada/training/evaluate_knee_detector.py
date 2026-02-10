"""
Evaluate trained Knee Detector on Test set.
"""

from ultralytics import YOLO
from pathlib import Path
import argparse


def evaluate_knee_detector(model_path: str, data_yaml: str):
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
        name="val_knee_detector",
        plots=True,
    )

    print("\nEvaluation Results:")
    print(f"mAP@50:    {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f} (Proxy for Mean IoU performance)")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")

    print("\nNote: mAP@50-95 averages precision over IoU thresholds 0.5-0.95.")
    print(
        "A high mAP@50-95 indicates that predicted boxes overlap well with ground truth (High IoU)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to best.pt")
    parser.add_argument(
        "--data",
        default="processed/knee_detection/dataset.yaml",
        help="Path to dataset.yaml",
    )
    args = parser.parse_args()

    evaluate_knee_detector(args.model, args.data)
