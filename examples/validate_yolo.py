"""
YOLO Model Validation with Prediction Export

Runs YOLO validation and exports predictions in COCO format for visualization.

Usage:
    python examples/validate_yolo.py \
        --model runs/detect/exp3_filtered_test/weights/best.pt \
        --data processed/yolo11_labels.yaml \
        --output runs/detect/exp3_filtered_test/validation
"""

import sys
from pathlib import Path
import argparse
import json
from ultralytics import YOLO

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def validate_yolo(
    model_path: str,
    data_yaml: str,
    output_dir: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.7,
):
    """
    Run YOLO validation and export predictions.

    Args:
        model_path: Path to YOLO model weights (.pt file)
        data_yaml: Path to data YAML file
        output_dir: Output directory for results
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
    """
    print("=" * 60)
    print("YOLO Model Validation")
    print("=" * 60)

    print(f"\nConfiguration:")
    print(f"  Model: {model_path}")
    print(f"  Data: {data_yaml}")
    print(f"  Conf threshold: {conf_threshold}")
    print(f"  IoU threshold: {iou_threshold}")

    # Load model
    print(f"\nLoading model...")
    model = YOLO(model_path)

    # Run validation
    print(f"\nRunning validation...")
    results = model.val(
        data=data_yaml,
        conf=conf_threshold,
        iou=iou_threshold,
        save_json=True,  # Save predictions in COCO format
        save_txt=False,
        save_conf=True,
        plots=True,
        device=0,
        project=output_dir,
        name="",
        exist_ok=True,
    )

    print(f"\n✅ Validation complete!")
    print(f"\nResults:")
    print(f"  mAP50-95: {results.box.map:.4f}")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  Precision: {results.box.mp:.4f}")
    print(f"  Recall: {results.box.mr:.4f}")

    # Find the predictions JSON file
    output_path = Path(output_dir)
    prediction_files = list(output_path.glob("**/predictions.json"))

    if prediction_files:
        pred_file = prediction_files[0]
        print(f"\n✅ Predictions saved to: {pred_file}")

        # Load and count predictions
        with open(pred_file, "r") as f:
            predictions = json.load(f)
        print(f"  Total predictions: {len(predictions)}")
    else:
        print(f"\n⚠️  No predictions.json file found")
        print(f"  Check: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate YOLO model and export predictions"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to YOLO model weights (.pt file)"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to data YAML file"
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS")
    parser.add_argument(
        "--output", type=str, required=True, help="Output directory for results"
    )

    args = parser.parse_args()

    validate_yolo(args.model, args.data, args.output, args.conf, args.iou)


if __name__ == "__main__":
    main()
