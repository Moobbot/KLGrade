"""
Evaluate All Trained YOLO Models

Runs validation on all 15 trained lesion detection models and generates
a comprehensive performance report.
"""

import subprocess
import sys
from pathlib import Path
import json
import csv
from datetime import datetime
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# All trained models with their configurations
MODELS = {
    "cropped_base": [
        {
            "name": "5-class Cropped (Base)",
            "model": "runs/detect/lesion_5class_base/weights/best.pt",
            "data": "datasets/splits/dataset_knees_cropped_70_20_10/dataset.yaml",
            "classes": 5,
        },
        {
            "name": "4-class Cropped (Base)",
            "model": "runs/detect/lesion_4class_base/weights/best.pt",
            "data": "datasets/splits/dataset_knees_cropped_4_class_70_20_10/dataset.yaml",
            "classes": 4,
        },
        {
            "name": "8-class Cropped (Base)",
            "model": "runs/detect/lesion_8class_base/weights/best.pt",
            "data": "datasets/splits/dataset_knees_cropped_8_class_70_20_10/dataset.yaml",
            "classes": 8,
        },
        {
            "name": "10-class Cropped (Base)",
            "model": "runs/detect/lesion_10class_base/weights/best.pt",
            "data": "datasets/splits/dataset_knees_cropped_10_class_60_20_20/dataset.yaml",
            "classes": 10,
        },
    ],
    "cropped_balanced": [
        {
            "name": "5-class Cropped (Balanced)",
            "model": "runs/detect/lesion_5class_balanced/weights/best.pt",
            "data": "datasets/splits/balanced_knees_cropped_70_20_10/dataset.yaml",
            "classes": 5,
        },
        {
            "name": "4-class Cropped (Balanced)",
            "model": "runs/detect/lesion_4class_balanced/weights/best.pt",
            "data": "datasets/splits/balanced_knees_cropped_4_class_70_20_10/dataset.yaml",
            "classes": 4,
        },
        {
            "name": "8-class Cropped (Balanced)",
            "model": "runs/detect/lesion_8class_balanced/weights/best.pt",
            "data": "datasets/splits/balanced_knees_cropped_8_class_70_20_10/dataset.yaml",
            "classes": 8,
        },
        {
            "name": "10-class Cropped (Balanced)",
            "model": "runs/detect/lesion_10class_balanced/weights/best.pt",
            "data": "datasets/splits/balanced_knees_cropped_10_class_60_20_20/dataset.yaml",
            "classes": 10,
        },
    ],
    "full_xray_base": [
        {
            "name": "4-class Full X-ray (Base)",
            "model": "runs/detect/lesion_full_4class_base/weights/best.pt",
            "data": "datasets/splits/knee_full_4_class_70_20_10/dataset.yaml",
            "classes": 4,
        },
        {
            "name": "8-class Full X-ray (Base)",
            "model": "runs/detect/lesion_full_8class_base/weights/best.pt",
            "data": "datasets/splits/knee_full_8_class_70_20_10/dataset.yaml",
            "classes": 8,
        },
        {
            "name": "10-class Full X-ray (Base)",
            "model": "runs/detect/lesion_full_10class_base/weights/best.pt",
            "data": "datasets/splits/knee_full_10_class_70_20_10/dataset.yaml",
            "classes": 10,
        },
    ],
    "full_xray_balanced": [
        {
            "name": "4-class Full X-ray (Balanced)",
            "model": "runs/detect/lesion_full_4class_balanced/weights/best.pt",
            "data": "datasets/splits/balanced_full_xray_4_class_70_20_10/dataset.yaml",
            "classes": 4,
        },
        {
            "name": "8-class Full X-ray (Balanced)",
            "model": "runs/detect/lesion_full_8class_balanced/weights/best.pt",
            "data": "datasets/splits/balanced_full_xray_8_class_70_20_10/dataset.yaml",
            "classes": 8,
        },
        {
            "name": "10-class Full X-ray (Balanced)",
            "model": "runs/detect/lesion_full_10class_balanced/weights/best.pt",
            "data": "datasets/splits/balanced_full_xray_10_class_60_20_20/dataset.yaml",
            "classes": 10,
        },
    ],
    "detection": [
        {
            "name": "Knee Detection",
            "model": "runs/detect/knee_detector/weights/best.pt",
            "data": "datasets/splits/knee/dataset.yaml",
            "classes": 1,
        },
    ],
}


def run_evaluation(config: dict):
    """Run YOLO validation for a single model."""
    print(f"\n{'='*80}")
    print(f"Evaluating: {config['name']}")
    print(f"{'='*80}")
    print(f"Model: {config['model']}")
    print(f"Data: {config['data']}")

    model_path = PROJECT_ROOT / config["model"]
    data_path = PROJECT_ROOT / config["data"]

    if not model_path.exists():
        print(f"⚠️  Model not found: {model_path}")
        return None

    if not data_path.exists():
        print(f"⚠️  Dataset not found: {data_path}")
        return None

    try:
        # Import YOLO here to ensure we use the library available in the execution environment
        from ultralytics import YOLO

        # Load model
        model = YOLO(str(model_path))

        # Run validation
        metrics_obj = model.val(
            data=str(data_path),
            split="test",
            save_json=True,
            save_conf=True,
            device="0",  # Assuming GPU 0
            verbose=False,
        )

        # Extract metrics
        metrics = {"mAP50": metrics_obj.box.map50, "mAP50-95": metrics_obj.box.map}

        print(f"✅ {config['name']} evaluated")
        print(f"   mAP@50: {metrics.get('mAP50', 'N/A')}")
        print(f"   mAP@50-95: {metrics.get('mAP50-95', 'N/A')}")

        return {
            "name": config["name"],
            "model": config["model"],
            "classes": config["classes"],
            **metrics,
        }

    except Exception as e:
        print(f"❌ {config['name']} failed")
        print(f"   Error: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    print("=" * 80)
    print("Evaluating All Trained YOLO Models")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    all_results = []

    # Evaluate all models
    for category, configs in MODELS.items():
        print(f"\n{'='*80}")
        print(f"Category: {category}")
        print(f"{'='*80}")

        for config in configs:
            result = run_evaluation(config)
            if result:
                result["category"] = category
                all_results.append(result)

    # Save results
    output_dir = PROJECT_ROOT / "doc-training" / datetime.now().strftime("%Y/%m_%d")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as CSV
    csv_path = output_dir / "evaluation_summary.csv"
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Results saved to: {csv_path}")

    # Save as JSON
    json_path = output_dir / "evaluation_details.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"✅ Details saved to: {json_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    print(f"Total models evaluated: {len(all_results)}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    if all_results:
        print("\nTop 5 Models by mAP@50:")
        sorted_results = sorted(
            all_results, key=lambda x: x.get("mAP50", 0), reverse=True
        )
        for i, result in enumerate(sorted_results[:5], 1):
            print(f"{i}. {result['name']}: mAP@50={result.get('mAP50', 'N/A'):.3f}")


if __name__ == "__main__":
    main()
