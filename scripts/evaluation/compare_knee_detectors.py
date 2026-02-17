#!/usr/bin/env python3
"""
Compare two knee detector models
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from ultralytics import YOLO
import json


def evaluate_model(model_path, model_name, data_yaml):
    """Evaluate a single model on all splits"""
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name}")
    print(f"Model: {model_path}")
    print(f"{'='*80}")

    model = YOLO(model_path)
    results = {}

    for split in ["train", "val", "test"]:
        print(f"\n{split.upper()} split...")
        r = model.val(
            data=data_yaml,
            split=split,
            imgsz=640,
            batch=16,
            device=0,
            conf=0.001,
            iou=0.6,
            plots=False,
            save_json=False,
            verbose=False,
        )

        results[split] = {
            "map50": float(r.box.map50),
            "map50_95": float(r.box.map),
            "precision": float(r.box.mp),
            "recall": float(r.box.mr),
        }

    return results


def main():
    data_yaml = "datasets/splits/knee/dataset.yaml"

    models = {
        "knee_detector": "runs/detect/knee_detector/weights/best.pt",
        "my_knee_run_resplit": "runs/my_knee_run_resplit/weights/best.pt",
    }

    all_results = {}

    for name, path in models.items():
        if Path(path).exists():
            all_results[name] = evaluate_model(path, name, data_yaml)
        else:
            print(f"\n⚠️  Model not found: {path}")

    # Print comparison
    print(f"\n{'='*80}")
    print("MODEL COMPARISON")
    print(f"{'='*80}\n")

    for model_name, results in all_results.items():
        print(f"\n{model_name.upper()}:")
        print(
            f"{'Split':<10} {'mAP@50':<12} {'mAP@50-95':<12} {'Precision':<12} {'Recall':<12}"
        )
        print("-" * 60)

        for split, metrics in results.items():
            print(
                f"{split:<10} {metrics['map50']:<12.4f} {metrics['map50_95']:<12.4f} "
                f"{metrics['precision']:<12.4f} {metrics['recall']:<12.4f}"
            )

        # Average
        avg_map50 = sum(m["map50"] for m in results.values()) / len(results)
        avg_map50_95 = sum(m["map50_95"] for m in results.values()) / len(results)
        avg_precision = sum(m["precision"] for m in results.values()) / len(results)
        avg_recall = sum(m["recall"] for m in results.values()) / len(results)

        print("-" * 60)
        print(
            f"{'AVERAGE':<10} {avg_map50:<12.4f} {avg_map50_95:<12.4f} "
            f"{avg_precision:<12.4f} {avg_recall:<12.4f}"
        )

    # Save results
    output_file = "docs/knee_detector_comparison.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✅ Results saved to: {output_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
