#!/usr/bin/env python3
"""
Evaluate knee detector on all splits (train, val, test)
"""
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from ultralytics import YOLO
import json


def evaluate_split(model, data_yaml, split_name):
    """Evaluate on a specific split"""
    print(f"\n{'='*60}")
    print(f"Evaluating on {split_name.upper()} split")
    print(f"{'='*60}")

    # Run validation
    results = model.val(
        data=data_yaml,
        split=split_name,
        imgsz=640,
        batch=16,
        device=0,
        conf=0.001,
        iou=0.6,
        max_det=300,
        plots=False,
        save_json=False,
    )

    return {
        "split": split_name,
        "images": results.seen if hasattr(results, "seen") else "N/A",
        "map50": float(results.box.map50),
        "map50_95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "speed_preprocess": results.speed["preprocess"],
        "speed_inference": results.speed["inference"],
        "speed_postprocess": results.speed["postprocess"],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate knee detector on all splits")
    parser.add_argument(
        "--model",
        type=str,
        default="runs/my_knee_run_resplit/weights/best.pt",
        help="Path to model weights",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="docs/knee_detector_full_evaluation.json",
        help="Output JSON file path",
    )
    args = parser.parse_args()

    model_path = args.model
    data_yaml = "datasets/splits/knee/dataset.yaml"

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # Evaluate on all splits
    all_results = []

    for split in ["train", "val", "test"]:
        try:
            result = evaluate_split(model, data_yaml, split)
            all_results.append(result)
        except Exception as e:
            print(f"Error evaluating {split}: {e}")
            continue

    # Print summary
    print(f"\n{'='*80}")
    print("COMPLETE EVALUATION SUMMARY")
    print(f"{'='*80}\n")

    print(
        f"{'Split':<10} {'Images':<10} {'mAP@50':<12} {'mAP@50-95':<12} {'Precision':<12} {'Recall':<12}"
    )
    print("-" * 80)

    for r in all_results:
        print(
            f"{r['split']:<10} {str(r['images']):<10} "
            f"{r['map50']:<12.4f} {r['map50_95']:<12.4f} "
            f"{r['precision']:<12.4f} {r['recall']:<12.4f}"
        )

    # Calculate average
    if len(all_results) > 0:
        avg_map50 = sum(r["map50"] for r in all_results) / len(all_results)
        avg_map50_95 = sum(r["map50_95"] for r in all_results) / len(all_results)
        avg_precision = sum(r["precision"] for r in all_results) / len(all_results)
        avg_recall = sum(r["recall"] for r in all_results) / len(all_results)

        print("-" * 80)
        print(
            f"{'AVERAGE':<10} {'':<10} "
            f"{avg_map50:<12.4f} {avg_map50_95:<12.4f} "
            f"{avg_precision:<12.4f} {avg_recall:<12.4f}"
        )

    print(f"\n{'='*80}")

    # Save results
    output_file = args.output
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Results saved to: {output_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
