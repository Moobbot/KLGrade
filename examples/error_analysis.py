"""
Error Analysis Script for Object Detection Models

Performs detailed error analysis on model predictions:
- Confusion matrix analysis (which classes are confused)
- False positives, false negatives, correct detections
- Localization errors (IoU-based)
- Per-image and per-class statistics
- CSV export and text report

Supports both YOLO and DETR models.

Usage:
    # For YOLO predictions
    python examples/error_analysis.py \
        --predictions runs/detect/exp1/predictions.json \
        --ground_truth processed/coco/annotations_val.json \
        --output runs/detect/exp1/error_analysis
    
    # For DETR predictions  
    python examples/error_analysis.py \
        --predictions runs/detr/exp1/evaluation/predictions.json \
        --ground_truth processed/coco/annotations_val.json \
        --output runs/detr/exp1/error_analysis
"""

import sys
from pathlib import Path
import argparse
import json
import csv
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute IoU between two boxes in [x, y, w, h] format.

    Args:
        box1: [x, y, w, h]
        box2: [x, y, w, h]

    Returns:
        IoU score
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert to [x1, y1, x2, y2]
    box1_x2 = x1 + w1
    box1_y2 = y1 + h1
    box2_x2 = x2 + w2
    box2_y2 = y2 + h2

    # Compute intersection
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)

    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    # Compute union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def match_predictions_to_ground_truth(
    pred_boxes: List[Dict], gt_boxes: List[Dict], iou_threshold: float = 0.5
) -> Tuple[List[Tuple], List[int], List[int]]:
    """
    Match predictions to ground truth boxes using greedy IoU matching.

    Args:
        pred_boxes: List of prediction dicts with 'bbox', 'category_id', 'score'
        gt_boxes: List of ground truth dicts with 'bbox', 'category_id'
        iou_threshold: IoU threshold for matching

    Returns:
        matches: List of (pred_idx, gt_idx, iou, pred_class, gt_class)
        unmatched_preds: List of prediction indices
        unmatched_gts: List of ground truth indices
    """
    if len(pred_boxes) == 0:
        return [], [], list(range(len(gt_boxes)))

    if len(gt_boxes) == 0:
        return [], list(range(len(pred_boxes))), []

    # Compute IoU matrix
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, pred in enumerate(pred_boxes):
        for j, gt in enumerate(gt_boxes):
            iou_matrix[i, j] = compute_iou(pred["bbox"], gt["bbox"])

    # Greedy matching: highest IoU first
    matches = []
    matched_preds = set()
    matched_gts = set()

    # Sort all (pred, gt) pairs by IoU
    pairs = []
    for i in range(len(pred_boxes)):
        for j in range(len(gt_boxes)):
            if iou_matrix[i, j] >= iou_threshold:
                pairs.append((i, j, iou_matrix[i, j]))

    pairs.sort(key=lambda x: x[2], reverse=True)

    for pred_idx, gt_idx, iou in pairs:
        if pred_idx not in matched_preds and gt_idx not in matched_gts:
            matches.append(
                (
                    pred_idx,
                    gt_idx,
                    iou,
                    pred_boxes[pred_idx]["category_id"],
                    gt_boxes[gt_idx]["category_id"],
                )
            )
            matched_preds.add(pred_idx)
            matched_gts.add(gt_idx)

    unmatched_preds = [i for i in range(len(pred_boxes)) if i not in matched_preds]
    unmatched_gts = [i for i in range(len(gt_boxes)) if i not in matched_gts]

    return matches, unmatched_preds, unmatched_gts


def analyze_errors(
    predictions_path: str,
    ground_truth_path: str,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.0,
) -> Dict:
    """
    Perform comprehensive error analysis.

    Args:
        predictions_path: Path to predictions JSON (COCO format)
        ground_truth_path: Path to ground truth COCO JSON
        iou_threshold: IoU threshold for matching
        conf_threshold: Confidence threshold for predictions

    Returns:
        Dictionary with error statistics
    """
    print("=" * 60)
    print("Error Analysis")
    print("=" * 60)

    # Load ground truth
    print(f"\nLoading ground truth from {ground_truth_path}...")
    coco_gt = COCO(ground_truth_path)

    # Load predictions
    print(f"Loading predictions from {predictions_path}...")
    with open(predictions_path, "r") as f:
        predictions = json.load(f)

    # Filter by confidence
    predictions = [p for p in predictions if p.get("score", 1.0) >= conf_threshold]
    print(f"✅ Loaded {len(predictions)} predictions (conf >= {conf_threshold})")

    # Get class names
    categories = {cat["id"]: cat["name"] for cat in coco_gt.dataset["categories"]}
    num_classes = len(categories)

    # Initialize statistics
    stats = {
        "total_gt": 0,
        "total_pred": len(predictions),
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "localization_errors": 0,  # Correct class but poor localization
        "classification_errors": 0,  # Good localization but wrong class
        "confusion_matrix": np.zeros((num_classes, num_classes), dtype=int),
        "per_class": defaultdict(
            lambda: {
                "gt_count": 0,
                "pred_count": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "loc_error": 0,
                "cls_error": 0,
            }
        ),
        "per_image": {},
        "error_examples": {
            "false_positives": [],
            "false_negatives": [],
            "classification_errors": [],
            "localization_errors": [],
        },
    }

    # Group predictions by image
    preds_by_image = defaultdict(list)
    for pred in predictions:
        preds_by_image[pred["image_id"]].append(pred)

    # Analyze each image
    print(f"\nAnalyzing {len(coco_gt.imgs)} images...")

    for img_id, img_info in coco_gt.imgs.items():
        # Get ground truth annotations for this image
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        gt_anns = coco_gt.loadAnns(ann_ids)

        # Get predictions for this image
        pred_anns = preds_by_image.get(img_id, [])

        # Match predictions to ground truth
        matches, unmatched_preds, unmatched_gts = match_predictions_to_ground_truth(
            pred_anns, gt_anns, iou_threshold
        )

        # Image-level statistics
        img_stats = {
            "image_id": img_id,
            "filename": img_info["file_name"],
            "gt_count": len(gt_anns),
            "pred_count": len(pred_anns),
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "loc_error": 0,
            "cls_error": 0,
        }

        # Process matches
        for pred_idx, gt_idx, iou, pred_cls, gt_cls in matches:
            stats["total_gt"] += 1

            if pred_cls == gt_cls:
                # True positive
                stats["true_positives"] += 1
                stats["per_class"][gt_cls]["tp"] += 1
                img_stats["tp"] += 1

                # Update confusion matrix (diagonal)
                stats["confusion_matrix"][gt_cls, pred_cls] += 1
            else:
                # Classification error
                stats["classification_errors"] += 1
                stats["per_class"][gt_cls]["cls_error"] += 1
                img_stats["cls_error"] += 1

                # Update confusion matrix (off-diagonal)
                stats["confusion_matrix"][gt_cls, pred_cls] += 1

                # Record example
                if len(stats["error_examples"]["classification_errors"]) < 100:
                    stats["error_examples"]["classification_errors"].append(
                        {
                            "image_id": img_id,
                            "filename": img_info["file_name"],
                            "gt_class": categories[gt_cls],
                            "pred_class": categories[pred_cls],
                            "iou": float(iou),
                            "confidence": float(pred_anns[pred_idx].get("score", 1.0)),
                        }
                    )

        # Process unmatched predictions (false positives)
        for pred_idx in unmatched_preds:
            stats["false_positives"] += 1
            pred_cls = pred_anns[pred_idx]["category_id"]
            stats["per_class"][pred_cls]["fp"] += 1
            img_stats["fp"] += 1

            # Record example
            if len(stats["error_examples"]["false_positives"]) < 100:
                stats["error_examples"]["false_positives"].append(
                    {
                        "image_id": img_id,
                        "filename": img_info["file_name"],
                        "pred_class": categories[pred_cls],
                        "confidence": float(pred_anns[pred_idx].get("score", 1.0)),
                        "bbox": pred_anns[pred_idx]["bbox"],
                    }
                )

        # Process unmatched ground truths (false negatives)
        for gt_idx in unmatched_gts:
            stats["false_negatives"] += 1
            gt_cls = gt_anns[gt_idx]["category_id"]
            stats["per_class"][gt_cls]["fn"] += 1
            img_stats["fn"] += 1
            stats["total_gt"] += 1

            # Record example
            if len(stats["error_examples"]["false_negatives"]) < 100:
                stats["error_examples"]["false_negatives"].append(
                    {
                        "image_id": img_id,
                        "filename": img_info["file_name"],
                        "gt_class": categories[gt_cls],
                        "bbox": gt_anns[gt_idx]["bbox"],
                    }
                )

        # Count ground truth per class
        for gt_ann in gt_anns:
            stats["per_class"][gt_ann["category_id"]]["gt_count"] += 1

        # Count predictions per class
        for pred_ann in pred_anns:
            stats["per_class"][pred_ann["category_id"]]["pred_count"] += 1

        stats["per_image"][img_id] = img_stats

    # Add category names
    stats["categories"] = categories

    return stats


def save_csv_report(stats: Dict, output_path: Path):
    """Save detailed CSV report."""
    csv_path = output_path / "error_analysis.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Per-image statistics
        writer.writerow(["Image-Level Statistics"])
        writer.writerow(
            [
                "Image ID",
                "Filename",
                "GT Count",
                "Pred Count",
                "True Positives",
                "False Positives",
                "False Negatives",
                "Classification Errors",
                "Precision",
                "Recall",
            ]
        )

        for img_id, img_stats in stats["per_image"].items():
            tp = img_stats["tp"]
            fp = img_stats["fp"]
            fn = img_stats["fn"]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            writer.writerow(
                [
                    img_id,
                    img_stats["filename"],
                    img_stats["gt_count"],
                    img_stats["pred_count"],
                    tp,
                    fp,
                    fn,
                    img_stats["cls_error"],
                    f"{precision:.4f}",
                    f"{recall:.4f}",
                ]
            )

        writer.writerow([])
        writer.writerow([])

        # Per-class statistics
        writer.writerow(["Class-Level Statistics"])
        writer.writerow(
            [
                "Class",
                "GT Count",
                "Pred Count",
                "True Positives",
                "False Positives",
                "False Negatives",
                "Classification Errors",
                "Precision",
                "Recall",
                "F1-Score",
            ]
        )

        for cls_id, cls_stats in stats["per_class"].items():
            tp = cls_stats["tp"]
            fp = cls_stats["fp"]
            fn = cls_stats["fn"]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            writer.writerow(
                [
                    stats["categories"][cls_id],
                    cls_stats["gt_count"],
                    cls_stats["pred_count"],
                    tp,
                    fp,
                    fn,
                    cls_stats["cls_error"],
                    f"{precision:.4f}",
                    f"{recall:.4f}",
                    f"{f1:.4f}",
                ]
            )

    print(f"✅ CSV report saved to {csv_path}")


def save_text_report(stats: Dict, output_path: Path):
    """Save human-readable text report."""
    report_path = output_path / "error_report.txt"

    total = stats["total_gt"]
    tp = stats["true_positives"]
    fp = stats["false_positives"]
    fn = stats["false_negatives"]
    cls_err = stats["classification_errors"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("ERROR ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")

        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Total Ground Truth Objects: {total}\n")
        f.write(f"Total Predictions: {stats['total_pred']}\n")
        f.write(f"True Positives: {tp} ({tp/total*100:.2f}%)\n")
        f.write(
            f"False Positives: {fp} ({fp/stats['total_pred']*100:.2f}% of predictions)\n"
        )
        f.write(f"False Negatives (Missed): {fn} ({fn/total*100:.2f}%)\n")
        f.write(f"Classification Errors: {cls_err} ({cls_err/total*100:.2f}%)\n")
        f.write(f"\nPrecision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")

        f.write("\n" + "=" * 60 + "\n\n")

        # Error breakdown
        f.write("ERROR BREAKDOWN\n")
        f.write("-" * 60 + "\n")
        f.write(f"1. Correct Detections (TP): {tp} ({tp/total*100:.1f}%)\n")
        f.write(f"   └─ Correctly detected and classified objects\n\n")

        f.write(
            f"2. False Positives: {fp} ({fp/stats['total_pred']*100:.1f}% of predictions)\n"
        )
        f.write(f"   └─ Model detected objects that don't exist\n\n")

        f.write(f"3. False Negatives: {fn} ({fn/total*100:.1f}% of ground truth)\n")
        f.write(f"   └─ Model failed to detect existing objects\n\n")

        f.write(f"4. Classification Errors: {cls_err} ({cls_err/total*100:.1f}%)\n")
        f.write(f"   └─ Object detected but wrong class predicted\n\n")

        f.write("\n" + "=" * 60 + "\n\n")

        # Per-class statistics
        f.write("PER-CLASS STATISTICS\n")
        f.write("-" * 60 + "\n")
        f.write(
            f"{'Class':<20} {'GT':>6} {'Pred':>6} {'TP':>6} {'FP':>6} {'FN':>6} {'Cls Err':>8} {'Prec':>7} {'Rec':>7} {'F1':>7}\n"
        )
        f.write("-" * 60 + "\n")

        for cls_id in sorted(stats["per_class"].keys()):
            cls_stats = stats["per_class"][cls_id]
            cls_name = stats["categories"][cls_id]

            tp_cls = cls_stats["tp"]
            fp_cls = cls_stats["fp"]
            fn_cls = cls_stats["fn"]
            cls_err_cls = cls_stats["cls_error"]

            prec = tp_cls / (tp_cls + fp_cls) if (tp_cls + fp_cls) > 0 else 0.0
            rec = tp_cls / (tp_cls + fn_cls) if (tp_cls + fn_cls) > 0 else 0.0
            f1_cls = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            f.write(
                f"{cls_name:<20} {cls_stats['gt_count']:>6} {cls_stats['pred_count']:>6} "
                f"{tp_cls:>6} {fp_cls:>6} {fn_cls:>6} {cls_err_cls:>8} "
                f"{prec:>7.3f} {rec:>7.3f} {f1_cls:>7.3f}\n"
            )

        f.write("\n" + "=" * 60 + "\n\n")

        # Confusion matrix
        f.write("CONFUSION MATRIX (Predicted vs Ground Truth)\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'':>20}")
        for cls_id in sorted(stats["categories"].keys()):
            f.write(f"{stats['categories'][cls_id][:8]:>10}")
        f.write("\n")

        for gt_id in sorted(stats["categories"].keys()):
            f.write(f"{stats['categories'][gt_id]:<20}")
            for pred_id in sorted(stats["categories"].keys()):
                count = stats["confusion_matrix"][gt_id, pred_id]
                f.write(f"{count:>10}")
            f.write("\n")

    print(f"✅ Text report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze detection errors")
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions JSON (COCO format)",
    )
    parser.add_argument(
        "--ground_truth", type=str, required=True, help="Path to ground truth COCO JSON"
    )
    parser.add_argument(
        "--iou_threshold", type=float, default=0.5, help="IoU threshold for matching"
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.0,
        help="Confidence threshold for predictions",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for analysis results",
    )

    args = parser.parse_args()

    # Run analysis
    stats = analyze_errors(
        args.predictions, args.ground_truth, args.iou_threshold, args.conf_threshold
    )

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_csv_report(stats, output_dir)
    save_text_report(stats, output_dir)

    # Save raw statistics
    stats_json = output_dir / "statistics.json"

    # Convert numpy arrays to lists for JSON serialization
    stats_for_json = {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in stats.items()
        if k != "per_image"  # Too large
    }

    with open(stats_json, "w") as f:
        json.dump(stats_for_json, f, indent=2)

    print(f"\n✅ Analysis complete! Results saved to {output_dir}")
    print(f"\nSummary:")
    print(f"  Total GT: {stats['total_gt']}")
    print(f"  True Positives: {stats['true_positives']}")
    print(f"  False Positives: {stats['false_positives']}")
    print(f"  False Negatives: {stats['false_negatives']}")
    print(f"  Classification Errors: {stats['classification_errors']}")


if __name__ == "__main__":
    main()
