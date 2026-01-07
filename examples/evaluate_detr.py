"""
DETR Model Evaluation Script

Evaluates a trained DETR model on validation dataset and computes COCO metrics:
- mAP50 (IoU=0.5)
- mAP50-95 (IoU=0.5:0.95)
- Precision and Recall (per class and overall)

Usage:
    python examples/evaluate_detr.py \
        --model_path runs/detr/exp1_test_5classes/best_model.pt \
        --img_dir dataset/dataset_v1/images \
        --label_dir dataset/dataset_v1/labels \
        --output runs/detr/exp1_test_5classes/evaluation
"""

import sys
from pathlib import Path
import argparse
import json
import torch
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import CocoDataset, create_coco_json, get_detr_processor
from config import CLASSES, CLASSES_LABEL_NEW, CLASSES_FILTERED
from transformers import DetrForObjectDetection
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def load_model(checkpoint_path: str, num_classes: int, device: str = "cuda"):
    """
    Load DETR model from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        num_classes: Number of object classes
        device: Device to load model on

    Returns:
        Loaded DETR model
    """
    print(f"Loading model from {checkpoint_path}...")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Initialize model architecture
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50", num_labels=num_classes, ignore_mismatched_sizes=True
    )

    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"✅ Model loaded successfully")
    return model


def run_inference(model, dataset, device: str = "cuda", conf_threshold: float = 0.5):
    """
    Run inference on dataset and collect predictions.

    Args:
        model: DETR model
        dataset: CocoDataset instance
        device: Device to run inference on
        conf_threshold: Confidence threshold for predictions

    Returns:
        List of predictions in COCO format
    """
    predictions = []

    print(f"\nRunning inference on {len(dataset)} images...")

    with torch.no_grad():
        for idx in tqdm(range(len(dataset))):
            # Get sample
            sample = dataset[idx]
            pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
            pixel_mask = sample["pixel_mask"].unsqueeze(0).to(device)

            # Get image info
            img_info = dataset.get_image_info(idx)
            img_id = img_info["id"]
            img_width = img_info["width"]
            img_height = img_info["height"]

            # Run inference
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

            # Post-process predictions
            # Convert outputs to probabilities
            logits = outputs.logits[0]  # (num_queries, num_classes)
            boxes = outputs.pred_boxes[0]  # (num_queries, 4)

            # Get probabilities and predicted classes
            probs = logits.softmax(-1)  # (num_queries, num_classes)

            # For each query, get the best class (excluding background/no-object)
            # DETR has num_classes + 1 outputs (including no-object class)
            scores, labels = probs[:, :-1].max(-1)  # Exclude last class (no-object)

            # Filter by confidence threshold
            keep = scores > conf_threshold

            scores = scores[keep].cpu().numpy()
            labels = labels[keep].cpu().numpy()
            boxes = boxes[keep].cpu().numpy()

            # Convert boxes from normalized [cx, cy, w, h] to [x, y, w, h] in pixels
            for score, label, box in zip(scores, labels, boxes):
                cx, cy, w, h = box

                # Convert to pixel coordinates
                cx = cx * img_width
                cy = cy * img_height
                w = w * img_width
                h = h * img_height

                # Convert from center format to top-left format
                x = cx - w / 2
                y = cy - h / 2

                # Clip to image boundaries
                x = max(0, min(x, img_width))
                y = max(0, min(y, img_height))
                w = max(0, min(w, img_width - x))
                h = max(0, min(h, img_height - y))

                predictions.append(
                    {
                        "image_id": int(img_id),
                        "category_id": int(label),
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "score": float(score),
                    }
                )

    print(f"✅ Generated {len(predictions)} predictions")
    return predictions


def evaluate_coco(gt_json_path: str, predictions: list, output_dir: Path):
    """
    Evaluate predictions using COCO metrics.

    Args:
        gt_json_path: Path to ground truth COCO JSON
        predictions: List of predictions in COCO format
        output_dir: Directory to save results

    Returns:
        Dictionary of metrics
    """
    print("\n" + "=" * 60)
    print("COCO Evaluation")
    print("=" * 60)

    # Load ground truth
    coco_gt = COCO(gt_json_path)

    # Create predictions in COCO result format
    if len(predictions) == 0:
        print("⚠️  No predictions to evaluate!")
        return {}

    # Load detections
    coco_dt = coco_gt.loadRes(predictions)

    # Run COCO evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract metrics
    metrics = {
        "mAP50-95": coco_eval.stats[0],  # AP @ IoU=0.50:0.95
        "mAP50": coco_eval.stats[1],  # AP @ IoU=0.50
        "mAP75": coco_eval.stats[2],  # AP @ IoU=0.75
        "mAP_small": coco_eval.stats[3],  # AP for small objects
        "mAP_medium": coco_eval.stats[4],  # AP for medium objects
        "mAP_large": coco_eval.stats[5],  # AP for large objects
        "AR_max1": coco_eval.stats[6],  # AR with max 1 detection
        "AR_max10": coco_eval.stats[7],  # AR with max 10 detections
        "AR_max100": coco_eval.stats[8],  # AR with max 100 detections
        "AR_small": coco_eval.stats[9],  # AR for small objects
        "AR_medium": coco_eval.stats[10],  # AR for medium objects
        "AR_large": coco_eval.stats[11],  # AR for large objects
    }

    # Per-class metrics
    per_class_metrics = {}
    for cat_id, cat_info in coco_gt.cats.items():
        cat_name = cat_info["name"]

        # Evaluate for this category only
        coco_eval_cat = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval_cat.params.catIds = [cat_id]
        coco_eval_cat.evaluate()
        coco_eval_cat.accumulate()

        per_class_metrics[cat_name] = {
            "mAP50-95": float(coco_eval_cat.stats[0]),
            "mAP50": float(coco_eval_cat.stats[1]),
        }

    metrics["per_class"] = per_class_metrics

    # Save metrics to JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ Metrics saved to {metrics_file}")

    # Save human-readable summary
    summary_file = output_dir / "results.txt"
    with open(summary_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("DETR Model Evaluation Results\n")
        f.write("=" * 60 + "\n\n")

        f.write("Overall Metrics:\n")
        f.write(f"  mAP50-95: {metrics['mAP50-95']:.4f}\n")
        f.write(f"  mAP50:    {metrics['mAP50']:.4f}\n")
        f.write(f"  mAP75:    {metrics['mAP75']:.4f}\n")
        f.write(f"  AR@100:   {metrics['AR_max100']:.4f}\n\n")

        f.write("Per-Class Metrics:\n")
        for class_name, class_metrics in per_class_metrics.items():
            f.write(
                f"  {class_name:20s} - mAP50: {class_metrics['mAP50']:.4f}, "
                f"mAP50-95: {class_metrics['mAP50-95']:.4f}\n"
            )

    print(f"✅ Summary saved to {summary_file}")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DETR model on validation set"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default="dataset/dataset_v1/images",
        help="Directory containing images",
    )
    parser.add_argument(
        "--label_dir",
        type=str,
        default="dataset/dataset_v1/labels",
        help="Base directory for labels",
    )
    parser.add_argument(
        "--use_labels_new",
        action="store_true",
        help="Use labels_new (10 classes) instead of labels (5 classes)",
    )
    parser.add_argument(
        "--use_filtered", action="store_true", help="Use filtered dataset (7 classes)"
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for predictions",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for evaluation results",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("DETR Model Evaluation")
    print("=" * 60)

    # Determine class configuration
    if args.use_filtered:
        class_names = CLASSES_FILTERED
        label_suffix = "_filtered"
    elif args.use_labels_new:
        class_names = CLASSES_LABEL_NEW
        label_suffix = "_new"
    else:
        class_names = CLASSES
        label_suffix = ""

    num_classes = len(class_names)

    print(f"\nConfiguration:")
    print(f"  Model: {args.model_path}")
    print(f"  Classes: {num_classes}")
    print(f"  Confidence threshold: {args.conf_threshold}")
    print(f"  Device: {args.device}")

    # Step 1: Prepare COCO annotations if not exists
    coco_dir = Path("processed/coco")
    val_json = coco_dir / f"annotations_val{label_suffix}.json"

    if not val_json.exists():
        print(f"\n⚠️  COCO validation annotations not found. Creating...")
        from datasets import create_coco_json

        actual_label_dir = (
            Path(args.label_dir).parent
            / f"labels{label_suffix.replace('_filtered', '')}"
        )
        if args.use_filtered:
            actual_label_dir = Path("dataset/dataset_filtered/labels")

        create_coco_json(
            yolo_label_dir=str(actual_label_dir),
            img_dir=args.img_dir,
            output_path=str(val_json),
            class_names=class_names,
            split_file="splits/val.txt",
        )

    # Step 2: Load model
    model = load_model(args.model_path, num_classes, args.device)

    # Step 3: Create dataset (without processor for manual inference)
    processor = get_detr_processor()
    dataset = CocoDataset(
        coco_json_path=str(val_json), img_dir=args.img_dir, processor=processor
    )

    # Step 4: Run inference
    predictions = run_inference(model, dataset, args.device, args.conf_threshold)

    # Step 5: Evaluate with COCO metrics
    output_dir = Path(args.output)
    metrics = evaluate_coco(str(val_json), predictions, output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"mAP50-95: {metrics.get('mAP50-95', 0):.4f}")
    print(f"mAP50:    {metrics.get('mAP50', 0):.4f}")
    print(f"mAP75:    {metrics.get('mAP75', 0):.4f}")
    print(f"AR@100:   {metrics.get('AR_max100', 0):.4f}")
    print("=" * 60)

    print(f"\n✅ Evaluation complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
