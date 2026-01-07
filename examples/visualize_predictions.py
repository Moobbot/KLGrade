"""
Visualize DETR/YOLO Predictions on Images

Draws predicted bounding boxes on images for visual analysis.
Supports COCO format predictions JSON.

Usage:
    python examples/visualize_predictions.py \
        --predictions runs/detr/exp1_test_5classes/evaluation/predictions.json \
        --ground_truth processed/coco/annotations_val.json \
        --img_dir dataset/dataset_v1/images \
        --output runs/detr/exp1_test_5classes/visualizations \
        --num_images 20 \
        --conf_threshold 0.3
"""

import sys
from pathlib import Path
import argparse
import json
import cv2
import numpy as np
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pycocotools.coco import COCO


# Color palette for classes (BGR format for OpenCV)
COLORS = [
    (0, 255, 0),  # Green - KL0
    (255, 0, 0),  # Blue - KL1
    (0, 0, 255),  # Red - KL2
    (255, 255, 0),  # Cyan - KL3
    (255, 0, 255),  # Magenta - KL4
    (0, 255, 255),  # Yellow - KL5
    (128, 0, 128),  # Purple
    (255, 165, 0),  # Orange
    (0, 128, 128),  # Teal
    (128, 128, 0),  # Olive
]


def draw_box(img, bbox, label, score, color, thickness=2):
    """Draw a single bounding box with label."""
    x, y, w, h = bbox
    x, y, w, h = int(x), int(y), int(w), int(h)

    # Draw rectangle
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

    # Prepare label text
    label_text = f"{label}: {score:.2f}"

    # Get text size for background
    (text_width, text_height), baseline = cv2.getTextSize(
        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )

    # Draw background rectangle for text
    cv2.rectangle(
        img,
        (x, y - text_height - baseline - 5),
        (x + text_width, y),
        color,
        -1,  # Filled
    )

    # Draw text
    cv2.putText(
        img,
        label_text,
        (x, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),  # White text
        1,
    )


def visualize_predictions(
    predictions_path: str,
    ground_truth_path: str,
    img_dir: str,
    output_dir: Path,
    num_images: int = 20,
    conf_threshold: float = 0.5,
    max_boxes_per_image: int = 50,
):
    """
    Visualize predictions on images.

    Args:
        predictions_path: Path to predictions JSON
        ground_truth_path: Path to ground truth COCO JSON
        img_dir: Directory containing images
        output_dir: Output directory for visualizations
        num_images: Number of images to visualize
        conf_threshold: Confidence threshold for predictions
        max_boxes_per_image: Maximum boxes to draw per image (for clarity)
    """
    print("=" * 60)
    print("Visualizing Predictions")
    print("=" * 60)

    # Load ground truth
    print(f"\nLoading ground truth...")
    coco_gt = COCO(ground_truth_path)

    # Load predictions
    print(f"Loading predictions...")
    with open(predictions_path, "r") as f:
        predictions = json.load(f)

    # Filter by confidence
    predictions = [p for p in predictions if p["score"] >= conf_threshold]
    print(f"✅ Loaded {len(predictions)} predictions (conf >= {conf_threshold})")

    # Group predictions by image
    preds_by_image = defaultdict(list)
    for pred in predictions:
        preds_by_image[pred["image_id"]].append(pred)

    # Get category names
    categories = {cat["id"]: cat["name"] for cat in coco_gt.dataset["categories"]}

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select images to visualize
    img_ids = list(preds_by_image.keys())[:num_images]
    if len(img_ids) == 0:
        # If no predictions, use first N images from ground truth
        img_ids = list(coco_gt.imgs.keys())[:num_images]

    print(f"\nVisualizing {len(img_ids)} images...")

    for idx, img_id in enumerate(img_ids):
        img_info = coco_gt.imgs[img_id]
        img_path = Path(img_dir) / img_info["file_name"]

        if not img_path.exists():
            print(f"  ⚠️  Image not found: {img_path}")
            continue

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ⚠️  Failed to load: {img_path}")
            continue

        # Create visualization with GT and Predictions side by side
        h, w = img.shape[:2]
        vis = np.zeros((h, w * 2, 3), dtype=np.uint8)

        # Left: Ground Truth
        img_gt = img.copy()
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        gt_anns = coco_gt.loadAnns(ann_ids)

        for ann in gt_anns:
            cat_id = ann["category_id"]
            color = COLORS[cat_id % len(COLORS)]
            label = categories[cat_id]
            draw_box(img_gt, ann["bbox"], label, 1.0, color, thickness=3)

        # Add title
        cv2.putText(
            img_gt,
            f"Ground Truth ({len(gt_anns)} boxes)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Right: Predictions
        img_pred = img.copy()
        pred_anns = preds_by_image.get(img_id, [])

        # Sort by score and limit
        pred_anns = sorted(pred_anns, key=lambda x: x["score"], reverse=True)
        pred_anns = pred_anns[:max_boxes_per_image]

        for pred in pred_anns:
            cat_id = pred["category_id"]
            color = COLORS[cat_id % len(COLORS)]
            label = categories.get(cat_id, f"Class {cat_id}")
            score = pred["score"]
            draw_box(img_pred, pred["bbox"], label, score, color, thickness=2)

        # Add title
        cv2.putText(
            img_pred,
            f"Predictions ({len(pred_anns)}/{len(preds_by_image.get(img_id, []))} shown)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        # Combine
        vis[:, :w] = img_gt
        vis[:, w:] = img_pred

        # Save
        output_path = output_dir / f"vis_{idx:03d}_img{img_id}.jpg"
        cv2.imwrite(str(output_path), vis)

        print(
            f"  [{idx+1}/{len(img_ids)}] Saved: {output_path.name} (GT: {len(gt_anns)}, Pred: {len(pred_anns)})"
        )

    print(f"\n✅ Visualizations saved to {output_dir}")

    # Create summary
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Visualization Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total predictions: {len(predictions)}\n")
        f.write(f"Images with predictions: {len(preds_by_image)}\n")
        f.write(f"Confidence threshold: {conf_threshold}\n")
        f.write(f"Max boxes per image: {max_boxes_per_image}\n\n")

        f.write("Predictions per image:\n")
        for img_id in sorted(preds_by_image.keys())[:num_images]:
            img_info = coco_gt.imgs.get(img_id, {})
            filename = img_info.get("file_name", f"ID_{img_id}")
            count = len(preds_by_image[img_id])
            f.write(f"  {filename}: {count} predictions\n")

    print(f"✅ Summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize predictions on images")
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
        "--img_dir", type=str, required=True, help="Directory containing images"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output directory for visualizations"
    )
    parser.add_argument(
        "--num_images", type=int, default=20, help="Number of images to visualize"
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for predictions",
    )
    parser.add_argument(
        "--max_boxes", type=int, default=50, help="Maximum boxes to draw per image"
    )

    args = parser.parse_args()

    visualize_predictions(
        args.predictions,
        args.ground_truth,
        args.img_dir,
        Path(args.output),
        args.num_images,
        args.conf_threshold,
        args.max_boxes,
    )


if __name__ == "__main__":
    main()
