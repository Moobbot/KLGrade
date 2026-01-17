#!/usr/bin/env python3
"""
Visualize Prediction Results

Draws knee and grade bounding boxes from prediction JSON.

Usage:
    python scripts/visualization/visualize_manual.py \\
        --prediction-json path/to/prediction.json \\
        --image-dir datasets/dataset/dataset_v0/images \\
        --output prediction_vis.jpg
"""

import cv2
import json
import argparse
import sys
from pathlib import Path


def visualize_prediction(image_path, predictions, output_path):
    """
    Visualize prediction results on image.

    Args:
        image_path: Path to input image
        predictions: List of prediction dictionaries
        output_path: Path to save visualization
    """
    print(f"Reading image from: {image_path}")
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Error reading image: {image_path}")
        return False

    for i, pred in enumerate(predictions):
        # 1. Draw Knee Box (Green)
        bbox = pred["knee_bbox"]
        x1, y1, x2, y2 = bbox

        knee_score = pred.get("knee_conf", 0.0)

        kl_info = pred.get("kl_grade", {})
        grade_class = kl_info.get("grade_class", "Unknown")
        grade_conf = kl_info.get("grade_conf", 0.0)
        grade_bbox = kl_info.get("grade_bbox", [])

        # Knee Color: Green
        knee_color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), knee_color, 4)

        # Knee Label
        label = f"Knee ({knee_score:.2f}) | {grade_class}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(img, (x1, y1 - h - 10), (x1 + w, y1), knee_color, -1)
        cv2.putText(
            img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2
        )

        # 2. Draw Grade Box (Red) if exists
        if grade_bbox and len(grade_bbox) == 4:
            gx1, gy1, gx2, gy2 = grade_bbox
            grade_color = (0, 0, 255)  # Red

            cv2.rectangle(img, (gx1, gy1), (gx2, gy2), grade_color, 3)

            # Grade Label
            g_label = f"{grade_class} ({grade_conf:.2f})"
            (gw, gh), _ = cv2.getTextSize(g_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            # Draw label near the grade box
            cv2.rectangle(img, (gx1, gy1 - gh - 5), (gx1 + gw, gy1), grade_color, -1)
            cv2.putText(
                img,
                g_label,
                (gx1, gy1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

    print(f"Saving visualization to: {output_path}")
    cv2.imwrite(str(output_path), img)
    print("✅ Visualization saved successfully")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Visualize prediction results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--prediction-json",
        type=str,
        required=True,
        help="Path to prediction JSON file",
    )

    parser.add_argument(
        "--image-dir",
        type=str,
        help="Directory containing images (optional if image path in JSON is absolute)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="prediction_vis.jpg",
        help="Output image path (default: prediction_vis.jpg)",
    )

    args = parser.parse_args()

    # Load prediction JSON
    json_path = Path(args.prediction_json)
    if not json_path.exists():
        print(f"❌ Prediction JSON not found: {json_path}")
        sys.exit(1)

    with open(json_path, "r") as f:
        json_data = json.load(f)

    # Determine image path
    filename = json_data.get("filename")
    if not filename:
        print("❌ No 'filename' field in JSON")
        sys.exit(1)

    if args.image_dir:
        image_path = Path(args.image_dir) / filename
    else:
        image_path = Path(filename)

    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)

    # Visualize
    predictions = json_data.get("predictions", [])
    if not predictions:
        print("⚠️  No predictions in JSON")

    success = visualize_prediction(image_path, predictions, Path(args.output))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
