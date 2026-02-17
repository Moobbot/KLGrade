#!/usr/bin/env python3
"""
Visualize detection results with bounding boxes.
Filters boxes by confidence threshold to show only high-quality detections.
"""

import json
import cv2
import numpy as np
from pathlib import Path
import sys


def draw_boxes(
    image_path,
    predictions,
    output_path,
    knee_conf_threshold=0.5,
    lesion_conf_threshold=0.5,
):
    """
    Draw bounding boxes on image.

    Args:
        image_path: Path to input image
        predictions: Prediction dict with boxes
        output_path: Path to save annotated image
        knee_conf_threshold: Min confidence for knee boxes
        lesion_conf_threshold: Min confidence for lesion boxes
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return

    h, w = img.shape[:2]

    # Draw knee boxes (green)
    knee_boxes = predictions.get("knee_boxes", [])
    high_conf_knees = [b for b in knee_boxes if b["confidence"] >= knee_conf_threshold]

    print(
        f"\nKnee boxes: {len(knee_boxes)} total, {len(high_conf_knees)} above {knee_conf_threshold} confidence"
    )

    for box in high_conf_knees:
        bbox = box["bbox"]  # [x1, y1, x2, y2] normalized
        conf = box["confidence"]

        # Convert to pixel coordinates
        x1 = int(bbox[0] * w)
        y1 = int(bbox[1] * h)
        x2 = int(bbox[2] * w)
        y2 = int(bbox[3] * h)

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        label = f"Knee {conf:.2f}"
        cv2.putText(
            img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

    # Draw lesion boxes (red for OST, blue for JS)
    lesion_boxes = predictions.get("lesion_boxes", [])
    high_conf_lesions = [
        b for b in lesion_boxes if b["confidence"] >= lesion_conf_threshold
    ]

    print(
        f"Lesion boxes: {len(lesion_boxes)} total, {len(high_conf_lesions)} above {lesion_conf_threshold} confidence"
    )

    for box in high_conf_lesions:
        bbox = box["bbox"]
        conf = box["confidence"]
        lesion_type = box.get("type", "Unknown")

        # Convert to pixel coordinates
        x1 = int(bbox[0] * w)
        y1 = int(bbox[1] * h)
        x2 = int(bbox[2] * w)
        y2 = int(bbox[3] * h)

        # Color based on type
        color = (
            (0, 0, 255) if lesion_type == "OST" else (255, 0, 0)
        )  # Red for OST, Blue for JS

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{lesion_type} {conf:.2f}"
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Draw prediction at top
    pred_class = predictions["predicted_class"]
    pred_conf = predictions["confidence"]
    pred_text = f"Prediction: {pred_class} ({pred_conf:.2%})"

    cv2.rectangle(img, (10, 10), (400, 50), (0, 0, 0), -1)
    cv2.putText(img, pred_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Save
    cv2.imwrite(output_path, img)
    print(f"\n✅ Saved visualization to {output_path}")


def main():
    # Load prediction results
    result_file = "outputs/api_test_with_boxes/result.json"

    with open(result_file, "r") as f:
        results = json.load(f)

    if not results:
        print("No results found")
        return

    pred = results[0]
    image_path = pred["image_path"]

    print(f"Image: {image_path}")
    print(f"Predicted: {pred['predicted_class']} ({pred['confidence']:.2%})")

    # Try different confidence thresholds
    thresholds = [
        (0.5, 0.5, "high_conf"),
        (0.3, 0.3, "medium_conf"),
        (0.1, 0.1, "low_conf"),
    ]

    for knee_th, lesion_th, suffix in thresholds:
        output_path = f"outputs/api_test_with_boxes/visualization_{suffix}.jpg"
        print(f"\n{'='*60}")
        print(
            f"Creating visualization with thresholds: knee={knee_th}, lesion={lesion_th}"
        )
        print(f"{'='*60}")
        draw_boxes(image_path, pred, output_path, knee_th, lesion_th)


if __name__ == "__main__":
    main()
