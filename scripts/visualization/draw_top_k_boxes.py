#!/usr/bin/env python3
"""
Visualize TOP-K highest confidence boxes to see what model is detecting.
"""

import json
import cv2
import numpy as np
from pathlib import Path


def draw_top_k_boxes(image_path, predictions, output_path, top_k=10):
    """Draw only the top-k highest confidence boxes."""
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return

    h, w = img.shape[:2]

    # Get knee boxes sorted by confidence
    knee_boxes = predictions.get("knee_boxes", [])
    knee_boxes_sorted = sorted(knee_boxes, key=lambda x: x["confidence"], reverse=True)
    top_knees = knee_boxes_sorted[:top_k]

    print(f"\nTop {top_k} Knee boxes:")
    for i, box in enumerate(top_knees, 1):
        bbox = box["bbox"]
        conf = box["confidence"]
        print(
            f"  {i}. Confidence: {conf:.6f}, BBox: [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]"
        )

        # Convert to pixel coordinates
        x1 = int(bbox[0] * w)
        y1 = int(bbox[1] * h)
        x2 = int(bbox[2] * w)
        y2 = int(bbox[3] * h)

        # Draw rectangle (green)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Draw label
        label = f"K{i}: {conf:.4f}"
        cv2.putText(
            img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

    # Get lesion boxes sorted by confidence
    lesion_boxes = predictions.get("lesion_boxes", [])
    lesion_boxes_sorted = sorted(
        lesion_boxes, key=lambda x: x["confidence"], reverse=True
    )
    top_lesions = lesion_boxes_sorted[:top_k]

    print(f"\nTop {top_k} Lesion boxes:")
    for i, box in enumerate(top_lesions, 1):
        bbox = box["bbox"]
        conf = box["confidence"]
        lesion_type = box.get("type", "Unknown")
        print(
            f"  {i}. Type: {lesion_type}, Confidence: {conf:.6f}, BBox: [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]"
        )

        # Convert to pixel coordinates
        x1 = int(bbox[0] * w)
        y1 = int(bbox[1] * h)
        x2 = int(bbox[2] * w)
        y2 = int(bbox[3] * h)

        # Color based on type (red for OST, blue for JS)
        color = (0, 0, 255) if lesion_type == "OST" else (255, 0, 0)

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"L{i}:{lesion_type[:1]} {conf:.4f}"
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw prediction at top
    pred_class = predictions["predicted_class"]
    pred_conf = predictions["confidence"]
    pred_text = f"Prediction: {pred_class} ({pred_conf:.2%})"

    cv2.rectangle(img, (10, 10), (500, 50), (0, 0, 0), -1)
    cv2.putText(img, pred_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

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
    print(f"Total knee boxes: {len(pred.get('knee_boxes', []))}")
    print(f"Total lesion boxes: {len(pred.get('lesion_boxes', []))}")

    # Draw top-20 boxes
    output_path = "outputs/api_test_with_boxes/visualization_top20.jpg"
    draw_top_k_boxes(image_path, pred, output_path, top_k=20)


if __name__ == "__main__":
    main()
