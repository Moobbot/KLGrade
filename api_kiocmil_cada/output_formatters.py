"""
Output formatters for KIOCMIL-CADA inference results.
"""

import json
import csv
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any


def save_json(results: List[Dict], output_path: str):
    """
    Save inference results to JSON file.

    Args:
        results: List of prediction dictionaries
        output_path: Path to save JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Results saved to {output_path}")


def save_csv(results: List[Dict], output_path: str):
    """
    Save inference results to CSV file.

    Args:
        results: List of prediction dictionaries
        output_path: Path to save CSV file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Flatten results for CSV
    rows = []
    for result in results:
        image_path = result.get("image_path", "")

        if "knees" in result:
            for i, knee in enumerate(result["knees"]):
                row = {
                    "image_path": image_path,
                    "knee_id": i,
                    "predicted_class": knee.get("predicted_class", ""),
                    "predicted_class_id": knee.get("predicted_class_id", ""),
                    "confidence": knee.get("confidence", 0.0),
                    "num_js_lesions": knee.get("num_js_lesions", 0),
                    "num_ost_lesions": knee.get("num_ost_lesions", 0),
                }
                rows.append(row)
        else:
            # Single prediction format
            row = {
                "image_path": image_path,
                "knee_id": 0,
                "predicted_class": result.get("predicted_class", ""),
                "predicted_class_id": result.get("predicted_class_id", ""),
                "confidence": result.get("confidence", 0.0),
                "num_js_lesions": result.get("num_js_lesions", 0),
                "num_ost_lesions": result.get("num_ost_lesions", 0),
            }
            rows.append(row)

    if rows:
        fieldnames = rows[0].keys()
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"✅ CSV saved to {output_path}")


def save_visualization(
    image_path: str,
    predictions: List[Dict],
    output_path: str,
    draw_lesions: bool = True,
):
    """
    Save visualization with bounding boxes and predictions.

    Args:
        image_path: Path to original image
        predictions: List of knee predictions
        output_path: Path to save visualization
        draw_lesions: Whether to draw lesion boxes
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"⚠️  Could not load image: {image_path}")
        return

    # Draw predictions
    for pred in predictions:
        # Draw knee box
        knee_bbox = pred.get("knee_bbox", None)
        if knee_bbox:
            x1, y1, x2, y2 = knee_bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add label
            label = f"{pred['predicted_class']} ({pred['confidence']:.2f})"
            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        if draw_lesions:
            # Draw JS lesion boxes (blue)
            for js_bbox in pred.get("js_lesion_bboxes", []):
                x1, y1, x2, y2 = js_bbox
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)

            # Draw OST lesion boxes (red)
            for ost_bbox in pred.get("ost_lesion_bboxes", []):
                x1, y1, x2, y2 = ost_bbox
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # Save
    cv2.imwrite(str(output_path), image)
    print(f"✅ Visualization saved to {output_path}")


def save_summary(results: List[Dict], output_path: str):
    """
    Save summary statistics to text file.

    Args:
        results: List of prediction dictionaries
        output_path: Path to save summary
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate statistics
    total_images = len(results)
    total_knees = sum(len(r.get("knees", [])) for r in results)

    class_counts = {}
    for result in results:
        for knee in result.get("knees", []):
            cls = knee.get("predicted_class", "Unknown")
            class_counts[cls] = class_counts.get(cls, 0) + 1

    # Write summary
    with open(output_path, "w") as f:
        f.write("KIOCMIL-CADA Inference Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Images: {total_images}\n")
        f.write(f"Total Knees Detected: {total_knees}\n\n")
        f.write("Class Distribution:\n")
        f.write("-" * 30 + "\n")
        for cls, count in sorted(class_counts.items()):
            f.write(f"{cls}: {count}\n")

    print(f"✅ Summary saved to {output_path}")
