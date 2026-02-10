"""
Output formatters for End-to-End KIOCMIL inference results.

Supports multiple output formats:
- JSON: Structured data with all predictions
- CSV: Tabular format for easy analysis
- Visualization: Annotated images with predictions
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Optional
import cv2
import numpy as np


def format_json(
    predictions: List[Dict],
    output_path: str,
    indent: int = 2,
) -> None:
    """
    Save predictions as JSON file.

    Args:
        predictions: List of prediction dictionaries
        output_path: Path to output JSON file
        indent: JSON indentation level
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=indent)

    print(f"✅ JSON results saved to {output_file}")


def format_csv(
    predictions: List[Dict],
    output_path: str,
    include_probabilities: bool = False,
) -> None:
    """
    Save predictions as CSV file.

    Args:
        predictions: List of prediction dictionaries
        output_path: Path to output CSV file
        include_probabilities: Whether to include class probabilities
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not predictions:
        print("⚠️  No predictions to save")
        return

    # Determine columns
    base_columns = [
        "image_path",
        "predicted_class",
        "predicted_class_id",
        "confidence",
        "num_classes",
    ]

    # Add optional columns if available
    optional_columns = []
    if "num_knees_detected" in predictions[0]:
        optional_columns.append("num_knees_detected")
    if "num_lesions_detected" in predictions[0]:
        optional_columns.append("num_lesions_detected")

    columns = base_columns + optional_columns

    # Add probability columns if requested
    if include_probabilities and "class_probabilities" in predictions[0]:
        prob_keys = list(predictions[0]["class_probabilities"].keys())
        columns.extend([f"prob_{k}" for k in prob_keys])

    # Write CSV
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for pred in predictions:
            if "error" in pred:
                # Skip failed predictions
                continue

            row = {col: pred.get(col, "") for col in base_columns + optional_columns}

            # Add probabilities if requested
            if include_probabilities and "class_probabilities" in pred:
                for class_name, prob in pred["class_probabilities"].items():
                    row[f"prob_{class_name}"] = f"{prob:.4f}"

            writer.writerow(row)

    print(f"✅ CSV results saved to {output_file}")


def format_visualization(
    predictions: List[Dict],
    output_dir: str,
    font_scale: float = 0.8,
    thickness: int = 2,
) -> None:
    """
    Create visualizations of predictions with annotations.

    Args:
        predictions: List of prediction dictionaries
        output_dir: Directory to save visualized images
        font_scale: Font scale for text annotations
        thickness: Line thickness for text
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for pred in predictions:
        if "error" in pred:
            continue

        # Load image
        img_path = pred["image_path"]
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠️  Could not load image: {img_path}")
            continue

        # Prepare annotation text
        pred_class = pred["predicted_class"]
        confidence = pred["confidence"]
        text = f"{pred_class} ({confidence:.2%})"

        # Add text background
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        # Draw background rectangle
        cv2.rectangle(
            img,
            (10, 10),
            (20 + text_width, 20 + text_height + baseline),
            (0, 0, 0),
            -1,
        )

        # Draw text
        cv2.putText(
            img,
            text,
            (15, 15 + text_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 0) if confidence > 0.7 else (0, 255, 255),
            thickness,
        )

        # Add additional info if available
        y_offset = 50 + text_height
        if "num_knees_detected" in pred:
            info_text = f"Knees: {pred['num_knees_detected']}"
            cv2.putText(
                img,
                info_text,
                (15, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )
            y_offset += 25

        if "num_lesions_detected" in pred:
            info_text = f"Lesions: {pred['num_lesions_detected']}"
            cv2.putText(
                img,
                info_text,
                (15, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

        # Save annotated image
        img_name = Path(img_path).stem
        output_file = output_path / f"{img_name}_predicted.jpg"
        cv2.imwrite(str(output_file), img)

    print(f"✅ Visualizations saved to {output_path}")


def save_summary(
    predictions: List[Dict],
    output_path: str,
) -> None:
    """
    Save summary statistics of predictions.

    Args:
        predictions: List of prediction dictionaries
        output_path: Path to output summary file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Calculate statistics
    total = len(predictions)
    successful = sum(1 for p in predictions if "error" not in p)
    failed = total - successful

    if successful == 0:
        print("⚠️  No successful predictions to summarize")
        return

    # Get class distribution
    class_counts = {}
    confidences = []

    for pred in predictions:
        if "error" in pred:
            continue

        pred_class = pred["predicted_class"]
        class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
        confidences.append(pred["confidence"])

    # Calculate confidence statistics
    avg_confidence = np.mean(confidences)
    min_confidence = np.min(confidences)
    max_confidence = np.max(confidences)

    # Write summary
    with open(output_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("End-to-End KIOCMIL Inference Summary\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Total images: {total}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {failed}\n\n")

        f.write("Confidence Statistics:\n")
        f.write(f"  Average: {avg_confidence:.4f}\n")
        f.write(f"  Min: {min_confidence:.4f}\n")
        f.write(f"  Max: {max_confidence:.4f}\n\n")

        f.write("Class Distribution:\n")
        for class_name, count in sorted(class_counts.items()):
            percentage = count / successful * 100
            f.write(f"  {class_name}: {count} ({percentage:.1f}%)\n")

    print(f"✅ Summary saved to {output_file}")
