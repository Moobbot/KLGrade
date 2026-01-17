"""
Data Preparation: Knee Cropping

Crops knee regions from full X-ray images using existing knee bounding box labels.
This is a DATA PREPARATION step (run once before preprocessing).

Flow:
    Raw Full X-rays → [Knee Cropping] → Cropped Knees → [Preprocessing] → Training Data

Usage:
    python scripts/prepare_knee_crops.py --input datasets/dataset/dataset_v0 --output datasets/dataset_knee_cropped --margin 0.15 2>&1 | head -n 30
"""

import sys
import os
import json
from pathlib import Path
from tqdm import tqdm
import shutil

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.data.preprocessing.core.knee_crop import (
    crop_knee_from_image,
)
from src.data.utils.yolo_utils import load_yolo_boxes
from src.data.preprocessing.core.base import load_image, save_image


def classify_box_shape(w: float, h: float, area: float) -> str:
    """
    Classify box as 'a' (osteophyte/spike) or 'b' (joint space).

    From class_split_report.py logic:
    - 'a': w/h < 1.2 or area < 0.01 (bone spike)
    - 'b': w/h > 2.0 or area > 0.03 (joint space)

    Args:
        w, h: Normalized width and height
        area: Normalized area

    Returns:
        'a' or 'b' or None if unclear
    """
    if h <= 0:
        return "a"

    ratio = w / h

    # Clear 'a' (bone spike)
    if ratio < 1.2 or area < 0.01:
        return "a"

    # Clear 'b' (joint space)
    if ratio > 2.0 or area > 0.03:
        return "b"

    # Unclear - needs relative comparison
    return None


def assign_relative_suffix(boxes: list):
    """
    For unclear boxes, assign suffix based on width comparison.

    Boxes with larger width → 'b' (joint space)
    Boxes with smaller width → 'a' (bone spike)
    """
    if len(boxes) != 2:
        return

    b1, b2 = boxes

    # Only process if at least one is unclear
    if b1["suffix"] is not None and b2["suffix"] is not None:
        return

    # Compare widths
    w1 = b1["w"]
    w2 = b2["w"]

    if w1 > w2:
        if b1["suffix"] is None:
            b1["suffix"] = "b"
        if b2["suffix"] is None:
            b2["suffix"] = "a"
    else:
        if b1["suffix"] is None:
            b1["suffix"] = "a"
        if b2["suffix"] is None:
            b2["suffix"] = "b"


def convert_5_class_to_10_class(labels_5_class: list) -> list:
    """
    Convert 5-class labels to 10-class labels using shape-based classification.

    Mapping:
    - class 0 → 0 (KL0-a) or 1 (KL0-b)
    - class 1 → 2 (KL1-a) or 3 (KL1-b)
    - class 2 → 4 (KL2-a) or 5 (KL2-b)
    - class 3 → 6 (KL3-a) or 7 (KL3-b)
    - class 4 → 8 (KL4-a) or 9 (KL4-b)

    Args:
        labels_5_class: List of 5-class labels (class_id 0-4)

    Returns:
        List of 10-class labels (class_id 0-9)
    """
    # Group by original class
    class_groups = {}
    for label in labels_5_class:
        cls = label["class_id"]
        if cls not in class_groups:
            class_groups[cls] = []

        # Add suffix field
        label_with_suffix = label.copy()
        area = label["w"] * label["h"]
        label_with_suffix["suffix"] = classify_box_shape(label["w"], label["h"], area)
        label_with_suffix["area"] = area
        class_groups[cls].append(label_with_suffix)

    # Process each class group
    for cls, group in class_groups.items():
        # Handle relative comparisons for unclear boxes
        if len(group) == 2:
            assign_relative_suffix(group)

        # Fallback for any remaining None suffixes
        for box in group:
            if box["suffix"] is None:
                # Fallback based on ratio
                ratio = box["w"] / box["h"] if box["h"] > 0 else 0
                box["suffix"] = "a" if ratio < 1.5 else "b"

    # Convert to 10-class labels
    labels_10_class = []
    for cls, group in class_groups.items():
        for box in group:
            new_label = {
                "class_id": cls * 2 + (0 if box["suffix"] == "a" else 1),
                "x": box["x"],
                "y": box["y"],
                "w": box["w"],
                "h": box["h"],
            }
            labels_10_class.append(new_label)

    return labels_10_class


def prepare_knee_crops(
    input_dir: Path,
    output_dir: Path,
    margin: float = 0.15,
    create_filtered: bool = True,
):
    """
    Crop knee regions from full X-ray dataset.

    Args:
        input_dir: Input directory (dataset_v0) with:
            - images/
            - labels/ (KL labels, 5-class: 0-4)
            - labels-knee/ (knee bounding boxes)
            - labels_new/ (optional, 10-class: 0-9)
        output_dir: Output directory for cropped knees
        margin: Margin around knee box (0.15 = 15%)
        create_filtered: Create 4-class and 8-class filtered versions
    """
    print("=" * 60)
    print("DATA PREPARATION: KNEE CROPPING")
    print("=" * 60)
    print(f"\nInput: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Margin: {margin * 100:.0f}%")
    print(f"Create filtered variants: {create_filtered}\n")

    # Setup paths
    input_images = input_dir / "images"
    input_labels = input_dir / "labels"  # KL labels (5-class: 0-4)
    input_knee_labels = input_dir / "labels-knee"  # Knee boxes
    input_labels_new = input_dir / "labels_new"  # 10-class labels (0-9)

    output_images = output_dir / "images"
    output_labels_5_class = output_dir / "labels"  # 5-class
    output_labels_10_class = output_dir / "labels_new"  # 10-class
    output_labels_4_class = output_dir / "labels_4_class"  # 4-class (no KL0)
    output_labels_8_class = output_dir / "labels_8_class"  # 8-class (no KL0-a/b)
    output_knee_labels = output_dir / "labels-knee"

    # Create output directories
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels_5_class.mkdir(parents=True, exist_ok=True)
    output_knee_labels.mkdir(parents=True, exist_ok=True)

    # We always create 10-class labels by converting from 5-class
    has_10_class = True  # Always generate 10-class labels
    output_labels_10_class.mkdir(parents=True, exist_ok=True)

    if create_filtered:
        output_labels_4_class.mkdir(parents=True, exist_ok=True)
        if has_10_class:
            output_labels_8_class.mkdir(parents=True, exist_ok=True)

    # Get all images
    image_files = list(input_images.glob("*.jpg")) + list(input_images.glob("*.png"))

    if not image_files:
        print(f"❌ No images found in {input_images}")
        return

    print(f"Found {len(image_files)} images\n")

    # Statistics
    stats = {
        "total_images": len(image_files),
        "total_knees_cropped": 0,
        "images_with_knees": 0,
        "images_without_knees": 0,
    }

    # Process each image
    for img_path in tqdm(image_files, desc="Cropping knees"):
        stem = img_path.stem

        # Check if knee labels exist
        knee_label_path = input_knee_labels / f"{stem}.txt"
        if not knee_label_path.exists():
            stats["images_without_knees"] += 1
            continue

        # Load image
        image = load_image(img_path, mode="grayscale")

        # Get all knee boxes for this image
        knee_boxes = load_yolo_boxes(knee_label_path)

        if not knee_boxes:
            stats["images_without_knees"] += 1
            continue

        stats["images_with_knees"] += 1

        # Process each knee in the image
        for knee_idx in range(len(knee_boxes)):
            # Crop knee
            kl_label_path = input_labels / f"{stem}.txt"
            cropped_img, transformed_kl_labels, metadata = crop_knee_from_image(
                image, knee_label_path, kl_label_path, margin, knee_idx
            )

            if not metadata["cropped"]:
                continue

            # Generate output filename
            if len(knee_boxes) == 1:
                # Single knee: keep original name
                output_stem = stem
            else:
                # Multiple knees: add index
                output_stem = f"{stem}_knee{knee_idx}"

            # Save cropped image
            output_img_path = output_images / f"{output_stem}.png"
            save_image(cropped_img, output_img_path)

            # Save transformed KL labels (5-class)
            output_kl_path = output_labels_5_class / f"{output_stem}.txt"
            save_yolo_labels(transformed_kl_labels, output_kl_path)

            # Create 4-class version (filter out KL0, remap 1-4 to 0-3)
            if create_filtered:
                labels_4_class = filter_class_0(transformed_kl_labels)
                output_4_class_path = output_labels_4_class / f"{output_stem}.txt"
                save_yolo_labels(labels_4_class, output_4_class_path)

            # Save knee label (class 0, full crop)
            output_knee_path = output_knee_labels / f"{output_stem}.txt"
            knee_full_box = {"class_id": 0, "x": 0.5, "y": 0.5, "w": 1.0, "h": 1.0}
            save_yolo_labels([knee_full_box], output_knee_path)

            # Process 10-class labels if available
            if has_10_class:
                # Generate 10-class labels from 5-class using shape classification
                labels_10_class = convert_5_class_to_10_class(transformed_kl_labels)
                output_10_class_path = output_labels_10_class / f"{output_stem}.txt"
                save_yolo_labels(labels_10_class, output_10_class_path)

                # Create 8-class version (filter out classes 0,1 which are KL0-a/b, remap 2-9 to 0-7)
                if create_filtered:
                    labels_8_class = filter_class_0_10_class(labels_10_class)
                    output_8_class_path = output_labels_8_class / f"{output_stem}.txt"
                    save_yolo_labels(labels_8_class, output_8_class_path)

            stats["total_knees_cropped"] += 1

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total images: {stats['total_images']}")
    print(f"Images with knees: {stats['images_with_knees']}")
    print(f"Images without knees: {stats['images_without_knees']}")
    print(f"Total knees cropped: {stats['total_knees_cropped']}")
    print(f"\n📁 Output directories:")
    print(f"  - images/")
    print(f"  - labels/ (5-class: KL0-4)")
    print(f"  - labels-knee/ (knee boxes)")
    if has_10_class:
        print(f"  - labels_new/ (10-class: KL0-a/b to KL4-a/b)")
    if create_filtered:
        print(f"  - labels_4_class/ (4-class: KL1-4 → 0-3)")
        if has_10_class:
            print(f"  - labels_8_class/ (8-class: KL1-a/b to KL4-a/b → 0-7)")
    print(f"\n✅ Output: {output_dir}")
    print("=" * 60)

    # Save report
    report_path = output_dir / "crop_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("KNEE CROPPING REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Input: {input_dir}\n")
        f.write(f"Output: {output_dir}\n")
        f.write(f"Margin: {margin * 100:.0f}%\n\n")
        f.write("Statistics:\n")
        f.write(f"  Total images: {stats['total_images']}\n")
        f.write(f"  Images with knees: {stats['images_with_knees']}\n")
        f.write(f"  Images without knees: {stats['images_without_knees']}\n")
        f.write(f"  Total knees cropped: {stats['total_knees_cropped']}\n")

    print(f"📄 Saved report: {report_path}")

    return stats


# Import centralized filter implementation
from src.data.filters import filter_empty_labels


def filter_class_0(labels):
    """
    Filter out class 0 (KL0) and remap classes 1-4 to 0-3.

    For 5-class → 4-class conversion.
    """
    filtered = []
    for label in labels:
        if label["class_id"] == 0:
            continue  # Skip KL0
        # Remap: 1→0, 2→1, 3→2, 4→3
        new_label = label.copy()
        new_label["class_id"] = label["class_id"] - 1
        filtered.append(new_label)
    return filtered


def filter_class_0_10_class(labels):
    """
    Filter out classes 0,1 (KL0-a, KL0-b) and remap classes 2-9 to 0-7.

    For 10-class → 8-class conversion.
    """
    filtered = []
    for label in labels:
        if label["class_id"] in [0, 1]:
            continue  # Skip KL0-a and KL0-b
        # Remap: 2→0, 3→1, ..., 9→7
        new_label = label.copy()
        new_label["class_id"] = label["class_id"] - 2
        filtered.append(new_label)
    return filtered


def save_yolo_labels(labels, output_path: Path):
    """Save labels in YOLO format."""
    if not labels:
        output_path.write_text("")
        return

    lines = []
    for label in labels:
        line = f"{label['class_id']} {label['x']:.6f} {label['y']:.6f} {label['w']:.6f} {label['h']:.6f}"
        lines.append(line)

    output_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare knee crops from full X-rays")
    parser.add_argument(
        "--input", type=str, required=True, help="Input directory (dataset_v0)"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output directory for cropped knees"
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.15,
        help="Margin around knee box (default: 0.15)",
    )
    parser.add_argument(
        "--auto-filter",
        action="store_true",
        help="Automatically run filter_no_labels.py after cropping",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        sys.exit(1)

    prepare_knee_crops(input_dir, output_dir, args.margin)

    # Auto-filter if requested
    if args.auto_filter:
        print("\n" + "=" * 60)
        print("AUTO-FILTERING EMPTY LABELS")
        print("=" * 60)

        import subprocess

        filter_script = project_root / "scripts/data_preparation/filter_no_labels.py"

        if filter_script.exists():
            result = subprocess.run(
                [sys.executable, str(filter_script), "--input", str(output_dir)],
                env={**os.environ, "PYTHONPATH": str(project_root)},
            )

            if result.returncode != 0:
                print("⚠️  Filter failed, but cropping completed successfully")
        else:
            print(f"⚠️  Filter script not found: {filter_script}")
