"""
Filter Dataset by Removing KL0 Classes

Creates filtered datasets by removing images with only KL0 labels
and remapping remaining class IDs.

Usage:
    python scripts/data_preparation/filter_kl0.py --input processed/knee --output processed/knee_4_class
    python scripts/data_preparation/filter_kl0.py --input processed/knee_10_class --output processed/knee_8_class --num_classes 10
"""

import sys
from pathlib import Path
import argparse
import shutil
import json
from typing import List, Dict


def load_yolo_boxes(label_path: Path) -> List[Dict]:
    """Load YOLO boxes from file."""
    boxes = []
    if not label_path.exists():
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    class_id = int(float(parts[0]))
                    x, y, w, h = map(float, parts[1:5])
                    boxes.append({"class_id": class_id, "x": x, "y": y, "w": w, "h": h})
                except ValueError:
                    continue
    return boxes


def save_yolo_boxes(boxes: List[Dict], output_path: Path):
    """Save boxes in YOLO format."""
    lines = []
    for box in boxes:
        line = f"{box['class_id']} {box['x']:.6f} {box['y']:.6f} {box['w']:.6f} {box['h']:.6f}"
        lines.append(line)

    output_path.write_text("\n".join(lines) + "\n" if lines else "")


def filter_kl0(input_dir: Path, output_dir: Path, num_classes: int = 5):
    """
    Filter dataset by removing KL0 classes.

    Args:
        input_dir: Input dataset directory
        output_dir: Output dataset directory
        num_classes: 5 (remove class 0) or 10 (remove classes 0,1)
    """
    print("=" * 80)
    print("FILTERING KL0 CLASSES")
    print("=" * 80)
    print(f"\nInput: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Mode: {num_classes}-class dataset")

    img_dir = input_dir / "images"
    label_dir = input_dir / "labels"

    output_img_dir = output_dir / "images"
    output_label_dir = output_dir / "labels"

    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    # Determine KL0 class IDs to remove
    if num_classes == 5:
        kl0_classes = {0}  # Remove class 0 (KL0)
        remap_offset = 1  # KL1→0, KL2→1, KL3→2, KL4→3
        output_classes = 4
    elif num_classes == 10:
        kl0_classes = {0, 1}  # Remove classes 0,1 (KL0-a, KL0-b)
        remap_offset = 2  # KL1-a→0, KL1-b→1, ..., KL4-b→7
        output_classes = 8
    else:
        raise ValueError(f"Unsupported num_classes: {num_classes}")

    print(f"\nRemoving classes: {kl0_classes}")
    print(f"Remapping: class_id → class_id - {remap_offset}")
    print(f"Output classes: {output_classes} (0-{output_classes-1})")

    # Get all images
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [f for f in img_dir.iterdir() if f.suffix.lower() in img_extensions]

    print(f"\nProcessing {len(images)} images...")

    stats = {
        "total_images": len(images),
        "kept_images": 0,
        "filtered_images": 0,
        "original_boxes": 0,
        "kept_boxes": 0,
        "filtered_boxes": 0,
    }

    filtered_files = []

    for img_file in images:
        stem = img_file.stem
        label_file = label_dir / f"{stem}.txt"

        # Load boxes
        boxes = load_yolo_boxes(label_file)
        stats["original_boxes"] += len(boxes)

        # Filter out KL0 boxes and remap
        kept_boxes = []
        for box in boxes:
            if box["class_id"] in kl0_classes:
                stats["filtered_boxes"] += 1
                continue

            # Remap class ID
            new_class_id = box["class_id"] - remap_offset
            kept_boxes.append(
                {
                    "class_id": new_class_id,
                    "x": box["x"],
                    "y": box["y"],
                    "w": box["w"],
                    "h": box["h"],
                }
            )
            stats["kept_boxes"] += 1

        # Keep image only if it has non-KL0 boxes
        if kept_boxes:
            # Copy image
            shutil.copy2(img_file, output_img_dir / img_file.name)

            # Save remapped labels
            save_yolo_boxes(kept_boxes, output_label_dir / f"{stem}.txt")

            stats["kept_images"] += 1
        else:
            # Image only had KL0 boxes
            filtered_files.append(stem)
            stats["filtered_images"] += 1

    # Summary
    print("\n" + "=" * 80)
    print("FILTERING SUMMARY")
    print("=" * 80)
    print(f"\nTotal images: {stats['total_images']}")
    print(
        f"✅ Kept: {stats['kept_images']} ({stats['kept_images']/stats['total_images']*100:.1f}%)"
    )
    print(
        f"🗑️  Filtered: {stats['filtered_images']} ({stats['filtered_images']/stats['total_images']*100:.1f}%)"
    )

    print(f"\nBoxes:")
    print(f"  Original: {stats['original_boxes']}")
    print(
        f"  Kept: {stats['kept_boxes']} ({stats['kept_boxes']/stats['original_boxes']*100:.1f}%)"
    )
    print(
        f"  Filtered: {stats['filtered_boxes']} ({stats['filtered_boxes']/stats['original_boxes']*100:.1f}%)"
    )

    # Save stats
    stats_path = output_dir / "filter_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n💾 Saved stats: {stats_path}")

    # Save filtered files list
    if filtered_files:
        filtered_path = output_dir / "filtered_kl0_files.json"
        with open(filtered_path, "w") as f:
            json.dump(filtered_files, f, indent=2)
        print(f"📝 Saved filtered files list: {filtered_path}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Filter KL0 classes from dataset")
    parser.add_argument(
        "--input", type=str, required=True, help="Input dataset directory"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output dataset directory"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=5,
        choices=[5, 10],
        help="Original number of classes (5 or 10)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return

    filter_kl0(input_dir, output_dir, args.num_classes)


if __name__ == "__main__":
    main()
