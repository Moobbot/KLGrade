"""
Filter Crops Without Labels

Moves cropped knee images without KL grade labels to a separate folder.
This is a post-processing step after crop_knee_regions.py.

Usage:
    python scripts/data_preparation/filter_no_labels.py --input processed/knee
"""

import sys
from pathlib import Path
import argparse
import shutil
import json


def filter_no_labels(input_dir: Path):
    """Move crops without labels to separate folder."""
    print("=" * 80)
    print("FILTERING CROPS WITHOUT LABELS")
    print("=" * 80)
    print(f"\nInput: {input_dir}")

    img_dir = input_dir / "images"
    label_dir = input_dir / "labels"

    no_label_img_dir = input_dir / "images-no-labels"
    no_label_label_dir = input_dir / "labels-no-labels"

    no_label_img_dir.mkdir(exist_ok=True)
    no_label_label_dir.mkdir(exist_ok=True)

    if not img_dir.exists() or not label_dir.exists():
        print(f"❌ Missing directories: {img_dir} or {label_dir}")
        return

    # Get all images
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [f for f in img_dir.iterdir() if f.suffix.lower() in img_extensions]

    print(f"\nFound {len(images)} cropped images")

    moved_count = 0
    no_label_files = []

    for img_file in images:
        stem = img_file.stem
        label_file = label_dir / f"{stem}.txt"

        # Check if label is empty or missing
        has_label = False
        if label_file.exists():
            with open(label_file, "r") as f:
                content = f.read().strip()
                has_label = len(content) > 0

        if not has_label:
            # Move image to no-labels folder
            dest_img_path = no_label_img_dir / img_file.name
            shutil.move(str(img_file), str(dest_img_path))

            # Move empty label file if exists
            if label_file.exists():
                dest_label_path = no_label_label_dir / label_file.name
                shutil.move(str(label_file), str(dest_label_path))

            no_label_files.append(stem)
            moved_count += 1

    # Summary
    print("\n" + "=" * 80)
    print("FILTERING SUMMARY")
    print("=" * 80)
    print(f"\nTotal crops: {len(images)}")
    print(f"🗂️  Moved to no-labels: {moved_count}")
    print(f"📊 Remaining with labels: {len(images) - moved_count}")

    # Save log
    if no_label_files:
        log_path = input_dir / "no_label_files.json"
        with open(log_path, "w") as f:
            json.dump(no_label_files, f, indent=2)
        print(f"\n📝 Saved no-label files list: {log_path}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Filter crops without labels")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory (e.g., processed/knee)",
    )

    args = parser.parse_args()
    input_dir = Path(args.input)

    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return

    filter_no_labels(input_dir)


if __name__ == "__main__":
    main()
