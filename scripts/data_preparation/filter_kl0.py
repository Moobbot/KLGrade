#!/usr/bin/env python3
"""
Filter Dataset by Removing KL0 Classes

Creates filtered datasets by removing images with only KL0 labels
and remapping remaining class IDs.

This is a CLI wrapper for the centralized filter implementation.

Usage:
    python scripts/data_preparation/filter_kl0.py --input processed/knee --output processed/knee_4_class
    python scripts/data_preparation/filter_kl0.py --input processed/knee_10_class --output processed/knee_8_class --num_classes 10
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.data.filters import filter_kl0_classes


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
        sys.exit(1)

    print("=" * 80)
    print("FILTERING KL0 CLASSES")
    print("=" * 80)
    print(f"\nInput: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Mode: {args.num_classes}-class dataset\n")

    # Determine what will be filtered
    if args.num_classes == 5:
        print("Removing class 0 (KL0)")
        print("Remapping: 1→0, 2→1, 3→2, 4→3")
        print("Output classes: 4 (0-3)")
    else:  # 10 classes
        print("Removing classes 0, 1 (KL0-a, KL0-b)")
        print("Remapping: 2→0, 3→1, ..., 9→7")
        print("Output classes: 8 (0-7)")

    # Call centralized implementation
    stats = filter_kl0_classes(input_dir, output_dir, args.num_classes)

    # Print summary
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

    print(f"\n💾 Saved stats: {output_dir}/filter_kl0_stats.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
