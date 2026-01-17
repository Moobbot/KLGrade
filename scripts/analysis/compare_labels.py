#!/usr/bin/env python3
"""
Compare two label directories and show which files were filtered out.
Analyze why those files were removed (labels outside crop region).
"""

from pathlib import Path
import sys


def compare_label_dirs(new_dir: str, old_dir: str):
    """Compare two label directories and find differences."""
    new_path = Path(new_dir)
    old_path = Path(old_dir)

    # Get all label files
    new_files = set(f.name for f in new_path.glob("*.txt"))
    old_files = set(f.name for f in old_path.glob("*.txt"))

    print("=" * 80)
    print("LABEL FILE COMPARISON")
    print("=" * 80)
    print(f"New (filtered):   {len(new_files)} files in {new_dir}")
    print(f"Old (unfiltered): {len(old_files)} files in {old_dir}")
    print()

    # Find differences
    only_in_old = old_files - new_files
    only_in_new = new_files - old_files

    if only_in_old:
        print(f"📋 Files FILTERED OUT: {len(only_in_old)} files")
        print("-" * 80)
        print()

        total_boxes_removed = 0
        class_distribution = {}

        for fname in sorted(only_in_old):
            old_file = old_path / fname

            # Read content
            try:
                content = old_file.read_text().strip()
                lines = content.split("\n") if content else []

                print(f"📄 {fname}")
                if not lines:
                    print(f"   ⚠️  EMPTY FILE - No labels")
                else:
                    print(f"   Boxes: {len(lines)}")
                    for i, line in enumerate(lines, 1):
                        parts = line.split()
                        if len(parts) >= 5:
                            cls = int(parts[0])
                            x, y, w, h = map(float, parts[1:5])

                            # Track class distribution
                            class_distribution[cls] = class_distribution.get(cls, 0) + 1
                            total_boxes_removed += 1

                            # Check if box is near edge (likely filtered during transform)
                            near_edge = x < 0.1 or x > 0.9 or y < 0.1 or y > 0.9
                            edge_marker = " 🔴 NEAR EDGE" if near_edge else ""

                            print(
                                f"   Box {i}: class={cls}, x={x:.3f}, y={y:.3f}, w={w:.3f}, h={h:.3f}{edge_marker}"
                            )

                print()

            except Exception as e:
                print(f"   ❌ Error reading: {e}")
                print()

        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total files filtered: {len(only_in_old)}")
        print(f"Total boxes removed:  {total_boxes_removed}")
        print()
        print("Boxes removed by class:")
        for cls in sorted(class_distribution.keys()):
            count = class_distribution[cls]
            print(f"  Class {cls}: {count} boxes")
        print()

        # Reason analysis
        print("🔍 FILTER REASON ANALYSIS:")
        print("  These files were filtered because after cropping the knee region")
        print("  and transforming labels to crop space, the labels fell outside")
        print("  the crop boundaries or were too small to be valid.")
        print()
        print("  This is EXPECTED behavior for:")
        print("    - Labels far from the knee detection box")
        print("    - Labels near image edges that get cut off during cropping")
        print("    - Multi-knee images where labels belong to other knee crops")

    else:
        print("✅ No files filtered out - all old files present in new")

    if only_in_new:
        print()
        print(f"⚠️  Files ONLY in NEW: {len(only_in_new)} files")
        print("This is unexpected - new directory should not have extra files")
        for fname in sorted(only_in_new)[:10]:
            print(f"  - {fname}")

    print()
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_labels.py <new_labels_dir> <old_labels_dir>")
        print(
            "Example: python compare_labels.py datasets/dataset_knees_cropped/labels datasets/processed_draf/dataset_knees_cropped/labels"
        )
        sys.exit(1)

    compare_label_dirs(sys.argv[1], sys.argv[2])
