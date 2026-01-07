#!/usr/bin/env python3
"""
Fix split files to use RELATIVE paths instead of absolute paths
This makes them work on both Windows and Linux
"""

from pathlib import Path


def fix_splits_to_relative():
    """Convert all split files to use relative paths."""

    base_dir = Path(".")
    splits_dir = base_dir / "splits"

    # Mapping: split name -> relative path to images
    datasets = {
        "knee_5_class": "processed/knee_5_class/images",
        "knee_10_class": "processed/knee_10_class/images",
        "knee_4_class": "processed/knee_4_class/images",
        "knee_8_class": "processed/knee_8_class/images",
    }

    print("🔧 Fixing split files to use relative paths...")
    print("=" * 60)

    for split_name, img_dir_rel in datasets.items():
        split_dir = splits_dir / split_name

        if not split_dir.exists():
            print(f"\n⏭️  Skipping {split_name} (not found)")
            continue

        print(f"\n📁 Processing {split_name}...")

        for split_type in ["train", "val", "test"]:
            txt_file = split_dir / f"{split_type}.txt"

            if not txt_file.exists():
                continue

            # Read existing lines
            lines = txt_file.read_text(encoding="utf-8").strip().splitlines()

            if not lines:
                continue

            # Convert to relative paths
            new_lines = []
            for line in lines:
                # Extract filename
                fname = Path(line).name

                # Create relative path (e.g., "processed/knee/dataset_yolo/images/filename.jpg")
                rel_path = f"{img_dir_rel}/{fname}"
                new_lines.append(rel_path)

            # Write back
            txt_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            print(f"   ✅ {split_type}.txt: {len(new_lines)} paths (now relative)")

    print("\n" + "=" * 60)
    print("✅ All split files now use relative paths!")
    print("\nExample:")
    print("  processed/knee/dataset_yolo/images/1234.jpg")
    print("  (works on both Windows and Linux)")


if __name__ == "__main__":
    fix_splits_to_relative()
