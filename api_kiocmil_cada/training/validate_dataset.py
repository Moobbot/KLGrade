"""
Validate YOLO dataset integrity and fix broken paths.
"""

import os
import yaml
from pathlib import Path
from tqdm import tqdm
import argparse


def validate_dataset(yaml_path: str, fix: bool = False):
    """
    Validate dataset defined in yaml file.
    Checks if images in train/val/test splits actually exist.

    Args:
        yaml_path: Path to dataset.yaml
        fix: If True, generate new split files removing missing images
    """
    print(f"Validating dataset: {yaml_path}")

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    base_path = Path(data.get("path", ""))

    splits = ["train", "val", "test"]
    for split in splits:
        if split not in data:
            continue

        split_file = data[split]
        # Handle relative path if needed
        if not os.path.exists(split_file) and not os.path.isabs(split_file):
            # Try relative to yaml location
            split_file = os.path.join(os.path.dirname(yaml_path), split_file)

        if not os.path.exists(split_file):
            print(f"⚠️  Split file not found: {split_file}")
            continue

        print(f"\nChecking {split} split: {split_file}")

        with open(split_file, "r") as f:
            lines = f.readlines()

        valid_lines = []
        missing_count = 0

        for line in tqdm(lines):
            img_path = line.strip()
            if not img_path:
                continue

            # Check absolute path
            if os.path.exists(img_path):
                valid_lines.append(img_path)
                continue

            # Check relative to base_path
            if base_path:
                full_path = base_path / img_path
                if full_path.exists():
                    valid_lines.append(str(full_path))
                    continue

            # Check relative to split file
            rel_path = Path(os.path.dirname(split_file)) / img_path
            if rel_path.exists():
                valid_lines.append(str(rel_path))
                continue

            missing_count += 1
            # print(f"Missing: {img_path}")

        print(f"  Total: {len(lines)}")
        print(f"  Valid: {len(valid_lines)}")
        print(f"  Missing: {missing_count}")

        if fix and missing_count > 0:
            # Backup original
            backup_path = f"{split_file}.bak"
            if not os.path.exists(backup_path):
                os.rename(split_file, backup_path)
                print(f"  Backed up original to {backup_path}")

            # Write fixed
            with open(split_file, "w") as f:
                for line in valid_lines:
                    f.write(f"{line}\n")
            print(f"  ✅ Fixed split file saved to {split_file}")

    # Also check labels symlink/dir if possible
    # YOLO requires labels/ dir parallel to images/ or substitute images->labels
    # We assume standard YOLO structure


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_path", help="Path to dataset.yaml")
    parser.add_argument(
        "--fix", action="store_true", help="Fix broken paths by removing missing files"
    )
    args = parser.parse_args()

    validate_dataset(args.yaml_path, args.fix)
