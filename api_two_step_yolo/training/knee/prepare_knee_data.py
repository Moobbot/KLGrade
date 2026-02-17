"""
Prepare Knee Detection Dataset:
1. Preprocess labels (map all classes to 0)
2. Split dataset into train/val/test
3. Update dataset.yaml
"""

import os
import sys
import yaml
import shutil
import random
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import Counter

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import split_dataset logic directly or re-implement for simplicity/customization
# Re-implementing simplified split logic here to be self-contained in the package


def preprocess_labels(input_dir: Path, output_dir: Path):
    """Map all class IDs to 0."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(input_dir.glob("*.txt"))
    print(f"Preprocessing {len(files)} labels from {input_dir}...")

    mapping_count = 0

    for file in tqdm(files, desc="Normalizing labels"):
        with open(file, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                # Map to class 0 (Knee)
                if parts[0] != "0":
                    parts[0] = "0"
                    mapping_count += 1
                new_lines.append(" ".join(parts) + "\n")

        with open(output_dir / file.name, "w") as f:
            f.writelines(new_lines)

    print(f"Mapped {mapping_count} objects to class 0.")
    print(f"Saved normalized labels to {output_dir}")


def split_dataset(
    img_dir: Path,
    label_dir: Path,
    out_dir: Path,
    train_ratio=0.7,
    val_ratio=0.15,
    seed=42,
):
    """Split dataset and save txt files."""
    random.seed(seed)

    # Get all valid image-label pairs
    images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    valid_pairs = []

    for img_path in images:
        label_path = label_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            valid_pairs.append(img_path)

    print(f"Found {len(valid_pairs)} valid image-label pairs.")

    # Shuffle
    random.shuffle(valid_pairs)

    # Split
    n_total = len(valid_pairs)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_imgs = valid_pairs[:n_train]
    val_imgs = valid_pairs[n_train : n_train + n_val]
    test_imgs = valid_pairs[n_train + n_val :]

    # Ensure output dir exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write split files (relative paths to project root if possible)
    # YOLO expects absolute paths or relative to dataset.yaml location (or 'path' in yaml)
    # We will use paths relative to project root, and set 'path' in yaml to project root.

    project_root = Path.cwd()

    def write_split(imgs, filename):
        with open(out_dir / filename, "w") as f:
            for img_path in imgs:
                try:
                    rel_path = img_path.relative_to(project_root)
                except ValueError:
                    rel_path = img_path  # Fallback to absolute

                # YOLO uses label replacement: images/ -> labels/
                # But our preprocessed labels are in a different folder!
                # YOLO doesn't support separate label dir easily via split txt unless we copy labels to parallel folder.
                # However, if we use `path` in yaml, YOLO looks for `labels` relative to that.

                # CRITICAL: We need the IMAGES to be in proper structure or use `yolo` format where labels are inferred.
                # If we list output images as `datasets/dataset_v0/images/img.jpg`,
                # YOLO will look for `datasets/dataset_v0/labels/img.txt`.
                # But we want it to look in `datasets/processed/knee-labels-single/`.

                # ISSUE: YOLO training loop replaces 'images' with 'labels' in path.
                # If we want to use custom labels, we generally need to duplicate the images or symlink them
                # to a folder structure where `labels` folder contains our processed labels.

                f.write(str(rel_path) + "\n")

    # Since we cannot easily redirect YOLO to look for labels in a arbitrary folder if images are in original folder,
    # UNLESS we copy images or symlink them.
    # OR we modify the image paths we write to split files to point to a new "virtual" dataset folder,
    # where images are symlinked and labels are our processed ones.

    print(
        "Saving splits (WARNING: YOLO requires labels to be in 'labels' folder parallel to 'images')..."
    )

    # To fix the label path issue properly:
    # 1. Create a `processed/knee_detection_final/images` (symlinks to real images)
    # 2. Create a `processed/knee_detection_final/labels` (our processed labels)

    final_dataset_dir = Path("datasets/processed/knee_detection_final")
    (final_dataset_dir / "images").mkdir(parents=True, exist_ok=True)
    (final_dataset_dir / "labels").mkdir(parents=True, exist_ok=True)

    # Copy processed labels
    for img_path in tqdm(valid_pairs, desc="Structuring dataset"):
        # Symlink image
        dst_img = final_dataset_dir / "images" / img_path.name
        if not dst_img.exists():
            os.symlink(img_path.absolute(), dst_img)

        # Copy label
        src_label = label_dir / f"{img_path.stem}.txt"
        dst_label = final_dataset_dir / "labels" / f"{img_path.stem}.txt"
        if not dst_label.exists():
            shutil.copy(src_label, dst_label)

    # Now write splits relative to this new structure
    # Actually, we can write paths to the new images

    def write_final_split(imgs, filename):
        with open(out_dir / filename, "w") as f:
            for img_path in imgs:
                # Point to the image in the new structured directory
                new_img_path = final_dataset_dir / "images" / img_path.name
                f.write(str(new_img_path.absolute()) + "\n")

    write_final_split(train_imgs, "train.txt")
    write_final_split(val_imgs, "val.txt")
    write_final_split(test_imgs, "test.txt")

    print(f"Created validation-ready dataset structure at {final_dataset_dir}")
    print(f"Saved splits to {out_dir}")
    print(f"  Train: {len(train_imgs)}")
    print(f"  Val: {len(val_imgs)}")
    print(f"  Test: {len(test_imgs)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_home", default="datasets/dataset_v0/images", help="Source images"
    )
    parser.add_argument(
        "--label_home", default="datasets/dataset_v0/labels-knee", help="Source labels"
    )
    parser.add_argument(
        "--out_dir", default="datasets/splits/knee", help="Output for split txts"
    )
    args = parser.parse_args()

    # 1. Preprocess labels to temp dir
    temp_label_dir = Path("datasets/processed/temp_knee_labels")
    preprocess_labels(Path(args.label_home), temp_label_dir)

    # 2. Split and restructure
    split_dataset(Path(args.img_home), temp_label_dir, Path(args.out_dir))

    # 3. Update dataset.yaml
    yaml_path = Path("processed/knee_detection/dataset.yaml")
    yaml_data = {
        "path": str(Path.cwd()),
        "train": str(Path(args.out_dir) / "train.txt"),
        "val": str(Path(args.out_dir) / "val.txt"),
        "test": str(Path(args.out_dir) / "test.txt"),
        "names": {0: "knee"},
    }

    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, sort_keys=False)

    print(f"\n✅ Dataset preparation complete. Config updated at {yaml_path}")

    # Cleanup temp
    shutil.rmtree(temp_label_dir)


if __name__ == "__main__":
    main()
