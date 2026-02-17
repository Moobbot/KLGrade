import os
import glob
import random
import yaml
from pathlib import Path


def setup_dataset(dataset_name, dataset_path, class_names):
    """
    Sets up a dataset for YOLO training:
    1. Scans images
    2. Creates train/val split (80/20)
    3. Writes train.txt and val.txt
    4. Creates data.yaml
    """
    print(f"Setting up {dataset_name} at {dataset_path}...")

    # Define paths
    images_dir = os.path.join(dataset_path, "images")
    output_dir = dataset_path  # data.yaml goes to dataset root usually, or can be anywhere. Let's put it in dataset path

    # Scan images
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    images = []
    for ext in image_extensions:
        images.extend(glob.glob(os.path.join(images_dir, ext)))

    if not images:
        print(f"Error: No images found in {images_dir}")
        return

    print(f"Found {len(images)} images.")

    # Shuffle and split
    random.seed(42)  # For reproducibility
    random.shuffle(images)

    total_images = len(images)
    train_idx = int(total_images * 0.7)
    val_idx = int(total_images * 0.9)  # 70% train, 20% val, 10% test

    train_images = images[:train_idx]
    val_images = images[train_idx:val_idx]
    test_images = images[val_idx:]

    print(
        f"Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}"
    )

    # Write split files with absolute paths
    train_txt_path = os.path.join(output_dir, "train.txt")
    val_txt_path = os.path.join(output_dir, "val.txt")
    test_txt_path = os.path.join(output_dir, "test.txt")

    with open(train_txt_path, "w") as f:
        f.writelines([f"{img}\n" for img in train_images])

    with open(val_txt_path, "w") as f:
        f.writelines([f"{img}\n" for img in val_images])

    with open(test_txt_path, "w") as f:
        f.writelines([f"{img}\n" for img in test_images])

    print(f"Created {train_txt_path}, {val_txt_path}, and {test_txt_path}")

    # Create data.yaml content
    data_yaml = {
        "path": os.path.abspath(output_dir),
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
        "names": {i: name for i, name in enumerate(class_names)},
    }

    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print(f"Created {yaml_path}")
    print("-" * 30)


def main():
    base_processed_path = "/home/ngoductam/KLGrade/datasets/processed_balanced"

    # 5-class setup
    # Using 'resize_only' as the representative variant for training for now
    # The user request was "datasets/processed_balanced/knees_cropped"
    dataset_5_path = os.path.join(base_processed_path, "knees_cropped", "resize_only")
    class_names_5 = ["KL0", "KL1", "KL2", "KL3", "KL4"]
    setup_dataset("5-class Balanced", dataset_5_path, class_names_5)

    # 10-class setup
    dataset_10_path = os.path.join(
        base_processed_path, "knees_cropped_10_class", "resize_only"
    )
    class_names_10 = [
        "KL0-a",
        "KL0-b",
        "KL1-a",
        "KL1-b",
        "KL2-a",
        "KL2-b",
        "KL3-a",
        "KL3-b",
        "KL4-a",
        "KL4-b",
    ]
    setup_dataset("10-class Balanced", dataset_10_path, class_names_10)


if __name__ == "__main__":
    main()
