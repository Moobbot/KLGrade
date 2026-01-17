"""
Test script for YOLO dataset loader.

This script:
1. Loads YoloDataset from processed/knee/
2. Tests with train/val splits
3. Visualizes samples
4. Validates bbox coordinates
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.datasets import (
    YoloDataset,
    visualize_dataset_sample,
    get_default_train_transform,
    get_default_val_transform,
)
from src.config import CLASSES, CLASSES_10_CLASS
import torch


def test_yolo_dataset():
    """Test YOLO dataset loading and visualization."""

    print("=" * 60)
    print("Testing YOLO Dataset Loader")
    print("=" * 60)

    # Test with original labels (CLASSES)
    print("\n1. Testing with original labels (5 classes)...")
    train_transform = get_default_train_transform(
        img_size=(640, 640), use_augmentation=False
    )

    train_dataset = YoloDataset(
        img_dir="dataset/dataset_v0/images",
        label_dir="dataset/dataset_v0/labels",
        transform=train_transform,
        split_file="splits/base/train.txt",
        use_labels_10_class=False,
        filter_no_label=True,
        cache_images=False,
        cache_labels=True,
        bbox_format="pascal_voc",
        return_dict=True,
    )

    print(f"✅ Train dataset size: {len(train_dataset)}")

    # Test a sample
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f"\n   Sample keys: {sample.keys()}")
        print(f"   Image shape: {sample['image'].shape}")
        print(f"   Boxes shape: {sample['boxes'].shape}")
        print(f"   Labels shape: {sample['labels'].shape}")
        print(f"   Image ID: {sample['image_id']}")

        # Validate bbox
        boxes = sample["boxes"]
        if len(boxes) > 0:
            print(f"\n   First box (Pascal VOC): {boxes[0]}")
            # Check if boxes are valid
            valid = all(
                b[0] < b[2] and b[1] < b[3] and b[0] >= 0 and b[1] >= 0 for b in boxes
            )
            print(f"   Boxes valid: {valid}")

    # Test validation dataset
    print("\n2. Testing validation dataset...")
    val_transform = get_default_val_transform(img_size=(640, 640))

    val_dataset = YoloDataset(
        img_dir="dataset/dataset_v0/images",
        label_dir="dataset/dataset_v0/labels",
        transform=val_transform,
        split_file="splits/base/val.txt",
        use_labels_10_class=False,
        filter_no_label=True,
        bbox_format="pascal_voc",
        return_dict=True,
    )

    print(f"✅ Val dataset size: {len(val_dataset)}")

    # Test with labels_10_class (CLASSES_10_CLASS - 10 classes)
    print("\n3. Testing with labels_10_class (10 classes)...")

    train_dataset_new = YoloDataset(
        img_dir="dataset/dataset_v0/images",
        label_dir="dataset/dataset_v0/labels",  # Will be changed to labels_10_class
        transform=train_transform,
        split_file="splits/new/train.txt",
        use_labels_10_class=True,
        filter_no_label=True,
        bbox_format="pascal_voc",
        return_dict=True,
    )

    print(f"✅ Train dataset (labels_10_class) size: {len(train_dataset_new)}")

    # Visualize samples
    print("\n4. Visualizing samples...")
    output_dir = Path("check_vis/test_yolo")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Visualize from original labels
    for i in range(min(3, len(train_dataset))):
        save_path = output_dir / f"train_sample_{i}.png"
        visualize_dataset_sample(
            train_dataset, idx=i, class_names=CLASSES, save_path=str(save_path)
        )
        print(f"   Saved: {save_path}")

    # Visualize from labels_10_class
    if len(train_dataset_new) > 0:
        for i in range(min(2, len(train_dataset_new))):
            save_path = output_dir / f"train_new_sample_{i}.png"
            visualize_dataset_sample(
                train_dataset_new,
                idx=i,
                class_names=CLASSES_10_CLASS,
                save_path=str(save_path),
            )
            print(f"   Saved: {save_path}")

    # Test DataLoader
    print("\n5. Testing DataLoader...")
    from torch.utils.data import DataLoader

    # Custom collate function for variable-sized tensors (different number of boxes per image)
    def custom_collate(batch):
        """Collate function that handles variable number of boxes per image."""
        images = torch.stack([item["image"] for item in batch])
        boxes = [item["boxes"] for item in batch]  # List of tensors
        labels = [item["labels"] for item in batch]  # List of tensors
        image_ids = [item["image_id"] for item in batch]

        return {
            "image": images,
            "boxes": boxes,
            "labels": labels,
            "image_id": image_ids,
        }

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=custom_collate,  # Use custom collate
    )

    # Get a batch
    batch = next(iter(train_loader))
    print(f"   Batch keys: {batch.keys()}")
    print(f"   Image batch shape: {batch['image'].shape}")
    print(f"   Boxes batch (list of tensors): {[b.shape for b in batch['boxes']]}")
    print(f"   Labels batch (list of tensors): {[l.shape for l in batch['labels']]}")

    print("\n" + "=" * 60)
    print("✅ All YOLO dataset tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_yolo_dataset()
