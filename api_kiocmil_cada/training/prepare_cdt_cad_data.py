#!/usr/bin/env python3
"""
Prepare CDT-CAD Dataset Configuration

Creates dataset.yaml files for CDT-CAD training with 5-class and 10-class variants.
"""

import argparse
import yaml
from pathlib import Path
import sys

def create_dataset_yaml(
    img_dir: Path,
    label_dir: Path,
    train_split: Path,
    val_split: Path,
    test_split: Path,
    num_classes: int,
    output_yaml: Path,
):
    """Create dataset.yaml for CDT-CAD training."""
    
    # Verify directories exist
    if not img_dir.exists():
        print(f"❌ Image directory not found: {img_dir}")
        return False
    
    if not label_dir.exists():
        print(f"❌ Label directory not found: {label_dir}")
        return False
    
    # Verify split files exist
    for split_file in [train_split, val_split, test_split]:
        if not split_file.exists():
            print(f"❌ Split file not found: {split_file}")
            return False
    
    # Count images in each split
    def count_images(split_file):
        with open(split_file, 'r') as f:
            return len([line for line in f if line.strip()])
    
    train_count = count_images(train_split)
    val_count = count_images(val_split)
    test_count = count_images(test_split)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Train: {train_count} images")
    print(f"  Val:   {val_count} images")
    print(f"  Test:  {test_count} images")
    print(f"  Total: {train_count + val_count + test_count} images")
    
    # Create class names
    if num_classes == 5:
        class_names = {i: f"KL{i}" for i in range(5)}  # KL0-KL4
    elif num_classes == 10:
        class_names = {
            0: "KL0-a", 1: "KL0-b",
            2: "KL1-a", 3: "KL1-b",
            4: "KL2-a", 5: "KL2-b",
            6: "KL3-a", 7: "KL3-b",
            8: "KL4-a", 9: "KL4-b",
        }
    else:
        class_names = {i: f"class_{i}" for i in range(num_classes)}
    
    # Create dataset config
    dataset_config = {
        'path': str(img_dir.parent.absolute()),
        'train': str(train_split.absolute()),
        'val': str(val_split.absolute()),
        'test': str(test_split.absolute()),
        'nc': num_classes,
        'names': class_names,
    }
    
    # Save to yaml
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(output_yaml, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✅ Created dataset config: {output_yaml}")
    print(f"   Classes: {num_classes}")
    print(f"   Images dir: {img_dir}")
    print(f"   Labels dir: {label_dir}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Prepare CDT-CAD dataset configuration")
    parser.add_argument(
        "--img-dir",
        type=str,
        default="datasets/dataset_knees_cropped/images",
        help="Path to images directory"
    )
    parser.add_argument(
        "--label-dir-5class",
        type=str,
        default="datasets/dataset_knees_cropped/labels",
        help="Path to 5-class labels directory"
    )
    parser.add_argument(
        "--label-dir-10class",
        type=str,
        default="datasets/dataset_knees_cropped/labels_10_class",
        help="Path to 10-class labels directory"
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default="datasets/splits/dataset_knees_cropped_70_20_10/train.txt",
        help="Path to train split file"
    )
    parser.add_argument(
        "--val-split",
        type=str,
        default="datasets/splits/dataset_knees_cropped_70_20_10/val.txt",
        help="Path to val split file"
    )
    parser.add_argument(
        "--test-split",
        type=str,
        default="datasets/splits/dataset_knees_cropped_70_20_10/test.txt",
        help="Path to test split file"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("CDT-CAD Dataset Preparation")
    print("="*60)
    
    img_dir = Path(args.img_dir)
    train_split = Path(args.train_split)
    val_split = Path(args.val_split)
    test_split = Path(args.test_split)
    
    # Create 5-class dataset config
    print("\n📦 Creating 5-class dataset config...")
    label_dir_5class = Path(args.label_dir_5class)
    output_yaml_5class = Path("datasets/dataset_knees_cropped/dataset_cdt_cad_5class.yaml")
    
    success_5class = create_dataset_yaml(
        img_dir=img_dir,
        label_dir=label_dir_5class,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        num_classes=5,
        output_yaml=output_yaml_5class,
    )
    
    # Create 10-class dataset config
    print("\n📦 Creating 10-class dataset config...")
    label_dir_10class = Path(args.label_dir_10class)
    output_yaml_10class = Path("datasets/dataset_knees_cropped/dataset_cdt_cad_10class.yaml")
    
    success_10class = create_dataset_yaml(
        img_dir=img_dir,
        label_dir=label_dir_10class,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        num_classes=10,
        output_yaml=output_yaml_10class,
    )
    
    print("\n" + "="*60)
    if success_5class and success_10class:
        print("✅ Dataset preparation complete!")
        print("\nYou can now train with:")
        print(f"  5-class:  --data {output_yaml_5class}")
        print(f"  10-class: --data {output_yaml_10class}")
    else:
        print("❌ Dataset preparation failed")
        return 1
    print("="*60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
