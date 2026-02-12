#!/usr/bin/env python3
"""
Prepare Balanced CDT-CAD Dataset Configurations

Creates dataset.yaml files for balanced 5-class and 10-class datasets.
"""

import argparse
import yaml
from pathlib import Path
import sys

def count_images(img_dir):
    """Count total images in directory."""
    count = 0
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        count += len(list(img_dir.glob(ext)))
    return count

def create_balanced_dataset_yaml(
    img_dir: Path,
    label_dir: Path,
    num_classes: int,
    output_yaml: Path,
):
    """Create dataset.yaml for balanced CDT-CAD training."""
    
    # Verify directories exist
    if not img_dir.exists():
        print(f"❌ Image directory not found: {img_dir}")
        return False
    
    if not label_dir.exists():
        print(f"❌ Label directory not found: {label_dir}")
        return False
    
    # Count images
    total_images = count_images(img_dir)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total: {total_images} images")
    
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
    
    # Get absolute paths for split files
    split_base = Path(f"datasets/splits/balanced_{img_dir.parent.name}").absolute()
    
    # Create dataset config with split files
    dataset_config = {
        'path': str(img_dir.parent.absolute()),
        'train': str(split_base / 'train.txt'),
        'val': str(split_base / 'val.txt'),
        'test': str(split_base / 'test.txt'),
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
    print("="*60)
    print("Balanced CDT-CAD Dataset Preparation")
    print("="*60)
    
    # Create 5-class balanced dataset config
    print("\n📦 Creating 5-class balanced dataset config...")
    img_dir_5class = Path("datasets/balanced/knees_cropped/images")
    label_dir_5class = Path("datasets/balanced/knees_cropped/labels")
    output_yaml_5class = Path("datasets/balanced/knees_cropped/dataset_cdt_cad_5class.yaml")
    
    success_5class = create_balanced_dataset_yaml(
        img_dir=img_dir_5class,
        label_dir=label_dir_5class,
        num_classes=5,
        output_yaml=output_yaml_5class,
    )
    
    # Create 10-class balanced dataset config
    print("\n📦 Creating 10-class balanced dataset config...")
    img_dir_10class = Path("datasets/balanced/knees_cropped_10_class/images")
    label_dir_10class = Path("datasets/balanced/knees_cropped_10_class/labels")
    output_yaml_10class = Path("datasets/balanced/knees_cropped_10_class/dataset_cdt_cad_10class.yaml")
    
    success_10class = create_balanced_dataset_yaml(
        img_dir=img_dir_10class,
        label_dir=label_dir_10class,
        num_classes=10,
        output_yaml=output_yaml_10class,
    )
    
    print("\n" + "="*60)
    if success_5class and success_10class:
        print("✅ Balanced dataset preparation complete!")
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
