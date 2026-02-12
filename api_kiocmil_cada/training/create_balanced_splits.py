#!/usr/bin/env python3
"""
Create Train/Val/Test Splits for Balanced Datasets

Creates split files for balanced CDT-CAD datasets.
"""

import argparse
from pathlib import Path
import random

def create_splits(img_dir: Path, output_dir: Path, train_ratio=0.7, val_ratio=0.2):
    """Create train/val/test splits."""
    
    # Get all image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        image_files.extend(list(img_dir.glob(ext)))
    
    # Get basenames (without extension)
    basenames = [f.stem for f in image_files]
    
    # Shuffle
    random.seed(42)
    random.shuffle(basenames)
    
    # Calculate split sizes
    total = len(basenames)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    # Split
    train_files = basenames[:train_size]
    val_files = basenames[train_size:train_size + val_size]
    test_files = basenames[train_size + val_size:]
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write split files
    with open(output_dir / "train.txt", "w") as f:
        f.write("\n".join(train_files) + "\n")
    
    with open(output_dir / "val.txt", "w") as f:
        f.write("\n".join(val_files) + "\n")
    
    with open(output_dir / "test.txt", "w") as f:
        f.write("\n".join(test_files) + "\n")
    
    print(f"✅ Created splits:")
    print(f"   Train: {len(train_files)} images")
    print(f"   Val:   {len(val_files)} images")
    print(f"   Test:  {len(test_files)} images")
    print(f"   Total: {total} images")
    
    return output_dir / "train.txt", output_dir / "val.txt", output_dir / "test.txt"

def main():
    print("="*60)
    print("Creating Balanced Dataset Splits")
    print("="*60)
    
    # 5-class
    print("\n📦 Creating 5-class splits...")
    img_dir_5class = Path("datasets/balanced/knees_cropped/images")
    output_dir_5class = Path("datasets/splits/balanced_knees_cropped")
    
    train_5, val_5, test_5 = create_splits(img_dir_5class, output_dir_5class)
    
    # 10-class
    print("\n📦 Creating 10-class splits...")
    img_dir_10class = Path("datasets/balanced/knees_cropped_10_class/images")
    output_dir_10class = Path("datasets/splits/balanced_knees_cropped_10_class")
    
    train_10, val_10, test_10 = create_splits(img_dir_10class, output_dir_10class)
    
    print("\n" + "="*60)
    print("✅ Splits created successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
