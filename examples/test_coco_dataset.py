"""
Test script for COCO dataset loader and DETR compatibility.

This script:
1. Converts YOLO labels to COCO JSON
2. Loads CocoDataset
3. Tests with DETR processor
4. Tests collate function
5. Visualizes samples
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import CocoDataset, create_coco_json, get_detr_processor, detr_collate_fn
from datasets.coco_dataset import visualize_coco_sample
from config import CLASSES, CLASSES_LABEL_NEW
import torch
from torch.utils.data import DataLoader


def test_coco_conversion():
    """Test YOLO to COCO conversion."""
    print("=" * 60)
    print("Step 1: Converting YOLO labels to COCO JSON")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("processed/coco")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert train split - original labels
    print("\n1.1. Converting train split (original 5 classes)...")
    train_json = create_coco_json(
        yolo_label_dir="processed/knee/labels",
        img_dir="processed/knee/images",
        output_path="processed/coco/annotations_train.json",
        class_names=CLASSES,
        split_file="splits/train.txt"
    )
    
    # Convert val split - original labels
    print("\n1.2. Converting val split (original 5 classes)...")
    val_json = create_coco_json(
        yolo_label_dir="processed/knee/labels",
        img_dir="processed/knee/images",
        output_path="processed/coco/annotations_val.json",
        class_names=CLASSES,
        split_file="splits/val.txt"
    )
    
    # Check if labels_new exists
    labels_new_dir = Path("processed/knee/labels_new")
    if labels_new_dir.exists():
        print("\n1.3. Converting train split (labels_new - 10 classes)...")
        train_new_json = create_coco_json(
            yolo_label_dir="processed/knee/labels_new",
            img_dir="processed/knee/images",
            output_path="processed/coco/annotations_train_new.json",
            class_names=CLASSES_LABEL_NEW,
            split_file="splits/train.txt"
        )
        
        print("\n1.4. Converting val split (labels_new - 10 classes)...")
        val_new_json = create_coco_json(
            yolo_label_dir="processed/knee/labels_new",
            img_dir="processed/knee/images",
            output_path="processed/coco/annotations_val_new.json",
            class_names=CLASSES_LABEL_NEW,
            split_file="splits/val.txt"
        )
    
    print("\n✅ COCO JSON conversion completed!")


def test_coco_dataset_raw():
    """Test COCO dataset loading without processor."""
    print("\n" + "=" * 60)
    print("Step 2: Testing COCO Dataset (Raw Format)")
    print("=" * 60)
    
    # Load dataset without processor
    dataset = CocoDataset(
        coco_json_path="processed/coco/annotations_train.json",
        img_dir="processed/knee/images"
    )
    
    print(f"\n✅ Dataset loaded: {len(dataset)} images")
    print(f"   Class names: {dataset.get_class_names()}")
    
    # Get a sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\n   Sample keys: {sample.keys()}")
        print(f"   Image type: {type(sample['image'])}")
        print(f"   Image size: {sample['image'].size}")
        print(f"   Num boxes: {len(sample['target']['boxes'])}")
        print(f"   Image ID: {sample['target']['image_id']}")
        
        if len(sample['target']['boxes']) > 0:
            print(f"   First box (COCO format): {sample['target']['boxes'][0]}")
            print(f"   First label: {sample['target']['class_labels'][0]}")
    
    # Visualize samples
    print("\n   Visualizing samples...")
    output_dir = Path("check_vis/test_coco")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(min(3, len(dataset))):
        save_path = output_dir / f"coco_sample_{i}.png"
        visualize_coco_sample(dataset, idx=i, save_path=str(save_path))


def test_coco_dataset_with_detr():
    """Test COCO dataset with DETR processor."""
    print("\n" + "=" * 60)
    print("Step 3: Testing COCO Dataset with DETR Processor")
    print("=" * 60)
    
    try:
        # Get DETR processor
        print("\n3.1. Loading DETR processor...")
        processor = get_detr_processor(model_name="facebook/detr-resnet-50")
        print(f"✅ Processor loaded: {type(processor)}")
        
        # Load dataset with processor
        print("\n3.2. Loading dataset with processor...")
        dataset = CocoDataset(
            coco_json_path="processed/coco/annotations_train.json",
            img_dir="processed/knee/images",
            processor=processor
        )
        
        print(f"✅ Dataset loaded: {len(dataset)} images")
        
        # Get a sample
        if len(dataset) > 0:
            print("\n3.3. Testing sample retrieval...")
            sample = dataset[0]
            print(f"   Sample keys: {sample.keys()}")
            print(f"   Pixel values shape: {sample['pixel_values'].shape}")
            print(f"   Pixel mask shape: {sample['pixel_mask'].shape}")
            print(f"   Labels keys: {sample['labels'].keys()}")
            
            labels = sample['labels']
            print(f"   Class labels shape: {labels['class_labels'].shape}")
            print(f"   Boxes shape: {labels['boxes'].shape}")
            print(f"   First box: {labels['boxes'][0] if len(labels['boxes']) > 0 else 'No boxes'}")
        
        # Test DataLoader with collate function
        print("\n3.4. Testing DataLoader with collate function...")
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=detr_collate_fn,
            num_workers=0
        )
        
        batch = next(iter(dataloader))
        print(f"   Batch keys: {batch.keys()}")
        print(f"   Pixel values batch shape: {batch['pixel_values'].shape}")
        print(f"   Pixel mask batch shape: {batch['pixel_mask'].shape}")
        print(f"   Number of labels in batch: {len(batch['labels'])}")
        
        print("\n✅ DETR compatibility test passed!")
        
    except ImportError as e:
        print(f"\n⚠️  Transformers library not installed: {e}")
        print("   Install with: pip install transformers")
        print("   Skipping DETR processor test...")
    except Exception as e:
        print(f"\n❌ Error during DETR test: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("COCO Dataset Loader Test Suite")
    print("=" * 60)
    
    try:
        # Step 1: Convert YOLO to COCO
        test_coco_conversion()
        
        # Step 2: Test raw COCO dataset
        test_coco_dataset_raw()
        
        # Step 3: Test with DETR processor
        test_coco_dataset_with_detr()
        
        print("\n" + "=" * 60)
        print("✅ All COCO dataset tests completed!")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        print("\nMake sure you have:")
        print("  1. Processed data in: processed/knee/images and processed/knee/labels")
        print("  2. Split files in: splits/train.txt and splits/val.txt")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
