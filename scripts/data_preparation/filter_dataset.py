"""
Filter dataset by class distribution threshold.

This script filters out rare classes (below a specified percentage threshold)
and copies only images/labels with sufficient class representation to a new directory.

Use case: Remove extremely rare classes that may cause training instability.
"""

import argparse
import shutil
import json
from pathlib import Path
from collections import Counter
from typing import Dict, List, Set, Tuple
from tqdm import tqdm


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Filter dataset by removing rare classes"
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        required=True,
        help="Source image directory"
    )
    parser.add_argument(
        "--label_dir",
        type=str,
        required=True,
        help="Source label directory"
    )
    parser.add_argument(
        "--output_img_dir",
        type=str,
        required=True,
        help="Output image directory"
    )
    parser.add_argument(
        "--output_label_dir",
        type=str,
        required=True,
        help="Output label directory"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Minimum percentage threshold (default: 1.0%%)"
    )
    parser.add_argument(
        "--analysis_json",
        type=str,
        default=None,
        help="Optional: analysis JSON file to load class counts from"
    )
    parser.add_argument(
        "--remove_rare_from_labels",
        action="store_true",
        help="Remove rare class boxes from labels (keep image if it has other valid boxes)"
    )
    
    return parser.parse_args()


def load_class_distribution(label_dir: Path) -> Dict[int, int]:
    """Load class distribution from label files."""
    class_counts = Counter()
    
    label_files = list(label_dir.glob("*.txt"))
    
    for label_file in tqdm(label_files, desc="Scanning labels"):
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    class_id = int(float(parts[0]))
                    class_counts[class_id] += 1
    
    return dict(class_counts)


def identify_rare_classes(
    class_counts: Dict[int, int],
    threshold_percent: float
) -> Tuple[Set[int], Set[int]]:
    """
    Identify rare and valid classes based on threshold.
    
    Returns:
        Tuple of (rare_classes, valid_classes)
    """
    total = sum(class_counts.values())
    rare_classes = set()
    valid_classes = set()
    
    print(f"\n📊 Class Analysis (Threshold: {threshold_percent}%)")
    print("=" * 60)
    
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        percentage = (count / total) * 100
        
        if percentage < threshold_percent:
            rare_classes.add(class_id)
            print(f"❌ Class {class_id}: {count} instances ({percentage:.2f}%) - FILTERED")
        else:
            valid_classes.add(class_id)
            print(f"✅ Class {class_id}: {count} instances ({percentage:.2f}%) - KEPT")
    
    print("=" * 60)
    print(f"Rare classes to filter: {sorted(rare_classes)}")
    print(f"Valid classes to keep: {sorted(valid_classes)}")
    
    return rare_classes, valid_classes


def filter_and_copy_dataset(
    img_dir: Path,
    label_dir: Path,
    output_img_dir: Path,
    output_label_dir: Path,
    rare_classes: Set[int],
    remove_rare_from_labels: bool = False
):
    """
    Filter and copy dataset to new directory.
    
    Args:
        img_dir: Source image directory
        label_dir: Source label directory
        output_img_dir: Output image directory
        output_label_dir: Output label directory
        rare_classes: Set of rare class IDs to filter
        remove_rare_from_labels: If True, remove rare bboxes but keep image if it has other boxes
    """
    # Create output directories
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all label files
    label_files = list(label_dir.glob("*.txt"))
    
    stats = {
        'total_images': 0,
        'copied_images': 0,
        'filtered_images': 0,
        'total_boxes_original': 0,
        'total_boxes_filtered': 0,
        'boxes_removed': 0
    }
    
    # Get image extensions
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    for label_file in tqdm(label_files, desc="Processing labels"):
        stem = label_file.stem
        stats['total_images'] += 1
        
        # Find corresponding image
        img_file = None
        for ext in img_extensions:
            candidate = img_dir / f"{stem}{ext}"
            if candidate.exists():
                img_file = candidate
                break
        
        if img_file is None:
            continue
        
        # Read and filter labels
        valid_boxes = []
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    class_id = int(float(parts[0]))
                    stats['total_boxes_original'] += 1
                    
                    if class_id not in rare_classes:
                        valid_boxes.append(line)
                    else:
                        stats['boxes_removed'] += 1
        
        # Decide whether to copy this image
        if remove_rare_from_labels:
            # Keep image if it has at least one valid box
            if valid_boxes:
                # Copy image
                shutil.copy2(img_file, output_img_dir / img_file.name)
                
                # Copy filtered labels
                output_label_file = output_label_dir / label_file.name
                with open(output_label_file, 'w') as f:
                    for box_line in valid_boxes:
                        f.write(box_line + '\n')
                
                stats['copied_images'] += 1
                stats['total_boxes_filtered'] += len(valid_boxes)
            else:
                stats['filtered_images'] += 1
        else:
            # Only keep image if ALL its boxes are valid (no rare classes)
            if len(valid_boxes) == stats['total_boxes_original']:
                # All boxes are valid
                shutil.copy2(img_file, output_img_dir / img_file.name)
                shutil.copy2(label_file, output_label_dir / label_file.name)
                
                stats['copied_images'] += 1
                stats['total_boxes_filtered'] += len(valid_boxes)
            else:
                stats['filtered_images'] += 1
    
    return stats


def save_filter_report(
    stats: Dict,
    class_counts: Dict[int, int],
    rare_classes: Set[int],
    valid_classes: Set[int],
    output_dir: Path
):
    """Save filtering report."""
    report = {
        'filtering_statistics': stats,
        'class_distribution': {
            'original': class_counts,
            'rare_classes_filtered': sorted(rare_classes),
            'valid_classes_kept': sorted(valid_classes)
        }
    }
    
    report_path = output_dir / 'filter_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    # Markdown report
    md_path = output_dir / 'FILTER_REPORT.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Dataset Filtering Report\n\n")
        
        f.write("## Statistics\n\n")
        f.write(f"- **Original Images**: {stats['total_images']}\n")
        f.write(f"- **Copied Images**: {stats['copied_images']}\n")
        f.write(f"- **Filtered Images**: {stats['filtered_images']}\n")
        f.write(f"- **Retention Rate**: {stats['copied_images']/stats['total_images']*100:.1f}%\n\n")
        
        f.write(f"- **Original Bounding Boxes**: {stats['total_boxes_original']}\n")
        f.write(f"- **Filtered Bounding Boxes**: {stats['total_boxes_filtered']}\n")
        f.write(f"- **Boxes Removed**: {stats['boxes_removed']}\n")
        f.write(f"- **Box Retention Rate**: {stats['total_boxes_filtered']/stats['total_boxes_original']*100:.1f}%\n\n")
        
        f.write("## Classes Filtered\n\n")
        f.write(f"Rare classes removed: {sorted(rare_classes)}\n\n")
        
        f.write("## Classes Kept\n\n")
        f.write(f"Valid classes kept: {sorted(valid_classes)}\n\n")
    
    print(f"\n✅ Saved reports:")
    print(f"   {report_path}")
    print(f"   {md_path}")


def main():
    """Main function."""
    args = parse_args()
    
    print("=" * 70)
    print("Dataset Filtering by Class Distribution")
    print("=" * 70)
    
    # Convert paths
    img_dir = Path(args.img_dir)
    label_dir = Path(args.label_dir)
    output_img_dir = Path(args.output_img_dir)
    output_label_dir = Path(args.output_label_dir)
    
    # Load class distribution
    print("\n📊 Analyzing class distribution...")
    
    if args.analysis_json:
        # Load from analysis JSON
        with open(args.analysis_json, 'r') as f:
            analysis = json.load(f)
            class_counts = {int(k): v for k, v in analysis['class_distribution']['class_counts'].items()}
    else:
        class_counts = load_class_distribution(label_dir)
    
    # Identify rare classes
    rare_classes, valid_classes = identify_rare_classes(
        class_counts=class_counts,
        threshold_percent=args.threshold
    )
    
    if not rare_classes:
        print("\n✅ No rare classes found! All classes meet the threshold.")
        return
    
    # Filter and copy
    print(f"\n🔄 Filtering dataset...")
    print(f"   Mode: {'Remove rare boxes, keep images with valid boxes' if args.remove_rare_from_labels else 'Skip images with any rare boxes'}")
    
    stats = filter_and_copy_dataset(
        img_dir=img_dir,
        label_dir=label_dir,
        output_img_dir=output_img_dir,
        output_label_dir=output_label_dir,
        rare_classes=rare_classes,
        remove_rare_from_labels=args.remove_rare_from_labels
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("Filtering Summary")
    print("=" * 70)
    print(f"Original images: {stats['total_images']}")
    print(f"Copied images: {stats['copied_images']}")
    print(f"Filtered images: {stats['filtered_images']}")
    print(f"Retention rate: {stats['copied_images']/stats['total_images']*100:.1f}%")
    print()
    print(f"Original bounding boxes: {stats['total_boxes_original']}")
    print(f"Filtered bounding boxes: {stats['total_boxes_filtered']}")
    print(f"Boxes removed: {stats['boxes_removed']}")
    print(f"Box retention rate: {stats['total_boxes_filtered']/stats['total_boxes_original']*100:.1f}%")
    
    # Save report
    save_filter_report(
        stats=stats,
        class_counts=class_counts,
        rare_classes=rare_classes,
        valid_classes=valid_classes,
        output_dir=output_label_dir.parent
    )
    
    print("\n" + "=" * 70)
    print("✅ Filtering completed successfully!")
    print(f"📁 Filtered dataset saved to:")
    print(f"   Images: {output_img_dir}")
    print(f"   Labels: {output_label_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()


# Example usage:
# Remove images that have ANY rare class (<1%):
# python filter_dataset_by_class.py --img_dir dataset/dataset_v1/images --label_dir dataset/dataset_v1/labels_new --output_img_dir dataset/dataset_filtered/images --output_label_dir dataset/dataset_filtered/labels --threshold 1.0 --analysis_json analysis_results_new/dataset_statistics.json

# Keep images, just remove rare class boxes:
# python filter_dataset_by_class.py --img_dir dataset/dataset_v1/images --label_dir dataset/dataset_v1/labels_new --output_img_dir dataset/dataset_filtered/images --output_label_dir dataset/dataset_filtered/labels --threshold 1.0 --remove_rare_from_labels --analysis_json analysis_results_new/dataset_statistics.json
