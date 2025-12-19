"""
Comprehensive Dataset Statistics and Analysis for Object Detection (YOLO format)

This script performs comprehensive Exploratory Data Analysis (EDA) on object detection
datasets to understand data characteristics and optimize model training.

Based on research best practices:
- Distribution analysis of bounding box dimensions
- Aspect ratio analysis for anchor box optimization
- Object size distribution (small, medium, large)
- Class distribution and imbalance detection
- Spatial distribution of objects in images
- Image quality statistics

References:
- COCO Dataset Analysis: https://coco

dataset.org
- YOLO Anchor Box Analysis: https://medium.com/@vijayabhaskar96/tutorial-on-yolo-hyperparameter-tuning
- Object Detection Dataset Best Practices: https://averroes.ai
"""

import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from tqdm import tqdm
import cv2

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive dataset analysis for object detection"
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        required=True,
        help="Directory containing images"
    )
    parser.add_argument(
        "--label_dir",
        type=str,
        required=True,
        help="Directory containing YOLO format labels"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset_analysis",
        help="Output directory for analysis results"
    )
    parser.add_argument(
        "--class_names",
        type=str,
        nargs="+",
        default=None,
        help="Class names (optional, for better visualization)"
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default=None,
        help="Optional split file to analyze specific subset"
    )
    
    return parser.parse_args()


def load_dataset_info(
    img_dir: Path,
    label_dir: Path,
    split_file: Path = None
) -> Tuple[Dict, List]:
    """
    Load dataset annotations and image information.
    
    Returns:
        Tuple of (annotations_dict, image_list)
    """
    # Get image list
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    if split_file and split_file.exists():
        with open(split_file, 'r') as f:
            stems = [line.strip() for line in f if line.strip()]
        image_files = []
        for stem in stems:
            for ext in img_extensions:
                img_path = img_dir / f"{stem}{ext}"
                if img_path.exists():
                    image_files.append(img_path)
                    break
    else:
        image_files = [
            f for f in img_dir.iterdir()
            if f.suffix.lower() in img_extensions
        ]
    
    print(f"Found {len(image_files)} images")
    
    # Load annotations
    annotations = {}
    bbox_data = []
    
    for img_file in tqdm(image_files, desc="Loading annotations"):
        stem = img_file.stem
        label_file = label_dir / f"{stem}.txt"
        
        if not label_file.exists():
            continue
        
        # Read image dimensions
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # Parse labels
        boxes = []
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    class_id = int(float(parts[0]))
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    box_w = float(parts[3])
                    box_h = float(parts[4])
                    
                    boxes.append({
                        'class_id': class_id,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': box_w,
                        'height': box_h,
                        'area': box_w * box_h,
                        'aspect_ratio': box_w / box_h if box_h > 0 else 0,
                        'abs_width': box_w * w,
                        'abs_height': box_h * h,
                        'abs_area': box_w * box_h * w * h
                    })
                    
                    bbox_data.append({
                        'image': stem,
                        'img_width': w,
                        'img_height': h,
                        **boxes[-1]
                    })
        
        if boxes:
            annotations[stem] = {
                'image_path': str(img_file),
                'width': w,
                'height': h,
                'boxes': boxes
            }
    
    print(f"Loaded {len(annotations)} images with annotations")
    print(f"Total bounding boxes: {len(bbox_data)}")
    
    return annotations, bbox_data


def analyze_class_distribution(bbox_data: List[Dict]) -> Dict:
    """Analyze class distribution."""
    class_counts = Counter(b['class_id'] for b in bbox_data)
    
    stats = {
        'class_counts': dict(sorted(class_counts.items())),
        'total_instances': len(bbox_data),
        'num_classes': len(class_counts),
        'max_instances': max(class_counts.values()),
        'min_instances': min(class_counts.values()),
        'imbalance_ratio': max(class_counts.values()) / min(class_counts.values())
    }
    
    return stats


def analyze_bbox_dimensions(bbox_data: List[Dict]) -> Dict:
    """Analyze bounding box dimensions."""
    widths = [b['width'] for b in bbox_data]
    heights = [b['height'] for b in bbox_data]
    areas = [b['area'] for b in bbox_data]
    aspect_ratios = [b['aspect_ratio'] for b in bbox_data if b['aspect_ratio'] > 0]
    
    abs_widths = [b['abs_width'] for b in bbox_data]
    abs_heights = [b['abs_height'] for b in bbox_data]
    abs_areas = [b['abs_area'] for b in bbox_data]
    
    stats = {
        'normalized': {
            'width': {
                'mean': float(np.mean(widths)),
                'std': float(np.std(widths)),
                'min': float(np.min(widths)),
                'max': float(np.max(widths)),
                'median': float(np.median(widths))
            },
            'height': {
                'mean': float(np.mean(heights)),
                'std': float(np.std(heights)),
                'min': float(np.min(heights)),
                'max': float(np.max(heights)),
                'median': float(np.median(heights))
            },
            'area': {
                'mean': float(np.mean(areas)),
                'std': float(np.std(areas)),
                'min': float(np.min(areas)),
                'max': float(np.max(areas)),
                'median': float(np.median(areas))
            }
        },
        'absolute_pixels': {
            'width': {
                'mean': float(np.mean(abs_widths)),
                'std': float(np.std(abs_widths)),
                'min': float(np.min(abs_widths)),
                'max': float(np.max(abs_widths)),
                'median': float(np.median(abs_widths))
            },
            'height': {
                'mean': float(np.mean(abs_heights)),
                'std': float(np.std(abs_heights)),
                'min': float(np.min(abs_heights)),
                'max': float(np.max(abs_heights)),
                'median': float(np.median(abs_heights))
            },
            'area': {
                'mean': float(np.mean(abs_areas)),
                'std': float(np.std(abs_areas)),
                'min': float(np.min(abs_areas)),
                'max': float(np.max(abs_areas)),
                'median': float(np.median(abs_areas))
            }
        },
        'aspect_ratio': {
            'mean': float(np.mean(aspect_ratios)),
            'std': float(np.std(aspect_ratios)),
            'min': float(np.min(aspect_ratios)),
            'max': float(np.max(aspect_ratios)),
            'median': float(np.median(aspect_ratios)),
            'percentile_25': float(np.percentile(aspect_ratios, 25)),
            'percentile_75': float(np.percentile(aspect_ratios, 75))
        }
    }
    
    return stats


def analyze_object_sizes(bbox_data: List[Dict]) -> Dict:
    """
    Categorize objects by size (COCO-style).
    Small: area < 0.01 (32x32 pixels at 640x640)
    Medium: 0.01 <= area < 0.05
    Large: area >= 0.05
    """
    small = sum(1 for b in bbox_data if b['area'] < 0.01)
    medium = sum(1 for b in bbox_data if 0.01 <= b['area'] < 0.05)
    large = sum(1 for b in bbox_data if b['area'] >= 0.05)
    
    total = len(bbox_data)
    
    return {
        'small': {'count': small, 'percentage': small / total * 100},
        'medium': {'count': medium, 'percentage': medium / total * 100},
        'large': {'count': large, 'percentage': large / total * 100}
    }


def analyze_spatial_distribution(bbox_data: List[Dict]) -> Dict:
    """Analyze spatial distribution of object centers."""
    x_centers = [b['x_center'] for b in bbox_data]
    y_centers = [b['y_center'] for b in bbox_data]
    
    stats = {
        'x_center': {
            'mean': float(np.mean(x_centers)),
            'std': float(np.std(x_centers)),
            'min': float(np.min(x_centers)),
            'max': float(np.max(x_centers))
        },
        'y_center': {
            'mean': float(np.mean(y_centers)),
            'std': float(np.std(y_centers)),
            'min': float(np.min(y_centers)),
            'max': float(np.max(y_centers))
        }
    }
    
    return stats


def analyze_image_properties(annotations: Dict) -> Dict:
    """Analyze image properties."""
    widths = [a['width'] for a in annotations.values()]
    heights = [a['height'] for a in annotations.values()]
    num_objects = [len(a['boxes']) for a in annotations.values()]
    
    stats = {
        'image_dimensions': {
            'width': {
                'mean': float(np.mean(widths)),
                'std': float(np.std(widths)),
                'min': int(np.min(widths)),
                'max': int(np.max(widths)),
                'unique': len(set(widths))
            },
            'height': {
                'mean': float(np.mean(heights)),
                'std': float(np.std(heights)),
                'min': int(np.min(heights)),
                'max': int(np.max(heights)),
                'unique': len(set(heights))
            }
        },
        'objects_per_image': {
            'mean': float(np.mean(num_objects)),
            'std': float(np.std(num_objects)),
            'min': int(np.min(num_objects)),
            'max': int(np.max(num_objects)),
            'median': float(np.median(num_objects))
        }
    }
    
    return stats


def create_visualizations(
    bbox_data: List[Dict],
    class_stats: Dict,
    bbox_stats: Dict,
    size_stats: Dict,
    output_dir: Path,
    class_names: List[str] = None
):
    """Create comprehensive visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Figure 1: Class Distribution
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    classes = sorted(class_stats['class_counts'].keys())
    counts = [class_stats['class_counts'][c] for c in classes]
    
    if class_names:
        labels = [class_names[c] if c < len(class_names) else f"Class {c}" for c in classes]
    else:
        labels = [f"Class {c}" for c in classes]
    
    # Bar chart
    axes[0].bar(labels, counts, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Class', fontsize=12)
    axes[0].set_ylabel('Number of Instances', fontsize=12)
    axes[0].set_title('Class Distribution', fontsize=14, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Pie chart
    axes[1].pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Bounding Box Dimensions
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig)
    
    widths = [b['width'] for b in bbox_data]
    heights = [b['height'] for b in bbox_data]
    areas = [b['area'] for b in bbox_data]
    aspect_ratios = [b['aspect_ratio'] for b in bbox_data if b['aspect_ratio'] > 0]
    
    # Width distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(widths, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Normalized Width', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.set_title('Width Distribution', fontsize=12, fontweight='bold')
    ax1.axvline(np.mean(widths), color='red', linestyle='--', label=f'Mean: {np.mean(widths):.3f}')
    ax1.legend()
    
    # Height distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(heights, bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Normalized Height', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.set_title('Height Distribution', fontsize=12, fontweight='bold')
    ax2.axvline(np.mean(heights), color='red', linestyle='--', label=f'Mean: {np.mean(heights):.3f}')
    ax2.legend()
    
    # Area distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(areas, bins=50, color='mediumseagreen', edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Normalized Area', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.set_title('Area Distribution', fontsize=12, fontweight='bold')
    ax3.axvline(np.mean(areas), color='red', linestyle='--', label=f'Mean: {np.mean(areas):.3f}')
    ax3.legend()
    
    # Aspect ratio distribution
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.hist(aspect_ratios, bins=50, color='orchid', edgecolor='black', alpha=0.7)
    ax4.set_xlabel('Aspect Ratio (Width/Height)', fontsize=10)
    ax4.set_ylabel('Frequency', fontsize=10)
    ax4.set_title('Aspect Ratio Distribution (for Anchor Box Optimization)', fontsize=12, fontweight='bold')
    ax4.axvline(np.mean(aspect_ratios), color='red', linestyle='--', label=f'Mean: {np.mean(aspect_ratios):.3f}')
    ax4.axvline(np.median(aspect_ratios), color='green', linestyle='--', label=f'Median: {np.median(aspect_ratios):.3f}')
    ax4.legend()
    
    # Object size categories
    ax5 = fig.add_subplot(gs[1, 2])
    size_categories = ['Small\n(<1%)', 'Medium\n(1-5%)', 'Large\n(>5%)']
    size_counts = [
        size_stats['small']['count'],
        size_stats['medium']['count'],
        size_stats['large']['count']
    ]
    ax5.bar(size_categories, size_counts, color=['gold', 'orange', 'darkred'], edgecolor='black')
    ax5.set_ylabel('Number of Objects', fontsize=10)
    ax5.set_title('Object Size Categories\n(% of Image Area)', fontsize=12, fontweight='bold')
    for i, v in enumerate(size_counts):
        ax5.text(i, v + max(size_counts) * 0.02, str(v), ha='center', fontweight='bold')
    
    # Width vs Height scatter
    ax6 = fig.add_subplot(gs[2, :2])
    scatter = ax6.scatter(widths, heights, c=areas, cmap='viridis', alpha=0.5, s=20)
    ax6.set_xlabel('Normalized Width', fontsize=10)
    ax6.set_ylabel('Normalized Height', fontsize=10)
    ax6.set_title('Width vs Height (colored by Area)', fontsize=12, fontweight='bold')
    ax6.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Square (AR=1)')
    ax6.legend()
    plt.colorbar(scatter, ax=ax6, label='Normalized Area')
    
    # Spatial distribution heatmap
    ax7 = fig.add_subplot(gs[2, 2])
    x_centers = [b['x_center'] for b in bbox_data]
    y_centers = [b['y_center'] for b in bbox_data]
    heatmap, xedges, yedges = np.histogram2d(x_centers, y_centers, bins=20)
    extent = [0, 1, 0, 1]
    im = ax7.imshow(heatmap.T, extent=extent, origin='lower', cmap='hot', aspect='auto')
    ax7.set_xlabel('X Center (normalized)', fontsize=10)
    ax7.set_ylabel('Y Center (normalized)', fontsize=10)
    ax7.set_title('Object Center Heatmap', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax7, label='Density')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'bbox_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved visualizations to {output_dir}")


def save_statistics_report(
    class_stats: Dict,
    bbox_stats: Dict,
    size_stats: Dict,
    spatial_stats: Dict,
    img_stats: Dict,
    output_dir: Path
):
    """Save comprehensive statistics report to JSON."""
    report = {
        'class_distribution': class_stats,
        'bounding_box_statistics': bbox_stats,
        'object_size_categories': size_stats,
        'spatial_distribution': spatial_stats,
        'image_properties': img_stats
    }
    
    report_path = output_dir / 'dataset_statistics.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved statistics report to {report_path}")
    
    # Also save a markdown summary
    md_path = output_dir / 'DATASET_ANALYSIS_REPORT.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Dataset Analysis Report\n\n")
        
        f.write("## Class Distribution\n\n")
        f.write(f"- **Total Classes**: {class_stats['num_classes']}\n")
        f.write(f"- **Total Instances**: {class_stats['total_instances']}\n")
        f.write(f"- **Imbalance Ratio**: {class_stats['imbalance_ratio']:.2f}:1\n\n")
        
        f.write("### Per-Class Counts\n\n")
        f.write("| Class | Count |\n")
        f.write("|-------|-------|\n")
        for cls, count in sorted(class_stats['class_counts'].items()):
            f.write(f"| {cls} | {count} |\n")
        
        f.write("\n## Bounding Box Statistics\n\n")
        f.write("### Normalized Dimensions\n\n")
        f.write(f"- **Width**: μ={bbox_stats['normalized']['width']['mean']:.4f}, σ={bbox_stats['normalized']['width']['std']:.4f}\n")
        f.write(f"- **Height**: μ={bbox_stats['normalized']['height']['mean']:.4f}, σ={bbox_stats['normalized']['height']['std']:.4f}\n")
        f.write(f"- **Area**: μ={bbox_stats['normalized']['area']['mean']:.4f}, σ={bbox_stats['normalized']['area']['std']:.4f}\n")
        f.write(f"- **Aspect Ratio**: μ={bbox_stats['aspect_ratio']['mean']:.4f}, median={bbox_stats['aspect_ratio']['median']:.4f}\n\n")
        
        f.write("## Object Size Categories\n\n")
        f.write(f"- **Small (<1% of image)**: {size_stats['small']['count']} ({size_stats['small']['percentage']:.1f}%)\n")
        f.write(f"- **Medium (1-5%)**: {size_stats['medium']['count']} ({size_stats['medium']['percentage']:.1f}%)\n")
        f.write(f"- **Large (>5%)**: {size_stats['large']['count']} ({size_stats['large']['percentage']:.1f}%)\n\n")
        
        f.write("## Recommendations\n\n")
        
        if class_stats['imbalance_ratio'] > 5:
            f.write("⚠️ **Class Imbalance Detected**: Consider using weighted loss or data augmentation for minority classes.\n\n")
        
        if size_stats['small']['percentage'] > 30:
            f.write("⚠️ **Many Small Objects**: Consider using feature pyramid networks or multi-scale training.\n\n")
        
        f.write("📊 **Anchor Box Suggestions**: Use k-means clustering on bbox dimensions to optimize anchor boxes for YOLO.\n\n")
    
    print(f"✅ Saved markdown report to {md_path}")


def main():
    """Main function."""
    args = parse_args()
    
    print("=" * 70)
    print("Dataset Statistics and Analysis for Object Detection")
    print("=" * 70)
    
    # Convert paths
    img_dir = Path(args.img_dir)
    label_dir = Path(args.label_dir)
    output_dir = Path(args.output_dir)
    split_file = Path(args.split_file) if args.split_file else None
    
    # Validate
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")
    
    # Load data
    print("\n📊 Loading dataset...")
    annotations, bbox_data = load_dataset_info(img_dir, label_dir, split_file)
    
    if not bbox_data:
        print("❌ No annotations found!")
        return
    
    # Analyze
    print("\n📈 Analyzing class distribution...")
    class_stats = analyze_class_distribution(bbox_data)
    
    print("\n📏 Analyzing bounding box dimensions...")
    bbox_stats = analyze_bbox_dimensions(bbox_data)
    
    print("\n📐 Categorizing object sizes...")
    size_stats = analyze_object_sizes(bbox_data)
    
    print("\n🗺️  Analyzing spatial distribution...")
    spatial_stats = analyze_spatial_distribution(bbox_data)
    
    print("\n🖼️  Analyzing image properties...")
    img_stats = analyze_image_properties(annotations)
    
    # Visualize
    print("\n🎨 Creating visualizations...")
    create_visualizations(
        bbox_data=bbox_data,
        class_stats=class_stats,
        bbox_stats=bbox_stats,
        size_stats=size_stats,
        output_dir=output_dir,
        class_names=args.class_names
    )
    
    # Save report
    print("\n💾 Saving statistics report...")
    save_statistics_report(
        class_stats=class_stats,
        bbox_stats=bbox_stats,
        size_stats=size_stats,
        spatial_stats=spatial_stats,
        img_stats=img_stats,
        output_dir=output_dir
    )
    
    print("\n" + "=" * 70)
    print("✅ Analysis completed successfully!")
    print(f"📁 Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()


# Example usage:
# python check_dataset/analyze_dataset.py --img_dir dataset/dataset_v1/images --label_dir dataset/dataset_v1/labels --output_dir analysis_results --class_names KL0 KL1 KL2 KL3 KL4
# python check_dataset/analyze_dataset.py --img_dir dataset/dataset_v1/images --label_dir dataset/dataset_v1/labels_new --output_dir analysis_results_new --class_names KL0-a KL0-b KL1-a KL1-b KL2-a KL2-b KL3-a KL3-b KL4-a KL4-b
