#!/usr/bin/env python3
"""
Generate dataset visualizations for documentation without cv2 dependency.
Uses matplotlib and PIL for image processing.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json
import sys
from collections import Counter

def create_class_distribution_chart(dataset_stats_file, output_path, title="Class Distribution"):
    """Create class distribution bar chart from statistics file."""
    
    # Read statistics
    with open(dataset_stats_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Parse class distribution
    classes = []
    counts = []
    in_5class = False
    in_10class = False
    
    for line in lines:
        if "5-Class Labels" in line:
            in_5class = True
            in_10class = False
        elif "10-Class Labels" in line:
            in_5class = False
            in_10class = True
        elif "Class Distribution:" in line:
            continue
        elif in_5class or in_10class:
            if ":" in line and "boxes" in line:
                parts = line.strip().split(":")
                if len(parts) == 2:
                    class_name = parts[0].strip()
                    count_str = parts[1].split("boxes")[0].strip()
                    try:
                        count = int(count_str)
                        classes.append(class_name)
                        counts.append(count)
                    except:
                        pass
            elif line.strip() and line.strip()[0] == '-':
                break
    
    if not classes:
        print(f"No class data found in {dataset_stats_file}")
        return
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(classes)), counts, color='skyblue', edgecolor='black', linewidth=1.5)
    
    # Customize
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Boxes', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Created: {output_path}")


def create_imbalance_visualization(stats_file, output_path):
    """Create visualization showing class imbalance problem."""
    
    # Read statistics
    with open(stats_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse 5-class data
    classes_5 = {'KL0': 0, 'KL1': 0, 'KL2': 0, 'KL3': 0, 'KL4': 0}
    for class_name in classes_5.keys():
        import re
        pattern = rf'{class_name}\s*:\s*(\d+)\s*boxes'
        match = re.search(pattern, content)
        if match:
            classes_5[class_name] = int(match.group(1))
    
    # Create visualization with adjusted sizes
    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.3])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # Define distinct colors for each class
    class_colors = {
        'KL0': '#FF6B6B',  # Red
        'KL1': '#4ECDC4',  # Teal
        'KL2': '#45B7D1',  # Blue
        'KL3': '#96CEB4',  # Green
        'KL4': '#FFEAA7'   # Yellow
    }
    
    # Left: Bar chart (smaller)
    classes = list(classes_5.keys())
    counts = list(classes_5.values())
    colors = [class_colors[cls] for cls in classes]
    
    bars = ax1.bar(classes, counts, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_title('Class Distribution', fontsize=13, fontweight='bold')
    ax1.set_xlabel('KL Grade', fontsize=11)
    ax1.set_ylabel('Number of Boxes', fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add count labels (smaller)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}',
                ha='center', va='bottom', fontsize=9)
    
    # Right: Pie chart with LARGER percentages
    wedges, texts, autotexts = ax2.pie(counts, labels=classes, autopct='%1.1f%%', 
            startangle=90, colors=colors,
            textprops={'fontsize': 12, 'fontweight': 'bold'},
            wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
    
    # Make percentage text LARGER
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(14)
        autotext.set_fontweight('bold')
    
    ax2.set_title('Class Proportion (5-class)', fontsize=13, fontweight='bold')
    
    # Add imbalance ratio annotation
    if counts:
        ratio = max(counts) / min(counts)
        fig.text(0.5, 0.02, f'Imbalance Ratio: {ratio:.2f}:1 (Max/Min)', 
                ha='center', fontsize=12, fontweight='bold', 
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Created: {output_path}")


def create_pipeline_diagram(output_path):
    """Create simple pipeline workflow diagram."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Define steps
    steps = [
        {"y": 10.5, "text": "1. Crop Knees", "desc": "~1,691 full → ~1,783 knees"},
        {"y": 8.5, "text": "2. Generate Labels", "desc": "5/10/4/8-class variants"},
        {"y": 6.5, "text": "3. Balance Data", "desc": "Flip augmentation"},
        {"y": 4.5, "text": "4. Preprocess", "desc": "4 variants: resize/blur/sharp"},
        {"y": 2.5, "text": "5. Create Splits", "desc": "70% / 15% / 15%"},
        {"y": 0.5, "text": "✓ Ready for Training", "desc": ""},
    ]
    
    # Draw steps
    for i, step in enumerate(steps):
        # Box
        color = '#4CAF50' if i == len(steps)-1 else '#2196F3'
        rect = patches.FancyBboxPatch((1, step["y"]-0.35), 8, 0.7,
                                      boxstyle="round,pad=0.1",
                                      linewidth=2, edgecolor=color,
                                      facecolor=color, alpha=0.3)
        ax.add_patch(rect)
        
        # Text
        ax.text(5, step["y"], step["text"], 
                ha='center', va='center', fontsize=13, fontweight='bold')
        if step["desc"]:
            ax.text(5, step["y"]-0.15, step["desc"],
                    ha='center', va='center', fontsize=9, style='italic')
        
        # Arrow
        if i < len(steps) - 1:
            ax.arrow(5, step["y"]-0.5, 0, -1.3, head_width=0.3, head_length=0.15,
                    fc='gray', ec='gray', linewidth=2)
    
    ax.set_title('Data Processing Pipeline', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Created: {output_path}")


def create_augmentation_simple_demo(output_path):
    """Create augmentation demonstration with visual examples."""
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    
    # Title
    fig.suptitle('Data Augmentation Strategy for Balancing', fontsize=16, fontweight='bold', y=0.98)
    
    # Color scheme for different augmentation types
    colors = ['#E8F5E9', '#FFF3E0', '#E3F2FD', '#F3E5F5', '#FCE4EC', '#E0F2F1']
    
    augmentation_info = [
        ("Original\nImage", "Baseline\n(no transform)", colors[0]),
        ("Horizontal\nFlip", "Left ↔ Right\nmirror", colors[1]),
        ("Vertical\nFlip", "Top ↔ Bottom\nmirror", colors[2]),
        ("Both Flips", "H + V\ncombined", colors[3]),
        ("Conservative\nAug", "Mild rotation\n+ flip", colors[4]),
        ("Result", "Class balanced\ndataset", colors[5])
    ]
    
    for idx, (title, desc, color) in enumerate(augmentation_info):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        # Draw colored background box
        rect = patches.FancyBboxPatch((0, 0), 1, 1,
                                      boxstyle="round,pad=0.05",
                                      linewidth=3, edgecolor='black',
                                      facecolor=color, alpha=0.6)
        ax.add_patch(rect)
        
        # Add main title
        ax.text(0.5, 0.65, title, ha='center', va='center',
                fontsize=14, fontweight='bold', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.5))
        
        # Add description
        ax.text(0.5, 0.35, desc, ha='center', va='center',
                fontsize=11, style='italic', color='#333')
        
        # Add icon/symbol
        if idx == 0:
            ax.text(0.5, 0.15, '📷', ha='center', fontsize=30)
        elif idx == 1:
            ax.text(0.5, 0.15, '↔️', ha='center', fontsize=30)
        elif idx == 2:
            ax.text(0.5, 0.15, '↕️', ha='center', fontsize=30)
        elif idx == 3:
            ax.text(0.5, 0.15, '↔️↕️', ha='center', fontsize=24)
        elif idx == 4:
            ax.text(0.5, 0.15, '🔄', ha='center', fontsize=30)
        else:
            ax.text(0.5, 0.15, '✅', ha='center', fontsize=30)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    # Add summary text at bottom
    fig.text(0.5, 0.02, 
             'Strategy: Upsample minority classes (KL0, KL3, KL4) using flip augmentation to balance with majority class (KL2)',
             ha='center', fontsize=11, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5, pad=0.5))
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Created: {output_path}")


def create_bbox_analysis_from_dataset(dataset_dir, output_path):
    """Create bbox analysis visualization from dataset labels."""
    from pathlib import Path
    import numpy as np
    
    dataset_path = Path(dataset_dir)
    labels_dir = dataset_path / "labels"
    
    if not labels_dir.exists():
        print(f"⚠️  Labels directory not found: {labels_dir}")
        return
    
    # Collect all bbox data
    all_widths = []
    all_heights = []
    all_areas = []
    all_x = []
    all_y = []
    
    for label_file in labels_dir.glob("*.txt"):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    _, x, y, w, h = parts[:5]
                    x, y, w, h = float(x), float(y), float(w), float(h)
                    all_widths.append(w)
                    all_heights.append(h)
                    all_areas.append(w * h)
                    all_x.append(x)
                    all_y.append(y)
    
    if not all_widths:
        print("⚠️  No bbox data found")
        return
    
    # Create multi-panel visualization
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Width distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(all_widths, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(all_widths), color='red', linestyle='--', 
                label=f'Mean: {np.mean(all_widths):.3f}')
    ax1.set_title('Width Distribution', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Normalized Width')
    ax1.set_ylabel('Count')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Height distribution  
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(all_heights, bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(all_heights), color='red', linestyle='--',
                label=f'Mean: {np.mean(all_heights):.3f}')
    ax2.set_title('Height Distribution', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Normalized Height')
    ax2.set_ylabel('Count')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Area distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(all_areas, bins=50, color='mediumseagreen', edgecolor='black', alpha=0.7)
    ax3.axvline(np.mean(all_areas), color='red', linestyle='--',
                label=f'Mean: {np.mean(all_areas):.3f}')
    ax3.set_title('Area Distribution', fontweight='bold', fontsize=12)
    ax3.set_xlabel('Normalized Area')
    ax3.set_ylabel('Count')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Aspect ratio
    aspects = [w/h if h > 0 else 0 for w, h in zip(all_widths, all_heights)]
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(aspects, bins=50, color='mediumpurple', edgecolor='black', alpha=0.7)
    ax4.axvline(np.mean(aspects), color='red', linestyle='--',
                label=f'Mean: {np.mean(aspects):.3f}')
    ax4.axvline(np.median(aspects), color='green', linestyle='--',
                label=f'Median: {np.median(aspects):.3f}')
    ax4.set_title('Aspect Ratio (Width/Height)', fontweight='bold', fontsize=12)
    ax4.set_xlabel('Aspect Ratio')
    ax4.set_ylabel('Count')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # 5. Width vs Height scatter
    ax5 = fig.add_subplot(gs[1, 1])
    scatter = ax5.scatter(all_widths, all_heights, c=all_areas, cmap='viridis',
                         alpha=0.5, s=20, edgecolors='black', linewidth=0.5)
    max_val = max(max(all_widths), max(all_heights))
    ax5.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Square (1:1)')
    ax5.set_title('Width vs Height', fontweight='bold', fontsize=12)
    ax5.set_xlabel('Width')
    ax5.set_ylabel('Height')
    ax5.legend()
    ax5.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax5)
    cbar.set_label('Area')
    
    # 6. Center heatmap
    ax6 = fig.add_subplot(gs[1, 2])
    heatmap, xedges, yedges = np.histogram2d(all_x, all_y, bins=20)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax6.imshow(heatmap.T, extent=extent, origin='lower', cmap='hot', aspect='auto')
    ax6.set_title('BBox Center Heatmap', fontweight='bold', fontsize=12)
    ax6.set_xlabel('X Center')
    ax6.set_ylabel('Y Center')
    ax6.grid(True, alpha=0.3, color='white', linestyle='--')
    plt.colorbar(im, ax=ax6, label='Density')
    
    fig.suptitle(f'Bounding Box Analysis - {dataset_dir}', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Created: {output_path}")


if __name__ == "__main__":
    import numpy as np
    
    output_dir = Path("DeAn/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("GENERATING DATASET VISUALIZATIONS")
    print("="*80 + "\n")
    
    # 1. Class distribution charts
    stats_file = Path("datasets/dataset_v0/dataset_statistics.txt")
    if stats_file.exists():
        create_class_distribution_chart(
            stats_file, 
            output_dir / "class_distribution_5class_chart.png",
            "Phân bố 5-class (KL0-KL4)"
        )
        create_imbalance_visualization(
            stats_file,
            output_dir / "class_imbalance.png"
        )
    
    # 2. Pipeline diagram
    create_pipeline_diagram(output_dir / "pipeline_workflow.png")
    
    # 3. Augmentation demo
    create_augmentation_simple_demo(output_dir / "augmentation_strategy.png")
    
    # 4. BBox analysis from dataset_v0
    create_bbox_analysis_from_dataset(
        "datasets/dataset_v0",
        output_dir / "bbox_analysis.png"
    )
    
    print("\n" + "="*80)
    print("✅ ALL VISUALIZATIONS GENERATED")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Total files: {len(list(output_dir.glob('*.png')))}")
