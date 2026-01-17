#!/usr/bin/env python3
"""
Visualize filtered knee crops to understand why they were removed.
Compare original image with knee boxes and labels to see crop boundaries.

1. Visualize Specific Files
python scripts/analysis/visualize_filtered_crops.py \
    --files file1_knee0.txt file2_knee1.txt
2. Visualize All Files trong Folder
python scripts/analysis/visualize_filtered_crops.py \
    --folder datasets/dataset_knees_cropped/labels \
    --limit 10  # Optional: giới hạn số lượng
3. Comparison Mode (Tự động tìm filtered files)
python scripts/analysis/visualize_filtered_crops.py \
    --comparison datasets/dataset_knees_cropped/labels \
                 datasets/processed_draf/dataset_knees_cropped/labels
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def load_yolo_boxes(label_path):
    """Load YOLO format boxes."""
    boxes = []
    if not label_path.exists():
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                boxes.append(
                    {
                        "class_id": int(parts[0]),
                        "x": float(parts[1]),
                        "y": float(parts[2]),
                        "w": float(parts[3]),
                        "h": float(parts[4]),
                    }
                )
    return boxes


def yolo_to_pixel(box, img_w, img_h):
    """Convert YOLO normalized coords to pixel coords."""
    x_center = box["x"] * img_w
    y_center = box["y"] * img_h
    w = box["w"] * img_w
    h = box["h"] * img_h

    x1 = int(x_center - w / 2)
    y1 = int(y_center - h / 2)
    x2 = int(x_center + w / 2)
    y2 = int(y_center + h / 2)

    return x1, y1, x2, y2


def expand_to_square(x1, y1, x2, y2, img_w, img_h, margin=0.15):
    """Expand box to square with margin."""
    w = x2 - x1
    h = y2 - y1

    # Make square
    size = max(w, h)

    # Add margin
    size = int(size * (1 + 2 * margin))

    # Center
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    x1 = cx - size // 2
    y1 = cy - size // 2
    x2 = x1 + size
    y2 = y1 + size

    # Clamp to image
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    return x1, y1, x2, y2


def visualize_filtered_crop(image_name, dataset_dir, output_dir):
    """Visualize why a specific crop was filtered."""
    # Remove _knee suffix and .txt extension to get original image name
    parts = image_name.replace(".txt", "").split("_knee")
    base_name = parts[0]
    knee_idx = int(parts[1]) if len(parts) > 1 else 0

    # Load original image
    img_path = dataset_dir / "images" / f"{base_name}.png"
    if not img_path.exists():
        img_path = dataset_dir / "images" / f"{base_name}.jpg"

    if not img_path.exists():
        print(f"⚠️  Image not found: {base_name}")
        return

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"⚠️  Failed to load: {img_path}")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w = img.shape[:2]

    # Load labels
    knee_labels_path = dataset_dir / "labels-knee" / f"{base_name}.txt"
    kl_labels_path = dataset_dir / "labels" / f"{base_name}.txt"

    knee_boxes = load_yolo_boxes(knee_labels_path)
    kl_boxes = load_yolo_boxes(kl_labels_path)

    # Check if knee index exists
    knee_not_found = knee_idx >= len(knee_boxes)

    if knee_not_found:
        print(f"⚠️  Knee index {knee_idx} not found in {base_name}")
        print(f"    Available knees: {len(knee_boxes)}")
        print(f"    This may happen if knee boxes were manually edited.")

        # If no knees at all, skip
        if len(knee_boxes) == 0:
            print(f"    No knee boxes available - skipping")
            return

        # Use first available knee as fallback for visualization
        print(f"    Using knee 0 for visualization (fallback)")
        fallback_knee_idx = 0
    else:
        fallback_knee_idx = knee_idx

    # Get knee box (use fallback if needed)
    knee_box = knee_boxes[fallback_knee_idx]
    kx1, ky1, kx2, ky2 = yolo_to_pixel(knee_box, img_w, img_h)

    # Expand to square crop (what crop_knee_regions.py does)
    crop_x1, crop_y1, crop_x2, crop_y2 = expand_to_square(
        kx1, ky1, kx2, ky2, img_w, img_h, margin=0.15
    )

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left: Full image with annotations
    ax = axes[0]
    ax.imshow(img)

    # Title with warning if using fallback
    if knee_not_found:
        title = f"Original Image: {base_name}\n⚠️ Requested Knee #{knee_idx} NOT FOUND - Showing Knee #{fallback_knee_idx} (fallback)"
        title_color = "red"
    else:
        title = f"Original Image: {base_name}\nKnee #{knee_idx}"
        title_color = "black"

    ax.set_title(title, fontsize=12, fontweight="bold", color=title_color)

    # Draw ALL knee boxes
    for i, kb in enumerate(knee_boxes):
        x1, y1, x2, y2 = yolo_to_pixel(kb, img_w, img_h)
        color = "lime" if i == knee_idx else "cyan"
        linewidth = 3 if i == knee_idx else 2
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=linewidth,
            edgecolor=color,
            facecolor="none",
            linestyle="--",
        )
        ax.add_patch(rect)
        ax.text(
            x1,
            y1 - 10,
            f"Knee {i}",
            color=color,
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
        )

    # Draw ALL KL labels
    class_colors = {0: "blue", 1: "green", 2: "yellow", 3: "orange", 4: "red"}
    for kl_box in kl_boxes:
        x1, y1, x2, y2 = yolo_to_pixel(kl_box, img_w, img_h)
        cls = kl_box["class_id"]
        color = class_colors.get(cls, "white")
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            x1,
            y2 + 15,
            f"KL{cls}",
            color=color,
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
        )

    # Draw crop region
    rect = patches.Rectangle(
        (crop_x1, crop_y1),
        crop_x2 - crop_x1,
        crop_y2 - crop_y1,
        linewidth=4,
        edgecolor="magenta",
        facecolor="none",
        linestyle="-",
    )
    ax.add_patch(rect)
    ax.text(
        crop_x1,
        crop_y1 - 25,
        "CROP REGION",
        color="magenta",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.8),
    )

    ax.axis("off")

    # Right: Cropped region zoomed
    ax = axes[1]
    cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]
    ax.imshow(cropped)
    ax.set_title(
        f"Cropped Region\nSize: {crop_x2-crop_x1}x{crop_y2-crop_y1}",
        fontsize=12,
        fontweight="bold",
    )

    # Draw KL labels in crop space
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1

    labels_in_crop = 0
    for kl_box in kl_boxes:
        # Transform to crop space
        x_center = kl_box["x"] * img_w
        y_center = kl_box["y"] * img_h
        w = kl_box["w"] * img_w
        h = kl_box["h"] * img_h

        # Check if in crop
        if (
            crop_x1 - w <= x_center <= crop_x2 + w
            and crop_y1 - h <= y_center <= crop_y2 + h
        ):

            # Transform to crop coordinates
            new_x = x_center - crop_x1
            new_y = y_center - crop_y1

            # Clamp
            box_x1 = max(0, new_x - w / 2)
            box_y1 = max(0, new_y - h / 2)
            box_x2 = min(crop_w, new_x + w / 2)
            box_y2 = min(crop_h, new_y + h / 2)

            if box_x2 > box_x1 and box_y2 > box_y1:
                labels_in_crop += 1
                cls = kl_box["class_id"]
                color = class_colors.get(cls, "white")
                rect = patches.Rectangle(
                    (box_x1, box_y1),
                    box_x2 - box_x1,
                    box_y2 - box_y1,
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none",
                )
                ax.add_patch(rect)
                ax.text(
                    box_x1,
                    box_y2 + 10,
                    f"KL{cls}",
                    color=color,
                    fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
                )

    ax.axis("off")

    # Add annotation
    if labels_in_crop == 0:
        fig.text(
            0.5,
            0.02,
            "⚠️  NO LABELS IN CROP - File would be FILTERED",
            ha="center",
            fontsize=14,
            color="red",
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.8),
        )
    else:
        fig.text(
            0.5,
            0.02,
            f"✅ {labels_in_crop} label(s) in crop - File would be KEPT",
            ha="center",
            fontsize=14,
            color="green",
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
        )

    plt.tight_layout()

    # Save
    output_path = output_dir / f'{image_name.replace(".txt", "")}_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved: {output_path}")
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize knee crops to understand filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize specific files
  python visualize_filtered_crops.py --files file1_knee0.txt file2_knee1.txt
  
  # Visualize all crops from a folder
  python visualize_filtered_crops.py --folder datasets/dataset_knees_cropped/labels
  
  # Visualize filtered files (comparison result)
  python visualize_filtered_crops.py --comparison datasets/dataset_knees_cropped/labels datasets/processed_draf/dataset_knees_cropped/labels
        """,
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--files",
        nargs="+",
        help="Specific label files to visualize (e.g., image_knee0.txt)",
    )
    input_group.add_argument(
        "--folder", type=str, help="Folder containing label files (visualize all)"
    )
    input_group.add_argument(
        "--comparison",
        nargs=2,
        metavar=("NEW_DIR", "OLD_DIR"),
        help="Compare two label directories and visualize filtered files",
    )

    # Dataset and output options
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="datasets/dataset_v0",
        help="Base dataset directory (default: datasets/dataset_v0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset_analysis/filtered_visualizations",
        help="Output directory for visualizations (default: dataset_analysis/filtered_visualizations)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of files to visualize (useful for large folders)",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which files to visualize
    files_to_viz = []

    if args.files:
        # Specific files provided
        files_to_viz = args.files
        print("=" * 80)
        print(f"VISUALIZING {len(files_to_viz)} SPECIFIC FILES")
        print("=" * 80)

    elif args.folder:
        # All files in folder
        folder_path = Path(args.folder)
        if not folder_path.exists():
            print(f"❌ Folder not found: {folder_path}")
            return

        files_to_viz = [f.name for f in folder_path.glob("*.txt")]
        print("=" * 80)
        print(f"VISUALIZING ALL FILES IN FOLDER: {args.folder}")
        print(f"Found {len(files_to_viz)} label files")
        print("=" * 80)

    elif args.comparison:
        # Compare two directories and visualize filtered files
        new_dir = Path(args.comparison[0])
        old_dir = Path(args.comparison[1])

        new_files = set(f.name for f in new_dir.glob("*.txt"))
        old_files = set(f.name for f in old_dir.glob("*.txt"))

        # Files only in old (filtered out)
        filtered_out = sorted(old_files - new_files)

        print("=" * 80)
        print("COMPARISON MODE: Visualizing Filtered Files")
        print("=" * 80)
        print(f"New directory:  {args.comparison[0]} ({len(new_files)} files)")
        print(f"Old directory:  {args.comparison[1]} ({len(old_files)} files)")
        print(f"Filtered out:   {len(filtered_out)} files")
        print("=" * 80)

        if not filtered_out:
            print("✅ No files filtered - nothing to visualize")
            return

        files_to_viz = filtered_out

    # Apply limit if specified
    if args.limit and len(files_to_viz) > args.limit:
        print(f"⚠️ Limiting to first {args.limit} files (out of {len(files_to_viz)})")
        files_to_viz = files_to_viz[: args.limit]

    print()

    # Visualize each file
    success_count = 0
    error_count = 0

    for i, fname in enumerate(files_to_viz, 1):
        print(f"[{i}/{len(files_to_viz)}] Processing: {fname}")
        try:
            visualize_filtered_crop(fname, dataset_dir, output_dir)
            success_count += 1
        except Exception as e:
            print(f"  ❌ Error: {e}")
            error_count += 1
        print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Successfully visualized: {success_count}")
    if error_count > 0:
        print(f"❌ Errors: {error_count}")
    print(f"📁 Output directory: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
