"""
Generate preprocessing comparison visualizations for cropped knees dataset.

Creates detailed visual comparisons showing:
1. Raw vs preprocessed images (multi-sample overview)
2. Detailed single-image comparison with histograms
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.data.preprocessing import (
    load_image,
    get_basic_pipeline,
    get_v0_pipeline,
    get_v3_legacy_pipeline,
    get_notebook_pipeline,
)

# Set matplotlib style
plt.style.use("seaborn-v0_8-darkgrid")


def create_multi_sample_comparison(image_paths, output_path):
    """Create multi-sample comparison across different preprocessing methods."""

    presets = {
        "Raw": (None, "Original"),
        "Basic": (get_basic_pipeline(), "Resize Only"),
        "v0": (get_v0_pipeline(), "Blur + CLAHE 2.0"),
        "v3_legacy": (get_v3_legacy_pipeline(), "Sharp + CLAHE 4.0"),
    }

    n_samples = len(image_paths)
    n_methods = len(presets)

    fig = plt.figure(figsize=(5 * n_methods, 5 * n_samples))
    gs = GridSpec(n_samples, n_methods, figure=fig, hspace=0.3, wspace=0.1)

    for row, img_path in enumerate(image_paths):
        # Load raw image
        raw_img = load_image(img_path, mode="grayscale")

        for col, (preset_name, (pipeline, desc)) in enumerate(presets.items()):
            ax = fig.add_subplot(gs[row, col])

            # Process image
            if pipeline is None:
                processed = raw_img
            else:
                processed, _ = pipeline(raw_img)

            # Display
            ax.imshow(processed, cmap="gray", vmin=0, vmax=255)
            ax.axis("off")

            # Title only on first row
            if row == 0:
                ax.set_title(f"{preset_name}\n{desc}", fontsize=12, fontweight="bold")

            # Sample label on first column
            if col == 0:
                ax.text(
                    -0.1,
                    0.5,
                    f"Sample {row+1}",
                    transform=ax.transAxes,
                    rotation=90,
                    verticalalignment="center",
                    fontsize=11,
                    fontweight="bold",
                )

    plt.suptitle(
        "Preprocessing Comparison - Cropped Knees Dataset",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved multi-sample comparison: {output_path}")


def create_detailed_comparison(image_path, output_path):
    """Create detailed single-image comparison with histograms and statistics."""

    presets = {
        "Raw": (None, "Original"),
        "Basic": (get_basic_pipeline(), "Resize Only"),
        "v0": (get_v0_pipeline(), "Blur + CLAHE 2.0"),
        "v3_legacy": (get_v3_legacy_pipeline(), "Sharp + CLAHE 4.0"),
    }

    # Load raw image
    raw_img = load_image(image_path, mode="grayscale")

    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.3, height_ratios=[2, 1])

    for col, (preset_name, (pipeline, desc)) in enumerate(presets.items()):
        # Process image
        if pipeline is None:
            processed = raw_img
        else:
            processed, _ = pipeline(raw_img)

        # Image display
        ax_img = fig.add_subplot(gs[0, col])
        ax_img.imshow(processed, cmap="gray", vmin=0, vmax=255)
        ax_img.axis("off")
        ax_img.set_title(f"{preset_name}\n{desc}", fontsize=14, fontweight="bold")

        # Statistics
        stats_text = f"Mean: {processed.mean():.1f}\n"
        stats_text += f"Std: {processed.std():.1f}\n"
        stats_text += f"Range: [{processed.min():.0f}, {processed.max():.0f}]"
        ax_img.text(
            0.02,
            0.98,
            stats_text,
            transform=ax_img.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Histogram
        ax_hist = fig.add_subplot(gs[1, col])
        ax_hist.hist(
            processed.ravel(), bins=50, color="steelblue", alpha=0.7, edgecolor="black"
        )
        ax_hist.set_xlabel("Pixel Intensity", fontsize=11)
        ax_hist.set_ylabel("Frequency", fontsize=11)
        ax_hist.set_xlim(0, 255)
        ax_hist.grid(True, alpha=0.3)

    plt.suptitle(
        f"Detailed Preprocessing Comparison - Cropped Knees\n{image_path.name}",
        fontsize=16,
        fontweight="bold",
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved detailed comparison: {output_path}")


def main():
    """Main comparison workflow for cropped knees."""

    # Paths
    input_dir = project_root / "datasets/dataset_knees_cropped/images"
    output_dir = project_root / "datasets/data_examples/knees_cropped"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get sample images
    image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))

    if not image_files:
        print(f"❌ No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images in {input_dir}")

    # Select samples (first 3 for multi-sample)
    sample_paths = image_files[:3]

    print("\n" + "=" * 60)
    print("GENERATING CROPPED KNEES PREPROCESSING COMPARISONS")
    print("=" * 60 + "\n")

    # 1. Multi-sample comparison
    print("Creating multi-sample comparison...")
    multi_output = output_dir / "comparison_raw_vs_processed.png"
    create_multi_sample_comparison(sample_paths, multi_output)

    # 2. Detailed single-image comparison
    print("\nCreating detailed comparison...")
    detailed_output = output_dir / "comparison_detailed.png"
    create_detailed_comparison(sample_paths[0], detailed_output)

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"  - comparison_raw_vs_processed.png")
    print(f"  - comparison_detailed.png")


if __name__ == "__main__":
    main()
