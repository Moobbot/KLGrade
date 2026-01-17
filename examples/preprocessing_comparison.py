"""
Preprocessing Comparison - Before/After Visualization.

Compares raw data with different preprocessing methods side-by-side.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[1]  # KLGrade directory
sys.path.append(str(project_root))

from src.data.preprocessing import (
    load_image,
    get_basic_pipeline,
    get_v0_pipeline,
    get_v3_legacy_pipeline,
    get_notebook_pipeline,
)


def create_comparison_visualization(num_samples=3):
    """
    Create before/after comparison visualization.

    Shows raw image vs. all preprocessing methods side-by-side.
    """
    print("Creating preprocessing comparison visualization...")

    # Load sample images
    image_path = project_root / "datasets" / "dataset_v0" / "images"
    image_files = list(image_path.glob("*.jpg")) + list(image_path.glob("*.png"))

    if not image_files:
        print(f"No .jpg files found in {image_path}")
        return

    # Take first N samples
    samples = image_files[:num_samples]

    # Define preprocessing methods with technical names
    presets = {
        "raw": (None, "Original"),
        "resize_only": (get_basic_pipeline(), "Resize Only"),
        "blur_clahe2": (get_v0_pipeline(), "Blur + CLAHE 2.0"),
        "noBlur_clahe4": (get_v3_legacy_pipeline(), "No Blur + CLAHE 4.0"),
        "blur_clahe2_notebook": (get_notebook_pipeline(), "Notebook Method"),
    }

    # Create figure
    n_methods = len(presets)
    n_samples = len(samples)

    fig, axes = plt.subplots(n_samples, n_methods, figsize=(20, 4 * n_samples))

    # Ensure axes is 2D even for single sample
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    # Process each sample
    for i, sample_img_path in enumerate(samples):
        print(f"Processing sample {i+1}/{n_samples}: {sample_img_path.name}")

        # Load raw image
        raw_image = load_image(sample_img_path, mode="grayscale")

        # Process with each method
        for col, (preset_name, (pipeline, desc)) in enumerate(presets.items()):
            ax = axes[i, col]

            if pipeline is None:
                # Show raw image
                processed_img = raw_image
            else:
                # Apply preprocessing
                processed_img, _ = pipeline(raw_image)

            # Display image
            ax.imshow(processed_img, cmap="gray")
            ax.axis("off")

            # Add title only on first row
            if i == 0:
                ax.set_title(
                    f"{preset_name}\n{desc}", fontsize=12, fontweight="bold", pad=10
                )

            # Add sample number on left
            if col == 0:
                ax.text(
                    -0.1,
                    0.5,
                    f"Sample {i+1}",
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=11,
                    fontweight="bold",
                )

    plt.tight_layout()

    # Save comparison
    output_path = (
        project_root
        / "datasets"
        / "data_examples"
        / "comparison"
        / "comparison_raw_vs_processed.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved comparison to: {output_path}")

    plt.close()


def create_detailed_comparison_single_image():
    """
    Create detailed comparison with statistics for a single image.
    """
    print("\nCreating detailed single-image comparison...")

    # Load one sample
    image_path = project_root / "datasets" / "dataset_v0" / "images"
    image_files = list(image_path.glob("*.jpg")) + list(image_path.glob("*.png"))

    if not image_files:
        print(f"No .jpg files found in {image_path}")
        return

    sample_img_path = image_files[0]
    raw_image = load_image(sample_img_path, mode="grayscale")

    # Define preprocessing methods with technical names
    presets = {
        "raw": (None, "Original"),
        "resize_only": (get_basic_pipeline(), "Resize Only"),
        "blur_clahe2": (get_v0_pipeline(), "Blur + CLAHE 2.0"),
        "noBlur_clahe4": (get_v3_legacy_pipeline(), "No Blur + CLAHE 4.0"),
    }

    # Create figure with 2 rows: images + histograms
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, len(presets), hspace=0.3, wspace=0.2)

    for col, (preset_name, (pipeline, desc)) in enumerate(presets.items()):
        # Process image
        if pipeline is None:
            processed_img = raw_image
        else:
            processed_img, _ = pipeline(raw_image)

        # Image
        ax_img = fig.add_subplot(gs[0, col])
        ax_img.imshow(processed_img, cmap="gray")
        ax_img.axis("off")
        ax_img.set_title(f"{preset_name}\n{desc}", fontsize=14, fontweight="bold")

        # Add statistics
        mean_val = np.mean(processed_img)
        std_val = np.std(processed_img)
        min_val = np.min(processed_img)
        max_val = np.max(processed_img)

        stats_text = f"Mean: {mean_val:.1f}\nStd: {std_val:.1f}\nRange: [{min_val:.0f}, {max_val:.0f}]"
        ax_img.text(
            0.02,
            0.98,
            stats_text,
            transform=ax_img.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=9,
            family="monospace",
        )

        # Histogram
        ax_hist = fig.add_subplot(gs[1, col])
        ax_hist.hist(
            processed_img.ravel(), bins=50, color="blue", alpha=0.7, edgecolor="black"
        )
        ax_hist.set_xlabel("Pixel Intensity", fontsize=10)
        ax_hist.set_ylabel("Frequency", fontsize=10)
        ax_hist.set_title("Histogram", fontsize=11)
        ax_hist.grid(True, alpha=0.3)

    # Main title
    fig.suptitle(
        "Detailed Preprocessing Comparison\n(Image + Histogram + Statistics)",
        fontsize=16,
        fontweight="bold",
    )

    # Save
    output_path = (
        project_root
        / "datasets"
        / "data_examples"
        / "comparison"
        / "comparison_detailed.png"
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved detailed comparison to: {output_path}")

    plt.close()


if __name__ == "__main__":
    # Create output directory
    output_dir = project_root / "datasets" / "data_examples"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PREPROCESSING COMPARISON VISUALIZATION")
    print("=" * 60)

    # Create multi-sample comparison
    create_comparison_visualization(num_samples=3)

    # Create detailed single-image comparison
    create_detailed_comparison_single_image()

    print("\n" + "=" * 60)
    print("✅ All comparisons generated successfully!")
    print("=" * 60)
