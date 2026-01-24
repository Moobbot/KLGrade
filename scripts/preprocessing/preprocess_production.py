"""
Production Dataset Preprocessing Script.

Processes entire dataset with different preprocessing methods and saves to production directory.
"""

import sys
from pathlib import Path
from tqdm import tqdm
import shutil

# Add project root to path
project_root = Path(__file__).resolve().parents[1]  # KLGrade directory
sys.path.append(str(project_root))

from src.data.preprocessing import (
    load_image,
    save_image,
    get_basic_pipeline,
    get_v0_pipeline,
    get_v3_legacy_pipeline,
    # get_notebook_pipeline,
)


def process_dataset(preset_name, pipeline, input_dir, output_base):
    """
    Process entire dataset with given preprocessing pipeline.

    Args:
        preset_name: Name of the preset (for output directory)
        pipeline: Preprocessing pipeline to apply
        input_dir: Input directory with images
        output_base: Base output directory
    """
    print(f"\n{'='*60}")
    print(f"Processing dataset with: {preset_name}")
    print(f"{'='*60}")

    # Setup directories
    output_dir = output_base / preset_name / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all images
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))

    if not image_files:
        print(f"⚠️  No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images to process")

    # Process each image
    for img_path in tqdm(image_files, desc=f"Processing {preset_name}"):
        # Load image
        image = load_image(img_path, mode="grayscale")

        # Apply preprocessing
        if pipeline is not None:
            processed_img, _ = pipeline(image)
        else:
            processed_img = image

        # Save processed image
        output_path = output_dir / img_path.name
        save_image(processed_img, output_path, format="png")

    print(f"✅ Completed: {len(image_files)} images saved to {output_dir}")

    return len(image_files)


def copy_labels(input_base, output_base, preset_name):
    """
    Copy label files to output directory.

    Args:
        input_base: Input base directory
        output_base: Output base directory
        preset_name: Name of the preset
    """
    print(f"\nCopying labels for {preset_name}...")

    output_preset_dir = output_base / preset_name

    # Copy lesion labels
    input_labels = input_base / "labels"
    output_labels = output_preset_dir / "labels"

    if input_labels.exists():
        if output_labels.exists():
            shutil.rmtree(output_labels)
        shutil.copytree(input_labels, output_labels)
        print(
            f"  ✅ Copied lesion labels: {len(list(output_labels.glob('*.txt')))} files"
        )

    # Copy knee labels
    input_knees = input_base / "labels-knee"
    output_knees = output_preset_dir / "labels-knee"

    if input_knees.exists():
        if output_knees.exists():
            shutil.rmtree(output_knees)
        shutil.copytree(input_knees, output_knees)
        print(f"  ✅ Copied knee labels: {len(list(output_knees.glob('*.txt')))} files")


def main():
    """Main production preprocessing workflow."""

    print(f"\n{'#'*60}")
    print("PRODUCTION DATASET PREPROCESSING")
    print(f"{'#'*60}\n")

    # Define paths
    input_base = project_root / "datasets" / "dataset" / "dataset_v0"
    input_images = input_base / "images"
    output_base = project_root / "datasets" / "data_processed"

    # Check input
    if not input_images.exists():
        print(f"❌ Error: Input directory not found: {input_images}")
        return

    print(f"📂 Input: {input_base}")
    print(f"📂 Output: {output_base}")
    print()

    # Define preprocessing presets to run
    presets = {
        "resize_only": ("Basic (Resize only)", get_basic_pipeline()),
        "blur_clahe2": ("Standard (Blur + CLAHE 2.0)", get_v0_pipeline()),
        "sharp_clahe4": (
            "Legacy Sharp (No Blur + CLAHE 4.0)",
            get_v3_legacy_pipeline(),
        ),
        # "blur_clahe2_notebook": (
        #     "Notebook Method (Blur + CLAHE 2.0)",
        #     get_notebook_pipeline(),
        # ),
    }

    # Ask user which presets to run
    print("Available presets:")
    for i, (key, (desc, _)) in enumerate(presets.items(), 1):
        print(f"  {i}. {key}: {desc}")
    print(f"  {len(presets)+1}. all: Run all presets")

    choice = input("\nSelect preset to run (1-5 or 'all'): ").strip().lower()

    if choice == "all" or choice == str(len(presets) + 1):
        selected_presets = list(presets.keys())
    elif choice.isdigit() and 1 <= int(choice) <= len(presets):
        selected_key = list(presets.keys())[int(choice) - 1]
        selected_presets = [selected_key]
    else:
        print("❌ Invalid choice. Exiting.")
        return

    print(f"\n🚀 Running presets: {', '.join(selected_presets)}")

    # Process each selected preset
    total_processed = 0
    for preset_name in selected_presets:
        desc, pipeline = presets[preset_name]

        # Process images
        n_processed = process_dataset(preset_name, pipeline, input_images, output_base)
        total_processed += n_processed

        # Copy labels
        copy_labels(input_base, output_base, preset_name)

    # Summary
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total images processed: {total_processed}")
    print(f"Output directory: {output_base}")
    print(f"\nGenerated datasets:")
    for preset_name in selected_presets:
        dataset_dir = output_base / preset_name
        print(f"  - {dataset_dir}")
    print(f"\n✅ Ready for training!")


if __name__ == "__main__":
    main()
