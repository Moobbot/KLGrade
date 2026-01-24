"""
Custom Preprocessing Pipeline Examples.

Demonstrates how to use the new modular preprocessing architecture.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]  # KLGrade directory
sys.path.append(str(project_root))

from src.data.preprocessing import (
    PreprocessingPipeline,
    get_basic_pipeline,
    get_v0_pipeline,
    get_v3_legacy_pipeline,
    # get_notebook_pipeline,
    get_custom_pipeline,
    load_image,
    save_image,
)
from src.data.preprocessing.core import (
    resize_image,
    gaussian_blur,
    median_blur,
    apply_clahe,
    adjust_brightness,
    horizontal_flip,
)


def example_1_basic_preprocessing():
    """Example: Basic preprocessing (resize only)"""
    print("\n=== Example 1: Basic Preprocessing ===")

    # Load image
    image_path = project_root / "datasets" / "dataset_v0" / "images"
    image_files = list(image_path.glob("*.jpg")) + list(image_path.glob("*.png"))
    if not image_files:
        print(f"No image files found in {image_path}")
        return
    sample_img = image_files[0]
    image = load_image(sample_img, mode="grayscale")

    # Apply basic preprocessing
    pipeline = get_basic_pipeline(target_size=(640, 640))
    processed_img, _ = pipeline(image)

    # Save result
    output_path = (
        project_root / "datasets" / "data_examples" / "comparison" / "basic_example.png"
    )
    save_image(processed_img, output_path)
    print(f"Saved to: {output_path}")


def example_2_v0_preprocessing():
    """Example: v0 preprocessing (Blur + CLAHE 2.0)"""
    print("\n=== Example 2: v0 Preprocessing ===")

    # Load image
    image_path = project_root / "datasets" / "dataset_v0" / "images"
    image_files = list(image_path.glob("*.jpg")) + list(image_path.glob("*.png"))
    if not image_files:
        print(f"No image files found in {image_path}")
        return
    sample_img = image_files[0]
    image = load_image(sample_img, mode="grayscale")

    # Apply v0 preprocessing
    pipeline = get_v0_pipeline(target_size=(640, 640))
    processed_img, _ = pipeline(image)

    # Save result
    output_path = (
        project_root / "datasets" / "data_examples" / "comparison" / "v0_example.png"
    )
    save_image(processed_img, output_path)
    print(f"Saved to: {output_path}")


def example_3_custom_pipeline():
    """Example: Custom pipeline with specific parameters"""
    print("\n=== Example 3: Custom Pipeline ===")

    # Load image
    image_path = project_root / "datasets" / "dataset_v0" / "images"
    sample_img = (list(image_path.glob("*.jpg")) + list(image_path.glob("*.png")))[0]
    image = load_image(sample_img, mode="grayscale")

    # Create custom pipeline: No blur, high CLAHE
    pipeline = get_custom_pipeline(
        use_blur=False, clahe_clip=4.0, target_size=(512, 512)
    )
    processed_img, _ = pipeline(image)

    # Save result
    output_path = (
        project_root
        / "datasets"
        / "data_examples"
        / "comparison"
        / "custom_example.png"
    )
    save_image(processed_img, output_path)
    print(f"Saved to: {output_path}")


def example_4_fully_custom():
    """Example: Fully custom pipeline composition"""
    print("\n=== Example 4: Fully Custom Composition ===")

    # Load image
    image_path = project_root / "datasets" / "dataset_v0" / "images"
    sample_img = (list(image_path.glob("*.jpg")) + list(image_path.glob("*.png")))[0]
    image = load_image(sample_img, mode="grayscale")

    # Build custom pipeline step by step
    pipeline = PreprocessingPipeline(
        [
            lambda img: resize_image(img, (640, 640)),
            lambda img: median_blur(
                img, kernel_size=5
            ),  # Use median instead of Gaussian
            lambda img: apply_clahe(img, clip_limit=3.0),  # Medium CLAHE
            lambda img: adjust_brightness(img, factor=1.1),  # Slightly brighter
        ]
    )

    processed_img, _ = pipeline(image)

    # Save result
    output_path = (
        project_root
        / "datasets"
        / "data_examples"
        / "comparison"
        / "fully_custom_example.png"
    )
    save_image(processed_img, output_path)
    print(f"Saved to: {output_path}")


def example_5_with_augmentation():
    """Example: Pipeline with augmentation"""
    print("\n=== Example 5: With Augmentation ===")

    # Load image
    image_path = project_root / "datasets" / "dataset_v0" / "images"
    sample_img = (list(image_path.glob("*.jpg")) + list(image_path.glob("*.png")))[0]
    image = load_image(sample_img, mode="grayscale")

    # Pipeline with horizontal flip
    pipeline = PreprocessingPipeline(
        [
            lambda img: resize_image(img, (640, 640)),
            lambda img: gaussian_blur(img, kernel_size=(5, 5)),
            lambda img: apply_clahe(img, clip_limit=2.0),
            lambda img, lbl: horizontal_flip(img, lbl),  # Flip augmentation
        ]
    )

    # Note: Pass labels if you have them
    processed_img, _ = pipeline(image, labels=None)

    # Save result
    output_path = (
        project_root
        / "datasets"
        / "data_examples"
        / "comparison"
        / "augmented_example.png"
    )
    save_image(processed_img, output_path)
    print(f"Saved to: {output_path}")


def example_6_compare_presets():
    """Example: Compare different presets side by side"""
    print("\n=== Example 6: Compare All Presets ===")

    # Load image
    image_path = project_root / "datasets" / "dataset_v0" / "images"
    sample_img = (list(image_path.glob("*.jpg")) + list(image_path.glob("*.png")))[0]
    image = load_image(sample_img, mode="grayscale")

    presets = {
        "resize_only": get_basic_pipeline(),
        "blur_clahe2": get_v0_pipeline(),
        "noBlur_clahe4": get_v3_legacy_pipeline(),
        # "blur_clahe2_notebook": get_notebook_pipeline(),
    }

    output_dir = project_root / "datasets" / "data_examples" / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, pipeline in presets.items():
        processed_img, _ = pipeline(image)
        output_path = output_dir / f"{name}.png"
        save_image(processed_img, output_path)
        print(f"Saved {name} to: {output_path}")


if __name__ == "__main__":
    # Create output directory
    output_dir = project_root / "datasets" / "data_examples" / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run all examples
    print("Running preprocessing examples...")

    try:
        example_1_basic_preprocessing()
        example_2_v0_preprocessing()
        example_3_custom_pipeline()
        example_4_fully_custom()
        example_5_with_augmentation()
        example_6_compare_presets()

        print("\n✅ All examples completed successfully!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
