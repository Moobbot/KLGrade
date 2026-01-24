#!/usr/bin/env python3
"""
Unified preprocessing script for all dataset variants.

Supports:
- Full X-rays (5-class + 10-class)
- Full X-rays 4-class (4-class + 8-class)
- Cropped knees (5-class + 10-class)
- Cropped knees 4-class (4-class + 8-class)

With multiple preprocessing variants:
- resize_only
- blur_clahe2
- sharp_clahe4
# - blur_clahe2_notebook
"""

import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import shutil
import json
import time

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.preprocessing import (
    load_image,
    save_image,
    get_basic_pipeline,
    get_v0_pipeline,
    get_v3_legacy_pipeline,
    # get_notebook_pipeline,
)


def preprocess_dataset(
    input_dir: Path,
    output_dir: Path,
    label_dirs: list,
    image_source_dir: Path = None,
    filter_by_labels: bool = False,
):
    """
    Preprocess a dataset with all preprocessing variants.

    Args:
        input_dir: Input dataset directory
        output_dir: Output base directory
        label_dirs: List of label directory names to copy
        image_source_dir: Source of images (if different from input_dir)
        filter_by_labels: If True, only process images that have labels
    """

    print("=" * 60)
    print(f"PREPROCESSING: {input_dir.name}")
    print("=" * 60)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print()

    # Get images
    if image_source_dir:
        image_dir = image_source_dir / "images"
    else:
        image_dir = input_dir / "images"

    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

    # Filter by labels if needed
    if filter_by_labels:
        label_files = set(f.stem for f in (input_dir / "labels").glob("*.txt"))
        image_files = [f for f in image_files if f.stem in label_files]

    print(f"Found {len(image_files)} images")

    # Define preprocessing variants
    presets = {
        "resize_only": ("Resize only", get_basic_pipeline()),
        "blur_clahe2": ("Blur + CLAHE 2.0", get_v0_pipeline()),
        "sharp_clahe4": ("No Blur + CLAHE 4.0", get_v3_legacy_pipeline()),
        # "blur_clahe2_notebook": (
        #     "Notebook (Blur + CLAHE 2.0)",
        #     get_notebook_pipeline(),
        # ),
    }

    # Track processing log
    processing_log = {
        "dataset": input_dir.name,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "total_images": len(image_files),
        "filter_by_labels": filter_by_labels,
        "variants": {},
    }

    # Process each preprocessing variant
    for preset_name, (desc, pipeline) in presets.items():
        print(f"\nProcessing {preset_name}: {desc}")
        start_time = time.time()

        # Create output image directory
        output_img_dir = output_dir / preset_name / "images"
        output_img_dir.mkdir(parents=True, exist_ok=True)

        # Process images
        for img_path in tqdm(image_files, desc=preset_name):
            image = load_image(img_path, mode="grayscale")
            processed_img, _ = pipeline(image) if pipeline else (image, None)
            output_path = output_img_dir / img_path.name
            save_image(processed_img, output_path, format="png")

        # Track label copying results
        variant_log = {
            "description": desc,
            "images_processed": len(image_files),
            "labels_copied": {},
            "labels_skipped": [],
            "processing_time_seconds": 0,
        }

        # Copy label directories
        for label_dir_name in label_dirs:
            input_labels = input_dir / label_dir_name
            if input_labels.exists():
                output_labels = output_dir / preset_name / label_dir_name
                if output_labels.exists():
                    shutil.rmtree(output_labels)
                shutil.copytree(input_labels, output_labels)
                label_count = len(list(output_labels.glob("*.txt")))
                print(f"  ✅ Copied {label_dir_name}: {label_count} files")
                variant_log["labels_copied"][label_dir_name] = label_count
            else:
                print(f"  ⚠️  Skipped {label_dir_name} (not found)")
                variant_log["labels_skipped"].append(label_dir_name)

        variant_log["processing_time_seconds"] = round(time.time() - start_time, 2)
        processing_log["variants"][preset_name] = variant_log

    # Save processing log
    log_file = output_dir / "preprocessing_log.json"
    with open(log_file, "w") as f:
        json.dump(processing_log, f, indent=2)

    print()
    print("=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"Output: {output_dir}")
    print(f"📄 Log saved: {log_file}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess dataset with multiple variants"
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "full_xrays",
            "full_xrays_4_class",
            "knees_cropped",
            "knees_cropped_4_class",
        ],
        help="Dataset to preprocess",
    )

    args = parser.parse_args()

    # Configuration for each dataset
    configs = {
        "full_xrays": {
            "input_dir": project_root / "datasets/dataset_v0",
            "output_dir": project_root / "datasets/processed/full_xray",
            "label_dirs": ["labels", "labels_10_class", "labels-knee"],
            "image_source_dir": None,
            "filter_by_labels": False,
        },
        "full_xrays_4_class": {
            "input_dir": project_root / "datasets/dataset_v0_4_class",
            "output_dir": project_root / "datasets/processed/full_xray_4_class",
            "label_dirs": ["labels", "labels_8_class", "labels-knee"],
            "image_source_dir": project_root / "datasets/dataset_v0",
            "filter_by_labels": True,
        },
        "knees_cropped": {
            "input_dir": project_root / "datasets/dataset_knees_cropped",
            "output_dir": project_root / "datasets/processed/knees_cropped",
            "label_dirs": ["labels", "labels_10_class", "labels-knee"],
            "image_source_dir": None,
            "filter_by_labels": False,
        },
        "knees_cropped_4_class": {
            "input_dir": project_root / "datasets/dataset_knees_cropped_4_class",
            "output_dir": project_root / "datasets/processed/knees_cropped_4_class",
            "label_dirs": ["labels", "labels_8_class", "labels-knee"],
            "image_source_dir": project_root / "datasets/dataset_knees_cropped",
            "filter_by_labels": True,
        },
    }

    config = configs[args.dataset]
    preprocess_dataset(**config)


if __name__ == "__main__":
    main()
