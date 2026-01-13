import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.data.preprocessing import balance_dataset_with_flip


def main():
    # Define paths
    dataset_name = "dataset_v0"
    base_dir = project_root / "dataset" / dataset_name
    processed_dir = project_root / "dataset" / f"{dataset_name}_processed"

    input_images = base_dir / "images"
    input_labels = base_dir / "labels"

    output_images = processed_dir / "images"
    output_labels = processed_dir / "labels"

    # Check if inputs exist
    if not input_images.exists() or not input_labels.exists():
        print(f"Error: Input directories not found at {base_dir}")
        return

    print(f"Processing {dataset_name}...")
    print(f"Input: {input_images}")
    print(f"Output: {output_images}")

    # Run preprocessing
    balance_dataset_with_flip(
        img_dir=str(input_images),
        label_dir=str(input_labels),
        output_img_dir=str(output_images),
        output_label_dir=str(output_labels),
        num_classes=5,
        target_size=(640, 640),
    )

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
