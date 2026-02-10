#!/usr/bin/env python3
"""
Example: Batch inference with CSV output.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api_end_to_end import EndToEndInference, format_csv, format_json


def main():
    # Initialize 5-class model
    print("Initializing 5-class End-to-End model...")
    inference = EndToEndInference(
        checkpoint_path="runs/end_to_end/e2e_5class_balanced/best.pt",
        num_classes=5,
        device="cuda",
    )

    # Get sample images
    image_dir = Path("datasets/balanced/full_xray/images")
    image_paths = sorted(list(image_dir.glob("*.jpg")))[:10]

    print(f"\nProcessing {len(image_paths)} images...")

    # Batch inference
    results = inference.predict_batch(image_paths, show_progress=True)

    # Save results
    output_dir = Path("outputs/batch_example")
    output_dir.mkdir(parents=True, exist_ok=True)

    format_json(results, str(output_dir / "predictions.json"))
    format_csv(results, str(output_dir / "predictions.csv"), include_probabilities=True)

    print(f"\n✅ Results saved to {output_dir}")


if __name__ == "__main__":
    main()
