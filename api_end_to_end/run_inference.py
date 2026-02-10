#!/usr/bin/env python3
"""
End-to-End KIOCMIL Inference CLI

Standalone command-line interface for running KL grade classification.

Usage:
    # Single image
    python api_end_to_end/run_inference.py \\
        --config api_end_to_end/configs/config_10class.json \\
        --image path/to/xray.jpg \\
        --output results.json

    # Batch inference
    python api_end_to_end/run_inference.py \\
        --config api_end_to_end/configs/config_5class.json \\
        --image-dir datasets/test/ \\
        --output-dir outputs/predictions/

    # With visualization
    python api_end_to_end/run_inference.py \\
        --config api_end_to_end/configs/config_8class.json \\
        --image path/to/xray.jpg \\
        --visualize \\
        --output-dir outputs/viz/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api_end_to_end.inference import EndToEndInference
from api_end_to_end.output_formatters import (
    format_json,
    format_csv,
    format_visualization,
    save_summary,
)


def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def validate_config(config: Dict) -> None:
    """Validate configuration has required fields."""
    required_fields = ["checkpoint_path", "num_classes"]
    missing = [f for f in required_fields if f not in config]

    if missing:
        raise ValueError(f"Missing required config fields: {missing}")

    if config["num_classes"] not in [4, 5, 8, 10]:
        raise ValueError(
            f"num_classes must be 4, 5, 8, or 10, got {config['num_classes']}"
        )


def get_image_paths(
    image_path: str = None, image_dir: str = None, max_images: int = None
) -> List[str]:
    """Get list of image paths from single image or directory."""
    if image_path:
        return [image_path]

    if image_dir:
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        image_paths = []
        for ext in extensions:
            image_paths.extend(image_dir.glob(ext))

        image_paths = sorted([str(p) for p in image_paths])

        if max_images:
            image_paths = image_paths[:max_images]

        if not image_paths:
            raise ValueError(f"No images found in {image_dir}")

        return image_paths

    raise ValueError("Must provide either --image or --image-dir")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-End KIOCMIL Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON configuration file",
    )

    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str, help="Path to single input image")
    input_group.add_argument(
        "--image-dir", type=str, help="Directory containing input images"
    )

    # Output arguments
    parser.add_argument(
        "--output", type=str, help="Output file path (for single image)"
    )
    parser.add_argument(
        "--output-dir", type=str, help="Output directory (for batch inference)"
    )

    # Optional arguments
    parser.add_argument(
        "--max-images", type=int, help="Maximum number of images to process"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Create visualizations"
    )
    parser.add_argument("--csv", action="store_true", help="Save results as CSV")
    parser.add_argument(
        "--no-json", action="store_true", help="Do not save JSON results"
    )
    parser.add_argument(
        "--device", type=str, choices=["cuda", "cpu"], help="Override device"
    )

    args = parser.parse_args()

    # Load and validate config
    print(f"Loading configuration from {args.config}...")
    config = load_config(args.config)
    validate_config(config)

    if args.device:
        config["device"] = args.device

    # Print config info
    print(f"\n{'='*60}")
    print(f"Model: {config.get('model_type', 'Unknown')}")
    print(f"Description: {config.get('description', 'N/A')}")
    print(f"Checkpoint: {config['checkpoint_path']}")
    print(f"Classes: {config['num_classes']}")
    print(f"Device: {config.get('device', 'cuda')}")
    print(f"{'='*60}\n")

    # Initialize inference
    inference = EndToEndInference(
        checkpoint_path=config["checkpoint_path"],
        num_classes=config["num_classes"],
        device=config.get("device", "cuda"),
        image_size=config.get("image_size", 640),
        confidence_threshold=config.get("confidence_threshold", 0.0),
    )

    # Get image paths
    image_paths = get_image_paths(args.image, args.image_dir, args.max_images)
    print(f"\nProcessing {len(image_paths)} image(s)...\n")

    # Run inference
    if len(image_paths) == 1:
        predictions = [inference.predict_single(image_paths[0])]
    else:
        predictions = inference.predict_batch(image_paths, show_progress=True)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.output:
        output_dir = Path(args.output).parent
    else:
        output_dir = Path(config.get("output_dir", "outputs/inference"))

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    print(f"\n{'='*60}")
    print("Saving results...")
    print(f"{'='*60}\n")

    if not args.no_json:
        json_path = args.output if args.output else output_dir / "predictions.json"
        format_json(predictions, str(json_path))

    if args.csv:
        format_csv(
            predictions, str(output_dir / "predictions.csv"), include_probabilities=True
        )

    if args.visualize:
        format_visualization(predictions, str(output_dir / "visualizations"))

    save_summary(predictions, str(output_dir / "summary.txt"))

    # Print summary
    print(f"\n{'='*60}")
    print("Inference completed!")
    print(f"{'='*60}")

    successful = sum(1 for p in predictions if "error" not in p)
    failed = len(predictions) - successful
    print(f"Total: {len(predictions)} | Successful: {successful} | Failed: {failed}")

    if successful > 0:
        print(f"\nSample predictions:")
        for i, pred in enumerate(predictions[:3]):
            if "error" in pred:
                continue
            img_name = Path(pred["image_path"]).name
            print(f"  {img_name}: {pred['predicted_class']} ({pred['confidence']:.2%})")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
