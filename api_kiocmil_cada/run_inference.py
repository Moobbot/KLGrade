#!/usr/bin/env python3
"""
KIOCMIL-CADA Inference CLI

Command-line interface for running inference with the 2-step pipeline:
1. ROI Detection (Knee + Lesion) using YOLO
2. Classification using KIOCMIL-CADA
"""

import argparse
import sys
import json
import cv2
import torch
import glob
import os
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from api_kiocmil_cada.inference import KiocmilInference
from api_kiocmil_cada.output_formatters import (
    save_json,
    save_csv,
    save_visualization,
    save_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(description="KIOCMIL-CADA Inference API")

    # Input/Output
    parser.add_argument("--image", type=str, help="Path to single input image")
    parser.add_argument("--image-dir", type=str, help="Path to directory of images")
    parser.add_argument(
        "--output-dir", type=str, default="outputs", help="Directory to save results"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to model configuration JSON"
    )

    # Output formats
    parser.add_argument("--json", action="store_true", help="Save results to JSON")
    parser.add_argument("--csv", action="store_true", help="Save results to CSV")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualized images with bounding boxes",
    )
    parser.add_argument("--summary", action="store_true", help="Save summary report")

    # Overrides (optional)
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--knee-conf",
        type=float,
        default=None,
        help="Knee detection confidence threshold",
    )
    parser.add_argument(
        "--lesion-conf",
        type=float,
        default=None,
        help="Lesion detection confidence threshold",
    )

    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return json.load(f)


def main():
    args = parse_args()

    # Validation
    if not args.image and not args.image_dir:
        print("Error: Must specify either --image or --image-dir")
        sys.exit(1)

    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)

    # Load configuration
    print(f"Loading configuration from {args.config}...")
    config = load_config(args.config)

    # Override config with args if provided
    device = args.device if args.device else config.get("device", "cuda")
    knee_conf = (
        args.knee_conf
        if args.knee_conf
        else config["inference_params"].get("knee_conf_threshold", 0.5)
    )
    lesion_conf = (
        args.lesion_conf
        if args.lesion_conf
        else config["inference_params"].get("lesion_conf_threshold", 0.5)
    )

    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Model
    try:
        inference_engine = KiocmilInference(
            kiocmil_model_path=config["checkpoints"]["kiocmil"],
            knee_model_path=config["checkpoints"]["knee_detector"],
            lesion_model_path=config["checkpoints"]["lesion_detector"],
            num_classes=config["num_classes"],
            ctx_size=tuple(config["inference_params"]["ctx_size"]),
            patch_size=tuple(config["inference_params"]["patch_size"]),
            device=device,
        )
    except Exception as e:
        print(f"Error initializing model: {e}")
        print("Please check your checkpoint paths in the config file.")
        sys.exit(1)

    # Collect images
    image_paths = []
    if args.image:
        image_paths.append(Path(args.image))
    elif args.image_dir:
        # Support common image extensions
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))
            image_paths.extend(glob.glob(os.path.join(args.image_dir, ext.upper())))

    print(f"Found {len(image_paths)} images to process.")

    results = []

    # Processing Loop
    for img_path in tqdm(image_paths, desc="Running Inference"):
        img_path_str = str(img_path)

        # Read image
        image = cv2.imread(img_path_str)
        if image is None:
            print(f"Warning: Could not read image {img_path_str}")
            continue

        # Convert BGR to RGB for inference
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run inference
        try:
            predictions = inference_engine.predict(
                image_rgb, knee_conf=knee_conf, lesion_conf=lesion_conf
            )

            # Add metadata
            result_entry = {
                "image_path": str(img_path.absolute()),
                "filename": img_path.name,
                "knees": predictions,
            }
            results.append(result_entry)

            # Visualization (per image)
            if args.visualize:
                viz_path = output_dir / "viz" / f"viz_{img_path.name}"
                save_visualization(img_path_str, predictions, str(viz_path))

        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            import traceback

            traceback.print_exc()

    # Save aggregated results
    if args.json:
        save_json(results, str(output_dir / "results.json"))

    if args.csv:
        save_csv(results, str(output_dir / "results.csv"))

    if args.summary:
        save_summary(results, str(output_dir / "summary.txt"))

    print(f"\nProcessing complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
