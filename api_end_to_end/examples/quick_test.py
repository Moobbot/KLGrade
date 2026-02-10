#!/usr/bin/env python3
"""
Example: Quick test with 10-class model on a single image.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api_end_to_end import EndToEndInference


def main():
    # Initialize 10-class model
    print("Initializing 10-class End-to-End model...")
    inference = EndToEndInference(
        checkpoint_path="runs/end_to_end/e2e_10class_balanced/best.pt",
        num_classes=10,
        device="cuda",
    )

    # Run inference on sample image
    image_path = "datasets/balanced/full_xray_10_class/images/9000000.jpg"
    print(f"\nRunning inference on: {image_path}")

    result = inference.predict_single(image_path)

    # Print results
    print(f"\n{'='*60}")
    print("Prediction Results")
    print(f"{'='*60}")
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nTop 3 Predictions:")

    # Sort probabilities
    probs = result["class_probabilities"]
    top3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]

    for class_name, prob in top3:
        print(f"  {class_name}: {prob:.2%}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
