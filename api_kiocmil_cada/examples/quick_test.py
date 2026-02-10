"""
Quick test script for KIOCMIL-CADA inference.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from api_kiocmil_cada.inference import KiocmilInference


def main():
    # Configuration
    # NOTE: You need to set valid paths to checkpoints here
    config = {
        "kiocmil_model_path": "checkpoints/kiocmil_cada_10class_balanced.pt",
        "knee_model_path": "checkpoints/knee_detector.pt",
        "lesion_model_path": "checkpoints/lesion_detector.pt",
        "num_classes": 10,
        "image_path": "examples/test_image.jpg",
    }

    # check if models exist
    for key in ["kiocmil_model_path", "knee_model_path", "lesion_model_path"]:
        if not os.path.exists(config[key]):
            print(f"⚠️  Warning: Checkpoint not found at {config[key]}")
            print(f"Please update the path in this script or download the model.")
            # We continue to show usage, but it will fail later if not fixed

    print("Initializing Inference Engine...")
    try:
        engine = KiocmilInference(
            kiocmil_model_path=config["kiocmil_model_path"],
            knee_model_path=config["knee_model_path"],
            lesion_model_path=config["lesion_model_path"],
            num_classes=config["num_classes"],
        )

        # Run inference
        if os.path.exists(config["image_path"]):
            import cv2

            image = cv2.imread(config["image_path"])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            print(f"Processing {config['image_path']}...")
            results = engine.predict(image)

            # Print results
            print("\nResults:")
            for i, knee in enumerate(results):
                print(f"Knee {i+1}:")
                print(f"  Class: {knee['predicted_class']}")
                print(f"  Confidence: {knee['confidence']:.4f}")
                print(f"  JS Lesions: {knee['num_js_lesions']}")
                print(f"  Ost Lesions: {knee['num_ost_lesions']}")
        else:
            print(f"\n⚠️  Test image not found at {config['image_path']}")
            print("Please place a test image there to run prediction.")

    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
