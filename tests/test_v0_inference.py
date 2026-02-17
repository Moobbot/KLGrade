import os
import cv2
import glob
from pathlib import Path
from api_two_step_yolo.inference.two_step_yolo_api import TwoStepYOLOInference

# Configuration
KNEE_MODEL = "runs/detect/knee_detector/weights/best.pt"
LESION_MODEL = "runs/detect/lesion_8class_balanced/weights/best.pt"
INPUT_DIR = "datasets/dataset_v0/images"
OUTPUT_DIR = "outputs/test_v0"
NUM_IMAGES = 5


def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize pipeline
    print(f"Initializing Two-Step Pipeline...")
    print(f"Step 1: Knee Model: {KNEE_MODEL}")
    print(f"Step 2: Lesion Model: {LESION_MODEL}")

    try:
        pipeline = TwoStepYOLOInference(
            knee_model_path=KNEE_MODEL, lesion_model_path=LESION_MODEL, device="cuda:0"
        )
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # Get images
    image_paths = glob.glob(os.path.join(INPUT_DIR, "*.jpg"))[:NUM_IMAGES]

    if not image_paths:
        print(f"No images found in {INPUT_DIR}")
        return

    print(f"Found {len(image_paths)} images. Processing...")

    for i, img_path in enumerate(image_paths):
        print(f"[{i+1}/{len(image_paths)}] Processing {os.path.basename(img_path)}...")

        try:
            # Run prediction
            result = pipeline.predict(img_path)

            # Print explicit results
            print(f"  - Knees: {len(result['knees'])}")
            print(f"  - Lesions: {len(result['lesions'])}")
            print(f"  - KL Grade: {result['kl_grade']}")

            # Save visualization
            output_name = f"result_{os.path.basename(img_path)}"
            output_path = os.path.join(OUTPUT_DIR, output_name)
            pipeline.visualize(img_path, output_path)

        except Exception as e:
            print(f"  Error processing image: {e}")

    print(f"\nProcessing complete! Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
