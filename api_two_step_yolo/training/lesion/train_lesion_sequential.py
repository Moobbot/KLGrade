"""
Sequential Lesion Detector Training Script

Trains lesion detectors sequentially:
1. 5-class (KL0-KL4)
2. 10-class (KL0-a to KL4-b)

Usage:
    python train_lesion_sequential.py
"""

import subprocess
import sys
from pathlib import Path
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


def run_training(data_yaml: str, name: str, description: str):
    """Run YOLO training with given configuration."""
    print("\n" + "=" * 60)
    print(f"Training: {description}")
    print("=" * 60)
    print(f"Dataset: {data_yaml}")
    print(f"Config: 100 epochs, batch=16, patience=20")
    print("-" * 60)

    cmd = [
        "python",
        "api_two_step_yolo/training/lesion/train_lesion_detector.py",
        "--data",
        data_yaml,
        "--epochs",
        "100",
        "--batch",
        "16",
        "--device",
        "0",
    ]

    print(f"Command: {' '.join(cmd)}\n")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=False,  # Show output in real-time
        )

        elapsed = time.time() - start_time
        print(f"\n✅ {description} completed in {elapsed/60:.1f} minutes")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with exit code {e.returncode}")
        return False


def main():
    print("\n" + "=" * 60)
    print("Sequential Lesion Detector Training")
    print("=" * 60)

    # Training configurations
    trainings = [
        {
            "data": "datasets/splits/dataset_knees_cropped_70_20_10/dataset.yaml",
            "name": "5-class",
            "description": "5-class Lesion Detector (KL0-KL4)",
        },
        {
            "data": "datasets/splits/dataset_knees_cropped_10_class_60_20_20/dataset.yaml",
            "name": "10-class",
            "description": "10-class Lesion Detector (KL0-a to KL4-b)",
        },
    ]

    results = {}

    # Run each training sequentially
    for i, config in enumerate(trainings, 1):
        print(f"\n{'='*60}")
        print(f"Step {i}/{len(trainings)}")
        print(f"{'='*60}")

        success = run_training(
            data_yaml=config["data"],
            name=config["name"],
            description=config["description"],
        )

        results[config["name"]] = success

        if not success:
            print(f"\n⚠️  Training failed at step {i}. Stopping.")
            break

    # Summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)

    for name, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"{name:15s}: {status}")

    print("\nResults saved to:")
    print("  5-class:  runs/detect/train_lesion_detector/weights/best.pt")
    print("  10-class: runs/detect/train_lesion_detector2/weights/best.pt")
    print("=" * 60)

    # Exit with appropriate code
    if all(results.values()):
        print("\n🎉 All training completed successfully!")
        sys.exit(0)
    else:
        print("\n⚠️  Some training failed. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
