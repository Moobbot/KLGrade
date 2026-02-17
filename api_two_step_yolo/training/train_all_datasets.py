"""
Automated Training Script for All Dataset Variants

Trains YOLO11l lesion detectors sequentially for all dataset configurations:
- Cropped knee datasets (base + balanced): 4/5/8/10-class
- Full X-ray datasets (base + balanced): 4/8/10-class
- Knee detection (single class)

Total: 15 dataset variants

Usage:
    # Train all datasets
    python train_all_datasets.py

    # Train specific category
    python train_all_datasets.py --category cropped
    python train_all_datasets.py --category full_xray
    python train_all_datasets.py --category detection
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime


# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# Training configurations for all datasets
TRAINING_CONFIGS = {
    "cropped_base": [
        {
            "name": "5-class Cropped (Base)",
            "data": "datasets/splits/dataset_knees_cropped_70_20_10/dataset.yaml",
            "run_name": "lesion_5class_base",
        },
        {
            "name": "4-class Cropped (Base)",
            "data": "datasets/splits/dataset_knees_cropped_4_class_70_20_10/dataset.yaml",
            "run_name": "lesion_4class_base",
        },
        {
            "name": "8-class Cropped (Base)",
            "data": "datasets/splits/dataset_knees_cropped_8_class_70_20_10/dataset.yaml",
            "run_name": "lesion_8class_base",
        },
        {
            "name": "10-class Cropped (Base)",
            "data": "datasets/splits/dataset_knees_cropped_10_class_60_20_20/dataset.yaml",
            "run_name": "lesion_10class_base",
        },
    ],
    "cropped_balanced": [
        {
            "name": "5-class Cropped (Balanced)",
            "data": "datasets/splits/balanced_knees_cropped_70_20_10/dataset.yaml",
            "run_name": "lesion_5class_balanced",
        },
        {
            "name": "4-class Cropped (Balanced)",
            "data": "datasets/splits/balanced_knees_cropped_4_class_70_20_10/dataset.yaml",
            "run_name": "lesion_4class_balanced",
        },
        {
            "name": "8-class Cropped (Balanced)",
            "data": "datasets/splits/balanced_knees_cropped_8_class_70_20_10/dataset.yaml",
            "run_name": "lesion_8class_balanced",
        },
        {
            "name": "10-class Cropped (Balanced)",
            "data": "datasets/splits/balanced_knees_cropped_10_class_60_20_20/dataset.yaml",
            "run_name": "lesion_10class_balanced",
        },
    ],
    "full_xray_base": [
        {
            "name": "4-class Full X-ray (Base)",
            "data": "datasets/splits/knee_full_4_class_70_20_10/dataset.yaml",
            "run_name": "lesion_full_4class_base",
        },
        {
            "name": "8-class Full X-ray (Base)",
            "data": "datasets/splits/knee_full_8_class_70_20_10/dataset.yaml",
            "run_name": "lesion_full_8class_base",
        },
        {
            "name": "10-class Full X-ray (Base)",
            "data": "datasets/splits/knee_full_10_class_70_20_10/dataset.yaml",
            "run_name": "lesion_full_10class_base",
        },
    ],
    "full_xray_balanced": [
        {
            "name": "4-class Full X-ray (Balanced)",
            "data": "datasets/splits/balanced_full_xray_4_class_70_20_10/dataset.yaml",
            "run_name": "lesion_full_4class_balanced",
        },
        {
            "name": "8-class Full X-ray (Balanced)",
            "data": "datasets/splits/balanced_full_xray_8_class_70_20_10/dataset.yaml",
            "run_name": "lesion_full_8class_balanced",
        },
        {
            "name": "10-class Full X-ray (Balanced)",
            "data": "datasets/splits/balanced_full_xray_10_class_60_20_20/dataset.yaml",
            "run_name": "lesion_full_10class_balanced",
        },
    ],
    "detection": [
        {
            "name": "Knee Detection",
            "data": "datasets/splits/knee/dataset.yaml",
            "run_name": "knee_detector",
        },
    ],
}


def run_training(config: dict, epochs: int = 100, batch: int = 16, device: str = "0"):
    """Run YOLO training for a single dataset configuration."""
    print("\n" + "=" * 80)
    print(f"Training: {config['name']}")
    print("=" * 80)
    print(f"Dataset: {config['data']}")
    print(f"Run name: {config['run_name']}")
    print(f"Config: {epochs} epochs, batch={batch}, patience=20")
    print("-" * 80)

    # Check if dataset.yaml exists
    data_path = PROJECT_ROOT / config["data"]
    if not data_path.exists():
        print(f"⚠️  Dataset not found: {data_path}")
        print(f"⚠️  Skipping {config['name']}")
        return False

    cmd = [
        "python",
        "api_two_step_yolo/training/lesion/train_lesion_detector.py",
        "--data",
        config["data"],
        "--epochs",
        str(epochs),
        "--batch",
        str(batch),
        "--device",
        device,
        "--name",
        config["run_name"],
    ]

    print(f"Command: {' '.join(cmd)}\n")

    start_time = time.time()

    try:
        # Run training
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=False,  # Show output in real-time
        )

        elapsed = time.time() - start_time
        print(f"\n✅ {config['name']} completed in {elapsed/60:.1f} minutes")
        print(f"   Model saved: runs/detect/{config['run_name']}/weights/best.pt")
        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ {config['name']} failed after {elapsed/60:.1f} minutes")
        print(f"   Exit code: {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Train all dataset variants")
    parser.add_argument(
        "--category",
        type=str,
        choices=["all", "cropped", "full_xray", "detection"],
        default="all",
        help="Category of datasets to train",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="Device ID")

    args = parser.parse_args()

    # Select configurations based on category
    if args.category == "all":
        selected_configs = []
        for category_configs in TRAINING_CONFIGS.values():
            selected_configs.extend(category_configs)
    elif args.category == "cropped":
        selected_configs = (
            TRAINING_CONFIGS["cropped_base"] + TRAINING_CONFIGS["cropped_balanced"]
        )
    elif args.category == "full_xray":
        selected_configs = (
            TRAINING_CONFIGS["full_xray_base"] + TRAINING_CONFIGS["full_xray_balanced"]
        )
    elif args.category == "detection":
        selected_configs = TRAINING_CONFIGS["detection"]

    # Print summary
    print("\n" + "=" * 80)
    print("Automated Training for All Dataset Variants")
    print("=" * 80)
    print(f"Category: {args.category}")
    print(f"Total datasets: {len(selected_configs)}")
    print(f"Config: {args.epochs} epochs, batch={args.batch}, patience=20")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Run training for each configuration
    results = {}
    total_start = time.time()

    for i, config in enumerate(selected_configs, 1):
        print(f"\n{'='*80}")
        print(f"Progress: {i}/{len(selected_configs)}")
        print(f"{'='*80}")

        success = run_training(
            config=config,
            epochs=args.epochs,
            batch=args.batch,
            device=args.device,
        )

        results[config["name"]] = success

        # Print intermediate summary
        completed = sum(1 for s in results.values() if s)
        failed = sum(1 for s in results.values() if not s)
        print(
            f"\n📊 Progress: {completed} completed, {failed} failed, {len(selected_configs) - i} remaining"
        )

    # Final summary
    total_elapsed = time.time() - total_start

    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)
    print(f"Total time: {total_elapsed/3600:.2f} hours")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Group results by status
    successful = [name for name, success in results.items() if success]
    failed = [name for name, success in results.items() if not success]

    if successful:
        print(f"✅ Successful ({len(successful)}):")
        for name in successful:
            print(f"   - {name}")
        print()

    if failed:
        print(f"❌ Failed ({len(failed)}):")
        for name in failed:
            print(f"   - {name}")
        print()

    print("=" * 80)
    print(f"Results: {len(successful)}/{len(results)} successful")
    print("=" * 80)

    # Exit with appropriate code
    if all(results.values()):
        print("\n🎉 All training completed successfully!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {len(failed)} training(s) failed. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
