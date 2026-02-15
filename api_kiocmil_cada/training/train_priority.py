"""
Priority Training Script - Train Best Models First

Based on training_summary.csv results, this script trains models
in order of best performance (mAP@50-95) to worst.

Top performers:
1. Knee Detection: 0.727 mAP@50-95
2. 8-class Cropped (Balanced): 0.663
3. 5-class Cropped (Balanced): 0.572
4. 10-class Cropped (Balanced): 0.571
5. 10-class Full X-ray (Balanced): 0.540
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# Training configurations sorted by Best_mAP50_95 (descending)
# Based on training_summary.csv results
PRIORITY_CONFIGS = [
    # Rank 1: Knee Detection (0.727 mAP@50-95)
    {
        "name": "Knee Detection",
        "data": "datasets/splits/knee/dataset.yaml",
        "run_name": "knee_detector",
        "priority": 1,
        "expected_map": 0.727,
    },
    # Rank 2: 8-class Cropped Balanced (0.663)
    {
        "name": "8-class Cropped (Balanced)",
        "data": "datasets/splits/balanced_knees_cropped_8_class_70_20_10/dataset.yaml",
        "run_name": "lesion_8class_balanced",
        "priority": 2,
        "expected_map": 0.663,
    },
    # Rank 3: 5-class Cropped Balanced (0.572)
    {
        "name": "5-class Cropped (Balanced)",
        "data": "datasets/splits/balanced_knees_cropped_70_20_10/dataset.yaml",
        "run_name": "lesion_5class_balanced",
        "priority": 3,
        "expected_map": 0.572,
    },
    # Rank 4: 10-class Cropped Balanced (0.571)
    {
        "name": "10-class Cropped (Balanced)",
        "data": "datasets/splits/balanced_knees_cropped_10_class_60_20_20/dataset.yaml",
        "run_name": "lesion_10class_balanced",
        "priority": 4,
        "expected_map": 0.571,
    },
    # Rank 5: 10-class Full X-ray Balanced (0.540)
    {
        "name": "10-class Full X-ray (Balanced)",
        "data": "datasets/splits/balanced_full_xray_10_class_60_20_20/dataset.yaml",
        "run_name": "lesion_full_10class_balanced",
        "priority": 5,
        "expected_map": 0.540,
    },
    # Rank 6: 8-class Full X-ray Balanced (0.507)
    {
        "name": "8-class Full X-ray (Balanced)",
        "data": "datasets/splits/balanced_full_xray_8_class_70_20_10/dataset.yaml",
        "run_name": "lesion_full_8class_balanced",
        "priority": 6,
        "expected_map": 0.507,
    },
    # Rank 7: 4-class Cropped Balanced (0.320)
    {
        "name": "4-class Cropped (Balanced)",
        "data": "datasets/splits/balanced_knees_cropped_4_class_70_20_10/dataset.yaml",
        "run_name": "lesion_4class_balanced",
        "priority": 7,
        "expected_map": 0.320,
    },
    # Rank 8: 4-class Full X-ray Balanced (0.191)
    {
        "name": "4-class Full X-ray (Balanced)",
        "data": "datasets/splits/balanced_full_xray_4_class_70_20_10/dataset.yaml",
        "run_name": "lesion_full_4class_balanced",
        "priority": 8,
        "expected_map": 0.191,
    },
    # Rank 9: 10-class Cropped Base (0.120)
    {
        "name": "10-class Cropped (Base)",
        "data": "datasets/splits/dataset_knees_cropped_10_class_60_20_20/dataset.yaml",
        "run_name": "lesion_10class_base",
        "priority": 9,
        "expected_map": 0.120,
    },
    # Rank 10: 8-class Cropped Base (0.118)
    {
        "name": "8-class Cropped (Base)",
        "data": "datasets/splits/dataset_knees_cropped_8_class_70_20_10/dataset.yaml",
        "run_name": "lesion_8class_base",
        "priority": 10,
        "expected_map": 0.118,
    },
    # Rank 11: 4-class Cropped Base (0.114)
    {
        "name": "4-class Cropped (Base)",
        "data": "datasets/splits/dataset_knees_cropped_4_class_70_20_10/dataset.yaml",
        "run_name": "lesion_4class_base",
        "priority": 11,
        "expected_map": 0.114,
    },
    # Rank 12: 5-class Cropped Base (0.102)
    {
        "name": "5-class Cropped (Base)",
        "data": "datasets/splits/dataset_knees_cropped_70_20_10/dataset.yaml",
        "run_name": "lesion_5class_base",
        "priority": 12,
        "expected_map": 0.102,
    },
    # Rank 13: 8-class Full X-ray Base (0.0964)
    {
        "name": "8-class Full X-ray (Base)",
        "data": "datasets/splits/knee_full_8_class_70_20_10/dataset.yaml",
        "run_name": "lesion_full_8class_base",
        "priority": 13,
        "expected_map": 0.0964,
    },
    # Rank 14: 4-class Full X-ray Base (0.0961)
    {
        "name": "4-class Full X-ray (Base)",
        "data": "datasets/splits/knee_full_4_class_70_20_10/dataset.yaml",
        "run_name": "lesion_full_4class_base",
        "priority": 14,
        "expected_map": 0.0961,
    },
    # Rank 15: 10-class Full X-ray Base (0.0729)
    {
        "name": "10-class Full X-ray (Base)",
        "data": "datasets/splits/knee_full_10_class_70_20_10/dataset.yaml",
        "run_name": "lesion_full_10class_base",
        "priority": 15,
        "expected_map": 0.0729,
    },
]


def run_training(config: dict, epochs: int = 100, batch: int = 16, device: str = "0"):
    """Run YOLO training for a single dataset configuration."""
    print("\n" + "=" * 80)
    print(f"Priority {config['priority']}: {config['name']}")
    print("=" * 80)
    print(f"Dataset: {config['data']}")
    print(f"Run name: {config['run_name']}")
    print(f"Expected mAP@50-95: {config['expected_map']:.3f}")
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
        "api_kiocmil_cada/training/train_lesion_detector.py",
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
    parser = argparse.ArgumentParser(
        description="Train models in priority order (best to worst)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Train only top N models (default: all 15)",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="Device ID")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if model already exists",
    )

    args = parser.parse_args()

    # Select configurations
    if args.top_n:
        selected_configs = PRIORITY_CONFIGS[: args.top_n]
    else:
        selected_configs = PRIORITY_CONFIGS

    # Print summary
    print("\n" + "=" * 80)
    print("Priority Training - Best Models First")
    print("=" * 80)
    print(f"Total models: {len(selected_configs)}")
    print(f"Config: {args.epochs} epochs, batch={args.batch}, patience=20")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print("\nTraining Order (by mAP@50-95):")
    for i, config in enumerate(selected_configs, 1):
        print(f"  {i}. {config['name']:<35} (mAP: {config['expected_map']:.3f})")
    print("=" * 80)

    # Run training for each configuration
    results = {}
    total_start = time.time()

    for i, config in enumerate(selected_configs, 1):
        # Check if model already exists
        model_path = PROJECT_ROOT / f"runs/detect/{config['run_name']}/weights/best.pt"
        if args.skip_existing and model_path.exists():
            print(f"\n⏭️  Skipping {config['name']} (model already exists)")
            results[config["name"]] = "skipped"
            continue

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
        completed = sum(1 for s in results.values() if s is True)
        failed = sum(1 for s in results.values() if s is False)
        skipped = sum(1 for s in results.values() if s == "skipped")
        print(
            f"\n📊 Progress: {completed} completed, {failed} failed, {skipped} skipped, {len(selected_configs) - i} remaining"
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
    successful = [name for name, success in results.items() if success is True]
    failed = [name for name, success in results.items() if success is False]
    skipped = [name for name, success in results.items() if success == "skipped"]

    if successful:
        print(f"✅ Successful ({len(successful)}):")
        for name in successful:
            print(f"   - {name}")
        print()

    if skipped:
        print(f"⏭️  Skipped ({len(skipped)}):")
        for name in skipped:
            print(f"   - {name}")
        print()

    if failed:
        print(f"❌ Failed ({len(failed)}):")
        for name in failed:
            print(f"   - {name}")
        print()

    print("=" * 80)
    print(
        f"Results: {len(successful)}/{len([r for r in results.values() if r != 'skipped'])} successful"
    )
    print("=" * 80)

    # Exit with appropriate code
    if all(r is True or r == "skipped" for r in results.values()):
        print("\n🎉 All training completed successfully!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {len(failed)} training(s) failed. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
