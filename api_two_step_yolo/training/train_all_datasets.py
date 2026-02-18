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


def discover_configs():
    """Dynamically discover dataset configurations from datasets/splits directory."""
    configs = {
        "cropped_base": [],
        "cropped_balanced": [],
        "full_xray_base": [],
        "full_xray_balanced": [],
        "detection": []
    }
    
    splits_dir = PROJECT_ROOT / "datasets/splits"
    if not splits_dir.exists():
        print(f"Warning: {splits_dir} not found")
        return configs
        
    for d in splits_dir.iterdir():
        if not d.is_dir() or not (d / "dataset.yaml").exists():
            continue
            
        name = d.name
        config = {
            "name": name,  # Will be updated with readable name
            "data": str(d / "dataset.yaml").replace(str(PROJECT_ROOT) + "/", ""),
            "run_name": name,
        }
        
        # Classification and naming logic
        parts = name.split("_")
        split = ""
        # Check for split ratio at the end (e.g. 70_20_10)
        if len(parts) >= 3 and parts[-3].isdigit() and parts[-2].isdigit() and parts[-1].isdigit():
            split = f"[{parts[-3]}/{parts[-2]}/{parts[-1]}]"
            
        if name == "knee":
            config["name"] = "Knee Detection"
            configs["detection"].append(config)
            continue
            
        # Determine class count
        nc = "5"  # Default
        if "class" in parts:
            try:
                class_idx = parts.index("class")
                if class_idx > 0 and parts[class_idx-1].isdigit():
                    nc = parts[class_idx-1]
            except ValueError:
                pass
        
        # Categorize
        if "balanced" in name:
            if "full" in name or "xray" in name:
                category = "full_xray_balanced"
                readable_name = f"{nc}-class Full X-ray (Balanced) {split}"
            else:
                category = "cropped_balanced"
                readable_name = f"{nc}-class Cropped (Balanced) {split}"
        else:
            if "full" in name or "xray" in name:
                category = "full_xray_base"
                readable_name = f"{nc}-class Full X-ray (Base) {split}"
            else:
                # Default to cropped base if not full/xray and not balanced
                # Check if it looks like cropped dataset
                if "cropped" in name or "knees" in name:
                    category = "cropped_base"
                    readable_name = f"{nc}-class Cropped (Base) {split}"
                else:
                    # Unknown category, skip or put in misc? attempting to categorise as cropped base
                    category = "cropped_base" 
                    readable_name = f"{nc}-class {name} {split}"

        config["name"] = readable_name.strip()
        configs[category].append(config)
    
    # Sort configs by name
    for key in configs:
        configs[key].sort(key=lambda x: x["name"])
        
    return configs

# Training configurations for all datasets
TRAINING_CONFIGS = discover_configs()


def run_training(config: dict, epochs: int = 100, batch: int = 16, device: str = "0", wandb_project: str = "klgrade-lesion-detection"):
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
        "--wandb-project",
        wandb_project,
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
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="klgrade-lesion-detection",
        help="WandB project name",
    )

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
            wandb_project=args.wandb_project,
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
