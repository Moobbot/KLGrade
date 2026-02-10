"""
Generate dataset.yaml files for YOLO training

This script automatically creates dataset.yaml files for ALL dataset configurations
listed in scripts/pipelines/step_5_create_splits.sh
"""

import yaml
from pathlib import Path
from typing import Dict, List


# Dataset configurations - matches step_5_create_splits.sh output structure
DATASETS = {
    # ========== CROPPED KNEE DATASETS ==========
    # Base (unbalanced) cropped knee datasets
    "dataset_knees_cropped_5class": {
        "path": "/home/ngoductam/KLGrade/datasets/dataset_knees_cropped",
        "splits_dir": "dataset_knees_cropped_70_20_10",
        "classes": {0: "KL0", 1: "KL1", 2: "KL2", 3: "KL3", 4: "KL4"},
    },
    "dataset_knees_cropped_4class": {
        "path": "/home/ngoductam/KLGrade/datasets/dataset_knees_cropped_4_class",
        "splits_dir": "dataset_knees_cropped_4_class_70_20_10",
        "classes": {0: "KL1", 1: "KL2", 2: "KL3", 3: "KL4"},
    },
    "dataset_knees_cropped_8class": {
        "path": "/home/ngoductam/KLGrade/datasets/dataset_knees_cropped_8_class",
        "splits_dir": "dataset_knees_cropped_8_class_70_20_10",
        "classes": {
            0: "KL1-a",
            1: "KL1-b",
            2: "KL2-a",
            3: "KL2-b",
            4: "KL3-a",
            5: "KL3-b",
            6: "KL4-a",
            7: "KL4-b",
        },
    },
    "dataset_knees_cropped_10class": {
        "path": "/home/ngoductam/KLGrade/datasets/dataset_knees_cropped_10_class",
        "splits_dir": "dataset_knees_cropped_10_class_60_20_20",
        "classes": {
            0: "KL0-a",
            1: "KL0-b",
            2: "KL1-a",
            3: "KL1-b",
            4: "KL2-a",
            5: "KL2-b",
            6: "KL3-a",
            7: "KL3-b",
            8: "KL4-a",
            9: "KL4-b",
        },
    },
    # Balanced cropped knee datasets
    "balanced_knees_cropped_5class": {
        "path": "/home/ngoductam/KLGrade/datasets/balanced/knees_cropped",
        "splits_dir": "balanced_knees_cropped_70_20_10",
        "classes": {0: "KL0", 1: "KL1", 2: "KL2", 3: "KL3", 4: "KL4"},
    },
    "balanced_knees_cropped_4class": {
        "path": "/home/ngoductam/KLGrade/datasets/balanced/knees_cropped_4_class",
        "splits_dir": "balanced_knees_cropped_4_class_70_20_10",
        "classes": {0: "KL1", 1: "KL2", 2: "KL3", 3: "KL4"},
    },
    "balanced_knees_cropped_8class": {
        "path": "/home/ngoductam/KLGrade/datasets/balanced/knees_cropped_8_class",
        "splits_dir": "balanced_knees_cropped_8_class_70_20_10",
        "classes": {
            0: "KL1-a",
            1: "KL1-b",
            2: "KL2-a",
            3: "KL2-b",
            4: "KL3-a",
            5: "KL3-b",
            6: "KL4-a",
            7: "KL4-b",
        },
    },
    "balanced_knees_cropped_10class": {
        "path": "/home/ngoductam/KLGrade/datasets/balanced/knees_cropped_10_class",
        "splits_dir": "balanced_knees_cropped_10_class_60_20_20",
        "classes": {
            0: "KL0-a",
            1: "KL0-b",
            2: "KL1-a",
            3: "KL1-b",
            4: "KL2-a",
            5: "KL2-b",
            6: "KL3-a",
            7: "KL3-b",
            8: "KL4-a",
            9: "KL4-b",
        },
    },
    # ========== FULL X-RAY DATASETS ==========
    # Base (unbalanced) full X-ray datasets
    "knee_full_4class": {
        "path": "/home/ngoductam/KLGrade/datasets/dataset_v0_4_class",
        "splits_dir": "knee_full_4_class_70_20_10",
        "classes": {0: "KL1", 1: "KL2", 2: "KL3", 3: "KL4"},
    },
    "knee_full_8class": {
        "path": "/home/ngoductam/KLGrade/datasets/dataset_v0_8_class",
        "splits_dir": "knee_full_8_class_70_20_10",
        "classes": {
            0: "KL1-a",
            1: "KL1-b",
            2: "KL2-a",
            3: "KL2-b",
            4: "KL3-a",
            5: "KL3-b",
            6: "KL4-a",
            7: "KL4-b",
        },
    },
    "knee_full_10class": {
        "path": "/home/ngoductam/KLGrade/datasets/dataset_v0_10_class",
        "splits_dir": "knee_full_10_class_70_20_10",
        "classes": {
            0: "KL0-a",
            1: "KL0-b",
            2: "KL1-a",
            3: "KL1-b",
            4: "KL2-a",
            5: "KL2-b",
            6: "KL3-a",
            7: "KL3-b",
            8: "KL4-a",
            9: "KL4-b",
        },
    },
    # Balanced full X-ray datasets
    "balanced_full_xray_4class": {
        "path": "/home/ngoductam/KLGrade/datasets/balanced/full_xray_4_class",
        "splits_dir": "balanced_full_xray_4_class_70_20_10",
        "classes": {0: "KL1", 1: "KL2", 2: "KL3", 3: "KL4"},
    },
    "balanced_full_xray_8class": {
        "path": "/home/ngoductam/KLGrade/datasets/balanced/full_xray_8_class",
        "splits_dir": "balanced_full_xray_8_class_70_20_10",
        "classes": {
            0: "KL1-a",
            1: "KL1-b",
            2: "KL2-a",
            3: "KL2-b",
            4: "KL3-a",
            5: "KL3-b",
            6: "KL4-a",
            7: "KL4-b",
        },
    },
    "balanced_full_xray_10class": {
        "path": "/home/ngoductam/KLGrade/datasets/balanced/full_xray_10_class",
        "splits_dir": "balanced_full_xray_10_class_60_20_20",
        "classes": {
            0: "KL0-a",
            1: "KL0-b",
            2: "KL1-a",
            3: "KL1-b",
            4: "KL2-a",
            5: "KL2-b",
            6: "KL3-a",
            7: "KL3-b",
            8: "KL4-a",
            9: "KL4-b",
        },
    },
    # ========== DETECTION DATASETS ==========
    "knee_detection": {
        "path": "/home/ngoductam/KLGrade/datasets/processed/knee_detection_final",
        "splits_dir": "knee",
        "classes": {0: "Knee"},
    },
}


def generate_dataset_yaml(
    name: str,
    data_path: str,
    splits_dir: str,
    classes: Dict[int, str],
    output_path: Path,
) -> None:
    """
    Generate a dataset.yaml file for YOLO training.

    Args:
        name: Dataset name
        data_path: Path to dataset directory
        splits_dir: Directory name containing train/val/test splits
        classes: Dictionary mapping class IDs to class names
        output_path: Path to save the dataset.yaml file
    """
    project_root = Path("/home/ngoductam/KLGrade")

    # Build split paths
    if splits_dir:
        splits_base = project_root / "datasets" / "splits" / splits_dir
        train_path = splits_base / "train.txt"
        val_path = splits_base / "val.txt"
        test_path = splits_base / "test.txt"
    else:
        # Internal splits (e.g., lesion_detection)
        data_dir = Path(data_path)
        train_path = data_dir / "train.txt"
        val_path = data_dir / "val.txt"
        test_path = data_dir / "test.txt"

    # Create dataset config
    config = {
        "path": data_path,
        "train": str(train_path),
        "val": str(val_path),
        "test": str(test_path),
        "names": classes,
        "nc": len(classes),
    }

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write YAML file
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✅ Created: {output_path}")
    print(f"   Classes: {len(classes)} ({', '.join(classes.values())})")


def main():
    """Generate all dataset.yaml files."""
    print("=" * 70)
    print("Generating dataset.yaml files for ALL datasets")
    print("=" * 70)
    print()

    project_root = Path("/home/ngoductam/KLGrade")
    created_count = 0
    skipped_count = 0

    for name, config in DATASETS.items():
        print(f"Processing: {name}")

        # Determine output path
        if config["splits_dir"]:
            output_dir = project_root / "datasets" / "splits" / config["splits_dir"]
        else:
            output_dir = Path(config["path"])

        output_path = output_dir / "dataset.yaml"

        # Check if splits directory exists
        if config["splits_dir"] and not output_dir.exists():
            print(f"⚠️  Skipped: Splits directory not found: {output_dir}")
            skipped_count += 1
            print()
            continue

        # Generate YAML
        try:
            generate_dataset_yaml(
                name=name,
                data_path=config["path"],
                splits_dir=config["splits_dir"],
                classes=config["classes"],
                output_path=output_path,
            )
            created_count += 1
        except Exception as e:
            print(f"❌ Error: {e}")
            skipped_count += 1

        print()

    print("=" * 70)
    print(f"✅ Generated {created_count} dataset.yaml files")
    if skipped_count > 0:
        print(f"⚠️  Skipped {skipped_count} datasets (splits not found)")
    print("=" * 70)


if __name__ == "__main__":
    main()
