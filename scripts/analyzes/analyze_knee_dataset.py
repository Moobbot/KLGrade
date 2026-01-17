"""
Analyze knee cropped dataset and generate statistics report.
"""

import sys
from pathlib import Path
from collections import Counter
from datetime import datetime

# Add project root
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))


def analyze_labels(label_dir: Path, class_names: dict) -> dict:
    """Analyze label distribution."""
    label_files = list(label_dir.glob("*.txt"))

    class_counts = Counter()
    total_boxes = 0
    images_with_labels = 0
    images_without_labels = 0

    for label_file in label_files:
        with open(label_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        if lines:
            images_with_labels += 1
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    class_id = int(float(parts[0]))
                    class_counts[class_id] += 1
                    total_boxes += 1
        else:
            images_without_labels += 1

    return {
        "total_images": len(label_files),
        "images_with_labels": images_with_labels,
        "images_without_labels": images_without_labels,
        "total_boxes": total_boxes,
        "class_distribution": dict(class_counts),
        "class_names": class_names,
    }


def generate_report(output_dir: Path):
    """Generate comprehensive statistics report."""

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("KNEE CROPPED DATASET - STATISTICS REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Dataset: {output_dir}")
    report_lines.append("")

    # Count images
    images_dir = output_dir / "images"
    image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
    report_lines.append(f"Total cropped knee images: {len(image_files)}")
    report_lines.append("")

    # Analyze each label variant
    label_variants = {
        "labels": {
            "name": "5-Class Labels (KL0-4)",
            "classes": {0: "KL0", 1: "KL1", 2: "KL2", 3: "KL3", 4: "KL4"},
        },
        "labels_10_class": {
            "name": "10-Class Labels (KL0-a/b to KL4-a/b)",
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
        "labels_4_class": {
            "name": "4-Class Labels (KL1-4, filtered KL0)",
            "classes": {0: "KL1", 1: "KL2", 2: "KL3", 3: "KL4"},
        },
        "labels_8_class": {
            "name": "8-Class Labels (KL1-a/b to KL4-a/b, filtered KL0)",
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
    }

    for label_dir_name, variant_info in label_variants.items():
        label_dir = output_dir / label_dir_name

        if not label_dir.exists():
            continue

        report_lines.append("-" * 80)
        report_lines.append(variant_info["name"])
        report_lines.append("-" * 80)

        stats = analyze_labels(label_dir, variant_info["classes"])

        report_lines.append(f"Total label files: {stats['total_images']}")
        report_lines.append(f"Images with labels: {stats['images_with_labels']}")
        report_lines.append(f"Images without labels: {stats['images_without_labels']}")
        report_lines.append(f"Total bounding boxes: {stats['total_boxes']}")
        report_lines.append("")

        if stats["class_distribution"]:
            report_lines.append("Class Distribution:")
            for class_id in sorted(stats["class_distribution"].keys()):
                count = stats["class_distribution"][class_id]
                class_name = variant_info["classes"].get(class_id, f"Class {class_id}")
                percentage = (
                    (count / stats["total_boxes"]) * 100
                    if stats["total_boxes"] > 0
                    else 0
                )
                report_lines.append(
                    f"  {class_name:12s}: {count:5d} boxes ({percentage:5.1f}%)"
                )
        else:
            report_lines.append("No labels found.")

        report_lines.append("")

    # Summary
    report_lines.append("=" * 80)
    report_lines.append("DATASET READY FOR PREPROCESSING & TRAINING")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("Next steps:")
    report_lines.append(
        "1. Apply preprocessing (blur, CLAHE, etc.) using scripts/preprocess_production.py"
    )
    report_lines.append("2. Create train/val/test splits")
    report_lines.append("3. Start training")
    report_lines.append("")

    return "\n".join(report_lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze knee cropped dataset")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to knee cropped dataset"
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset)

    if not dataset_dir.exists():
        print(f"❌ Dataset not found: {dataset_dir}")
        sys.exit(1)

    # Generate report
    report = generate_report(dataset_dir)

    # Print to console
    print(report)

    # Save to file
    report_path = dataset_dir / "dataset_statistics.txt"
    report_path.write_text(report)
    print(f"\n✅ Report saved to: {report_path}")
