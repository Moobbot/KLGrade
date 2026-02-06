import torch
import argparse
import sys
import json
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.kiocmil_with_detection import KiocmilWithDetection
from src.datasets.kiocmil_dataset_end_to_end import (
    KiocmilDatasetEndToEnd,
    collate_end_to_end,
)


def evaluate_model(
    checkpoint_path,
    img_dir,
    knee_label_dir,
    lesion_label_dir,
    split_file,
    num_classes=10,
    device="cuda",
    batch_size=32,
):
    print(f"\nEvaluating: {checkpoint_path}")

    # Load Model
    model = KiocmilWithDetection(
        backbone_name="yolo11l",  # Assuming yolo1l was used as per run script
        num_classes=num_classes,
        pretrained_kiocmil=None,
        freeze_kiocmil=False,
    )

    # Load Weights
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  ✅ Weights loaded (Epoch {checkpoint.get('epoch', 'N/A')})")
    except Exception as e:
        print(f"  ❌ Failed to load weights: {e}")
        return None

    model = model.to(device)
    model.eval()

    # Load Dataset
    dataset = KiocmilDatasetEndToEnd(
        img_dir=img_dir,
        knee_label_dir=knee_label_dir,
        lesion_label_dir=lesion_label_dir,
        split_file=split_file,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_end_to_end,
        num_workers=4,
    )

    # Inference
    all_preds = []
    all_targets = []

    print("  Running inference...")
    with torch.no_grad():
        for images, targets in tqdm(loader, leave=False):
            images = images.to(device)
            global_labels = torch.stack([t["label"] for t in targets]).cpu().numpy()

            # Forward
            outputs = model(images)
            if "logits_10" in outputs:
                img_logits = outputs["logits_10"]
                preds = torch.argmax(img_logits, dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_targets.extend(global_labels)
            else:
                print("  ⚠️ No logits_10 in output")

    # Metrics
    if len(all_preds) == 0:
        return None

    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro")
    cm = confusion_matrix(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, output_dict=True)

    results = {
        "accuracy": acc,
        "f1_macro": f1,
        "confusion_matrix": cm.tolist(),
        "report": report,
    }

    print(f"  ✅ Accuracy: {acc:.4f}")
    print(f"  ✅ F1 Macro: {f1:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs/end_to_end")
    parser.add_argument("--output", default="end_to_end_results_summary.json")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    all_experiments = sorted([d for d in runs_dir.iterdir() if d.is_dir()])

    final_results = {}

    # Define dataset paths mapping (reverse engineered from run_all_end_to_end_experiments.sh)
    # This is rough, ideally we should read the config used, but for now strict mapping

    MAP = {
        "e2e_10class_unbalanced": {
            "num_classes": 10,
            "img_dir": "datasets/dataset_v0/images",
            "knee_label_dir": "datasets/dataset_v0/labels-knee",
            "lesion_label_dir": "datasets/dataset_v0/labels_10_class",
            "split_file": "datasets/splits/knee_full_10_class/val.txt",
        },
        "e2e_10class_balanced": {
            "num_classes": 10,
            "img_dir": "datasets/balanced/full_xray_10_class/images",
            "knee_label_dir": "datasets/balanced/full_xray_10_class/labels-knee",
            "lesion_label_dir": "datasets/balanced/full_xray_10_class/labels",
            "split_file": "datasets/splits/balanced_full_xray_10_class/val.txt",
        },
        "e2e_10class_balanced_resize": {
            "num_classes": 10,
            "img_dir": "datasets/processed_balanced/full_xray/resize_only/images",
            "knee_label_dir": "datasets/processed_balanced/full_xray/resize_only/labels-knee",
            "lesion_label_dir": "datasets/processed_balanced/full_xray/resize_only/labels_10_class",
            "split_file": "datasets/splits/balanced_full_xray_10_class/val.txt",
        },
        "e2e_10class_balanced_blur": {
            "num_classes": 10,
            "img_dir": "datasets/processed_balanced/full_xray/blur_clahe2/images",
            "knee_label_dir": "datasets/processed_balanced/full_xray/blur_clahe2/labels-knee",
            "lesion_label_dir": "datasets/processed_balanced/full_xray/blur_clahe2/labels_10_class",
            "split_file": "datasets/splits/balanced_full_xray_10_class/val.txt",
        },
        "e2e_10class_balanced_sharp": {
            "num_classes": 10,
            "img_dir": "datasets/processed_balanced/full_xray/sharp_clahe4/images",
            "knee_label_dir": "datasets/processed_balanced/full_xray/sharp_clahe4/labels-knee",
            "lesion_label_dir": "datasets/processed_balanced/full_xray/sharp_clahe4/labels_10_class",
            "split_file": "datasets/splits/balanced_full_xray_10_class/val.txt",
        },
        "e2e_5class_unbalanced": {
            "num_classes": 5,
            "img_dir": "datasets/dataset_v0/images",
            "knee_label_dir": "datasets/dataset_v0/labels-knee",
            "lesion_label_dir": "datasets/dataset_v0/labels",
            "split_file": "datasets/splits/knee_full_10_class/val.txt",
        },
    }

    print("\nStarting Batch Evaluation...")

    for exp_name, config in MAP.items():
        exp_dir = runs_dir / exp_name
        checkpoint = exp_dir / "best.pt"

        if not checkpoint.exists():
            print(f"Skipping {exp_name} (No best.pt found)")
            continue

        print(f"Processing experiment: {exp_name}")

        res = evaluate_model(
            str(checkpoint),
            img_dir=config["img_dir"],
            knee_label_dir=config["knee_label_dir"],
            lesion_label_dir=config["lesion_label_dir"],
            split_file=config["split_file"],
            num_classes=config["num_classes"],
        )

        if res:
            final_results[exp_name] = res

    # Save results
    with open(args.output, "w") as f:
        json.dump(final_results, f, indent=4)

    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
