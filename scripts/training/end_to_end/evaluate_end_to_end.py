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
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.models.kiocmil_with_detection import KiocmilWithDetection
from src.datasets.kiocmil_dataset_end_to_end import (
    KiocmilDatasetEndToEnd,
    collate_end_to_end,
)


def evaluate_model(
    checkpoint_path: str,
    img_dir: str,
    knee_label_dir: str,
    lesion_label_dir: str,
    split_file: str,
    num_classes: int = 10,
    device: str = "cuda",
    batch_size: int = 16,
    backbone: str = "yolo11l",
):
    print(f"\n{'='*60}")
    print(f"Evaluating End-to-End Model")
    print(f"{'='*60}\n")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {img_dir}")
    print(f"Split: {split_file}")
    print(f"Classes: {num_classes}")

    # Load Model
    print("\n1. Loading Model...")
    model = KiocmilWithDetection(
        backbone_name=backbone,
        num_classes=num_classes,
        pretrained_kiocmil=None,
        freeze_kiocmil=False,
    )

    # Load Weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Handle state dict structure
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        epoch = checkpoint.get("epoch", "N/A")
        print(f"✅ Loaded checkpoint from Epoch {epoch}")
    else:
        state_dict = checkpoint
        print("✅ Loaded state_dict directly")

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Load Dataset
    print("\n2. Loading Dataset...")
    dataset = KiocmilDatasetEndToEnd(
        img_dir=img_dir,
        knee_label_dir=knee_label_dir,
        lesion_label_dir=lesion_label_dir,
        split_file=split_file,
    )
    print(f"✅ Found {len(dataset)} samples")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_end_to_end,
        num_workers=4,
        pin_memory=True,
    )

    # Inference
    all_preds_10 = []
    all_targets = []

    # TODO: Add metrics for 5-class and others if needed

    print("\n3. Running Inference...")
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating"):
            images = images.to(device)

            # Get global labels
            global_labels = torch.stack([t["label"] for t in targets]).cpu().numpy()

            # Forward
            # We don't skip classification during eval
            outputs = model(images)

            if "logits_10" in outputs:
                img_logits = outputs["logits_10"]
                # For N-class models, logits_10 contains N logits
                preds = torch.argmax(img_logits, dim=1).cpu().numpy()

                all_preds_10.extend(preds)
                all_targets.extend(global_labels)
            else:
                print("⚠️  Warning: No logits found in output")

    # Metrics
    if len(all_preds_10) == 0:
        print("❌ No predictions made!")
        return None

    print("\n4. Calculating Metrics...")

    # Classification Metrics
    acc = accuracy_score(all_targets, all_preds_10)
    f1_macro = f1_score(all_targets, all_preds_10, average="macro")
    f1_weighted = f1_score(all_targets, all_preds_10, average="weighted")
    cm = confusion_matrix(all_targets, all_preds_10)
    report = classification_report(all_targets, all_preds_10, output_dict=True)
    report_text = classification_report(all_targets, all_preds_10)

    print(f"\n✅ Accuracy: {acc:.4f}")
    print(f"✅ F1 (Macro): {f1_macro:.4f}")
    print(f"✅ F1 (Weighted): {f1_weighted:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(report_text)

    results = {
        "metrics": {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted},
        "confusion_matrix": cm.tolist(),
        "report": report,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate End-to-End Model")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--knee-label-dir", required=True)
    parser.add_argument("--lesion-label-dir", required=True)
    parser.add_argument(
        "--split-file", required=True, help="Path to test/val split txt file"
    )

    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument(
        "--backbone", type=str, default="yolo11l"
    )  # Match training default
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", help="Path to save JSON results")

    args = parser.parse_args()

    results = evaluate_model(
        args.checkpoint,
        args.img_dir,
        args.knee_label_dir,
        args.lesion_label_dir,
        args.split_file,
        num_classes=args.num_classes,
        device=args.device,
        batch_size=args.batch_size,
        backbone=args.backbone,
    )

    if args.output and results:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\n✅ Results saved to {out_path}")


if __name__ == "__main__":
    main()
