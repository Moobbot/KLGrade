import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import sys
from tqdm import tqdm
import sklearn.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.datasets.kiocmil_dataset_v3 import KiocmilDatasetV3, collate_kiocmil_v3
from src.datasets.kiocmil_transforms_v2 import get_photometric_transforms
from src.models.kiocmil_model_cada import KiocmilModelCADA
from src.config import PROJECT_ROOT


def plot_confusion_matrix(cm, classes, save_path, title="Confusion Matrix"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def plot_roc_curve(targets, probs, n_classes, save_path, title="ROC Curve"):
    # Targets should be one-hot for ROC
    # Probs should be (N, n_classes)
    targets_one_hot = np.eye(n_classes)[targets]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(10, 8))

    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(targets_one_hot[:, i], probs[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f"Class {i} (area = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved ROC curve to {save_path}")


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Data
    print(f"Loading dataset from {args.split_file}...")

    # Use valid transform logic from trainer
    transform = get_photometric_transforms(level="none", use_clahe=True)

    dataset = KiocmilDatasetV3(
        img_dir=args.img_dir,
        knee_label_dir=args.knee_label_dir,
        lesion_label_dir=args.lesion_label_dir,
        split_file=args.split_file,
        geometric_transform=None,
        photometric_transform=transform,
        ctx_size=(384, 384),
        patch_size=(224, 224),
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_kiocmil_v3,
    )

    # Model
    print(f"Loading model from {args.model_path}...")
    model = KiocmilModelCADA(
        backbone_name="yolo11l",
        num_classes=10,
        feature_dim=256,
        num_deformable_points=4,
        num_context_scales=3,
        use_positional_encoding=True,
        dropout=0.1,
    ).to(device)

    # Load weights
    checkpoint = torch.load(args.model_path, map_location=device)
    # Handle both full checkpoint dict and model state dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Metrics containers
    all_preds_10 = []
    all_targets_10 = []
    all_probs_10 = []

    all_preds_grade = []
    all_targets_grade = []

    print("Running inference...")
    with torch.no_grad():
        for batch_data in tqdm(loader):
            if not batch_data or not batch_data[0].get("knees"):
                continue

            # Forward
            outputs = model(batch_data)

            # --- 10-Class Metrics ---
            logits_10 = outputs["logits_10"]
            probs_10 = torch.softmax(logits_10, dim=1)
            preds_10 = torch.argmax(logits_10, dim=1)

            # Get targets
            labels = [item["label"] for item in batch_data]
            targets_10 = torch.tensor(labels, device=device).long()

            all_preds_10.extend(preds_10.cpu().numpy())
            all_targets_10.extend(targets_10.cpu().numpy())
            all_probs_10.extend(probs_10.cpu().numpy())

            # --- Grade Metrics ---
            # Derived from 10-class (0-9 -> 0-4)
            preds_grade = preds_10 // 2
            targets_grade = targets_10 // 2

            all_preds_grade.extend(preds_grade.cpu().numpy())
            all_targets_grade.extend(targets_grade.cpu().numpy())

    # Convert to arrays
    all_targets_10 = np.array(all_targets_10)
    all_preds_10 = np.array(all_preds_10)
    all_probs_10 = np.array(all_probs_10)

    all_targets_grade = np.array(all_targets_grade)
    all_preds_grade = np.array(all_preds_grade)

    # --- Print Text Reports ---
    print("\n" + "=" * 60)
    print(" EVALUATION RESULTS (KIOCMIL CADA)")
    print("=" * 60)

    # 10-Class Report
    print(f"\n--- 10-Class Classification (Detailed) ---")
    acc_10 = metrics.accuracy_score(all_targets_10, all_preds_10)
    print(f"Accuracy: {acc_10:.4f}")
    print(metrics.classification_report(all_targets_10, all_preds_10, digits=4))

    # Grade Report (5-class)
    print(f"\n--- 5-Grade Classification (KL0-KL4) ---")
    acc_grade = metrics.accuracy_score(all_targets_grade, all_preds_grade)
    print(f"Accuracy: {acc_grade:.4f}")
    print(metrics.classification_report(all_targets_grade, all_preds_grade, digits=4))

    # --- Generate Plots ---
    print("\nGenerating plots...")

    # Confusion Matrix - 10 Class
    cm_10 = metrics.confusion_matrix(all_targets_10, all_preds_10)
    classes_10 = [str(i) for i in range(10)]  # Or use names from config if available
    plot_confusion_matrix(
        cm_10, classes_10, save_dir / "cm_10_class.png", "Confusion Matrix (10-Class)"
    )

    # Confusion Matrix - 5 Grade
    cm_grade = metrics.confusion_matrix(all_targets_grade, all_preds_grade)
    classes_grade = ["KL0", "KL1", "KL2", "KL3", "KL4"]
    plot_confusion_matrix(
        cm_grade,
        classes_grade,
        save_dir / "cm_5_grade.png",
        "Confusion Matrix (KL Grade)",
    )

    # ROC Curve - 10 Class
    plot_roc_curve(
        all_targets_10,
        all_probs_10,
        10,
        save_dir / "roc_10_class.png",
        "ROC Curve (10-Class)",
    )

    print(f"\nEvaluation complete. Plots saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Default paths from training wrapper
    parser.add_argument("--img_dir", default="processed/knee/images")
    parser.add_argument("--knee_label_dir", default="processed/knee/labels")
    parser.add_argument("--lesion_label_dir", default="processed/knee/labels")
    parser.add_argument(
        "--split_file", default="processed/splits/knee_10_class/val.txt"
    )

    parser.add_argument(
        "--model_path",
        default="runs/kiocmil_cada/best_model.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument(
        "--save_dir", default="analysis/plots", help="Directory to save plots"
    )
    args = parser.parse_args()
    evaluate(args)
