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
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def plot_roc_curve(
    targets, probs, n_classes, class_names, save_path, title="ROC Curve"
):
    # Targets should be one-hot for ROC
    # Probs should be (N, n_classes)
    targets_one_hot = np.eye(n_classes)[targets]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(10, 8))

    for i in range(n_classes):
        label = class_names[i] if class_names else f"Class {i}"
        fpr[i], tpr[i], _ = metrics.roc_curve(targets_one_hot[:, i], probs[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f"{label} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved ROC curve to {save_path}")


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Evaluating with {args.num_classes} classes")

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

    # Load checkpoint first to check class count
    print(f"Loading checkpoint from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = (
        checkpoint["model_state_dict"]
        if "model_state_dict" in checkpoint
        else checkpoint
    )

    # Auto-detect num_classes from head_10.weight
    saved_num_classes = state_dict["head_10.weight"].shape[0]
    print(
        f"Detected {saved_num_classes} classes in checkpoint (Override requested: {args.num_classes})"
    )

    # Initialize model with DETECTED classes to match weights
    model = KiocmilModelCADA(
        backbone_name="yolo11l",
        num_classes=saved_num_classes,
        feature_dim=256,
        num_deformable_points=4,
        num_context_scales=3,
        use_positional_encoding=True,
        dropout=0.1,
    ).to(device)

    model.load_state_dict(state_dict)

    # Use the detected classes for evaluation loop to avoid shape errors
    eval_num_classes = saved_num_classes

    model.eval()

    # Metrics containers
    all_preds_main = []
    all_targets_main = []
    all_probs_main = []

    # Derived metrics containers (if applicable)
    all_preds_grade = []
    all_targets_grade = []

    print("Running inference...")
    with torch.no_grad():
        for batch_data in tqdm(loader):
            if not batch_data or not batch_data[0].get("knees"):
                continue

            # Forward
            outputs = model(batch_data)

            # --- Main Metrics ---
            # Output key depends on num_classes in model definition, but typically is "logits_10"
            # However, if we change num_classes, the model output key MIGHT change if the model code does.
            # Looking at KiocmilModelCADA, it returns "logits_10" but the size matches num_classes.
            # Let's trust "logits_10" is the main head regardless of name, or check keys.
            logits_main = outputs["logits_10"]

            probs_main = torch.softmax(logits_main, dim=1)
            preds_main = torch.argmax(logits_main, dim=1)

            # Get targets
            labels = [item["label"] for item in batch_data]
            targets_main = torch.tensor(labels, device=device).long()

            all_preds_main.extend(preds_main.cpu().numpy())
            all_targets_main.extend(targets_main.cpu().numpy())
            all_probs_main.extend(probs_main.cpu().numpy())

            # --- Derived Grade Metrics ---
            if args.num_classes in [8, 10]:
                # 10-class (0-9) -> 5-Grade (0-4)
                # 8-class (0-7) -> 4-Grade (0-3) [which represents KL1-4]
                preds_grade = preds_main // 2
                targets_grade = targets_main // 2

                all_preds_grade.extend(preds_grade.cpu().numpy())
                all_targets_grade.extend(targets_grade.cpu().numpy())

    # Convert to arrays
    all_targets_main = np.array(all_targets_main)
    all_preds_main = np.array(all_preds_main)
    all_probs_main = np.array(all_probs_main)

    if args.num_classes in [8, 10]:
        all_targets_grade = np.array(all_targets_grade)
        all_preds_grade = np.array(all_preds_grade)

    # --- Print Text Reports ---
    print("\n" + "=" * 60)
    print(f" EVALUATION RESULTS (KIOCMIL CADA - {args.num_classes} Classes)")
    print("=" * 60)

    # Determine Class Names
    if args.num_classes == 10:
        class_names_main = [
            "KL0-a",
            "KL0-b",
            "KL1-a",
            "KL1-b",
            "KL2-a",
            "KL2-b",
            "KL3-a",
            "KL3-b",
            "KL4-a",
            "KL4-b",
        ]
        grade_names = ["KL0", "KL1", "KL2", "KL3", "KL4"]
    elif args.num_classes == 8:
        class_names_main = [
            "KL1-a",
            "KL1-b",
            "KL2-a",
            "KL2-b",
            "KL3-a",
            "KL3-b",
            "KL4-a",
            "KL4-b",
        ]
        grade_names = ["KL1", "KL2", "KL3", "KL4"]
    elif args.num_classes == 5:
        class_names_main = ["KL0", "KL1", "KL2", "KL3", "KL4"]
        grade_names = None
    elif args.num_classes == 4:
        class_names_main = ["KL1", "KL2", "KL3", "KL4"]
        grade_names = None
    else:
        class_names_main = [str(i) for i in range(args.num_classes)]
        grade_names = None

    # Main Report
    print(f"\n--- Detailed Classification ({args.num_classes}-Class) ---")

    if len(all_targets_main) == 0:
        print("⚠️  No samples evaluated! Check dataset paths or knee detection.")
        return

    acc_main = metrics.accuracy_score(all_targets_main, all_preds_main)
    print(f"Accuracy: {acc_main:.4f}")

    # Generate list of all possible label indices
    main_labels = list(range(args.num_classes))
    print(
        metrics.classification_report(
            all_targets_main,
            all_preds_main,
            labels=main_labels,
            target_names=class_names_main,
            digits=4,
            zero_division=0,
        )
    )

    # Derived Report
    if args.num_classes in [8, 10]:
        print(f"\n--- Aggregate Grade Classification ---")
        acc_grade = metrics.accuracy_score(all_targets_grade, all_preds_grade)
        print(f"Accuracy: {acc_grade:.4f}")

        # Determine number of grade classes
        num_grade_classes = 5 if args.num_classes == 10 else 4
        grade_labels = list(range(num_grade_classes))

        print(
            metrics.classification_report(
                all_targets_grade,
                all_preds_grade,
                labels=grade_labels,
                target_names=grade_names,
                digits=4,
                zero_division=0,
            )
        )

    # --- Generate Plots ---
    print("\nGenerating plots...")

    # Confusion Matrix - Main
    cm_main = metrics.confusion_matrix(all_targets_main, all_preds_main)
    plot_confusion_matrix(
        cm_main,
        class_names_main,
        save_dir / f"cm_{args.num_classes}_class.png",
        f"Confusion Matrix ({args.num_classes}-Class)",
    )

    # ROC Curve - Main
    plot_roc_curve(
        all_targets_main,
        all_probs_main,
        args.num_classes,
        class_names_main,
        save_dir / f"roc_{args.num_classes}_class.png",
        f"ROC Curve ({args.num_classes}-Class)",
    )

    # Confusion Matrix - Grade (Derived)
    if args.num_classes in [8, 10]:
        cm_grade = metrics.confusion_matrix(all_targets_grade, all_preds_grade)
        plot_confusion_matrix(
            cm_grade,
            grade_names,
            save_dir / "cm_grade_derived.png",
            "Confusion Matrix (Derived Grade)",
        )

    print(f"\nEvaluation complete. Plots saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Default paths from training wrapper
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--knee_label_dir", required=True)
    parser.add_argument("--lesion_label_dir", required=True)
    parser.add_argument("--split_file", required=True)

    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument(
        "--save_dir", default="analysis/plots", help="Directory to save plots"
    )
    args = parser.parse_args()
    evaluate(args)
