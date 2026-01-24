import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import sys
from tqdm import tqdm
import sklearn.metrics as metrics
import numpy as np
import os

# Add src to path FIRST before importing from src
sys.path.append(str(Path(__file__).parent.parent.parent))

# Now import from src modules
from src.datasets.kiocmil_dataset_v3 import KiocmilDatasetV3, collate_kiocmil_v3
from src.datasets.kiocmil_transforms_v2 import get_photometric_transforms
from src.models.kiocmil_model_cada import KiocmilModelCADA
from src.config import PROJECT_ROOT
from src.utils.visualization import (
    plot_confusion_matrix,
    plot_confusion_matrix_normalized,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_metric_curves,
    plot_label_distribution,
)


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
    all_preds_type = []
    all_targets_type = []

    print("Running inference...")
    with torch.no_grad():
        for batch_data in tqdm(loader):
            if not batch_data or not batch_data[0].get("knees"):
                continue

            # Forward
            outputs = model(batch_data)

            # --- Explicit Heads ---
            # The model outputs explicit heads for grade and type in addition to the 10-class head
            # We should evaluate these as well if available

            # 1. Main 10-class Head
            logits_main = outputs["logits_10"]
            probs_main = torch.softmax(logits_main, dim=1)
            preds_main = torch.argmax(logits_main, dim=1)

            # 2. explicit Grade Head (5-class)
            logits_grade_head = outputs.get("logits_grade")
            if logits_grade_head is not None:
                preds_grade_head = torch.argmax(logits_grade_head, dim=1)
                all_preds_grade.extend(preds_grade_head.cpu().numpy())
            else:
                # Fallback to deriving from 10-class if head missing (backward compatibility)
                if args.num_classes in [8, 10]:
                    all_preds_grade.extend((preds_main // 2).cpu().numpy())

            # 3. Explicit Type Head (binary)
            logits_type_head = outputs.get("logits_type")
            # Note: The model code uses BCEWithLogitsLoss for type, so shape is (B, 1) usually?
            # Model definition: self.head_type = nn.Linear(feature_dim, 1)
            # So output is (B, 1) logits.
            if logits_type_head is not None:
                probs_type_head = torch.sigmoid(logits_type_head)
                preds_type_head = (probs_type_head > 0.5).long().squeeze(-1)
                # Need to store this for type metrics
                # We'll use a temporary list or just repurpose the loop
                # Let's add a container for explicit type preds
                all_preds_type.extend(preds_type_head.cpu().numpy())
                # For now, let's keep the logic below simple and just add containers

            # Get targets
            labels = [item["label"] for item in batch_data]
            targets_main = torch.tensor(labels, device=device).long()

            all_preds_main.extend(preds_main.cpu().numpy())
            all_targets_main.extend(targets_main.cpu().numpy())
            all_probs_main.extend(probs_main.cpu().numpy())

            # Targets for Grade and Type
            if args.num_classes in [8, 10]:
                targets_grade = targets_main // 2
                all_targets_grade.extend(targets_grade.cpu().numpy())

                targets_type_derived = targets_main % 2
                all_targets_type.extend(targets_type_derived.cpu().numpy())

    # Convert to arrays
    all_targets_main = np.array(all_targets_main)
    all_preds_main = np.array(all_preds_main)
    all_probs_main = np.array(all_probs_main)

    if args.num_classes in [8, 10]:
        all_targets_grade = np.array(all_targets_grade)
        all_preds_grade = np.array(all_preds_grade)
        all_targets_type = np.array(all_targets_type)
        # If we used explicit head, `all_preds_grade` has correct length.
        # If we fell back, it also has correct length.
        # If all_preds_type is empty, it means the explicit head was not used.
        # In that case, we will derive preds_type from all_preds_main later.

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

    # --- Metrics Calculation ---
    print("\nCalculating metrics...")

    # 1. Main Classification Metrics
    acc_main = metrics.accuracy_score(all_targets_main, all_preds_main)
    kappa_main = metrics.cohen_kappa_score(all_targets_main, all_preds_main)

    # F1, Precision, Recall (Macro & Weighted)
    f1_macro = metrics.f1_score(
        all_targets_main, all_preds_main, average="macro", zero_division=0
    )
    precision_macro = metrics.precision_score(
        all_targets_main, all_preds_main, average="macro", zero_division=0
    )
    recall_macro = metrics.recall_score(
        all_targets_main, all_preds_main, average="macro", zero_division=0
    )

    # AUC (requires probabilities)
    try:
        if args.num_classes == 2:
            auc_macro = metrics.roc_auc_score(all_targets_main, all_probs_main[:, 1])
        else:
            auc_macro = metrics.roc_auc_score(
                all_targets_main, all_probs_main, multi_class="ovr", average="macro"
            )
    except Exception as e:
        print(f"⚠️  Could not calculate AUC: {e}")
        auc_macro = 0.0

    # Classification Report (Dict)
    cls_report_dict = metrics.classification_report(
        all_targets_main,
        all_preds_main,
        labels=list(range(args.num_classes)),
        target_names=class_names_main,
        output_dict=True,
        zero_division=0,
    )

    # Confusion Matrix (List of Lists for JSON)
    cm_main = metrics.confusion_matrix(all_targets_main, all_preds_main)
    cm_list = cm_main.tolist()

    # --- Derived Metrics (Grade Level) ---
    derived_metrics = None
    acc_grade = 0.0  # Access for plots later

    if args.num_classes in [8, 10]:
        print("\nCalculating derived grade metrics...")
        acc_grade = metrics.accuracy_score(all_targets_grade, all_preds_grade)
        kappa_grade = metrics.cohen_kappa_score(all_targets_grade, all_preds_grade)

        # Macro F1, Prec, Recall for grade
        f1_macro_grade = metrics.f1_score(
            all_targets_grade, all_preds_grade, average="macro", zero_division=0
        )
        prec_macro_grade = metrics.precision_score(
            all_targets_grade, all_preds_grade, average="macro", zero_division=0
        )
        rec_macro_grade = metrics.recall_score(
            all_targets_grade, all_preds_grade, average="macro", zero_division=0
        )

        # Aggregate probabilities for Grade AUC
        try:
            num_grades = 5 if args.num_classes == 10 else 4
            probs_grade = np.zeros((len(all_targets_grade), num_grades))
            for g in range(num_grades):
                # Sum prob of class 2*g and 2*g+1 (e.g. 0a+0b -> KL0)
                probs_grade[:, g] = (
                    all_probs_main[:, 2 * g] + all_probs_main[:, 2 * g + 1]
                )

            auc_macro_grade = metrics.roc_auc_score(
                all_targets_grade, probs_grade, multi_class="ovr", average="macro"
            )
        except Exception as e:
            print(f"⚠️  Could not calculate Derived Grade AUC: {e}")
            auc_macro_grade = 0.0

        # Grade Report Dict
        num_grade_classes = 5 if args.num_classes == 10 else 4
        grade_labels = list(range(num_grade_classes))

        grade_report_dict = metrics.classification_report(
            all_targets_grade,
            all_preds_grade,
            labels=grade_labels,
            target_names=grade_names,
            output_dict=True,
            zero_division=0,
        )

        cm_grade_val = metrics.confusion_matrix(all_targets_grade, all_preds_grade)

        derived_metrics = {
            "accuracy": acc_grade,
            "kappa": kappa_grade,
            "auc_macro": auc_macro_grade,
            "f1_macro": f1_macro_grade,
            "precision_macro": prec_macro_grade,
            "recall_macro": rec_macro_grade,
            "classification_report": grade_report_dict,
            "confusion_matrix": cm_grade_val.tolist(),
        }

    # --- Derived Metric: Type (0=a, 1=b) ---
    if args.num_classes in [8, 10]:
        print("\nCalculating derived type (box/compartment) metrics...")
        # 10-class: 0->0(a), 1->1(b), 2->0(a), 3->1(b)... => pred % 2
        # 8-class: 0->0(a), 1->1(b)...
        preds_type = all_preds_main % 2
        targets_type = all_targets_main % 2

        acc_type = metrics.accuracy_score(targets_type, preds_type)
        kappa_type = metrics.cohen_kappa_score(targets_type, preds_type)

        f1_type = metrics.f1_score(
            targets_type, preds_type, average="macro", zero_division=0
        )
        prec_type = metrics.precision_score(
            targets_type, preds_type, average="macro", zero_division=0
        )
        rec_type = metrics.recall_score(
            targets_type, preds_type, average="macro", zero_division=0
        )

        try:
            # Aggregate probs for Type AUC
            # Prob(type=0) = Sum(Prob(class k)) where k%2==0
            # Prob(type=1) = Sum(Prob(class k)) where k%2==1
            probs_type = np.zeros((len(all_probs_main), 2))
            probs_type[:, 0] = np.sum(
                all_probs_main[:, ::2], axis=1
            )  # Sum even columns
            probs_type[:, 1] = np.sum(
                all_probs_main[:, 1::2], axis=1
            )  # Sum odd columns

            auc_type = metrics.roc_auc_score(targets_type, probs_type[:, 1])
        except Exception as e:
            print(f"⚠️  Could not calculate Derived Type AUC: {e}")
            auc_type = 0.0

        type_names = ["Type a (Osteophytes)", "Type b (Joint Space)"]
        type_report_dict = metrics.classification_report(
            targets_type,
            preds_type,
            target_names=type_names,
            output_dict=True,
            zero_division=0,
        )
        cm_type = metrics.confusion_matrix(targets_type, preds_type)

        derived_metrics["type_metrics"] = {
            "accuracy": acc_type,
            "kappa": kappa_type,
            "auc_macro": auc_type,
            "f1_macro": f1_type,
            "classification_report": type_report_dict,
            "confusion_matrix": cm_type.tolist(),
        }

    # --- Construct Results Dictionary ---
    results = {
        "accuracy": acc_main,
        "kappa": kappa_main,
        "auc_macro": auc_macro,
        "f1_macro": f1_macro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "classification_report": cls_report_dict,
        "confusion_matrix": cm_list,
        "derived_metrics": derived_metrics,
    }

    # --- Print & Save Text Report ---
    report_str = "=" * 60 + "\n"
    report_str += f" EVALUATION RESULTS (KIOCMIL CADA - {args.num_classes} Classes)\n"
    report_str += "=" * 60 + "\n\n"

    report_str += f"Accuracy:        {acc_main:.4f}\n"
    report_str += f"Kappa Score:     {kappa_main:.4f}\n"
    report_str += f"AUC (Macro):     {auc_macro:.4f}\n"
    report_str += f"F1 (Macro):      {f1_macro:.4f}\n"

    report_str += "\n--- Detailed Classification Report ---\n"
    report_str += metrics.classification_report(
        all_targets_main,
        all_preds_main,
        labels=list(range(args.num_classes)),
        target_names=class_names_main,
        digits=4,
        zero_division=0,
    )
    report_str += "\n\n--- Confusion Matrix ---\n"
    report_str += str(cm_main)
    report_str += "\n"

    # Append Derived Reports
    if derived_metrics:
        # Grade Report
        report_str += "\n" + "=" * 60 + "\n"
        report_str += f" EXPLICIT GRADE REPORT ({num_grade_classes}-Class)\n"
        report_str += "=" * 60 + "\n\n"
        report_str += f"Accuracy:        {derived_metrics['accuracy']:.4f}\n"
        report_str += f"Kappa Score:     {derived_metrics['kappa']:.4f}\n"

        report_str += "\n--- Detailed Grade Report ---\n"
        # We need to reconstruct the string report for Grade manually or re-run classification_report with string output
        # Re-running for string output
        grade_report_str = metrics.classification_report(
            all_targets_grade,
            all_preds_grade,
            labels=list(range(num_grade_classes)),
            target_names=grade_names,
            digits=4,
            zero_division=0,
        )
        report_str += grade_report_str
        report_str += "\n\n--- Grade Confusion Matrix ---\n"
        report_str += str(np.array(derived_metrics["confusion_matrix"]))
        report_str += "\n"

        # Type Report
        if "type_metrics" in derived_metrics:
            tm = derived_metrics["type_metrics"]
            report_str += "\n" + "=" * 60 + "\n"
            report_str += f" DERIVED TYPE REPORT (a/b)\n"
            report_str += "=" * 60 + "\n\n"
            report_str += f"Accuracy:        {tm['accuracy']:.4f}\n"
            report_str += f"Kappa Score:     {tm['kappa']:.4f}\n"
            report_str += f"AUC:             {tm['auc_macro']:.4f}\n"

            report_str += "\n--- Detailed Type Report ---\n"
            type_report_str = metrics.classification_report(
                targets_type,
                preds_type,
                target_names=type_names,
                digits=4,
                zero_division=0,
            )
            report_str += type_report_str
            report_str += "\n\n--- Type Confusion Matrix ---\n"
            report_str += str(np.array(tm["confusion_matrix"]))
            report_str += "\n"

    # Save Results
    json_path = save_dir / "metrics.json"
    txt_path = save_dir / "metrics.txt"

    import json

    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    with open(txt_path, "w") as f:
        f.write(report_str)

    print(f"saved metrics to {json_path}")
    print(f"saved report to {txt_path}")
    print(report_str)

    # --- Generate Plots ---
    print("\nGenerating plots...")

    # Confusion Matrix - Main
    plot_confusion_matrix(
        cm_main,
        class_names_main,
        save_dir / f"cm_{args.num_classes}_class.png",
        f"Confusion Matrix ({args.num_classes}-Class)",
    )

    # Normalized Confusion Matrix
    plot_confusion_matrix_normalized(
        cm_main,
        class_names_main,
        save_dir / "confusion_matrix_normalized.png",
        f"Normalized Confusion Matrix ({args.num_classes}-Class)",
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

    # Precision-Recall Curve
    plot_precision_recall_curve(
        all_targets_main,
        all_probs_main,
        args.num_classes,
        class_names_main,
        save_dir / "PR_curve.png",
        f"Precision-Recall Curve ({args.num_classes}-Class)",
    )

    # F1, Precision, Recall Curves
    plot_metric_curves(
        all_targets_main,
        all_probs_main,
        all_preds_main,
        args.num_classes,
        class_names_main,
        save_dir,
    )

    # Label Distribution
    plot_label_distribution(all_targets_main, class_names_main, save_dir / "labels.jpg")

    # Confusion Matrix - Grade (Derived)
    if args.num_classes in [8, 10]:
        acc_grade = metrics.accuracy_score(all_targets_grade, all_preds_grade)
        cm_grade = metrics.confusion_matrix(all_targets_grade, all_preds_grade)
        plot_confusion_matrix(
            cm_grade,
            grade_names,
            save_dir / "cm_grade_derived.png",
            f"Confusion Matrix (Derived Grade) - Acc: {acc_grade:.4f}",
        )

        # Confusion Matrix - Type (Derived)
        if "type_metrics" in derived_metrics:
            cm_type = np.array(derived_metrics["type_metrics"]["confusion_matrix"])
            acc_type = derived_metrics["type_metrics"]["accuracy"]
            plot_confusion_matrix(
                cm_type,
                type_names,
                save_dir / "cm_type_derived.png",
                f"Confusion Matrix (Derived Type) - Acc: {acc_type:.4f}",
            )

    print(f"\nEvaluation complete. Results saved to {save_dir}")


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
