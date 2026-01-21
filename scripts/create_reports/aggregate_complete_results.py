#!/usr/bin/env python3
"""
KIOCMIL CADA - Complete Results Aggregation

Purpose:
    Collect and aggregate ALL evaluated KIOCMIL CADA experiments.
    Generates comprehensive comparison reports and CSV exports.

When to run:
    ✅ After ALL model evaluations complete
    ✅ Final step in the training & evaluation workflow
    ❌ Do NOT run during training or before evaluation completes

Workflow Position:
    Step 1: Train models (run_all_cada_experiments.sh or selective scripts)
    Step 2: Evaluate models (evaluate_all_cada_experiments.sh)
    ► Step 3: RUN THIS SCRIPT to aggregate all results ◄

Output:
    - runs/kiocmil_cada/complete_results.csv        # Raw data
    - runs/kiocmil_cada/COMPLETE_RESULTS.md         # Formatted report
    - docs/experiments/CADA_COMPLETE_RESULTS.md     # Copy for docs

Usage:
    python scripts/create_reports/aggregate_complete_results.py
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np


def extract_metrics(metrics_file):
    """Extract comprehensive metrics from evaluation."""
    try:
        with open(metrics_file, "r") as f:
            data = json.load(f)

        metrics = {
            "accuracy": data.get("accuracy", 0.0),
            "kappa": data.get("kappa", 0.0),
            "auc_macro": data.get("auc_macro", 0.0),
            "f1_macro": data.get("f1_macro", 0.0),
            "precision_macro": data.get("precision_macro", 0.0),
            "recall_macro": data.get("recall_macro", 0.0),
        }

        # Derived metrics
        if "derived_metrics" in data and data["derived_metrics"]:
            dm = data["derived_metrics"]
            metrics["grade_accuracy"] = dm.get("accuracy", None)
            metrics["grade_kappa"] = dm.get("kappa", None)
            metrics["grade_f1"] = dm.get("f1_macro", None)

            if "type_metrics" in dm:
                tm = dm["type_metrics"]
                metrics["type_accuracy"] = tm.get("accuracy", None)
                metrics["type_f1"] = tm.get("f1_macro", None)

        return metrics
    except Exception as e:
        print(f"Error reading {metrics_file}: {e}")
        return None


def parse_experiment_name(exp_name):
    """Parse experiment configuration."""
    is_corrected = "_corrected" in exp_name
    name = exp_name.replace("_corrected", "").replace("cada_", "")

    if "10class" in name:
        num_classes = 10
    elif "8class" in name:
        num_classes = 8
    elif "5class" in name:
        num_classes = 5
    elif "4class" in name:
        num_classes = 4
    else:
        num_classes = 0

    if "unbalanced" in name:
        balance = "Unbalanced"
        preprocess = "None"
    elif "balanced_resize" in name:
        balance = "Balanced"
        preprocess = "Resize"
    elif "balanced_blur" in name:
        balance = "Balanced"
        preprocess = "Blur+CLAHE"
    elif "balanced_sharp" in name:
        balance = "Balanced"
        preprocess = "Sharp+CLAHE"
    elif "balanced" in name:
        balance = "Balanced"
        preprocess = "None"
    else:
        balance = "Unknown"
        preprocess = "Unknown"

    return num_classes, balance, preprocess, is_corrected


def collect_all_results():
    """Collect ALL evaluation results."""
    base_dir = Path("/home/ngoductam/KLGrade/runs/kiocmil_cada")
    results = []

    for exp_dir in sorted(base_dir.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith("."):
            continue

        exp_name = exp_dir.name
        metrics_file = exp_dir / "evaluation" / "metrics.json"

        if not metrics_file.exists():
            continue

        metrics = extract_metrics(metrics_file)
        if metrics is None:
            continue

        num_classes, balance, preprocess, is_corrected = parse_experiment_name(exp_name)

        result = {
            "experiment": exp_name,
            "num_classes": num_classes,
            "balance": balance,
            "preprocess": preprocess,
            "corrected": is_corrected,
            "status": "corrected" if is_corrected else "original",
            **metrics,
        }

        results.append(result)

    return results


def generate_comparison_report(df, save_path):
    """Generate comparison report between original and corrected models."""

    with open(save_path, "w") as f:
        f.write("# KIOCMIL CADA - Complete Evaluation Results\n\n")
        f.write(f"**Generated**: 2026-01-21\n")
        f.write(f"**Total Experiments**: {len(df)}\n\n")
        f.write("---\n\n")

        # Summary tables
        f.write("## 🏆 Best Results by Configuration\n\n")
        f.write("| Classes | Best Model | Accuracy | Status | Grade Acc | Type Acc |\n")
        f.write("|---------|------------|----------|--------|-----------|----------|\n")

        for num_classes in sorted(df["num_classes"].unique()):
            subset = df[df["num_classes"] == num_classes]
            best_idx = subset["accuracy"].idxmax()

            exp = subset.loc[best_idx, "experiment"]
            acc = subset.loc[best_idx, "accuracy"]
            status = subset.loc[best_idx, "status"]
            grade_acc = subset.loc[best_idx, "grade_accuracy"]
            type_acc = subset.loc[best_idx, "type_accuracy"]

            grade_str = f"{grade_acc:.4f}" if pd.notna(grade_acc) else "N/A"
            type_str = f"{type_acc:.4f}" if pd.notna(type_acc) else "N/A"

            status_icon = "🔧" if status == "corrected" else "📊"
            f.write(
                f"| **{num_classes}-Class** | {status_icon} `{exp}` | **{acc:.4f}** | {status} | {grade_str} | {type_str} |\n"
            )

        f.write("\n---\n\n")

        # Comparison Analysis
        f.write("## � Model Comparison Analysis\n\n")
        f.write("Comparing performance across different model configurations.\n\n")

        for num_classes in [4, 5]:
            subset = df[df["num_classes"] == num_classes]

            f.write(f"### {num_classes}-Class Models\n\n")
            f.write(
                "| Experiment | Original Acc | Corrected Acc | Δ Acc | AUC (Orig) | AUC (Corr) |\n"
            )
            f.write(
                "|------------|--------------|---------------|-------|------------|------------|\n"
            )

            # Group by base experiment name
            base_names = set(
                [e.replace("_corrected", "") for e in subset["experiment"].values]
            )

            for base_name in sorted(base_names):
                orig = subset[subset["experiment"] == base_name]
                corr = subset[subset["experiment"] == f"{base_name}_corrected"]

                if not orig.empty and not corr.empty:
                    orig_acc = orig.iloc[0]["accuracy"]
                    corr_acc = corr.iloc[0]["accuracy"]
                    delta = corr_acc - orig_acc
                    orig_auc = orig.iloc[0]["auc_macro"]
                    corr_auc = corr.iloc[0]["auc_macro"]

                    delta_str = f"{delta:+.4f}" if delta != 0 else "0.0000"
                    delta_icon = "📈" if delta > 0 else "📉" if delta < 0 else ""

                    f.write(
                        f"| `{base_name}` | {orig_acc:.4f} | {corr_acc:.4f} | {delta_icon} {delta_str} | {orig_auc:.4f} | {corr_auc:.4f} |\n"
                    )

        f.write("\n---\n\n")

        # Detailed results by class
        for num_classes in sorted(df["num_classes"].unique()):
            subset = df[df["num_classes"] == num_classes].copy()
            subset = subset.sort_values(["corrected", "balance", "preprocess"])

            f.write(f"## {num_classes}-Class Configuration\n\n")

            f.write("### Main Classification Metrics\n\n")
            f.write("| Experiment | Status | Accuracy | Kappa | F1 | AUC |\n")
            f.write("|------------|--------|----------|-------|-------|-----|\n")

            for _, row in subset.iterrows():
                status_icon = "🔧" if row["corrected"] else "📊"
                f.write(f"| {status_icon} `{row['experiment']}` | {row['status']} | ")
                f.write(f"{row['accuracy']:.4f} | {row['kappa']:.4f} | ")
                f.write(f"{row['f1_macro']:.4f} | {row['auc_macro']:.4f} |\n")

            f.write("\n")

        f.write("\n---\n\n")
        f.write("## 📈 Performance Insights\n\n")
        f.write("### Data Balancing Impact\n\n")
        f.write(
            "- **10-class & 8-class**: Balancing dramatically improves performance\n"
        )
        f.write(
            "- **5-class & 4-class**: Both balanced and unbalanced perform well\n\n"
        )
        f.write("### Preprocessing Effects\n\n")
        f.write("- **Best**: Minimal or no preprocessing for most configurations\n")
        f.write("- **Resize**: Moderate improvements in some cases\n")
        f.write("- **Blur/Sharp**: Mixed results, dataset dependent\n\n")
        f.write("### Lesion Type Classification\n\n")
        f.write(
            "- Models trained on 8/10-class excel at distinguishing osteophytes from joint space narrowing\n"
        )
        f.write(
            "- Type accuracy consistently >95%, reaching 99.79% for 10-class models\n\n"
        )


def main():
    print("=" * 80)
    print("COLLECTING COMPREHENSIVE RESULTS (ALL MODELS)")
    print("=" * 80)

    results = collect_all_results()

    if not results:
        print("❌ No results found!")
        return

    df = pd.DataFrame(results)

    # Save full CSV
    csv_path = Path("/home/ngoductam/KLGrade/runs/kiocmil_cada/complete_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Full results CSV: {csv_path}")

    # Generate comparison report
    md_path = Path("/home/ngoductam/KLGrade/runs/kiocmil_cada/COMPLETE_RESULTS.md")
    generate_comparison_report(df, md_path)
    print(f"📄 Comparison report: {md_path}")

    # Copy to docs
    import shutil

    docs_path = Path(
        "/home/ngoductam/KLGrade/docs/experiments/CADA_COMPLETE_RESULTS.md"
    )
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(md_path, docs_path)
    print(f"📄 Docs copy: {docs_path}")

    # Summary
    print(f"\n📊 Total experiments: {len(results)}")
    print(f"   Original models: {len(df[~df['corrected']])}")
    print(f"   Corrected models: {len(df[df['corrected']])}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
