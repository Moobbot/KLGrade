#!/usr/bin/env python3
"""
Script to summarize all training experiments
Reads results from runs/detect/ and creates a comparison table
"""

import pandas as pd
import yaml
from pathlib import Path
import json


def summarize_experiment(exp_dir):
    """Extract key metrics from a training run"""
    exp_path = Path(exp_dir)

    if not exp_path.exists():
        return None

    summary = {"experiment": exp_path.name, "path": str(exp_path)}

    # Read config
    args_file = exp_path / "args.yaml"
    if args_file.exists():
        with open(args_file) as f:
            args = yaml.safe_load(f)
            summary["dataset"] = args.get("data", "unknown")
            summary["epochs"] = args.get("epochs", 0)
            summary["batch"] = args.get("batch", 0)
            summary["imgsz"] = args.get("imgsz", 0)

    # Read final metrics
    results_file = exp_path / "results.csv"
    if results_file.exists():
        df = pd.read_csv(results_file)
        last_row = df.iloc[-1]

        summary["final_mAP50"] = f"{last_row.get('metrics/mAP50(B)', 0):.4f}"
        summary["final_mAP50-95"] = f"{last_row.get('metrics/mAP50-95(B)', 0):.4f}"
        summary["final_precision"] = f"{last_row.get('metrics/precision(B)', 0):.4f}"
        summary["final_recall"] = f"{last_row.get('metrics/recall(B)', 0):.4f}"

        # Best mAP50
        best_idx = df["metrics/mAP50(B)"].idxmax()
        best_row = df.iloc[best_idx]
        summary["best_mAP50"] = f"{best_row.get('metrics/mAP50(B)', 0):.4f}"
        summary["best_epoch"] = int(best_row.get("epoch", 0))

    return summary


def main():
    runs_dir = Path("runs/detect")

    if not runs_dir.exists():
        print(f"❌ Directory {runs_dir} not found!")
        return

    # Find all experiment directories
    experiments = [d for d in runs_dir.iterdir() if d.is_dir()]

    if not experiments:
        print("❌ No experiments found!")
        return

    print(f"📊 Found {len(experiments)} experiments\n")

    # Collect summaries
    summaries = []
    for exp_dir in sorted(experiments):
        summary = summarize_experiment(exp_dir)
        if summary:
            summaries.append(summary)

    if not summaries:
        print("❌ No valid experiment data found!")
        return

    # Create DataFrame
    df = pd.DataFrame(summaries)

    # Print summary table
    print("=" * 120)
    print("TRAINING EXPERIMENTS SUMMARY")
    print("=" * 120)
    print(df.to_string(index=False))
    print("=" * 120)

    # Save to file
    output_file = "docs/TRAINING_SUMMARY.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Training Experiments Summary\n\n")
        f.write(f"Total Experiments: {len(summaries)}\n\n")
        f.write("## Results Table\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Experiment Details\n\n")

        for summary in summaries:
            f.write(f"### {summary['experiment']}\n\n")
            f.write(f"- **Dataset**: {summary.get('dataset', 'N/A')}\n")
            f.write(f"- **Epochs**: {summary.get('epochs', 'N/A')}\n")
            f.write(
                f"- **Best mAP50**: {summary.get('best_mAP50', 'N/A')} (epoch {summary.get('best_epoch', 'N/A')})\n"
            )
            f.write(f"- **Final mAP50**: {summary.get('final_mAP50', 'N/A')}\n")
            f.write(f"- **Final mAP50-95**: {summary.get('final_mAP50-95', 'N/A')}\n")
            f.write(f"- **Path**: `{summary['path']}`\n\n")

    print(f"\n✅ Summary saved to: {output_file}")

    # Save JSON for programmatic access
    json_file = "docs/TRAINING_SUMMARY.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print(f"✅ JSON saved to: {json_file}")


if __name__ == "__main__":
    main()
