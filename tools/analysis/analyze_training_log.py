import re
import pandas as pd
import json
import os

LOG_FILE = "/home/ngoductam/KLGrade/training_all.log"
OUTPUT_CSV = "training_summary.csv"
OUTPUT_JSON = "training_details.json"


def parse_log(log_path):
    if not os.path.exists(log_path):
        print(f"Error: Log file not found at {log_path}")
        return []

    runs = []
    current_run = {}

    # Regex patterns
    # pattern for engine/trainer line to extract config
    trainer_pattern = re.compile(r"engine/trainer:\s+(.*)")
    # pattern for results line: "all" followed by numbers
    # Example: all        337        626      0.427     0.0351     0.0103    0.00441
    results_pattern = re.compile(
        r"^\s*all\s+(\d+)\s+(\d+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)"
    )

    # pattern for "Starting training for X epochs..."
    start_pattern = re.compile(r"Starting training for (\d+) epochs")

    # pattern for model loading
    model_pattern = re.compile(r"Loading (\S+) model")

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # New run detection (start of a new block usually has 'engine/trainer' or 'Starting training')
        # However, 'engine/trainer' contains the most config info, so we use it as a primary anchor if available.

        if "engine/trainer:" in line:
            # If we were processing a run, check if it has results before saving,
            # or if it was just a setup phase.
            if current_run and current_run.get("metrics"):
                runs.append(current_run)

            # Start a new run object
            current_run = {
                "config": {},
                "metrics": [],
                "epochs": 0,
                "dataset": "Unknown",
                "model": "Unknown",
            }

            match = trainer_pattern.search(line)
            if match:
                config_str = match.group(1)
                # Parse config string "key=value, key=value"
                config_items = [item.strip() for item in config_str.split(",")]
                for item in config_items:
                    if "=" in item:
                        k, v = item.split("=", 1)
                        current_run["config"][k] = v
                        if k == "data":
                            current_run["dataset"] = v
                        if k == "model":
                            current_run["model"] = v
                        if k == "epochs":
                            current_run["epochs"] = int(v) if v.isdigit() else v

        # specific epoch results
        match_results = results_pattern.search(line)
        if match_results and current_run:
            # images, instances, P, R, mAP50, mAP50-95
            images = int(match_results.group(1))
            instances = int(match_results.group(2))
            p = float(match_results.group(3))
            r = float(match_results.group(4))
            map50 = float(match_results.group(5))
            map50_95 = float(match_results.group(6))

            current_run["metrics"].append(
                {
                    "images": images,
                    "instances": instances,
                    "P": p,
                    "R": r,
                    "mAP50": map50,
                    "mAP50-95": map50_95,
                }
            )

    # Append the last run if it exists
    if current_run and current_run.get("metrics"):
        runs.append(current_run)

    return runs


def summarize_runs(runs):
    summary_data = []

    for i, run in enumerate(runs):
        metrics = run["metrics"]
        if not metrics:
            continue

        # Get best mAP50-95
        best_map50_95 = max(m["mAP50-95"] for m in metrics)
        best_epoch_idx = next(
            i for i, m in enumerate(metrics) if m["mAP50-95"] == best_map50_95
        )
        # Epochs are 1-indexed in display, but list is 0-indexed
        best_epoch = best_epoch_idx + 1

        # Last epoch metrics
        last_metrics = metrics[-1]

        summary_data.append(
            {
                "Run_ID": i + 1,
                "Dataset": run.get("dataset", "Unknown"),
                "Model": run.get("model", "Unknown"),
                "Total_Epochs_Planned": run.get("epochs", "Unknown"),
                "Epochs_Trained": len(metrics),
                "Best_mAP50_95": best_map50_95,
                "Best_Epoch": best_epoch,
                "Last_mAP50_95": last_metrics["mAP50-95"],
                "Last_mAP50": last_metrics["mAP50"],
                "Last_Precision": last_metrics["P"],
                "Last_Recall": last_metrics["R"],
            }
        )

    return pd.DataFrame(summary_data)


def main():
    print(f"Parsing {LOG_FILE}...")
    runs = parse_log(LOG_FILE)
    print(f"Found {len(runs)} training runs.")

    if not runs:
        print("No runs found. Exiting.")
        return

    # Save detailed JSON
    print(f"Saving details to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(runs, f, indent=4)

    # Generate and save CSV summary
    print("Generating summary...")
    df_summary = summarize_runs(runs)

    if not df_summary.empty:
        print(f"Saving summary to {OUTPUT_CSV}...")
        df_summary.to_csv(OUTPUT_CSV, index=False)
        print("\nSummary Preview:")
        print(df_summary.to_markdown(index=False))
    else:
        print("No valid metrics found to summarize.")


if __name__ == "__main__":
    main()
