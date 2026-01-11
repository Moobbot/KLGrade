#!/usr/bin/env python3
"""
Evaluate All Models

Scans the runs/ directory for trained YOLO models and evaluates them.
Attempts to automatically determine the correct dataset configuration.
"""

import os
from pathlib import Path
import yaml
import subprocess
import argparse
import sys

def get_best_pt_files(runs_dir: Path):
    """Find all best.pt files in runs directory."""
    return list(runs_dir.glob("**/weights/best.pt"))

def parse_yaml(yaml_path: Path):
    """Parse a YAML file."""
    try:
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error reading {yaml_path}: {e}")
        return None

def find_dataset_config_from_args(model_dir: Path, processed_dir: Path):
    """
    Attempt to find dataset configuration from args.yaml or directory name.
    Returns (dataset_dir, nc, names) tuple or None.
    """
    args_path = model_dir / "args.yaml"
    experiment_name = model_dir.name
    
    # 1. Try to find dataset matching experiment name in processed/matches
    # This is a heuristic: if we named the experiment 'knee_4_class', maybe the dataset is 'splits/knee_4_class'
    
    # Common locations for splits
    potential_split_dirs = [
        processed_dir / "splits" / experiment_name,
        processed_dir / "enhanced" / "splits" / "dataset_v0" / experiment_name,
        processed_dir / "enhanced" / "splits" / "knee_10_class" / experiment_name,
        processed_dir / "enhanced" / "splits" / "knee_5_class" / experiment_name,
    ]
    
    for split_dir in potential_split_dirs:
        if split_dir.exists() and (split_dir / "val.txt").exists():
            print(f"  Found matching split dir: {split_dir}")
            # We still need nc and names. Try to find a config in the run dir or assume defaults?
            # Better to get nc/names from args.yaml
            break
    else:
        split_dir = None

    if not args_path.exists():
        print(f"  ⚠️ No args.yaml found in {model_dir}")
        return None

    args = parse_yaml(args_path)
    if not args:
        return None

    data_config_path = args.get('data')
    if not data_config_path:
        print("  ⚠️ No data config in args.yaml")
        return None
        
    # Fix relative paths from args.yaml (which might be relative to where training started)
    # We assume training started from project root
    project_root = Path.cwd() 
    full_data_config_path = project_root / data_config_path
    
    if not full_data_config_path.exists():
        print(f"  ⚠️ Config file not found: {full_data_config_path}")
        # Try to find it in configs/ dir if it's just a filename
        if (project_root / "configs" / Path(data_config_path).name).exists():
           full_data_config_path = project_root / "configs" / Path(data_config_path).name
           print(f"  Found config in configs/: {full_data_config_path}")
        else:
           return None

    data_config = parse_yaml(full_data_config_path)
    if not data_config:
        return None
        
    nc = data_config.get('nc')
    names = data_config.get('names')
    
    # If we found a split dir, use it strictly for the paths
    if split_dir:
        return split_dir, nc, names
        
    # Otherwise, try to infer real paths from the config
    # The config might point to 'processed/splits/...'
    # We need to verify if those paths exist
    
    val_path = data_config.get('val')
    if not val_path:
        return None
        
    # Check if val_path is absolute or relative
    # If it depends on 'path', construct it
    base_path = data_config.get('path', '')
    
    if base_path:
        # data_config['path'] could be absolute or relative
        candidate_path = Path(str(base_path)) / str(val_path)
        if not candidate_path.exists():
             # Try relative to project root
             candidate_path = project_root / str(base_path) / str(val_path)
    else:
        candidate_path = project_root / str(val_path)
        
    if candidate_path.exists():
        # Parent directory of the split file is what we want
        return candidate_path.parent, nc, names
        
    print(f"  Could not resolve valid dataset path from config: {full_data_config_path}")
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate all trained models")
    parser.add_argument("--runs-dir", type=str, default="runs", help="Directory containing runs")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    project_root = Path.cwd()
    runs_dir = project_root / args.runs_dir
    processed_dir = project_root / "processed"
    
    best_weights = get_best_pt_files(runs_dir)
    print(f"Found {len(best_weights)} models.")
    
    success_count = 0
    fail_count = 0
    skipped_count = 0
    
    results_list = []

    for weight_path in sorted(best_weights):
        model_dir = weight_path.parent.parent
        print(f"\nProcessing: {model_dir.name}")
        
        # Check if already evaluated (check for metrics.json)
        eval_output_dir = project_root / "runs/evaluate" / model_dir.name
        metrics_file = eval_output_dir / "metrics.json"
        
        if eval_output_dir.exists() and metrics_file.exists():
             print("  ✅ Already evaluated (metrics.json exists). Skipping re-run.")
             skipped_count += 1
             # Read metrics for summary
             import json
             try:
                 with open(metrics_file, 'r') as f:
                     m = json.load(f)
                     results_list.append({
                         "Model": model_dir.name,
                         "mAP50-95": m.get("map50_95", 0),
                         "mAP50": m.get("map50", 0),
                         "Precision": m.get("precision", 0),
                         "Recall": m.get("recall", 0)
                     })
             except:
                 pass
             continue

        dataset_info = find_dataset_config_from_args(model_dir, processed_dir)
        
        if not dataset_info:
            print("  ❌ Could not determine dataset configuration. Skipping.")
            fail_count += 1
            continue
            
        dataset_dir, nc, names = dataset_info
        
        # Format names list for CLI argument
        names_str = ",".join(str(n) for n in names) if isinstance(names, list) else str(names)
        if isinstance(names, dict):
             names_str = ",".join(names.values())

        cmd = [
            sys.executable, "scripts/evaluate_yolo_standalone.py",
            "--model", str(weight_path),
            "--dataset-dir", str(dataset_dir),
            "--nc", str(nc),
            "--class-names", names_str,
            "--split", "test",
            "--name", model_dir.name
        ]
        
        print(f"  Running evaluation on {dataset_dir.name}...")
        
        if args.dry_run:
            print("  CMD:", " ".join(cmd))
        else:
            try:
                # Run evaluation
                subprocess.run(cmd, check=True)
                success_count += 1
                
                # Read newly created metrics
                import json
                if metrics_file.exists():
                     with open(metrics_file, 'r') as f:
                         m = json.load(f)
                         results_list.append({
                             "Model": model_dir.name,
                             "mAP50-95": m.get("map50_95", 0),
                             "mAP50": m.get("map50", 0),
                             "Precision": m.get("precision", 0),
                             "Recall": m.get("recall", 0)
                         })
            except subprocess.CalledProcessError:
                print("  ❌ Evaluation script failed.")
                fail_count += 1

    # Write summary to file
    summary_file = project_root / "evaluation_summary.txt"
    if results_list:
        # Sort by mAP50-95 descending
        results_list.sort(key=lambda x: x["mAP50-95"], reverse=True)
        
        with open(summary_file, "w") as f:
            f.write(f"{'Model':<40} | {'mAP50-95':<10} | {'mAP50':<10} | {'Precision':<10} | {'Recall':<10}\n")
            f.write("-" * 90 + "\n")
            for r in results_list:
                f.write(f"{r['Model']:<40} | {r['mAP50-95']:.4f}     | {r['mAP50']:.4f}     | {r['Precision']:.4f}     | {r['Recall']:.4f}\n")
        
        print(f"\n📄 Summary written to: {summary_file}")
                
    print("\n" + "="*30)
    print("Summary")
    print("="*30)
    print(f"Total Models: {len(best_weights)}")
    print(f"Computed: {success_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Failed: {fail_count}")

if __name__ == "__main__":
    main()
