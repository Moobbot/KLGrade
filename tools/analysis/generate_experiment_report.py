import json
from pathlib import Path
import glob
from datetime import datetime
import os

def to_markdown_table(records, headers):
    if not records:
        return ""
    
    # Create header row
    md = "| " + " | ".join(headers) + " |\n"
    md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    
    # Create rows
    for rec in records:
        row = []
        for h in headers:
            val = rec.get(h, "-")
            if isinstance(val, float):
                row.append(f"{val:.4f}")
            else:
                row.append(str(val))
        md += "| " + " | ".join(row) + " |\n"
    return md

def generate_report():
    runs_dir = Path("runs/classify")
    json_files = sorted(glob.glob(str(runs_dir / "*/evaluation/metrics.json")))
    
    records = []
    
    for jf in json_files:
        path = Path(jf)
        exp_name = path.parent.parent.name
        
        with open(path, 'r') as f:
            data = json.load(f)
            
        # Determine num classes
        if "10class" in exp_name: num_classes = 10
        elif "8class" in exp_name: num_classes = 8
        elif "5class" in exp_name: num_classes = 5
        elif "4class" in exp_name: num_classes = 4
        else: num_classes = 0
            
        # Basic Metrics
        rec = {
            "Experiment": exp_name,
            "Num_Classes": num_classes,
            "Accuracy": data.get("accuracy", 0),
            "Kappa": data.get("kappa", 0),
            "AUC_Macro": data.get("auc_macro", 0),
            "F1_Macro": data.get("f1_macro", 0),
            "Precision_Macro": data.get("precision_macro", 0),
            "Recall_Macro": data.get("recall_macro", 0),
        }
            
        # Derived Metrics
        if "derived_metrics" in data and data["derived_metrics"]:
            dm = data["derived_metrics"]
            rec["Derived_Acc"] = dm.get("accuracy", 0)
            rec["Derived_Kappa"] = dm.get("kappa", 0)
            rec["Derived_F1"] = dm.get("f1_macro", 0)
            rec["Derived_Prec"] = dm.get("precision_macro", 0)
            rec["Derived_Rec"] = dm.get("recall_macro", 0)
        else:
            rec["Derived_Acc"] = "-"
            rec["Derived_Kappa"] = "-"
            rec["Derived_F1"] = "-"
            rec["Derived_Prec"] = "-"
            rec["Derived_Rec"] = "-"
            
        records.append(rec)
        
    # Sort: Num_Classes DESC, Experiment ASC
    records.sort(key=lambda x: (-x["Num_Classes"], x["Experiment"]))
    
    # Generate Markdown
    md = "# KIOCMIL CADA Experiment Report\n\n"
    md += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    headers = [
        "Experiment", "Num_Classes", "Accuracy", "Kappa", "AUC_Macro", "F1_Macro", "Precision_Macro", "Recall_Macro",
        "Derived_Acc", "Derived_Kappa", "Derived_F1", "Derived_Prec", "Derived_Rec"
    ]
    
    md += "## Summary Table\n\n"
    md += to_markdown_table(records, headers)
    
    md += "\n\n## Analysis by Class Configuration\n"
    
    unique_classes = sorted(set(r["Num_Classes"] for r in records), reverse=True)
    
    for nc in unique_classes:
        md += f"\n### {nc}-Class Experiments\n\n"
        sub_records = [r for r in records if r["Num_Classes"] == nc]
        
        # Determine best model based on Main Accuracy
        best_model = max(sub_records, key=lambda x: x["Accuracy"]) if sub_records else None
        
        md += to_markdown_table(sub_records, headers)
        
        if best_model:
            md += f"\n**🏆 Best {nc}-Class Model:** `{best_model['Experiment']}` (Acc: {best_model['Accuracy']:.4f})\n"

    # ---------------------------------------------------------
    # YOLO Detection Metrics
    # ---------------------------------------------------------
    md += "\n\n## YOLO Detection Performance (Knee/Lesion Localization)\n\n"
    md += "Performance of the YOLO models used to generate the bounding boxes for the above experiments.\n\n"
    
    yolo_records = []
    detect_dirs = glob.glob("runs/detect/runs/detect/*")
    
    for d_dir in detect_dirs:
        exp_name = os.path.basename(d_dir)
        csv_path = os.path.join(d_dir, "results.csv")
        
        if os.path.exists(csv_path):
            try:
                # Simple CSV parser to avoid pandas dependency
                with open(csv_path, "r") as f:
                    lines = f.readlines()
                    
                if len(lines) < 2:
                    continue
                    
                headers = [h.strip() for h in lines[0].strip().split(",")]
                # Indices
                try:
                    idx_prec = headers.index("metrics/precision(B)")
                    idx_rec = headers.index("metrics/recall(B)")
                    idx_map50 = headers.index("metrics/mAP50(B)")
                    idx_map5095 = headers.index("metrics/mAP50-95(B)")
                except ValueError:
                    continue # Column not found
                
                best_map50 = -1.0
                best_row = None
                
                for line in lines[1:]:
                    parts = [p.strip() for p in line.strip().split(",")]
                    if len(parts) != len(headers):
                        continue
                    try:
                        map50 = float(parts[idx_map50])
                        if map50 > best_map50:
                            best_map50 = map50
                            best_row = parts
                    except ValueError:
                        continue
                
                if best_row:
                    yolo_records.append({
                        "Model": exp_name,
                        "Precision": float(best_row[idx_prec]),
                        "Recall": float(best_row[idx_rec]),
                        "mAP50": float(best_row[idx_map50]),
                        "mAP50-95": float(best_row[idx_map5095])
                    })
                    
            except Exception as e:
                print(f"Error parsing YOLO {exp_name}: {e}")
                
    if yolo_records:
        yolo_headers = ["Model", "mAP50", "mAP50-95", "Precision", "Recall"]
        
        # Sort by mAP50 desc
        yolo_records.sort(key=lambda x: -x["mAP50"])
        
        # Build Table
        # Header
        md += "| " + " | ".join(yolo_headers) + " |\n"
        md += "| " + " | ".join(["---"] * len(yolo_headers)) + " |\n"
        
        # Rows
        for rec in yolo_records:
            row = [
                rec["Model"],
                f"{rec['mAP50']:.4f}",
                f"{rec['mAP50-95']:.4f}",
                f"{rec['Precision']:.4f}",
                f"{rec['Recall']:.4f}"
            ]
            md += "| " + " | ".join(row) + " |\n"
    else:
        md += "No YOLO results found in `runs/detect/runs/detect/`.\n"

    # Save Report
    with open("docs/EXPERIMENT_REPORT_FULL.md", "w") as f:
        f.write(md)
        
    print("Report generated at docs/EXPERIMENT_REPORT_FULL.md")
    
    # Print table to verify
    for r in records:
        print(f"{r['Experiment']}: {r['Accuracy']:.4f}")

if __name__ == "__main__":
    generate_report()
