import json
from pathlib import Path
import glob
from datetime import datetime

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
    runs_dir = Path("runs/kiocmil_cada")
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

    output_file = "docs/EXPERIMENT_REPORT_FULL.md"
    with open(output_file, "w") as f:
        f.write(md)
        
    print(f"Report generated at {output_file}")
    
    # Print table to verify
    for r in records:
        print(f"{r['Experiment']}: {r['Accuracy']:.4f}")

if __name__ == "__main__":
    generate_report()
