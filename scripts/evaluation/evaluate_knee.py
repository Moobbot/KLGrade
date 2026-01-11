
import sys
import argparse
from pathlib import Path
import json
from ultralytics import YOLO

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def evaluate_knee_detection(
    model_path: str,
    data_path: str = "processed/knee_detection/dataset.yaml",
    split: str = "test",
    batch: int = 16,
    img_size: int = 640,
    device: str = "0",
    project: str = "runs/evaluate",
    name: str = "knee_eval"
):
    """
    Evaluates a trained YOLO model on the knee detection dataset.
    """
    print("=" * 60)
    print(f"Knee Detection Evaluation")
    print("=" * 60)
    
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"❌ Error: Model weights not found at {model_file}")
        return
        
    data_file = Path(data_path)
    if not data_file.exists():
        print(f"❌ Error: Dataset YAML not found at {data_file}")
        print("   Did you run scripts/training/train_knee_detection.py first?")
        return

    print(f"Configuration:")
    print(f"  Model: {model_file}")
    print(f"  Data:  {data_file}")
    print(f"  Split: {split}")
    
    try:
        print(f"\n📦 Loading model...")
        model = YOLO(str(model_file))
        
        print(f"\n🚀 Running evaluation on '{split}' set...")
        metrics = model.val(
            data=str(data_file),
            split=split,
            batch=batch,
            imgsz=img_size,
            device=device,
            project=project,
            name=name,
            save=True,
            plots=True,
            exist_ok=True
        )
        
        print("\n" + "=" * 60)
        print("✅ Evaluation Complete")
        print("=" * 60)
        
        # Extended Metrics Logging
        results = {
            "model": str(model_file),
            "split": split,
            "map50": round(metrics.box.map50, 4),
            "map50_95": round(metrics.box.map, 4),
            "precision": round(metrics.box.mp, 4),
            "recall": round(metrics.box.mr, 4),
            "fitness": round(metrics.fitness, 4),
            "class_results": []
        }
        
        # Per-class metrics (we only have one class 'knee' usually, but good for robustness)
        # metrics.box.maps is an array of mAP50-95 for each class
        if hasattr(metrics.box, 'maps') and len(metrics.names) > 0:
            for i, class_name in metrics.names.items():
                if i < len(metrics.box.maps):
                    results["class_results"].append({
                        "class_id": i,
                        "class_name": class_name,
                        "map50_95": round(metrics.box.maps[i], 4)
                    })

        print(f"\n📊 Summary Metrics:")
        print(f"  mAP50:    {results['map50']}")
        print(f"  mAP50-95: {results['map50_95']}")
        print(f"  Precision: {results['precision']}")
        print(f"  Recall:    {results['recall']}")
        
        # Save detailed JSON
        save_dir = Path(metrics.save_dir)
        json_path = save_dir / "evaluation_results.json"
        
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)
            
        print(f"\n💾 Detailed results saved to: {json_path}")
        print(f"🖼️  Confusion matrix and plots saved to: {save_dir}")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Knee Detection Model")
    parser.add_argument("--model", type=str, required=True, help="Path to .pt model file")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to use")
    parser.add_argument("--project", type=str, default="runs/evaluate", help="Output directory")
    parser.add_argument("--name", type=str, default="knee_eval", help="Experiment name")
    parser.add_argument("--device", type=str, default="0", help="GPU device")
    
    args = parser.parse_args()
    
    evaluate_knee_detection(
        model_path=args.model,
        split=args.split,
        project=args.project,
        name=args.name,
        device=args.device
    )
