from pathlib import Path
from typing import List, Optional
import yaml
import json
from ultralytics import YOLO

class YOLOEvaluator:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = YOLO(str(self.model_path))
        
    def create_temp_config(self, dataset_dir: Path, nc: int, names: List[str], output_dir: Path) -> Path:
        """Create temporary YAML config for evaluation."""
        dataset_dir = dataset_dir.absolute()
        
        train_txt = dataset_dir / "train.txt"
        val_txt = dataset_dir / "val.txt"
        test_txt = dataset_dir / "test.txt"
        
        config = {
            "path": str(dataset_dir.parent.parent.parent),
            "train": str(train_txt) if train_txt.exists() else str(val_txt),
            "val": str(val_txt),
            "test": str(test_txt) if test_txt.exists() else str(val_txt),
            "nc": nc,
            "names": names
        }
        
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "temp_eval_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
            
        print(f"Created temp config: {config_path}")
        return config_path
        
    def evaluate(
        self,
        data_config: Path,
        split: str = "test",
        project: str = "runs/evaluate",
        name: str = "eval",
        device: str = "0",
        batch_size: int = 16
    ):
        """Run validation."""
        print(f"Running evaluation on {split}...")
        results = self.model.val(
            data=str(data_config),
            split=split,
            project=project,
            name=name,
            device=device,
            batch=batch_size,
            exist_ok=True,
            plots=True,
            save=True,
            verbose=True
        )
        
        return results

    def save_metrics(self, results, output_dir: Path):
        """Save metrics to JSON."""
        metrics = {
            "map50_95": results.box.map,
            "map50": results.box.map50,
            "precision": results.box.mp,
            "recall": results.box.mr
        }
        
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {output_dir}/metrics.json")
