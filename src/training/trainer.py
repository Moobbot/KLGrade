from pathlib import Path
import argparse
from ultralytics import YOLO

class Trainer:
    def __init__(self):
        self.args = self.parse_args()
        
    def parse_args(self):
        parser = argparse.ArgumentParser(description="YOLO Training")
        parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
        parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained model or yaml")
        parser.add_argument("--epochs", type=int, default=50)
        parser.add_argument("--imgsz", type=int, default=640)
        parser.add_argument("--batch", type=int, default=16)
        parser.add_argument("--project", type=str, default="runs/detect")
        parser.add_argument("--name", type=str, default="exp")
        parser.add_argument("--device", type=str, default="0")
        return parser.parse_args()

    def train(self):
        print(f"Starting training for {self.args.name}...")
        model = YOLO(self.args.model)
        
        results = model.train(
            data=self.args.data,
            epochs=self.args.epochs,
            imgsz=self.args.imgsz,
            batch=self.args.batch,
            project=self.args.project,
            name=self.args.name,
            device=self.args.device,
            exist_ok=True
        )
        return results
