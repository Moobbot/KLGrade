#!/usr/bin/env python3
"""
Inference API Script
Refactored to use src.api logic.

Usage:
    python scripts/inference_api.py \
        --source path/to/image.dcm \
        --model path/to/model.pt \
        --output-dir path/to/output
"""

import argparse
import sys
import os
import json
import cv2
from pathlib import Path

# Ensure project root in python path
sys.path.append(os.getcwd())

from src.api.utils import read_image_file
from src.api.inference import YOLOModel

def run_inference(source: str, model_path: str, output_dir: str, conf_threshold: float = 0.25):
    """
    Run inference and save results.
    """
    source_path = Path(source)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"I: Processing {source_path}")
    
    # Read Image using shared util
    try:
        with open(source_path, "rb") as f:
            file_bytes = f.read()
        image_rgb = read_image_file(file_bytes, source_path.name)
    except Exception as e:
        print(f"E: Failed to read image: {e}")
        return False

    # Load Model
    try:
        model = YOLOModel(model_path)
    except Exception as e:
        print(f"E: Failed to load model: {e}")
        return False
        
    # Run Inference
    results = model.predict(image_rgb, conf=conf_threshold)
    
    if not results:
        print("W: No results returned")
        return False
        
    result = results[0]
    
    # Save Annotated Image (BGR for cv2)
    annotated_bgr = result.plot()
    output_image_path = output_path / "result.png"
    cv2.imwrite(str(output_image_path), annotated_bgr)
    print(f"S: Saved image to {output_image_path}")
    
    # Extract Data for JSON
    json_data = {
        "source": str(source_path),
        "predictions": []
    }
    
    boxes = result.boxes
    for i in range(len(boxes)):
        box = boxes[i]
        xyxy = box.xyxy[0].cpu().numpy().tolist()
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        
        # Use class mapping from YOLOModel wrapper
        class_name = model.class_mapping.get(cls_id, str(cls_id))
        
        prediction = {
            "class_id": cls_id,
            "class_name": class_name,
            "confidence": conf,
            "bbox": {
                "x1": xyxy[0],
                "y1": xyxy[1],
                "x2": xyxy[2],
                "y2": xyxy[3]
            }
        }
        json_data["predictions"].append(prediction)
        
    output_json_path = output_path / "result.json"
    with open(output_json_path, "w") as f:
        json.dump(json_data, f, indent=4)
    print(f"S: Saved JSON to {output_json_path}")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference API")
    parser.add_argument("--source", type=str, required=True, help="Input image path")
    parser.add_argument("--model", type=str, required=True, help="YOLO model path")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    
    args = parser.parse_args()
    
    success = run_inference(args.source, args.model, args.output_dir, args.conf)
    
    sys.exit(0 if success else 1)
