#!/usr/bin/env python3
"""
Inference API Script

Run inference on a single image (PNG, JPG, DICOM) and output:
1. Annotated image (result.png)
2. JSON result (result.json)

Usage:
    python scripts/inference_api.py \
        --source path/to/image.dcm \
        --model path/to/model.pt \
        --output-dir path/to/output
"""

import argparse
from pathlib import Path
import json
import numpy as np
import cv2
from PIL import Image
import sys

# Try imports
try:
    import pydicom
except ImportError:
    pydicom = None

from ultralytics import YOLO

def read_dicom_image(dicom_path):
    """
    Read a DICOM file and convert to RGB numpy array (0-255).
    """
    if pydicom is None:
        raise ImportError("pydicom is not installed. Please run: pip install pydicom")
        
    ds = pydicom.dcmread(dicom_path)
    pixel_array = ds.pixel_array
    
    # Normalize to 0-255
    if pixel_array.max() > 0:
        pixel_array = (pixel_array / pixel_array.max()) * 255.0
    
    pixel_array = pixel_array.astype(np.uint8)
    
    # Convert to 3-channel RGB if it's grayscale
    if len(pixel_array.shape) == 2:
        image_rgb = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2RGB)
    else:
        # Assuming already RGB or handled
        image_rgb = pixel_array
        
    return image_rgb

def run_inference(source: str, model_path: str, output_dir: str, conf_threshold: float = 0.25):
    """
    Run inference and save results.
    """
    source_path = Path(source)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"I: Processing {source_path}")
    
    # Load Image
    if source_path.suffix.lower() == '.dcm':
        print("I: Detected DICOM file")
        try:
            image_obj = read_dicom_image(source_path)
        except Exception as e:
            print(f"E: Failed to read DICOM: {e}")
            return False
    else:
        # Standard image
        try:
            image_obj = cv2.imread(str(source_path))
            if image_obj is None:
                raise ValueError("Could not read image with cv2")
            image_obj = cv2.cvtColor(image_obj, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"E: Failed to read image: {e}")
            return False

    # Load Model
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"E: Failed to load model: {e}")
        return False
        
    # Run Inference
    results = model.predict(image_obj, conf=conf_threshold, verbose=False)
    
    if not results:
        print("W: No results returned")
        return False
        
    result = results[0]
    
    # Save Annotated Image
    annotated_frame = result.plot()
    # plot returns BGR usually if using cv2 backend, but let's check. 
    # Ultralytics plot() returns numpy array in BGR.
    output_image_path = output_path / "result.png"
    cv2.imwrite(str(output_image_path), annotated_frame)
    print(f"S: Saved image to {output_image_path}")
    
    # Extract Data for JSON
    json_data = {
        "source": str(source_path),
        "predictions": []
    }
    
    # Boxes
    boxes = result.boxes
    for i in range(len(boxes)):
        box = boxes[i]
        # box.xyxy provides [x1, y1, x2, y2]
        # box.cls provides class index
        # box.conf provides confidence
        
        xyxy = box.xyxy[0].cpu().numpy().tolist()
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        class_name = result.names[cls_id]
        
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
        
    # Save JSON
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
