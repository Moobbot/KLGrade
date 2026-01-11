
import cv2
import json
import numpy as np
from pathlib import Path

def visualize_prediction(image_path, predictions, output_path):
    print(f"Reading image from: {image_path}")
    img = cv2.imread(str(image_path))
    if img is None:
        print("Error reading image")
        return

    for i, pred in enumerate(predictions):
        # 1. Draw Knee Box (Green)
        bbox = pred["knee_bbox"]
        x1, y1, x2, y2 = bbox
        
        knee_score = pred.get("knee_conf", 0.0)
        
        kl_info = pred.get("kl_grade", {})
        grade_class = kl_info.get("grade_class", "Unknown")
        grade_conf = kl_info.get("grade_conf", 0.0)
        grade_bbox = kl_info.get("grade_bbox", [])
        
        # Knee Color: Green
        knee_color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), knee_color, 4)
        
        # Knee Label
        label = f"Knee ({knee_score:.2f}) | {grade_class}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(img, (x1, y1 - h - 10), (x1 + w, y1), knee_color, -1)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        
        # 2. Draw Grade Box (Red) if exists
        if grade_bbox and len(grade_bbox) == 4:
            gx1, gy1, gx2, gy2 = grade_bbox
            grade_color = (0, 0, 255) # Red
            
            cv2.rectangle(img, (gx1, gy1), (gx2, gy2), grade_color, 3)
            
            # Grade Label
            g_label = f"{grade_class} ({grade_conf:.2f})"
            (gw, gh), _ = cv2.getTextSize(g_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            # Draw label near the grade box
            cv2.rectangle(img, (gx1, gy1 - gh - 5), (gx1 + gw, gy1), grade_color, -1)
            cv2.putText(img, g_label, (gx1, gy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
    print(f"Saving visualization to: {output_path}")
    cv2.imwrite(str(output_path), img)

# Data from User Request
json_data = {
  "filename": "1.2.392.200036.9107.307.24972.20230130.143304.1042023.jpg",
  "predictions": [
    {
      "knee_bbox": [
        652,
        1690,
        1648,
        2824
      ],
      "knee_conf": 0.8988385200500488,
      "kl_grade": {
        "grade_class": "KL2-a",
        "grade_conf": 0.5045002698898315,
        "grade_id": 4,
        "grade_bbox": [
          1516,
          2377,
          1576,
          2458
        ]
      }
    },
    {
      "knee_bbox": [
        2414,
        1722,
        3354,
        2862
      ],
      "knee_conf": 0.8760183453559875,
      "kl_grade": {
        "grade_class": "KL3-a",
        "grade_conf": 0.6329189538955688,
        "grade_id": 6,
        "grade_bbox": [
          2443,
          2218,
          2495,
          2282
        ]
      }
    }
  ],
  "image_base64": None
}

# Paths
image_file = Path("dataset/dataset_v0/images") / json_data["filename"]
output_file = Path("prediction_vis.jpg")

if __name__ == "__main__":
    visualize_prediction(image_file, json_data["predictions"], output_file)
