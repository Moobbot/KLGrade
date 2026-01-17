#!/usr/bin/env python3
"""
GradCAM Visualization Script

Wrapper script for GradCAM visualization on knee images.
Uses src.visualization.gradcam module for the actual implementation.

Usage:
    python scripts/visualization/visualize_gradcam.py \\
        --source path/to/image.jpg \\
        --knee-model models/knee_detector.pt \\
        --grade-model models/grade_classifier.pt
"""

import argparse
import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.pipeline import KneePipeline
from src.visualization.gradcam import YOLOGradCAM, apply_colormap


def run_gradcam_pipeline(args):
    """Run GradCAM visualization pipeline."""
    # 1. Pipeline Setup
    print("Loading models...")
    pipeline = KneePipeline(args.knee_model, args.grade_model)

    img_path = Path(args.source)
    print(f"Reading {img_path}")
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Knee Detection (Standard)
    print("Detecting knee...")
    knee_results = pipeline.knee_model.predict(img_rgb, conf=0.25)
    if not knee_results:
        print("No knee detected.")
        return

    # Prepare Grading Model for GradCAM
    grade_pt_model = (
        pipeline.grade_model.model.model
    )  # Ultralytics wrapper -> Model -> nn.Module
    grade_pt_model.eval()

    # Target Layer: model.9 (SPPF) usually good
    target_layer = grade_pt_model.model[9]
    print(f"Hooking layer: {target_layer}")

    grad_cam = YOLOGradCAM(grade_pt_model, target_layer)

    vis_img = img.copy()

    for box in knee_results[0].boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy

        # Validate crop
        h, w, _ = img.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = img_rgb[y1:y2, x1:x2]

        # Preprocess crop for PyTorch
        crop_resized = cv2.resize(crop, (640, 640))
        input_tensor = (
            torch.from_numpy(crop_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        )
        input_tensor = input_tensor.to(pipeline.grade_model.model.device)
        input_tensor.requires_grad = True

        # Generate CAM
        print("Computing GradCAM for crop...")
        cam = grad_cam(input_tensor)

        # Apply colormap and overlay
        cam_resized = cv2.resize(cam, (x2 - x1, y2 - y1))
        cam_crop = apply_colormap(cam_resized, crop, alpha=0.4)

        # Convert back to BGR for OpenCV
        cam_crop_bgr = cv2.cvtColor(cam_crop, cv2.COLOR_RGB2BGR)

        # Paste back
        vis_img[y1:y2, x1:x2] = cam_crop_bgr

        # Draw Knee Box (Green)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis_img,
            "GradCAM",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        # Draw Grade Box (Red)
        grade_results = pipeline.grade_model.predict(crop, conf=0.25)
        if grade_results and len(grade_results[0].boxes) > 0:
            best_grade_box = max(grade_results[0].boxes, key=lambda b: b.conf[0].item())
            gx1, gy1, gx2, gy2 = best_grade_box.xyxy[0].cpu().numpy().astype(int)

            # Map back to full image
            global_gx1 = gx1 + x1
            global_gy1 = gy1 + y1
            global_gx2 = gx2 + x1
            global_gy2 = gy2 + y1

            # Draw Red Box
            cv2.rectangle(
                vis_img,
                (global_gx1, global_gy1),
                (global_gx2, global_gy2),
                (0, 0, 255),
                2,
            )

            # Label
            cls_id = int(best_grade_box.cls[0].item())
            grade_name = pipeline.grade_model.class_mapping.get(cls_id, str(cls_id))
            g_conf = float(best_grade_box.conf[0].item())
            g_label = f"{grade_name} {g_conf:.2f}"

            cv2.putText(
                vis_img,
                g_label,
                (global_gx1, global_gy1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

    out_path = "gradcam_vis.jpg"
    cv2.imwrite(out_path, vis_img)
    print(f"Saved GradCAM visualization to {out_path}")


if __name__ == "__main__":
    import torch  # Import here to avoid circular dependency

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to input image")
    parser.add_argument(
        "--knee-model", required=True, help="Path to knee detection model"
    )
    parser.add_argument(
        "--grade-model", required=True, help="Path to grade classification model"
    )
    args = parser.parse_args()

    run_gradcam_pipeline(args)
