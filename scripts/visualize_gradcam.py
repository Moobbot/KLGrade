
import argparse
import sys
import os
import cv2
import numpy as np
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.pipeline import KneePipeline
from src.api.inference import YOLOModel

class YOLOGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple, usually (grad,)
        self.gradients = grad_output[0]

    def __call__(self, input_tensor, target_category=None):
        # Forward pass
        # Note: We need to run the model in a way that allows backward.
        # Ultralytics model.predict() usually runs in no_grad.
        # We need to call model.model(input_tensor) directly.
        
        preds = self.model(input_tensor)
        # Preds structure for YOLOv8/11 detect: list of [batch, 4+cls, anchors] or similar depending on head
        # Ultralytics DetectionHead returns multiple outputs.
        # Usually preds[0] is the inference output.
        
        # We need to find the max score.
        # Simplify: Assume single image batch
        output = preds[0] # tensor
        
        # Output shape is [1, 84, 8400] usually (for 80 classes + 4 box)
        # 84 = 4 box + 80 classes.
        # For our custom model it's 4 + num_classes.
        
        # We want to maximize the score of the target class.
        # Let's find the max score for the target category.
        
        # Transpose to [1, 8400, 84] for easier indexing
        output = output.transpose(1, 2)
        
        # Find best detection
        # scores are from index 4 onwards
        scores = output[0, :, 4:]
        max_score, max_idx = torch.max(scores.flatten(), dim=0)
        
        # If target_category specific is needed, filter. 
        # But usually we just want "why did you predict this class?"
        # So using the max prediction is correct.
        
        self.model.zero_grad()
        max_score.backward()
        
        # Generate CAM
        gradients = self.gradients[0] # [C, H, W]
        activations = self.activations[0] # [C, H, W]
        
        # Global Average Pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))
        
        # Weighted combination of activations
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = torch.relu(cam)
        cam = cam.cpu().detach().numpy()
        
        # Normalize
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

def run_gradcam_pipeline(args):
    # 1. Pipeline Setup
    print("Loading models...")
    # Use standard pipeline mainly for loading, but we need raw access for Grading
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
    # Access internal pytorch model
    grade_pt_model = pipeline.grade_model.model.model # Ultralytics wrapper -> Model -> nn.Module
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
        if x2<=x1 or y2<=y1: continue
        
        crop = img_rgb[y1:y2, x1:x2]
        
        # Preprocess crop for Torch
        # Resize to model size (640 typically)
        crop_resized = cv2.resize(crop, (640, 640))
        input_tensor = torch.from_numpy(crop_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        input_tensor = input_tensor.to(pipeline.grade_model.model.device)
        input_tensor.requires_grad = True # Enable grad for input flow? Not strictly needed if weights have grad, but model needs to allow backward.
        
        # Generate CAM
        print("Computing GradCAM for crop...")
        mask = grad_cam(input_tensor)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        
        # Resize heatmap back to crop size
        heatmap = cv2.resize(heatmap, (x2-x1, y2-y1))
        
        # Superimpose
        cam_crop = heatmap + np.float32(crop) / 255 # RGB crop
        cam_crop = cam_crop / np.max(cam_crop)
        cam_crop = np.uint8(255 * cam_crop)
        
        # Convert back to BGR for opencv drawing
        cam_crop_bgr = cv2.cvtColor(cam_crop, cv2.COLOR_RGB2BGR)
        
        # Paste back
        vis_img[y1:y2, x1:x2] = cam_crop_bgr
        
        # Draw Knee Box (Green) 
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis_img, "GradCAM", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Draw Grade Box (Red)
        # We need to predict on the crop to get the box
        # Note: input_tensor was resized to 640, but predict expects original image size logic or handles resizing internally
        # Let's use the pipeline's grade_model which wraps ultralytics
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
            cv2.rectangle(vis_img, (global_gx1, global_gy1), (global_gx2, global_gy2), (0, 0, 255), 2)
            
            # Label
            cls_id = int(best_grade_box.cls[0].item())
            grade_name = pipeline.grade_model.class_mapping.get(cls_id, str(cls_id))
            g_conf = float(best_grade_box.conf[0].item())
            g_label = f"{grade_name} {g_conf:.2f}"
            
            cv2.putText(vis_img, g_label, (global_gx1, global_gy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
    out_path = "gradcam_vis.jpg"
    cv2.imwrite(out_path, vis_img)
    print(f"Saved GradCAM visualization to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--knee-model", required=True)
    parser.add_argument("--grade-model", required=True)
    args = parser.parse_args()
    
    run_gradcam_pipeline(args)
