"""
KIOCMIL-CADA Model Inference API

Handles inference for KIOCMIL-CADA models which require:
- Knee detection (context patches)
- Lesion detection (JS and OST patches)
- Structured batch data with bboxes
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from ultralytics import YOLO

from .models.kiocmil_model_cada import KiocmilModelCADA
from .preprocessing.transforms import get_photometric_transforms


class KiocmilInference:
    """
    Inference wrapper for KIOCMIL-CADA models.

    Requires:
    1. Knee detection model (YOLO) to localize knee regions
    2. Lesion detection model (YOLO) to find JS/OST lesions
    3. KIOCMIL-CADA model for final classification
    """

    def __init__(
        self,
        kiocmil_model_path: str,
        knee_model_path: str,
        lesion_model_path: str,
        num_classes: int = 10,
        ctx_size: Tuple[int, int] = (384, 384),
        patch_size: Tuple[int, int] = (224, 224),
        device: str = "cuda",
    ):
        """
        Initialize KIOCMIL inference pipeline.

        Args:
            kiocmil_model_path: Path to KIOCMIL-CADA checkpoint
            knee_model_path: Path to knee detection YOLO model
            lesion_model_path: Path to lesion detection YOLO model
            num_classes: Number of output classes
            ctx_size: Context (knee) patch size
            patch_size: Lesion patch size
            device: Device for inference
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.ctx_size = ctx_size
        self.patch_size = patch_size
        self.num_classes = num_classes

        # Load models
        print(f"Loading KIOCMIL-CADA model from {kiocmil_model_path}...")
        self.model = self._load_kiocmil_model(kiocmil_model_path, num_classes)

        print(f"Loading knee detection model from {knee_model_path}...")
        self.knee_detector = YOLO(knee_model_path)

        print(f"Loading lesion detection model from {lesion_model_path}...")
        self.lesion_detector = YOLO(lesion_model_path)

        # Load transforms
        self.transform = get_photometric_transforms(level="light", use_clahe=True)

        print("✅ KIOCMIL inference pipeline loaded successfully!")

    def _load_kiocmil_model(
        self, model_path: str, num_classes: int
    ) -> KiocmilModelCADA:
        """Load KIOCMIL-CADA model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = (
            checkpoint["model_state_dict"]
            if "model_state_dict" in checkpoint
            else checkpoint
        )

        # Auto-detect num_classes from checkpoint
        if "head_10.weight" in state_dict:
            saved_num_classes = state_dict["head_10.weight"].shape[0]
            print(f"Detected {saved_num_classes} classes in checkpoint")
            num_classes = saved_num_classes

        model = KiocmilModelCADA(
            backbone_name="yolo11l",
            num_classes=num_classes,
            feature_dim=256,
            num_deformable_points=4,
            num_context_scales=3,
            use_positional_encoding=True,
            dropout=0.1,
        ).to(self.device)

        model.load_state_dict(state_dict)
        model.eval()

        return model

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply photometric transforms to image."""
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image

    def _yolo_to_pascal(
        self, box: List[float], w: int, h: int
    ) -> Tuple[int, int, int, int]:
        """Convert YOLO format [cx, cy, w, h] to Pascal VOC [x1, y1, x2, y2]."""
        cx, cy, bw, bh = box
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        return x1, y1, x2, y2

    def _extract_knee_patches(
        self, image: np.ndarray, knee_conf: float = 0.25
    ) -> List[Dict]:
        """
        Detect knees and extract context patches.

        Returns:
            List of knee dictionaries with context patches, bboxes, and confidence
        """
        results = self.knee_detector.predict(image, conf=knee_conf, verbose=False)

        if not results or len(results[0].boxes) == 0:
            return []

        h, w = image.shape[:2]
        knees = []

        for box in results[0].boxes:
            # Get bbox and confidence
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            knee_confidence = float(box.conf[0].item())  # Get knee confidence

            # Add padding
            pad_w = int((x2 - x1) * 0.1)
            pad_h = int((y2 - y1) * 0.1)
            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(w, x2 + pad_w)
            y2 = min(h, y2 + pad_h)

            # Extract and resize context patch
            ctx_patch = image[y1:y2, x1:x2]
            ctx_patch = cv2.resize(ctx_patch, self.ctx_size)

            # Apply transforms
            ctx_patch = self._preprocess_image(ctx_patch)

            # Convert to tensor
            ctx_tensor = torch.from_numpy(ctx_patch).permute(2, 0, 1).float() / 255.0

            # Normalize bbox to [0, 1]
            ctx_bbox = [
                (x1 + x2) / 2 / w,  # cx
                (y1 + y2) / 2 / h,  # cy
                (x2 - x1) / w,  # w
                (y2 - y1) / h,  # h
            ]

            knees.append(
                {
                    "ctx": ctx_tensor,
                    "ctx_bbox": torch.tensor(ctx_bbox, dtype=torch.float32),
                    "knee_region": (x1, y1, x2, y2),
                    "knee_confidence": knee_confidence,  # Store knee confidence
                }
            )

        return knees

    def _extract_lesion_patches(
        self,
        image: np.ndarray,
        knee_region: Tuple[int, int, int, int],
        lesion_conf: float = 0.25,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Tuple], List[Tuple]
    ]:
        """
        Detect lesions within knee region and extract patches.

        Args:
            image: Full image
            knee_region: (x1, y1, x2, y2) of knee bbox
            lesion_conf: Confidence threshold for lesion detection

        Returns:
            (js_patches, js_bboxes, ost_patches, ost_bboxes, js_global_bboxes, ost_global_bboxes)
        """
        x1, y1, x2, y2 = knee_region
        knee_crop = image[y1:y2, x1:x2]

        # Detect lesions in knee region
        results = self.lesion_detector.predict(
            knee_crop, conf=lesion_conf, verbose=False
        )

        js_patches = []
        js_bboxes = []
        ost_patches = []
        ost_bboxes = []
        js_global_bboxes = []  # Global coordinates for API response
        ost_global_bboxes = []  # Global coordinates for API response

        if results and len(results[0].boxes) > 0:
            h, w = knee_crop.shape[:2]

            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                lx1, ly1, lx2, ly2 = xyxy

                # Extract lesion patch
                lesion_patch = knee_crop[ly1:ly2, lx1:lx2]

                if lesion_patch.size == 0:
                    continue

                # Resize to patch size
                lesion_patch = cv2.resize(lesion_patch, self.patch_size)
                lesion_patch = self._preprocess_image(lesion_patch)
                lesion_tensor = (
                    torch.from_numpy(lesion_patch).permute(2, 0, 1).float() / 255.0
                )

                # Normalize bbox
                bbox = [
                    (lx1 + lx2) / 2 / w,
                    (ly1 + ly2) / 2 / h,
                    (lx2 - lx1) / w,
                    (ly2 - ly1) / h,
                ]
                bbox_tensor = torch.tensor(bbox, dtype=torch.float32)

                # Calculate global coordinates (relative to full image)
                global_x1 = x1 + lx1
                global_y1 = y1 + ly1
                global_x2 = x1 + lx2
                global_y2 = y1 + ly2
                global_bbox = (
                    int(global_x1),
                    int(global_y1),
                    int(global_x2),
                    int(global_y2),
                )

                # Categorize by class (JS vs OST)
                # Assuming classes 4,5 are JS and 0,1,2,3 are OST (from config)
                if cls_id in [4, 5]:  # JS classes
                    js_patches.append(lesion_tensor)
                    js_bboxes.append(bbox_tensor)
                    js_global_bboxes.append(global_bbox)
                else:  # OST classes
                    ost_patches.append(lesion_tensor)
                    ost_bboxes.append(bbox_tensor)
                    ost_global_bboxes.append(global_bbox)

        # Stack or create empty tensors
        js_tensor = (
            torch.stack(js_patches)
            if js_patches
            else torch.empty(0, 3, *self.patch_size)
        )
        js_bbox_tensor = torch.stack(js_bboxes) if js_bboxes else torch.empty(0, 4)
        ost_tensor = (
            torch.stack(ost_patches)
            if ost_patches
            else torch.empty(0, 3, *self.patch_size)
        )
        ost_bbox_tensor = torch.stack(ost_bboxes) if ost_bboxes else torch.empty(0, 4)

        return (
            js_tensor,
            js_bbox_tensor,
            ost_tensor,
            ost_bbox_tensor,
            js_global_bboxes,
            ost_global_bboxes,
        )

    def predict(
        self,
        image: np.ndarray,
        knee_conf: float = 0.5,
        lesion_conf: float = 0.5,
    ) -> List[Dict]:
        """
        Run full KIOCMIL inference pipeline.

        Args:
            image: Input image (RGB numpy array)
            knee_conf: Confidence threshold for knee detection
            lesion_conf: Confidence threshold for lesion detection

        Returns:
            List of predictions per knee with class probabilities
        """
        # Step 1: Detect knees
        knees = self._extract_knee_patches(image, knee_conf)

        if not knees:
            return []

        # Step 2: For each knee, detect lesions and store bboxes
        for knee in knees:
            js, js_bbox, ost, ost_bbox, js_global, ost_global = (
                self._extract_lesion_patches(image, knee["knee_region"], lesion_conf)
            )
            knee["js"] = js
            knee["js_bboxes"] = js_bbox
            knee["ost"] = ost
            knee["ost_bboxes"] = ost_bbox
            knee["js_global_bboxes"] = js_global  # Store global coordinates
            knee["ost_global_bboxes"] = ost_global  # Store global coordinates

        # Step 3: Prepare batch data for KIOCMIL model
        batch_data = [{"knees": knees, "label": 0}]  # Dummy label for inference

        # Step 4: Run KIOCMIL-CADA model
        with torch.no_grad():
            outputs = self.model(batch_data)

        # Step 5: Extract predictions
        logits = outputs["logits_10"]
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        # Format results
        results = []
        # Class names from config.py CLASSES_10_CLASS
        class_names_10 = [
            "KL0-a",  # Osteophyte (gai xương)
            "KL0-b",  # Joint space (khe khớp)
            "KL1-a",
            "KL1-b",
            "KL2-a",
            "KL2-b",
            "KL3-a",
            "KL3-b",
            "KL4-a",
            "KL4-b",
        ]

        for i, knee in enumerate(knees):
            pred_idx = preds[i].item()
            confidence = probs[i, pred_idx].item()

            results.append(
                {
                    "knee_bbox": knee["knee_region"],
                    "predicted_class": (
                        class_names_10[pred_idx]
                        if pred_idx < len(class_names_10)
                        else str(pred_idx)
                    ),
                    "predicted_class_id": pred_idx,
                    "confidence": confidence,
                    "class_probabilities": {
                        class_names_10[j]: probs[i, j].item()
                        for j in range(min(len(class_names_10), probs.shape[1]))
                    },
                    "num_js_lesions": len(knee["js"]),
                    "num_ost_lesions": len(knee["ost"]),
                    "js_lesion_bboxes": knee.get("js_global_bboxes", []),
                    "ost_lesion_bboxes": knee.get("ost_global_bboxes", []),
                }
            )

        return results
