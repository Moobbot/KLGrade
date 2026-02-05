"""
End-to-End Knee Detection and Classification Model

This module provides two approaches for end-to-end knee OA grading:
1. KIOCMIL with integrated detection heads
2. YOLO with classification head

Both approaches aim to eliminate the 2-stage pipeline by combining
detection and classification in a single model.
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from typing import Dict, List, Tuple, Optional
import warnings

from src.models.kiocmil_model_cada import KiocmilModelCADA
from src.models.attention_modules import (
    DeformableAttention,
    CrossAttentionWithDeformable,
)


# ============================================================================
# APPROACH 1: KIOCMIL with Detection Heads
# ============================================================================


class DetectionHead(nn.Module):
    """
    Simple detection head for knee/lesion detection.

    This is a lightweight detection head that can be added to KIOCMIL backbone.
    For production, consider using YOLO/Faster R-CNN detection heads.
    """

    def __init__(
        self,
        in_channels: int = 1024,
        num_classes: int = 1,  # 1 for knee, 2 for JS+OST
        num_anchors: int = 3,
    ):
        super().__init__()

        # Detection layers
        self.conv1 = nn.Conv2d(in_channels, 512, 3, padding=1)
        self.conv2 = nn.Conv2d(512, 256, 3, padding=1)

        # Prediction heads
        self.bbox_head = nn.Conv2d(256, num_anchors * 4, 1)  # x, y, w, h
        self.conf_head = nn.Conv2d(256, num_anchors * num_classes, 1)  # confidence

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) feature map

        Returns:
            bboxes: (B, N, 4) predicted boxes
            confs: (B, N, num_classes) confidence scores
        """
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))

        bboxes = self.bbox_head(x)  # (B, num_anchors*4, H, W)
        confs = self.conf_head(x)  # (B, num_anchors*num_classes, H, W)

        # Reshape to (B, N, 4) and (B, N, num_classes)
        B, _, H, W = bboxes.shape
        bboxes = bboxes.permute(0, 2, 3, 1).reshape(B, -1, 4)
        confs = confs.permute(0, 2, 3, 1).reshape(
            B, -1, self.conf_head.out_channels // 3
        )

        return bboxes, confs


class KiocmilEndToEnd(nn.Module):
    """
    APPROACH 1: KIOCMIL with integrated detection heads.

    Architecture:
    1. Shared backbone (YOLO11L)
    2. Knee detection head → knee boxes
    3. For each knee:
       - Crop knee region
       - Lesion detection head → lesion boxes
    4. KIOCMIL-CADA classification → KL grade

    Advantages:
    - Single model for detection + classification
    - End-to-end training possible
    - Shared backbone reduces parameters

    Disadvantages:
    - Complex training (multi-task loss)
    - Harder to debug
    - May need careful loss balancing
    """

    def __init__(
        self,
        backbone_name: str = "yolo11l",
        num_classes: int = 10,
        feature_dim: int = 256,
        pretrained_kiocmil: Optional[str] = None,
        pretrained_knee_detector: Optional[str] = None,
    ):
        super().__init__()

        # 1. Shared Backbone
        print(f"Loading {backbone_name} backbone...")
        yolo = YOLO(f"weight/{backbone_name}.pt")
        self.backbone = yolo.model.model[:10]  # Extract backbone layers

        # 2. Detection Heads
        self.knee_detector = DetectionHead(
            in_channels=1024,
            num_classes=1,  # Only knee class
            num_anchors=3,
        )

        self.lesion_detector = DetectionHead(
            in_channels=1024,
            num_classes=2,  # JS and OST
            num_anchors=3,
        )

        # 3. KIOCMIL-CADA for classification
        self.kiocmil = KiocmilModelCADA(
            backbone_name=backbone_name,
            num_classes=num_classes,
            feature_dim=feature_dim,
        )

        # Load pretrained weights if available
        if pretrained_kiocmil:
            self._load_kiocmil_weights(pretrained_kiocmil)

        if pretrained_knee_detector:
            self._load_knee_detector_weights(pretrained_knee_detector)

    def _load_kiocmil_weights(self, path: str):
        """Load pretrained KIOCMIL weights."""
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.kiocmil.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded KIOCMIL weights from {path}")

    def _load_knee_detector_weights(self, path: str):
        """Load pretrained YOLO knee detector weights."""
        # This would need custom logic to transfer YOLO detection head weights
        print(f"⚠️  Knee detector weight loading not implemented yet")

    def forward(
        self,
        images: torch.Tensor,
        mode: str = "train",
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for end-to-end detection + classification.

        Args:
            images: (B, 3, H, W) input images
            mode: "train" or "inference"

        Returns:
            Dict containing:
            - knee_boxes: (B, N_knee, 4) detected knee boxes
            - knee_confs: (B, N_knee, 1) knee confidences
            - lesion_boxes: List of (N_lesion, 4) per knee
            - lesion_confs: List of (N_lesion, 2) per knee
            - logits_10: (B, 10) KL grade predictions
            - logits_grade: (B, 5) grade predictions
            - logits_type: (B, 1) type predictions
        """
        B = images.shape[0]
        device = images.device

        # 1. Extract features
        features = self.backbone(images)  # (B, 1024, H/32, W/32)

        # 2. Detect knees
        knee_boxes, knee_confs = self.knee_detector(features)

        # 3. For each image, detect lesions in knee regions
        batch_data = []

        for b in range(B):
            # Get valid knee detections (confidence > threshold)
            valid_knees = knee_confs[b, :, 0] > 0.5
            knees_b = knee_boxes[b][valid_knees]  # (N_knee, 4)

            if len(knees_b) == 0:
                # No knees detected, use full image
                knees_b = torch.tensor([[0.5, 0.5, 1.0, 1.0]], device=device)

            knees_data = []

            for knee_box in knees_b:
                # Crop knee region
                x, y, w, h = knee_box
                x1 = int((x - w / 2) * images.shape[3])
                y1 = int((y - h / 2) * images.shape[2])
                x2 = int((x + w / 2) * images.shape[3])
                y2 = int((y + h / 2) * images.shape[2])

                knee_crop = images[b : b + 1, :, y1:y2, x1:x2]

                # Detect lesions in knee crop
                knee_features = self.backbone(knee_crop)
                lesion_boxes, lesion_confs = self.lesion_detector(knee_features)

                # Filter lesions by confidence
                valid_lesions = lesion_confs[0].max(dim=1)[0] > 0.3
                lesions_b = lesion_boxes[0][valid_lesions]
                lesion_classes = lesion_confs[0][valid_lesions].argmax(dim=1)

                # Separate JS and OST
                js_mask = lesion_classes == 0
                ost_mask = lesion_classes == 1

                js_boxes = (
                    lesions_b[js_mask]
                    if js_mask.any()
                    else torch.empty(0, 4, device=device)
                )
                ost_boxes = (
                    lesions_b[ost_mask]
                    if ost_mask.any()
                    else torch.empty(0, 4, device=device)
                )

                # Create knee data structure for KIOCMIL
                # TODO: Need to crop actual patches from image
                # For now, use dummy patches
                ctx_patch = torch.randn(3, 384, 384, device=device)
                js_patches = (
                    torch.randn(len(js_boxes), 3, 224, 224, device=device)
                    if len(js_boxes) > 0
                    else torch.empty(0, 3, 224, 224, device=device)
                )
                ost_patches = (
                    torch.randn(len(ost_boxes), 3, 224, 224, device=device)
                    if len(ost_boxes) > 0
                    else torch.empty(0, 3, 224, 224, device=device)
                )

                knees_data.append(
                    {
                        "ctx": ctx_patch,
                        "ctx_bbox": knee_box,
                        "js": js_patches,
                        "js_bboxes": js_boxes,
                        "ost": ost_patches,
                        "ost_bboxes": ost_boxes,
                    }
                )

            batch_data.append(
                {
                    "knees": knees_data,
                    "label": 0,  # Dummy label
                }
            )

        # 4. KIOCMIL classification
        kiocmil_output = self.kiocmil(batch_data)

        # 5. Combine outputs
        return {
            "knee_boxes": knee_boxes,
            "knee_confs": knee_confs,
            "logits_10": kiocmil_output["logits_10"],
            "logits_grade": kiocmil_output["logits_grade"],
            "logits_type": kiocmil_output["logits_type"],
            "embedding": kiocmil_output["embedding"],
        }


# ============================================================================
# APPROACH 2: YOLO with Classification Head
# ============================================================================


class YOLOWithClassification(nn.Module):
    """
    APPROACH 2: YOLO model with added classification head.

    Architecture:
    1. YOLO backbone + neck + detection head
       - Detects: knees, JS lesions, OST lesions
    2. Classification head on detected knee features
       - Classifies: KL grade (10 classes)

    Advantages:
    - Leverages proven YOLO detection
    - Simpler architecture
    - Easier to train (can use pretrained YOLO)

    Disadvantages:
    - Less sophisticated than KIOCMIL attention mechanism
    - May lose some accuracy on classification
    """

    def __init__(
        self,
        yolo_model_path: str = "yolo11n.pt",
        num_classes: int = 10,
        feature_dim: int = 256,
    ):
        super().__init__()

        # 1. YOLO for detection
        self.yolo = YOLO(yolo_model_path)

        # 2. Classification head
        # Extract features from YOLO backbone
        self.feature_extractor = self.yolo.model.model[:10]  # Backbone

        # Classification MLP
        self.classifier = nn.Sequential(
            nn.Linear(1024, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(
        self,
        images: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for YOLO detection + classification.

        Args:
            images: (B, 3, H, W) input images

        Returns:
            Dict containing:
            - detections: YOLO detection results
            - logits: (B, num_classes) classification logits
        """
        # 1. YOLO detection
        detections = self.yolo(images)

        # 2. Extract features for classification
        features = self.feature_extractor(images)  # (B, 1024, H/32, W/32)

        # Global average pooling
        pooled = features.mean(dim=[2, 3])  # (B, 1024)

        # 3. Classification
        logits = self.classifier(pooled)  # (B, num_classes)

        return {
            "detections": detections,
            "logits": logits,
        }


# ============================================================================
# Helper Functions
# ============================================================================


def load_pretrained_models(
    kiocmil_path: str,
    knee_detector_path: str,
) -> Tuple[nn.Module, nn.Module]:
    """
    Load pretrained KIOCMIL and knee detector models.

    Returns:
        kiocmil_model, knee_detector
    """
    # Load KIOCMIL
    kiocmil = KiocmilModelCADA(num_classes=10)
    checkpoint = torch.load(kiocmil_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    kiocmil.load_state_dict(state_dict)

    # Load YOLO knee detector
    knee_detector = YOLO(knee_detector_path)

    return kiocmil, knee_detector
