"""
Approach 1: KIOCMIL with Integrated Detection Heads

End-to-end model that combines:
1. Knee detection head
2. Lesion detection head
3. KIOCMIL-CADA classification

This approach maintains the sophisticated attention mechanism of KIOCMIL
while adding detection capabilities for a single unified model.
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from typing import Dict, List, Tuple, Optional
import warnings

from src.models.kiocmil_model_cada import KiocmilModelCADA


class DetectionHead(nn.Module):
    """
    Lightweight detection head for knee/lesion detection.

    This is a simple detection head that can be added to KIOCMIL backbone.
    For production, consider using more sophisticated detection architectures.
    """

    def __init__(
        self,
        in_channels: int = 1024,
        num_classes: int = 1,
        num_anchors: int = 3,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Detection layers
        self.conv1 = nn.Conv2d(in_channels, 512, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 256, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        # Prediction heads
        self.bbox_head = nn.Conv2d(256, num_anchors * 4, 1)  # x, y, w, h
        self.conf_head = nn.Conv2d(256, num_anchors * num_classes, 1)  # confidence

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) feature map

        Returns:
            bboxes: (B, N, 4) predicted boxes in [cx, cy, w, h] format
            confs: (B, N, num_classes) confidence scores
        """
        # Feature extraction
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))

        # Predictions
        bboxes = self.bbox_head(x)  # (B, num_anchors*4, H, W)
        confs = self.conf_head(x)  # (B, num_anchors*num_classes, H, W)

        # Reshape to (B, N, 4) and (B, N, num_classes)
        B, _, H, W = bboxes.shape
        N = H * W * self.num_anchors

        bboxes = bboxes.permute(0, 2, 3, 1).reshape(B, N, 4)
        confs = confs.permute(0, 2, 3, 1).reshape(B, N, self.num_classes)

        # Apply sigmoid to boxes and confidence
        bboxes = torch.sigmoid(bboxes)
        confs = torch.sigmoid(confs)

        return bboxes, confs


class KiocmilWithDetection(nn.Module):
    """
    APPROACH 1: KIOCMIL-CADA with integrated detection heads.

    Architecture:
    1. Shared YOLO backbone for feature extraction
    2. Knee detection head → detects knee regions
    3. Lesion detection head → detects JS and OST lesions within knees
    4. KIOCMIL-CADA module → classifies KL grade using detected regions

    Advantages:
    - Single unified model (end-to-end)
    - Maintains KIOCMIL's sophisticated attention mechanism
    - Can leverage pretrained weights from both KIOCMIL and YOLO
    - Shared backbone reduces total parameters

    Disadvantages:
    - Complex training (multi-task loss balancing required)
    - Higher memory requirements
    - Harder to debug than separate models
    """

    def __init__(
        self,
        backbone_name: str = "yolo11l",
        num_classes: int = 10,
        feature_dim: int = 256,
        pretrained_kiocmil: Optional[str] = None,
        pretrained_knee_detector: Optional[str] = None,
        freeze_kiocmil: bool = True,
    ):
        """
        Initialize end-to-end model.

        Args:
            backbone_name: YOLO backbone to use
            num_classes: Number of classification classes (10 for KL grading)
            feature_dim: Feature dimension for KIOCMIL
            pretrained_kiocmil: Path to pretrained KIOCMIL checkpoint (None = train from scratch)
            pretrained_knee_detector: Path to pretrained YOLO knee detector
            freeze_kiocmil: Whether to freeze KIOCMIL weights initially
        """
        super().__init__()

        self.num_classes = num_classes
        self.feature_dim = feature_dim

        # 1. Shared Backbone (YOLO)
        print(f"Loading {backbone_name} backbone...")
        yolo = YOLO(f"weight/{backbone_name}.pt")
        self.backbone, self.backbone_dim = self._extract_backbone(yolo)
        print(f"✅ Backbone loaded. Feature dim: {self.backbone_dim}")

        # 2. Detection Heads (use actual backbone output dimension)
        self.knee_detector = DetectionHead(
            in_channels=self.backbone_dim,
            num_classes=1,  # Only "knee" class
            num_anchors=3,
        )
        print("✅ Knee detection head initialized")

        self.lesion_detector = DetectionHead(
            in_channels=self.backbone_dim,
            num_classes=2,  # JS (0) and OST (1)
            num_anchors=3,
        )
        print("✅ Lesion detection head initialized")

        # 3. KIOCMIL-CADA Classification Module (share the backbone)
        self.kiocmil = KiocmilModelCADA(
            backbone_name=backbone_name,
            num_classes=num_classes,
            feature_dim=feature_dim,
            external_backbone=(self.backbone, self.backbone_dim),  # Share backbone!
        )
        print("✅ KIOCMIL-CADA module initialized")

        # Load pretrained weights if provided
        if pretrained_kiocmil:
            self._load_kiocmil_weights(pretrained_kiocmil)
            if freeze_kiocmil:
                self._freeze_kiocmil()
        else:
            print("⚠️  Training KIOCMIL from scratch (no pretrained weights)")

        if pretrained_knee_detector:
            self._load_knee_detector_weights(pretrained_knee_detector)

    def _extract_backbone(self, yolo_model):
        """Extract feature extraction backbone from YOLO and determine output dimension."""
        model = yolo_model.model
        # Extract first 10 layers (backbone before neck)
        if hasattr(model, "model"):
            layers = list(model.model.children())
        else:
            layers = list(model.children())

        backbone = nn.Sequential(*layers[:10])

        # Determine output dimension with dummy forward pass
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_output = backbone(dummy_input)

            if isinstance(dummy_output, (list, tuple)):
                dummy_output = dummy_output[-1]

            if dummy_output.dim() == 4:
                feature_dim = dummy_output.shape[1]
            else:
                feature_dim = dummy_output.shape[-1]

        return backbone, feature_dim

    def _load_kiocmil_weights(self, path: str):
        """Load pretrained KIOCMIL-CADA weights."""
        print(f"Loading KIOCMIL weights from {path}...")
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.kiocmil.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded KIOCMIL weights")

    def _load_knee_detector_weights(self, path: str):
        """
        Load pretrained YOLO knee detector weights.

        Note: This requires custom logic to transfer YOLO detection head
        weights to our DetectionHead. Currently not implemented.
        """
        print(f"⚠️  Knee detector weight transfer not implemented yet")
        print(f"   Will train detection head from scratch")

    def _freeze_kiocmil(self):
        """Freeze KIOCMIL parameters for transfer learning."""
        for param in self.kiocmil.parameters():
            param.requires_grad = False
        print("🔒 KIOCMIL weights frozen")

    def unfreeze_kiocmil(self):
        """Unfreeze KIOCMIL for fine-tuning."""
        for param in self.kiocmil.parameters():
            param.requires_grad = True
        print("🔓 KIOCMIL weights unfrozen")

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
            - knee_boxes: (B, N_knee, 4) detected knee boxes [cx, cy, w, h]
            - knee_confs: (B, N_knee, 1) knee detection confidences
            - lesion_boxes: (B, N_lesion, 4) detected lesion boxes
            - lesion_confs: (B, N_lesion, 2) lesion confidences [JS, OST]
            - logits_10: (B, 10) KL grade predictions (10-class)
            - logits_grade: (B, 5) KL grade predictions (5-class)
            - logits_type: (B, 1) lesion type predictions
            - embedding: (B, feature_dim) final embeddings
        """
        B = images.shape[0]
        device = images.device

        # 1. Extract features from backbone
        features = self.backbone(images)  # (B, 1024, H/32, W/32)

        # 2. Detect knees
        knee_boxes, knee_confs = self.knee_detector(features)

        # 3. Detect lesions
        # Note: In full implementation, should detect lesions within knee regions
        # For now, detect on full image
        lesion_boxes, lesion_confs = self.lesion_detector(features)

        # 4. Create batch_data for KIOCMIL
        # TODO: Implement proper cropping and patch extraction
        # For now, use dummy data structure
        batch_data = self._create_batch_data(
            images, knee_boxes, knee_confs, lesion_boxes, lesion_confs
        )

        # 5. KIOCMIL classification
        kiocmil_output = self.kiocmil(batch_data)

        # 6. Combine outputs
        return {
            "knee_boxes": knee_boxes,
            "knee_confs": knee_confs,
            "lesion_boxes": lesion_boxes,
            "lesion_confs": lesion_confs,
            "logits_10": kiocmil_output["logits_10"],
            "logits_grade": kiocmil_output["logits_grade"],
            "logits_type": kiocmil_output["logits_type"],
            "embedding": kiocmil_output["embedding"],
        }

    def _create_batch_data(
        self,
        images: torch.Tensor,
        knee_boxes: torch.Tensor,
        knee_confs: torch.Tensor,
        lesion_boxes: torch.Tensor,
        lesion_confs: torch.Tensor,
    ) -> List[Dict]:
        """
        Create batch_data structure for KIOCMIL from detected boxes.

        TODO: Implement proper patch cropping from detected boxes.
        Currently returns dummy structure.
        """
        B = images.shape[0]
        device = images.device

        batch_data = []
        for b in range(B):
            # Filter valid detections
            valid_knees = knee_confs[b, :, 0] > 0.5
            knees_b = knee_boxes[b][valid_knees]

            if len(knees_b) == 0:
                # No knees detected, use full image
                knees_b = torch.tensor([[0.5, 0.5, 1.0, 1.0]], device=device)

            knees_data = []
            for knee_box in knees_b:
                # TODO: Crop actual patches from image using knee_box
                # For now, use dummy patches
                ctx_patch = torch.randn(3, 384, 384, device=device)
                js_patches = torch.empty(0, 3, 224, 224, device=device)
                ost_patches = torch.empty(0, 3, 224, 224, device=device)

                knees_data.append(
                    {
                        "ctx": ctx_patch,
                        "ctx_bbox": knee_box,
                        "js": js_patches,
                        "js_bboxes": torch.empty(0, 4, device=device),
                        "ost": ost_patches,
                        "ost_bboxes": torch.empty(0, 4, device=device),
                    }
                )

            batch_data.append(
                {
                    "knees": knees_data,
                    "label": 0,  # Dummy label
                }
            )

        return batch_data


# Example usage
if __name__ == "__main__":
    # Initialize model
    model = KiocmilWithDetection(
        backbone_name="yolo11l",
        num_classes=10,
        pretrained_kiocmil="runs/kiocmil_cada/cada_10class_balanced/best_acc_model.pt",
        freeze_kiocmil=True,  # Freeze for Phase 1 training
    )

    # Test forward pass
    dummy_images = torch.randn(2, 3, 640, 640)
    outputs = model(dummy_images)

    print("\nOutput shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
