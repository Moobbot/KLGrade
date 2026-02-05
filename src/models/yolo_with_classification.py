"""
Approach 2: YOLO with Classification Head

End-to-end model that combines:
1. YOLO detection (knees + lesions)
2. Classification head for KL grading

This approach leverages proven YOLO detection with a simple
classification head for a lightweight end-to-end solution.
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from typing import Dict, List, Tuple, Optional


class ClassificationHead(nn.Module):
    """
    Classification head for KL grading.

    Takes pooled features from YOLO backbone and predicts KL grade.
    """

    def __init__(
        self,
        in_features: int = 1024,
        hidden_dim: int = 256,
        num_classes: int = 10,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: (B, in_features) pooled features

        Returns:
            logits: (B, num_classes) classification logits
        """
        return self.classifier(x)


class YOLOWithClassification(nn.Module):
    """
    APPROACH 2: YOLO detection with classification head.

    Architecture:
    1. YOLO model for detection
       - Detects: knees (class 0), JS lesions (class 1), OST lesions (class 2)
    2. Feature extraction from YOLO backbone
    3. Classification head for KL grading

    Advantages:
    - Simpler architecture than Approach 1
    - Leverages proven YOLO detection
    - Easier to train and debug
    - Lower memory requirements
    - Faster inference

    Disadvantages:
    - Less sophisticated than KIOCMIL attention mechanism
    - May have lower classification accuracy
    - Simple global pooling instead of attention
    """

    def __init__(
        self,
        yolo_model_path: str = "yolo11n.pt",
        num_classes: int = 10,
        feature_dim: int = 256,
        freeze_yolo: bool = False,
    ):
        """
        Initialize YOLO with classification head.

        Args:
            yolo_model_path: Path to YOLO model (pretrained or trained detector)
            num_classes: Number of classification classes (10 for KL grading)
            feature_dim: Hidden dimension for classification head
            freeze_yolo: Whether to freeze YOLO weights initially
        """
        super().__init__()

        self.num_classes = num_classes

        # 1. YOLO Model for Detection
        print(f"Loading YOLO model from {yolo_model_path}...")
        self.yolo = YOLO(yolo_model_path)
        print("✅ YOLO model loaded")

        # 2. Feature Extractor (YOLO backbone)
        self.feature_extractor = self._extract_backbone()
        print("✅ Feature extractor initialized")

        # 3. Classification Head
        self.classifier = ClassificationHead(
            in_features=1024,  # YOLO11 backbone output
            hidden_dim=feature_dim,
            num_classes=num_classes,
            dropout=0.2,
        )
        print("✅ Classification head initialized")

        # Freeze YOLO if requested
        if freeze_yolo:
            self._freeze_yolo()

    def _extract_backbone(self):
        """Extract backbone from YOLO for feature extraction."""
        model = self.yolo.model
        if hasattr(model, "model"):
            layers = list(model.model.children())
        else:
            layers = list(model.children())
        # Extract first 10 layers (backbone)
        return nn.Sequential(*layers[:10])

    def _freeze_yolo(self):
        """Freeze YOLO parameters for transfer learning."""
        for param in self.yolo.model.parameters():
            param.requires_grad = False
        print("🔒 YOLO weights frozen")

    def unfreeze_yolo(self):
        """Unfreeze YOLO for fine-tuning."""
        for param in self.yolo.model.parameters():
            param.requires_grad = True
        print("🔓 YOLO weights unfrozen")

    def forward(
        self,
        images: torch.Tensor,
        return_detections: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for detection + classification.

        Args:
            images: (B, 3, H, W) input images
            return_detections: Whether to return YOLO detection results

        Returns:
            Dict containing:
            - logits: (B, num_classes) classification logits
            - detections: YOLO detection results (if return_detections=True)
            - features: (B, 1024) extracted features
        """
        # 1. YOLO Detection (optional)
        detections = None
        if return_detections:
            detections = self.yolo(images, verbose=False)

        # 2. Extract features from backbone
        features = self.feature_extractor(images)  # (B, 1024, H/32, W/32)

        # 3. Global average pooling
        pooled = features.mean(dim=[2, 3])  # (B, 1024)

        # 4. Classification
        logits = self.classifier(pooled)  # (B, num_classes)

        return {
            "logits": logits,
            "detections": detections,
            "features": pooled,
        }

    def predict(
        self,
        images: torch.Tensor,
        conf_threshold: float = 0.25,
    ) -> Dict:
        """
        Inference mode prediction.

        Args:
            images: (B, 3, H, W) input images
            conf_threshold: Confidence threshold for detections

        Returns:
            Dict containing:
            - predicted_class: (B,) predicted KL grade class
            - confidence: (B,) classification confidence
            - detections: YOLO detection results
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(images, return_detections=True)

            # Get predicted class
            probs = torch.softmax(outputs["logits"], dim=1)
            predicted_class = torch.argmax(probs, dim=1)
            confidence = probs.max(dim=1)[0]

            return {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "probabilities": probs,
                "detections": outputs["detections"],
            }


class YOLOMultiTask(nn.Module):
    """
    Alternative: YOLO with multi-task heads.

    This variant adds multiple classification heads for different tasks:
    - 10-class KL grading
    - 5-class KL grade
    - Binary lesion type

    Similar to KIOCMIL's multi-head design but simpler architecture.
    """

    def __init__(
        self,
        yolo_model_path: str = "yolo11n.pt",
        feature_dim: int = 256,
    ):
        super().__init__()

        # YOLO detection
        self.yolo = YOLO(yolo_model_path)
        self.feature_extractor = self._extract_backbone()

        # Multiple classification heads
        self.head_10 = ClassificationHead(1024, feature_dim, num_classes=10)
        self.head_grade = ClassificationHead(1024, feature_dim, num_classes=5)
        self.head_type = nn.Sequential(
            nn.Linear(1024, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim, 1),
        )

    def _extract_backbone(self):
        """Extract backbone from YOLO."""
        model = self.yolo.model
        if hasattr(model, "model"):
            layers = list(model.model.children())
        else:
            layers = list(model.children())
        return nn.Sequential(*layers[:10])

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multiple heads.

        Returns:
            Dict containing:
            - logits_10: (B, 10) 10-class predictions
            - logits_grade: (B, 5) grade predictions
            - logits_type: (B, 1) type predictions
            - detections: YOLO detection results
        """
        # Detection
        detections = self.yolo(images, verbose=False)

        # Feature extraction
        features = self.feature_extractor(images)
        pooled = features.mean(dim=[2, 3])  # (B, 1024)

        # Multiple heads
        logits_10 = self.head_10(pooled)
        logits_grade = self.head_grade(pooled)
        logits_type = self.head_type(pooled)

        return {
            "logits_10": logits_10,
            "logits_grade": logits_grade,
            "logits_type": logits_type,
            "detections": detections,
        }


# Example usage
if __name__ == "__main__":
    # Initialize model
    model = YOLOWithClassification(
        yolo_model_path="runs/my_knee_run_resplit/weights/best.pt",
        num_classes=10,
        freeze_yolo=True,  # Freeze for initial training
    )

    # Test forward pass
    dummy_images = torch.randn(2, 3, 640, 640)
    outputs = model(dummy_images, return_detections=False)

    print("\nOutput shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")

    # Test prediction
    predictions = model.predict(dummy_images)
    print(f"\nPredicted classes: {predictions['predicted_class']}")
    print(f"Confidences: {predictions['confidence']}")
