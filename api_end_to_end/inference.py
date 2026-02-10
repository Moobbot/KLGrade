"""
End-to-End KIOCMIL Inference API

Provides inference capabilities for End-to-End KIOCMIL models supporting
4-class, 5-class, 8-class, and 10-class KL grade classification.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple
from PIL import Image
import torchvision.transforms as T

from src.models.kiocmil_with_detection import KiocmilWithDetection
from .utils.config import get_class_names, get_class_list


class EndToEndInference:
    """
    Inference wrapper for End-to-End KIOCMIL models.

    Supports all classification types:
    - 10-class: KL0-a through KL4-b (detailed structure classification)
    - 8-class: KL1-a through KL4-b (pathological cases only)
    - 5-class: KL0 through KL4 (traditional KL grading)
    - 4-class: KL1 through KL4 (pathological cases, merged)
    """

    def __init__(
        self,
        checkpoint_path: str,
        num_classes: int,
        device: str = "cuda",
        image_size: int = 640,
        confidence_threshold: float = 0.0,
    ):
        """
        Initialize End-to-End inference pipeline.

        Args:
            checkpoint_path: Path to model checkpoint (.pt file)
            num_classes: Number of output classes (4, 5, 8, or 10)
            device: Device for inference ('cuda' or 'cpu')
            image_size: Input image size (default: 640)
            confidence_threshold: Minimum confidence for predictions (default: 0.0)
        """
        if num_classes not in [4, 5, 8, 10]:
            raise ValueError(f"num_classes must be 4, 5, 8, or 10, got {num_classes}")

        self.num_classes = num_classes
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Get class names
        self.class_names = get_class_list(num_classes)

        # Load model
        print(f"Loading End-to-End model from {checkpoint_path}...")
        self.model = self._load_model(checkpoint_path, num_classes)
        print(f"✅ Model loaded successfully on {self.device}")
        print(
            f"   Classes: {num_classes}-class ({self.class_names[0]} to {self.class_names[-1]})"
        )

    def _load_model(
        self, checkpoint_path: str, num_classes: int
    ) -> KiocmilWithDetection:
        """Load model from checkpoint."""
        # Initialize model
        model = KiocmilWithDetection(
            backbone_name="yolo11l",
            num_classes=num_classes,
            pretrained_kiocmil=None,
            freeze_kiocmil=False,
        )

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()

        # Print checkpoint info
        epoch = checkpoint.get("epoch", "N/A")
        metrics = checkpoint.get("metrics", {})
        val_acc = metrics.get("val_accuracy", "N/A")
        val_loss = metrics.get("val_loss", "N/A")

        print(f"   Checkpoint: Epoch {epoch}")
        if val_acc != "N/A":
            print(f"   Val Accuracy: {val_acc:.4f}")
        if val_loss != "N/A":
            print(f"   Val Loss: {val_loss:.4f}")

        return model

    def _preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Load and preprocess image for inference.

        Args:
            image_path: Path to input image

        Returns:
            Preprocessed image tensor (1, 3, H, W)
        """
        # Load image
        img = Image.open(image_path).convert("RGB")

        # Transform
        transform = T.Compose(
            [
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
            ]
        )

        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
        return img_tensor

    def _format_prediction(
        self,
        outputs: Dict[str, torch.Tensor],
        image_path: str,
    ) -> Dict:
        """
        Format model outputs into structured prediction.

        Args:
            outputs: Raw model outputs
            image_path: Path to input image

        Returns:
            Formatted prediction dictionary
        """
        # Get logits based on num_classes
        if self.num_classes == 10:
            logits = outputs["logits_10"]
        elif self.num_classes == 8:
            logits = outputs["logits_8"]
        elif self.num_classes == 5:
            logits = outputs["logits_5"]
        elif self.num_classes == 4:
            logits = outputs["logits_4"]
        else:
            raise ValueError(f"Unsupported num_classes: {self.num_classes}")

        # Get prediction
        probs = torch.softmax(logits, dim=1)[0]
        pred_class_id = torch.argmax(probs).item()
        confidence = probs[pred_class_id].item()

        # Format result
        result = {
            "image_path": str(image_path),
            "predicted_class": self.class_names[pred_class_id],
            "predicted_class_id": pred_class_id,
            "confidence": float(confidence),
            "num_classes": self.num_classes,
        }

        # Add class probabilities if confidence threshold is met
        if confidence >= self.confidence_threshold:
            result["class_probabilities"] = {
                self.class_names[i]: float(probs[i].item())
                for i in range(self.num_classes)
            }

        # Add knee detection info with bounding boxes
        if "knee_boxes" in outputs and "knee_confs" in outputs:
            knee_boxes = outputs["knee_boxes"][
                0
            ]  # Shape: (num_knees, 4) - [x1, y1, x2, y2]
            knee_confs = outputs["knee_confs"][0]  # Shape: (num_knees,)

            num_knees = knee_boxes.shape[0]
            result["num_knees_detected"] = num_knees

            # Format knee boxes with confidence scores
            knee_detections = []
            for i in range(num_knees):
                box = knee_boxes[i].cpu().numpy()
                conf = knee_confs[i].cpu().item()
                knee_detections.append(
                    {
                        "bbox": [
                            float(box[0]),
                            float(box[1]),
                            float(box[2]),
                            float(box[3]),
                        ],  # [x1, y1, x2, y2]
                        "confidence": float(conf),
                    }
                )
            result["knee_boxes"] = knee_detections

        # Add lesion detection info
        if "lesion_boxes" in outputs and "lesion_confs" in outputs:
            lesion_boxes = outputs["lesion_boxes"][0]  # Shape: (num_lesions, 4)
            lesion_confs = outputs["lesion_confs"][
                0
            ]  # Shape: (num_lesions, 2) - [JS_conf, OST_conf]

            num_lesions = lesion_boxes.shape[0]
            result["num_lesions_detected"] = num_lesions

            # Format lesion boxes with confidence scores
            lesion_detections = []
            for i in range(num_lesions):
                box = lesion_boxes[i].cpu().numpy()
                # Get max confidence over 2 classes (JS, OST)
                conf_js = lesion_confs[i, 0].cpu().item()
                conf_ost = lesion_confs[i, 1].cpu().item()
                max_conf = max(conf_js, conf_ost)
                lesion_type = "JS" if conf_js > conf_ost else "OST"

                lesion_detections.append(
                    {
                        "bbox": [
                            float(box[0]),
                            float(box[1]),
                            float(box[2]),
                            float(box[3]),
                        ],
                        "confidence": float(max_conf),
                        "type": lesion_type,
                        "conf_js": float(conf_js),
                        "conf_ost": float(conf_ost),
                    }
                )
            result["lesion_boxes"] = lesion_detections

        return result

    def predict_single(self, image_path: str) -> Dict:
        """
        Run inference on a single image.

        Args:
            image_path: Path to input image

        Returns:
            Prediction dictionary with class, confidence, and probabilities
        """
        # Preprocess
        img_tensor = self._preprocess_image(image_path).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(img_tensor)

        # Format prediction
        result = self._format_prediction(outputs, image_path)

        return result

    def predict_batch(
        self,
        image_paths: List[str],
        show_progress: bool = True,
    ) -> List[Dict]:
        """
        Run inference on multiple images.

        Args:
            image_paths: List of paths to input images
            show_progress: Whether to show progress bar

        Returns:
            List of prediction dictionaries
        """
        results = []

        if show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(image_paths, desc="Running inference")
            except ImportError:
                iterator = image_paths
                print(f"Processing {len(image_paths)} images...")
        else:
            iterator = image_paths

        for img_path in iterator:
            try:
                result = self.predict_single(img_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                results.append(
                    {
                        "image_path": str(img_path),
                        "error": str(e),
                    }
                )

        return results

    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model configuration
        """
        return {
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "image_size": self.image_size,
            "device": str(self.device),
            "confidence_threshold": self.confidence_threshold,
        }
