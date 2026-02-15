"""
Two-Step YOLO Inference API

Step 1: Knee Detection (YOLO11l)
Step 2: Lesion Detection on cropped knees (YOLO11l)

This API uses two YOLO models sequentially for end-to-end KL grading.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
import torch


class TwoStepYOLOInference:
    """
    Two-step YOLO inference pipeline:
    1. Detect knees in full X-ray
    2. Detect lesions in cropped knee regions
    """

    def __init__(
        self,
        knee_model_path: str,
        lesion_model_path: str,
        device: str = "cuda:0",
        knee_conf_threshold: float = 0.75,
        lesion_conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        """
        Initialize two-step YOLO pipeline.

        Args:
            knee_model_path: Path to knee detection model
            lesion_model_path: Path to lesion detection model
            device: Device to run inference on
            knee_conf_threshold: Confidence threshold for knee detection (default 0.75)
            lesion_conf_threshold: Confidence threshold for lesion detection (default 0.25)
            iou_threshold: IoU threshold for NMS
        """
        self.device = device
        self.knee_conf_threshold = knee_conf_threshold
        self.lesion_conf_threshold = lesion_conf_threshold
        self.iou_threshold = iou_threshold

        # Load models
        print(f"Loading knee detector: {knee_model_path}")
        self.knee_model = YOLO(knee_model_path)

        print(f"Loading lesion detector: {lesion_model_path}")
        self.lesion_model = YOLO(lesion_model_path)

        print(f"Models loaded on {device}")

    def detect_knees(self, image: np.ndarray) -> List[Dict]:
        """
        Step 1: Detect knee regions in full X-ray.

        Args:
            image: Input image (H, W, 3) RGB

        Returns:
            List of knee detections with boxes and confidence
        """
        results = self.knee_model.predict(
            image,
            conf=self.knee_conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        knees = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                conf = boxes.conf[i].cpu().item()

                knees.append(
                    {
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": float(conf),
                        "knee_id": i,
                    }
                )

        return knees

    def detect_lesions(self, knee_image: np.ndarray, knee_id: int = 0) -> List[Dict]:
        """
        Step 2: Detect lesions in cropped knee region.

        Args:
            knee_image: Cropped knee image (H, W, 3) RGB
            knee_id: ID of the knee (for tracking)

        Returns:
            List of lesion detections with boxes, class, and confidence
        """
        results = self.lesion_model.predict(
            knee_image,
            conf=self.lesion_conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        lesions = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                conf = boxes.conf[i].cpu().item()
                cls = int(boxes.cls[i].cpu().item())

                lesions.append(
                    {
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "class": cls,
                        "class_name": result.names[cls],
                        "confidence": float(conf),
                        "knee_id": knee_id,
                    }
                )

        return lesions

    def predict(
        self, image_input: str | np.ndarray, return_crops: bool = False
    ) -> Dict:
        """
        Run full two-step inference pipeline.

        Args:
            image_input: Path to input X-ray image OR numpy array (H, W, 3) RGB
            return_crops: Whether to return cropped knee images

        Returns:
            Dictionary with knees, lesions, and optional crops
        """
        image_path = "memory"

        # Load image
        if isinstance(image_input, str):
            image_path = image_input
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, np.ndarray):
            # Assume input is BGR (opencv standard) or check channel count
            image_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Input must be file path or numpy array")

        h, w = image_rgb.shape[:2]

        # Step 1: Detect knees
        knees = self.detect_knees(image_rgb)

        if len(knees) == 0:
            return {
                "image_path": image_path,
                "image_size": [h, w],
                "knees": [],
                "lesions": [],
                "kl_grade": None,
                "message": "No knees detected",
            }

        # Step 2: Detect lesions in each knee
        all_lesions = []
        knee_crops = []

        for knee in knees:
            x1, y1, x2, y2 = knee["bbox"]

            # Crop knee region
            knee_crop = image_rgb[y1:y2, x1:x2]

            if return_crops:
                knee_crops.append(knee_crop.copy())

            # Detect lesions
            lesions = self.detect_lesions(knee_crop, knee["knee_id"])

            # Convert lesion boxes to full image coordinates
            for lesion in lesions:
                lx1, ly1, lx2, ly2 = lesion["bbox"]
                lesion["bbox_global"] = [
                    lx1 + x1,
                    ly1 + y1,
                    lx2 + x1,
                    ly2 + y1,
                ]

            all_lesions.extend(lesions)

        # Determine KL grade from lesions
        kl_grade = self._determine_kl_grade(all_lesions)

        result = {
            "image_path": image_path,
            "image_size": [h, w],
            "knees": knees,
            "lesions": all_lesions,
            "kl_grade": kl_grade,
        }

        if return_crops:
            result["knee_crops"] = knee_crops

        return result

    def _determine_kl_grade(self, lesions: List[Dict]) -> Optional[int]:
        """
        Determine KL grade from detected lesions.

        Mapping strategy based on model class count:
        - 5-class (0-4): Direct mapping (0->KL0, ... 4->KL4)
        - 8-class (0-7): Assumes KL1-KL4 (Osteophytes/JSN).
                         0,1->KL1; 2,3->KL2; 4,5->KL3; 6,7->KL4.
                         Returns 0 if no lesions detected.
        - 10-class (0-9): 0,1->KL0; 2,3->KL1; 4,5->KL2; 6,7->KL3; 8,9->KL4.

        Args:
            lesions: List of lesion detections

        Returns:
            KL grade (0-4)
        """
        if not lesions:
            return 0  # Default to KL0 if no lesions detected

        # Get highest class
        max_class = max(lesion["class"] for lesion in lesions)

        # Determine mapping based on model vocabulary size
        num_classes = len(self.lesion_model.names)

        if num_classes == 8:
            # 8-class: 0-7 mapping to KL1-4
            # 0,1 -> 1 (0//2 + 1 = 1)
            # ...
            # 6,7 -> 4 (6//2 + 1 = 4)
            return (max_class // 2) + 1

        elif num_classes == 10:
            # 10-class: 0-9 mapping to KL0-4
            # 0,1 -> 0
            # 2,3 -> 1
            return max_class // 2

        elif num_classes == 5:
            # 5-class: Direct mapping
            return max_class

        else:
            # Fallback for unknown config, assume direct mapping or warn
            # For safety, return max_class clipped to 4?
            # Or assume it matches 5-class logic if small count.
            return min(max_class, 4)

    def visualize(
        self, image_path: str, output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Run inference and visualize results.

        Args:
            image_path: Path to input image
            output_path: Optional path to save visualization

        Returns:
            Annotated image
        """
        # Run inference
        result = self.predict(image_path)

        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Draw knee boxes (green)
        for knee in result["knees"]:
            x1, y1, x2, y2 = knee["bbox"]
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                image_rgb,
                f"Knee {knee['knee_id']}: {knee['confidence']:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # Draw lesion boxes (red)
        for lesion in result["lesions"]:
            x1, y1, x2, y2 = lesion["bbox_global"]
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                image_rgb,
                f"{lesion['class_name']}: {lesion['confidence']:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )

        # Draw KL grade
        kl_grade = result["kl_grade"]
        if kl_grade is not None:
            cv2.putText(
                image_rgb,
                f"KL Grade: {kl_grade}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                3,
            )

        # Save if output path provided
        if output_path:
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, image_bgr)
            print(f"Saved visualization to: {output_path}")

        return image_rgb


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Two-Step YOLO Inference")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument(
        "--knee-model",
        type=str,
        default="runs/detect/knee_detector/weights/best.pt",
        help="Knee detection model",
    )
    parser.add_argument(
        "--lesion-model",
        type=str,
        default="runs/detect/lesion_8class_balanced/weights/best.pt",
        help="Lesion detection model",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output visualization path"
    )
    parser.add_argument(
        "--knee-conf",
        type=float,
        default=0.75,
        help="Knee detection confidence threshold",
    )
    parser.add_argument(
        "--lesion-conf",
        type=float,
        default=0.25,
        help="Lesion detection confidence threshold",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = TwoStepYOLOInference(
        knee_model_path=args.knee_model,
        lesion_model_path=args.lesion_model,
        device=args.device,
        knee_conf_threshold=args.knee_conf,
        lesion_conf_threshold=args.lesion_conf,
    )

    # Run inference
    result = pipeline.predict(args.image)

    # Print results
    print("\n" + "=" * 60)
    print("Two-Step YOLO Inference Results")
    print("=" * 60)
    print(f"Image: {result['image_path']}")
    print(f"Knees detected: {len(result['knees'])}")
    print(f"Lesions detected: {len(result['lesions'])}")
    print(f"KL Grade: {result['kl_grade']}")
    print("=" * 60)

    # Visualize
    if args.output or True:
        output_path = args.output or "two_step_yolo_result.jpg"
        pipeline.visualize(args.image, output_path)
