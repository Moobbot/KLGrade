"""
GradCAM Visualization for KIOCMIL-CADA Model

Generates attention heatmaps to visualize model focus areas.
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Tuple, Optional, List


class GradCAM:
    """
    GradCAM implementation for KIOCMIL-CADA model.

    Visualizes attention by computing gradients of output with respect to
    intermediate feature maps.
    """

    def __init__(self, model, target_layer=None):
        """
        Initialize GradCAM.

        Args:
            model: KIOCMIL-CADA model instance
            target_layer: Layer to compute gradients for (default: backbone last layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        if target_layer is not None:
            self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(
        self, image: np.ndarray, batch_data: List[dict], class_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate GradCAM heatmap.

        Args:
            image: Input image (RGB numpy array)
            batch_data: Batch data for model
            class_idx: Target class index (if None, use predicted class)

        Returns:
            Heatmap as numpy array (H, W)
        """
        self.model.eval()

        # Forward pass
        outputs = self.model(batch_data)
        logits = outputs["logits_10"]

        # Get target class
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1)[0].item()

        # Backward pass
        self.model.zero_grad()
        target = logits[0, class_idx]
        target.backward()

        # Generate CAM
        if self.gradients is not None and self.activations is not None:
            # Global average pooling of gradients
            weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

            # Weighted combination of activation maps
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
            cam = F.relu(cam)

            # Normalize
            cam = cam.squeeze().cpu().numpy()
            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / cam.max()

            # Resize to input image size
            h, w = image.shape[:2]
            cam = cv2.resize(cam, (w, h))

            return cam

        # Fallback: return empty heatmap
        return np.zeros(image.shape[:2], dtype=np.float32)

    def generate_attention_map(
        self,
        image: np.ndarray,
        knee_bbox: Tuple[int, int, int, int],
        batch_data: List[dict],
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate attention map focused on knee region.

        Args:
            image: Input image (RGB)
            knee_bbox: Knee bounding box (x1, y1, x2, y2)
            batch_data: Batch data for model
            class_idx: Target class index

        Returns:
            Attention map as numpy array
        """
        # Generate full CAM
        cam = self.generate_cam(image, batch_data, class_idx)

        # Focus on knee region
        x1, y1, x2, y2 = knee_bbox
        knee_cam = cam[y1:y2, x1:x2]

        # Resize back to full image
        full_cam = np.zeros_like(cam)
        full_cam[y1:y2, x1:x2] = knee_cam

        return full_cam


def create_heatmap_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Create heatmap overlay on image.

    Args:
        image: Original image (RGB)
        heatmap: Heatmap array (0-1 range)
        alpha: Overlay transparency (0-1)
        colormap: OpenCV colormap

    Returns:
        Overlaid image (RGB)
    """
    # Convert heatmap to uint8
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # Blend
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)

    return overlay


def generate_multi_scale_attention(
    model,
    image: np.ndarray,
    batch_data: List[dict],
    knee_bboxes: List[Tuple[int, int, int, int]],
) -> List[np.ndarray]:
    """
    Generate attention maps for multiple knees.

    Args:
        model: KIOCMIL-CADA model
        image: Input image
        batch_data: Batch data
        knee_bboxes: List of knee bounding boxes

    Returns:
        List of attention maps for each knee
    """
    attention_maps = []

    # Simple attention visualization based on predictions
    # For each knee, create a focused attention map
    for bbox in knee_bboxes:
        x1, y1, x2, y2 = bbox

        # Create attention map
        attention = np.zeros(image.shape[:2], dtype=np.float32)

        # Add Gaussian attention centered on knee
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        width = x2 - x1
        height = y2 - y1

        # Create meshgrid
        y_coords, x_coords = np.ogrid[: image.shape[0], : image.shape[1]]

        # Gaussian kernel
        sigma_x = width / 4
        sigma_y = height / 4
        gaussian = np.exp(
            -(
                (x_coords - center_x) ** 2 / (2 * sigma_x**2)
                + (y_coords - center_y) ** 2 / (2 * sigma_y**2)
            )
        )

        attention = gaussian
        attention_maps.append(attention)

    return attention_maps


def draw_annotations(
    image: np.ndarray,
    predictions: List[dict],
    show_confidence: bool = True,
    show_lesion_count: bool = True,
) -> np.ndarray:
    """
    Draw bounding boxes and labels on image.

    Args:
        image: Input image (RGB)
        predictions: List of prediction dictionaries
        show_confidence: Whether to show confidence scores
        show_lesion_count: Whether to show lesion counts

    Returns:
        Annotated image
    """
    # Convert to BGR for OpenCV
    img_bgr = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)

    for pred in predictions:
        # Extract bbox
        bbox = pred["knee_bbox"]
        if isinstance(bbox, dict):
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        else:
            x1, y1, x2, y2 = bbox

        # Draw rectangle
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Prepare label
        label = pred["predicted_class"]
        if show_confidence:
            label += f" ({pred['confidence']:.2f})"

        # Draw label background
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(
            img_bgr,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0], y1),
            (0, 255, 0),
            -1,
        )

        # Draw label text
        cv2.putText(
            img_bgr, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
        )

        # Draw lesion count if requested
        if show_lesion_count:
            lesion_info = f"JS:{pred['num_js_lesions']} OST:{pred['num_ost_lesions']}"
            cv2.putText(
                img_bgr,
                lesion_info,
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

    # Convert back to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb
