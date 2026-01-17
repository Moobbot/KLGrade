"""
GradCAM Visualization Module

Provides GradCAM (Gradient-weighted Class Activation Mapping) for YOLO models.
"""

import cv2
import numpy as np
import torch


class YOLOGradCAM:
    """
    GradCAM implementation for YOLO models.

    Generates heatmaps showing which regions of the image the model focuses on
    when making predictions.
    """

    def __init__(self, model, target_layer):
        """
        Initialize GradCAM.

        Args:
            model: PyTorch model (e.g., YOLO model.model)
            target_layer: Target layer for CAM generation (e.g., model.model[9])
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        """Hook to save forward pass activations."""
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients."""
        # grad_output is a tuple, usually (grad,)
        self.gradients = grad_output[0]

    def __call__(self, input_tensor, target_category=None):
        """
        Generate GradCAM heatmap.

        Args:
            input_tensor: Input image tensor [1, C, H, W]
            target_category: Optional target class (not used in current implementation)

        Returns:
            numpy.ndarray: Normalized CAM heatmap [H, W] in range [0, 1]
        """
        # Forward pass
        preds = self.model(input_tensor)
        output = preds[0]  # tensor

        # Output shape is usually [1, num_classes+4, num_anchors]
        # Transpose to [1, num_anchors, num_classes+4] for easier indexing
        output = output.transpose(1, 2)

        # Find best detection - scores are from index 4 onwards
        scores = output[0, :, 4:]
        max_score, max_idx = torch.max(scores.flatten(), dim=0)

        # Backpropagate
        self.model.zero_grad()
        max_score.backward()

        # Generate CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]

        # Global Average Pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))

        # Weighted combination of activations
        cam = torch.zeros(
            activations.shape[1:], dtype=torch.float32, device=activations.device
        )
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam = cam.cpu().detach().numpy()

        # Normalize
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        if np.max(cam) > 0:
            cam = cam - np.min(cam)
            cam = cam / np.max(cam)

        return cam


def apply_colormap(cam, image, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Apply colormap to CAM and overlay on image.

    Args:
        cam: CAM heatmap [H, W] in range [0, 1]
        image: Original image (numpy array, RGB or BGR)
        alpha: Blending factor (default: 0.5)
        colormap: OpenCV colormap (default: COLORMAP_JET)

    Returns:
        numpy.ndarray: Heatmap overlaid on image
    """
    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), colormap)
    heatmap = np.float32(heatmap) / 255

    # Ensure image is float
    if image.dtype == np.uint8:
        image_float = np.float32(image) / 255
    else:
        image_float = np.float32(image)

    # Overlay
    overlaid = heatmap * alpha + image_float * (1 - alpha)
    overlaid = overlaid / np.max(overlaid)
    overlaid = np.uint8(255 * overlaid)

    return overlaid


def resize_cam_to_crop(cam, crop_size):
    """
    Resize CAM to match crop dimensions.

    Args:
        cam: CAM heatmap
        crop_size: (width, height) tuple

    Returns:
        numpy.ndarray: Resized CAM
    """
    return cv2.resize(cam, crop_size)
