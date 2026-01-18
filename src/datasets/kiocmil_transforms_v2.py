"""
Augmentation transforms for KIOCMIL Dataset V2.

This module separates geometric and photometric transforms to fix the augmentation pipeline:
- Geometric transforms: Applied to full image BEFORE cropping patches (rotation, flip, shift)
- Photometric transforms: Applied to individual patches AFTER cropping (brightness, CLAHE, blur)
- Normalization and tensor conversion: Applied LAST after all augmentations

Date: 2026-01-13
"""

import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Optional


class GeometricAugmentation:
    """
    Geometric augmentation transforms applied to the full image before cropping.

    These transforms modify spatial properties (rotation, flip, shift) and should
    be applied before any bbox operations to ensure consistency.
    """

    def __init__(self, level: str = "strong"):
        """
        Initialize geometric augmentation pipeline.

        Args:
            level: Augmentation strength ('none', 'light', 'medium', 'strong')
        """
        self.level = level
        self.transform = self._build_transform()

    def _build_transform(self) -> A.Compose:
        """Build the augmentation pipeline based on level."""
        if self.level == "none":
            return A.Compose([])

        transforms = []

        if self.level == "light":
            transforms = [
                A.HorizontalFlip(p=0.3),
                A.Rotate(limit=10, p=0.3),
            ]
        elif self.level == "medium":
            transforms = [
                A.HorizontalFlip(p=0.4),
                A.Rotate(limit=15, p=0.4),
                A.ShiftScaleRotate(
                    shift_limit=0.03, scale_limit=0.03, rotate_limit=10, p=0.2
                ),
            ]
        elif self.level == "strong":
            transforms = [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.3
                ),
            ]
        else:
            raise ValueError(f"Unknown augmentation level: {self.level}")

        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format="yolo",  # Input format is YOLO (cx, cy, w, h) normalized [0-1]
                label_fields=[
                    "class_labels"
                ],  # Track class IDs with bbox transformations
                min_visibility=0.3,  # Drop bboxes that are <30% visible after transform
                min_area=100,  # Drop very small bboxes (in pixels after transform)
            ),
        )

    def __call__(
        self, image: np.ndarray, bboxes: list = None, class_labels: list = None
    ) -> dict:
        """
        Apply geometric augmentation to image and optionally to bboxes.

        Args:
            image: Input image as numpy array (H, W, 3) uint8
            bboxes: Optional list of bboxes in YOLO format [[cx, cy, w, h], ...]
            class_labels: Optional list of class IDs for each bbox

        Returns:
            Dictionary with keys:
                - 'image': Augmented image (H, W, 3) uint8
                - 'bboxes': Transformed bboxes (if provided)
                - 'class_labels': Class labels for transformed bboxes (if provided)
        """
        # Handle None inputs
        if bboxes is None:
            bboxes = []

        if class_labels is None:
            if len(bboxes) > 0:
                class_labels = [0] * len(bboxes)
            else:
                class_labels = []

        # If level is none, no bbox_params are set in Compose
        if self.level == "none":
            # Transform image only
            result = self.transform(image=image)
            # Pass through bboxes/labels as they are not transformed
            result["bboxes"] = bboxes
            result["class_labels"] = class_labels
            return result

        # For non-none levels, bbox_params are set with label_fields.
        # Albumentations expects bboxes and class_labels to be passed,
        # even if they are empty lists.
        return self.transform(image=image, bboxes=bboxes, class_labels=class_labels)


class PhotometricAugmentation:
    """
    Photometric augmentation transforms applied to individual patches after cropping.

    These transforms modify color and intensity properties (brightness, contrast, CLAHE)
    and are applied independently to each cropped patch.
    """

    def __init__(self, level: str = "strong", use_clahe: bool = True):
        """
        Initialize photometric augmentation pipeline.

        Args:
            level: Augmentation strength ('none', 'light', 'medium', 'strong')
            use_clahe: Whether to include CLAHE (Contrast Limited Adaptive Histogram Equalization)
        """
        self.level = level
        self.use_clahe = use_clahe
        self.transform = self._build_transform()

    def _build_transform(self) -> A.Compose:
        """Build the augmentation pipeline based on level."""
        if self.level == "none":
            return A.Compose([])

        transforms = []

        if self.level == "light":
            transforms = [
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.3
                ),
                A.RandomGamma(gamma_limit=(90, 110), p=0.2),
            ]
            if self.use_clahe:
                transforms.append(A.CLAHE(clip_limit=2.0, p=0.2))

        elif self.level == "medium":
            transforms = [
                A.RandomBrightnessContrast(
                    brightness_limit=0.15, contrast_limit=0.15, p=0.4
                ),
                A.RandomGamma(gamma_limit=(85, 115), p=0.25),
                A.GaussianBlur(blur_limit=3, p=0.15),
            ]
            if self.use_clahe:
                transforms.append(A.CLAHE(clip_limit=2.0, p=0.25))

        elif self.level == "strong":
            transforms = [
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.2),
            ]
            if self.use_clahe:
                transforms.append(A.CLAHE(clip_limit=2.0, p=0.3))
        else:
            raise ValueError(f"Unknown augmentation level: {self.level}")

        return A.Compose(transforms)

    def __call__(self, image: np.ndarray) -> dict:
        """
        Apply photometric augmentation to an image patch.

        Args:
            image: Input patch as numpy array (H, W, 3) uint8

        Returns:
            Dictionary with 'image' key containing augmented patch (H, W, 3) uint8
        """
        return self.transform(image=image)


def normalize_patch(patch: np.ndarray) -> np.ndarray:
    """
    Normalize a patch with ImageNet statistics.

    Converts from uint8 [0, 255] to float32 with ImageNet normalization:
    - mean = [0.485, 0.456, 0.406]
    - std = [0.229, 0.224, 0.225]

    Args:
        patch: Input patch as numpy array (H, W, 3) uint8

    Returns:
        Normalized patch as numpy array (H, W, 3) float32
    """
    # Convert to float32 and scale to [0, 1]
    patch = patch.astype(np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    patch = (patch - mean) / std

    return patch


def patch_to_tensor(patch: np.ndarray) -> torch.Tensor:
    """
    Convert a normalized patch to PyTorch tensor.

    Args:
        patch: Normalized patch as numpy array (H, W, 3) float32

    Returns:
        PyTorch tensor (3, H, W) float32
    """
    # Convert to tensor and permute dimensions
    tensor = torch.from_numpy(patch).permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)
    return tensor


def get_geometric_transforms(level: str = "strong") -> Optional[GeometricAugmentation]:
    """
    Factory function to create geometric augmentation pipeline.

    Args:
        level: Augmentation strength ('none', 'light', 'medium', 'strong')

    Returns:
        GeometricAugmentation instance or None if level is 'none'
    """
    if level == "none":
        return None
    return GeometricAugmentation(level=level)


def get_photometric_transforms(
    level: str = "strong", use_clahe: bool = True
) -> Optional[PhotometricAugmentation]:
    """
    Factory function to create photometric augmentation pipeline.

    Args:
        level: Augmentation strength ('none', 'light', 'medium', 'strong')
        use_clahe: Whether to include CLAHE augmentation

    Returns:
        PhotometricAugmentation instance or None if level is 'none'
    """
    if level == "none":
        return None
    return PhotometricAugmentation(level=level, use_clahe=use_clahe)
