"""
Data augmentation transforms for KIOCMIL dataset.

Includes enhanced augmentations learned from preprocessing.py:
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Rotation
- Brightness/Contrast
- Gaussian Blur
"""

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform(
    img_size=(640, 640), augmentation_level="strong", use_clahe=True
):
    """
    Create training augmentation pipeline.

    Args:
        img_size: Target image size (height, width)
        augmentation_level: 'basic', 'medium', or 'strong'
        use_clahe: Whether to use CLAHE preprocessing

    Returns:
        Albumentations Compose transform
    """
    transforms = []

    if augmentation_level == "strong":
        # Geometric augmentations
        transforms.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.Rotate(
                    limit=15,
                    p=0.5,
                    border_mode=cv2.BORDER_REFLECT,
                    interpolation=cv2.INTER_LINEAR,
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=10,
                    p=0.5,
                    border_mode=cv2.BORDER_REFLECT,
                    interpolation=cv2.INTER_LINEAR,
                ),
            ]
        )

        # Brightness/Contrast augmentations
        transforms.extend(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            ]
        )

        # CLAHE - learned from preprocessing.py
        if use_clahe:
            transforms.append(
                A.CLAHE(
                    clip_limit=2.0,
                    tile_grid_size=(8, 8),
                    p=0.8,  # Apply frequently for better contrast
                )
            )

        # Noise reduction
        transforms.append(A.GaussianBlur(blur_limit=(3, 5), p=0.3))

    elif augmentation_level == "medium":
        transforms.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.3, border_mode=cv2.BORDER_REFLECT),
                A.RandomBrightnessContrast(
                    brightness_limit=0.15, contrast_limit=0.15, p=0.4
                ),
            ]
        )

        if use_clahe:
            transforms.append(A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.6))

    elif augmentation_level == "basic":
        transforms.extend(
            [
                A.HorizontalFlip(p=0.5),
            ]
        )

        if use_clahe:
            transforms.append(A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5))

    # NOTE: Do NOT add Normalize or ToTensorV2 here!
    # Dataset needs to crop patches from numpy array first,
    # then normalize and convert to tensor for each patch separately

    return A.Compose(transforms)


def get_val_transform(img_size=(640, 640), use_clahe=True):
    """
    Create validation transform pipeline.
    Only CLAHE + normalization, no random augmentations.

    Args:
        img_size: Target image size
        use_clahe: Whether to apply CLAHE

    Returns:
        Albumentations Compose transform
    """
    transforms = []

    if use_clahe:
        transforms.append(A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0))

    # NOTE: Do NOT add Normalize or ToTensorV2 here!
    # Dataset handles these after cropping patches

    return A.Compose(transforms)


# Convenience function for quick setup
def get_kiocmil_transforms(train=True, augmentation_level="strong", use_clahe=True):
    """
    Get transform for KIOCMIL dataset.

    Args:
        train: If True, return training transform, else validation
        augmentation_level: 'basic', 'medium', or 'strong' (training only)
        use_clahe: Whether to use CLAHE preprocessing

    Returns:
        Albumentations Compose transform
    """
    if train:
        return get_train_transform(
            augmentation_level=augmentation_level, use_clahe=use_clahe
        )
    else:
        return get_val_transform(use_clahe=use_clahe)
