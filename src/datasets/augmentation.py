"""
Data Augmentation Transforms for Knee OA Detection

Provides conservative augmentation pipelines specifically designed for
medical X-ray images, preserving anatomical correctness while increasing
dataset diversity.

Safe transformations:
- Horizontal flip (L/R knee symmetry)
- Slight rotation (±5°)
- Intensity adjustments (brightness, contrast, gamma)
- Minimal blur and noise

Avoid:
- Vertical flip (anatomically incorrect)
- Heavy rotation (>20°)
- Color jittering (grayscale X-rays)
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


def get_conservative_train_transform(img_size=(640, 640)):
    """
    Conservative augmentation for medical X-ray images.

    Safe transformations that preserve anatomical correctness:
    - Horizontal flip (L/R knee symmetry)
    - Slight rotation (±5°)
    - Brightness/contrast adjustment
    - Minimal blur/noise

    Args:
        img_size: Target image size (height, width)

    Returns:
        Albumentations Compose object with bbox support
    """
    return A.Compose(
        [
            # Geometric (anatomically valid)
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=5, p=0.5, border_mode=0),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=0,  # Rotation handled above
                p=0.5,
                border_mode=0,
            ),
            # Intensity (X-ray exposure variation)
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            # Image quality (simulate capture conditions)
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(var_limit=(5, 15), p=0.2),
            # Resize to target size
            A.Resize(img_size[0], img_size[1]),
            # Normalize
            A.Normalize(mean=0.5, std=0.5),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="yolo", label_fields=["class_labels"], min_area=0, min_visibility=0
        ),
    )


def get_moderate_train_transform(img_size=(640, 640)):
    """
    Moderate augmentation with heavier transformations.

    For experimentation to see if heavier augmentation helps
    with small dataset size.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20, p=0.7, border_mode=0),
            A.ShiftScaleRotate(
                shift_limit=0.15, scale_limit=0.15, rotate_limit=0, p=0.7, border_mode=0
            ),
            # Intensity
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.RandomGamma(gamma_limit=(70, 130), p=0.5),
            # Quality
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=5, p=1.0),
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                ],
                p=0.3,
            ),
            A.GaussNoise(var_limit=(5, 20), p=0.3),
            A.Resize(img_size[0], img_size[1]),
            A.Normalize(mean=0.5, std=0.5),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="yolo", label_fields=["class_labels"], min_area=0, min_visibility=0
        ),
    )


def get_val_transform(img_size=(640, 640)):
    """
    Validation transform - no augmentation.

    Only resize and normalize to match training preprocessing.
    """
    return A.Compose(
        [
            A.Resize(img_size[0], img_size[1]),
            A.Normalize(mean=0.5, std=0.5),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="yolo", label_fields=["class_labels"], min_area=0, min_visibility=0
        ),
    )


if __name__ == "__main__":
    # Test transforms
    import cv2

    # Load sample image
    img = cv2.imread(
        "processed/knee/images/0_P00001_20190701_knee0.jpg", cv2.IMREAD_GRAYSCALE
    )
    bboxes = [[0.5, 0.5, 0.2, 0.3]]
    class_labels = [2]

    # Test conservative transform
    transform = get_conservative_train_transform()
    result = transform(image=img, bboxes=bboxes, class_labels=class_labels)

    print("✅ Augmentation test passed!")
    print(f"   Image shape: {result['image'].shape}")
    print(f"   Bboxes: {result['bboxes']}")
    print(f"   Classes: {result['class_labels']}")
