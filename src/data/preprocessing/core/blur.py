"""
Core preprocessing operations - Blur module.

Provides various blur operations for noise reduction.
"""

import cv2
import numpy as np
from typing import Tuple


def gaussian_blur(
    image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5)
) -> np.ndarray:
    """
    Apply Gaussian blur to image.

    Args:
        image: Input image
        kernel_size: Size of Gaussian kernel (must be odd)

    Returns:
        Blurred image
    """
    return cv2.GaussianBlur(image, kernel_size, 0)


def median_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply median blur to image.

    Args:
        image: Input image
        kernel_size: Size of median filter (must be odd)

    Returns:
        Blurred image
    """
    return cv2.medianBlur(image, kernel_size)


def bilateral_filter(
    image: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75
) -> np.ndarray:
    """
    Apply bilateral filter (edge-preserving blur).

    Args:
        image: Input image
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in color space
        sigma_space: Filter sigma in coordinate space

    Returns:
        Filtered image
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
