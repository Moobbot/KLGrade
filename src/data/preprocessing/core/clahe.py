"""
Core preprocessing operations - CLAHE module.

Provides histogram equalization operations for contrast enhancement.
"""

import cv2
import numpy as np
from typing import Tuple


def apply_clahe(
    image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Args:
        image: Input grayscale image
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization

    Returns:
        Enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def adaptive_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Apply standard adaptive histogram equalization (no clipping).

    Args:
        image: Input grayscale image

    Returns:
        Enhanced image
    """
    # Use CLAHE with very high clip limit (effectively no limiting)
    clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Apply global histogram equalization.

    Args:
        image: Input grayscale image

    Returns:
        Equalized image
    """
    return cv2.equalizeHist(image)
