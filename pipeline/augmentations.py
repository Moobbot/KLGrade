"""
Augmentations and imbalance utilities for classification.

- Albumentations pipelines (train/val)
- Mixup and CutMix collates
- WeightedRandomSampler builder from class frequencies
"""
import os
import sys
from typing import Tuple

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import IMG_SIZE


def get_train_transforms(size: int | None = None):
    if size is None:
        size = IMG_SIZE
    return A.Compose(
        [
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            # ShiftScaleRotate is deprecated in favor of Affine; replace to avoid warnings
            A.Affine(
                translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
                scale=(1 - 0.05, 1 + 0.05),
                rotate=(-5, 5),
                shear=None,
                fit_output=False,
                mode=0,   # cv2.BORDER_CONSTANT
                cval=0,
                p=0.7,
            ),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.15, p=0.5),
            # Use GaussianNoise class to avoid var_limit warning on some versions
            A.GaussianNoise(var_limit=(5.0, 15.0), p=0.2),
            A.MotionBlur(blur_limit=3, p=0.2),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
            ToTensorV2(),
        ]
    )


def get_val_transforms(size: int | None = None):
    if size is None:
        size = IMG_SIZE
    return A.Compose(
        [
            A.Resize(size, size),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
            ToTensorV2(),
        ]
    )


def build_weighted_sampler(csv_path: str):
    df = pd.read_csv(csv_path)
    counts = df["label"].value_counts().to_dict()
    total = len(df)
    # inverse frequency weights
    weights = df["label"].map(lambda c: total / (len(counts) * counts[int(c)])).values
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=total, replacement=True)


def build_weighted_sampler_from_labels(labels):
    """Build a WeightedRandomSampler for a given list/array of labels (subset-aware)."""
    import numpy as np

    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)
    freq = {int(k): int(v) for k, v in zip(unique, counts)}
    total = len(labels)
    weights = [total / (len(freq) * freq[int(c)]) for c in labels]
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=total, replacement=True)


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return x, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, (y_a, y_b), lam


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return x, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size, _, H, W = x.size()
    index = torch.randperm(batch_size, device=x.device)
    # Compute cutout region
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    cut_w = int(W * np.sqrt(1 - lam))
    cut_h = int(H * np.sqrt(1 - lam))
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    # Apply CutMix
    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    y_a, y_b = y, y[index]
    return x, (y_a, y_b), lam


def mix_criterion(criterion, pred, target):
    if isinstance(target, tuple):
        y_a, y_b, lam = target[0], target[1], target[2]
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    return criterion(pred, target)
