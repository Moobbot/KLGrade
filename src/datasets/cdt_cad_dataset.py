"""
CDT-CAD Dataset Adapter for KLGrade

Converts YOLO format annotations to CDT-CAD format (similar to COCO).
Handles data loading, augmentation, and batch collation.
"""

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random


class CDTCADDataset(Dataset):
    """
    Dataset for CDT-CAD training/validation

    Reads images and YOLO format annotations, applies augmentations,
    and returns data in the format expected by CDT-CAD model.
    """

    def __init__(
        self,
        img_dir: str,
        label_dir: str,
        split_file: Optional[str] = None,
        image_size: Tuple[int, int] = (800, 800),
        augment: bool = False,
        clahe: bool = False,
        num_classes: int = 5,
    ):
        """
        Args:
            img_dir: Directory containing images
            label_dir: Directory containing YOLO format labels
            split_file: Optional .txt file with image names for this split
            image_size: Target image size (height, width)
            augment: Apply data augmentation
            clahe: Apply CLAHE preprocessing (medical imaging enhancement)
            num_classes: Number of classes
        """
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.image_size = image_size
        self.num_classes = num_classes

        # Get image list
        if split_file:
            with open(split_file, "r") as f:
                self.image_names = [line.strip() for line in f.readlines()]
        else:
            self.image_names = [f.stem for f in self.img_dir.glob("*.jpg")]

        # Define transforms
        self.transform = self._get_transforms(augment, clahe)

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Returns:
            image: [3, H, W] normalized tensor
            target: Dict with 'labels' and 'boxes'
        """
        img_name = self.image_names[idx]

        # Load image
        img_path = self.img_dir / f"{img_name}.jpg"
        if not img_path.exists():
            img_path = self.img_dir / f"{img_name}.png"

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load labels
        label_path = self.label_dir / f"{img_name}.txt"
        boxes, labels = self._load_yolo_labels(label_path, image.shape)

        # Apply transforms
        if boxes.size > 0:
            transformed = self.transform(image=image, bboxes=boxes, class_labels=labels)
            image = transformed["image"]
            boxes = np.array(transformed["bboxes"])
            labels = np.array(transformed["class_labels"])
        else:
            # No annotations
            transformed = self.transform(image=image, bboxes=[], class_labels=[])
            image = transformed["image"]
            boxes = np.zeros((0, 4))
            labels = np.array([])

        # Convert to tensors
        if boxes.size > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,  # [num_objects, 4] in (cx, cy, w, h) normalized format
            "labels": labels,  # [num_objects]
            "image_id": idx,
        }

        return image, target

    def _load_yolo_labels(
        self, label_path: Path, img_shape: Tuple
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load YOLO format labels

        Args:
            label_path: Path to label file
            img_shape: (H, W, C) of original image

        Returns:
            boxes: [N, 4] in (cx, cy, w, h) normalized format
            labels: [N] class indices
        """
        if not label_path.exists():
            return np.zeros((0, 4)), np.array([])

        with open(label_path, "r") as f:
            lines = f.readlines()

        if len(lines) == 0:
            return np.zeros((0, 4)), np.array([])

        annotations = []
        class_labels = []

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])

            annotations.append([cx, cy, w, h])
            class_labels.append(class_id)

        if len(annotations) == 0:
            return np.zeros((0, 4)), np.array([])

        boxes = np.array(annotations)  # Already in normalized (cx, cy, w, h) format
        labels = np.array(class_labels)

        return boxes, labels

    def _get_transforms(self, augment: bool, clahe: bool):
        """Define image transforms"""
        transforms_list = []

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if clahe:
            transforms_list.append(
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5)
            )

        # Augmentations
        if augment:
            transforms_list.extend(
                [
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=0.5
                    ),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                    A.Rotate(limit=10, p=0.3),
                    A.Affine(scale=(0.9, 1.1), translate_percent=0.1, p=0.3),
                ]
            )

        # Resize and normalize
        transforms_list.extend(
            [
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

        return A.Compose(
            transforms_list,
            bbox_params=A.BboxParams(
                format="yolo", label_fields=["class_labels"], min_visibility=0.3
            ),
        )


def collate_fn(
    batch: List[Tuple[torch.Tensor, Dict]],
) -> Tuple[torch.Tensor, List[Dict]]:
    """
    Custom collate function for variable-size annotations

    Args:
        batch: List of (image, target) tuples

    Returns:
        images: Batched images [B, 3, H, W]
        targets: List of target dicts
    """
    images = []
    targets = []

    for img, target in batch:
        images.append(img)
        targets.append(target)

    images = torch.stack(images, dim=0)

    return images, targets
