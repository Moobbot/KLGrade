"""
KIOCMIL Dataset V3 - Enhanced for CADA Architecture

Includes bbox information for deformable attention:
- Context bbox (knee region)
- JS (Joint Space) lesion bboxes
- Ost (Osteophyte) lesion bboxes

This enables the CADA model to use spatial information for adaptive attention.
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional
import warnings

from src.config import OST_CLASSES, JS_CLASSES, KNEE_CLASS_ID, CLASSES_10_CLASS


from src.datasets.kiocmil_transforms_v2 import normalize_patch, patch_to_tensor


class KiocmilDatasetV3(Dataset):
    """
    KIOCMIL Dataset V3 with bbox information for deformable attention.

    Enhanced from V2 to include:
    - Context bbox coordinates
    - Lesion bbox coordinates (JS and Ost)
    - Positional encoding support

    Dataset structure:
    {
      'knees': [
        {
          'ctx': (3, H, W),
          'ctx_bbox': (4,) - [cx, cy, w, h] normalized
          'js': (n, 3, h, w),
          'js_bboxes': (n, 4),
          'ost': (m, 3, h, w),
          'ost_bboxes': (m, 4),
        },
        ...
      ]
    }
    """

    def __init__(
        self,
        img_dir: str,
        knee_label_dir: str,
        lesion_label_dir: str,
        split_file: Optional[str] = None,
        geometric_transform=None,
        photometric_transform=None,
        ctx_size: Tuple[int, int] = (384, 384),
        patch_size: Tuple[int, int] = (224, 224),
        pad_ratio: float = 0.1,
        min_box_area_px: int = 100,
    ):
        self.img_dir = Path(img_dir)
        self.knee_label_dir = Path(knee_label_dir)
        self.lesion_label_dir = Path(lesion_label_dir)
        self.geometric_transform = geometric_transform
        self.photometric_transform = photometric_transform
        self.ctx_size = ctx_size
        self.patch_size = patch_size
        self.pad_ratio = pad_ratio
        self.min_box_area_px = min_box_area_px

        # Load image list
        self.image_files = self._load_image_list(split_file)

        # Load labels
        self.labels_map = self._load_all_labels()

    def _load_image_list(self, split_file: Optional[str]) -> List[str]:
        """Load image file list from split file or directory."""
        if split_file and os.path.exists(split_file):
            with open(split_file, "r") as f:
                stems = [Path(line.strip()).stem for line in f if line.strip()]

            files = []
            img_exts = {".jpg", ".png", ".jpeg", ".bmp"}
            for stem in stems:
                found = False
                for ext in img_exts:
                    p = self.img_dir / f"{stem}{ext}"
                    if p.exists():
                        files.append(p.name)
                        found = True
                        break
            return files
        else:
            return [
                f.name
                for f in self.img_dir.iterdir()
                if f.suffix.lower() in {".jpg", ".png", ".jpeg"}
            ]

    def _load_all_labels(self) -> Dict[str, int]:
        """Pre-load image-level labels."""
        labels = {}
        for filename in self.image_files:
            stem = Path(filename).stem
            path = self.lesion_label_dir / f"{stem}.txt"

            max_cls = 0
            if path.exists():
                with open(path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 1:
                            c = int(float(parts[0]))
                            if c > max_cls:
                                max_cls = c
            labels[filename] = max_cls
        return labels

    def _load_yolo_boxes(self, path: Path) -> Tuple[List[List[float]], List[int]]:
        """Load YOLO format boxes."""
        boxes = []
        classes = []
        if not path.exists():
            return boxes, classes

        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(float(parts[0]))
                    cx, cy, w, h = map(float, parts[1:5])
                    boxes.append([cx, cy, w, h])
                    classes.append(cls)
        return boxes, classes

    def _yolo_to_pascal(self, box, w, h):
        """Convert YOLO format to Pascal VOC."""
        cx, cy, bw, bh = box
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        return max(0, x1), max(0, y1), min(w, x2), min(h, y2)

    def _pascal_to_yolo(self, x1, y1, x2, y2, w, h):
        """Convert Pascal VOC to YOLO."""
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        return [cx, cy, bw, bh]

    def _clamp_box(self, box: List[float]) -> List[float]:
        """Clamp box coordinates to [0, 1] range."""
        cx, cy, w, h = box

        # Convert to corners
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # Clamp corners
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))

        # Re-convert to YOLO
        new_w = x2 - x1
        new_h = y2 - y1
        new_cx = x1 + new_w / 2
        new_cy = y1 + new_h / 2

        if new_w <= 0 or new_h <= 0:
            return [0.0, 0.0, 0.0, 0.0]

        return [new_cx, new_cy, new_w, new_h]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Get dataset item with bbox information.

        Returns dict with structure:
        {
          'knees': [
            {
              'ctx': (3, H, W) tensor,
              'ctx_bbox': (4,) tensor [cx, cy, w, h],
              'js': (n, 3, h, w) tensor or empty,
              'js_bboxes': (n, 4) tensor or empty,
              'ost': (m, 3, h, w) tensor or empty,
              'ost_bboxes': (m, 4) tensor or empty,
              'label': int,
            },
            ...
          ]
        }
        """
        filename = self.image_files[idx]
        stem = Path(filename).stem

        # Load image
        img_path = self.img_dir / filename
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Cannot read image: {img_path}")

        # Robustly handle image shape (some environments/files might load as 3-channel even with flag)
        if image.ndim == 2:
            H, W = image.shape
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            # Handle multi-channel (H, W, C)
            H, W = image.shape[:2]
            if image.shape[2] == 3:
                # Assuming BGR from opencv
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.shape[2] == 4:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                # Fallback for 1-channel 3D or other
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Load bounding boxes first
        knee_path = self.knee_label_dir / f"{stem}.txt"
        lesion_path = self.lesion_label_dir / f"{stem}.txt"

        knee_boxes, _ = self._load_yolo_boxes(knee_path)
        lesion_boxes, lesion_classes = self._load_yolo_boxes(lesion_path)

        # Apply geometric transforms to full image AND bboxes
        if self.geometric_transform:
            # Combine bboxes for transform
            all_boxes = []
            all_classes = []

            # Add knee boxes (class 999 to distinguish)
            for box in knee_boxes:
                clamped_box = self._clamp_box(box)
                if clamped_box[2] > 0 and clamped_box[3] > 0:
                    all_boxes.append(clamped_box)
                    all_classes.append(999)

            # Add lesion boxes
            for box, cls in zip(lesion_boxes, lesion_classes):
                clamped_box = self._clamp_box(box)
                if clamped_box[2] > 0 and clamped_box[3] > 0:
                    all_boxes.append(clamped_box)
                    all_classes.append(cls)

            # Transform
            if all_boxes:
                transformed = self.geometric_transform(
                    image=image_rgb, bboxes=all_boxes, class_labels=all_classes
                )
                image_rgb = transformed["image"]
                transformed_boxes = transformed["bboxes"]
                transformed_classes = transformed["class_labels"]

                # Separate back
                knee_boxes = []
                lesion_boxes = []
                lesion_classes = []

                for box, cls in zip(transformed_boxes, transformed_classes):
                    if cls == 999:
                        knee_boxes.append(list(box))
                    else:
                        lesion_boxes.append(list(box))
                        lesion_classes.append(cls)
            else:
                # No boxes, just transform image
                transformed = self.geometric_transform(image=image_rgb)
                image_rgb = transformed["image"]

        if not knee_boxes:
            # No knees, return empty
            return {"knees": []}

        # Process each knee
        knees = []

        for knee_box in knee_boxes:
            knee_dict = self._process_knee(
                image_rgb,
                knee_box,
                lesion_boxes,
                lesion_classes,
                H,
                W,
            )
            if knee_dict is not None:
                knees.append(knee_dict)

        # Get image-level label
        label = self.labels_map.get(filename, 0)

        return {"knees": knees, "label": label}

    def _process_knee(
        self,
        image: np.ndarray,
        knee_box: List[float],
        lesion_boxes: List[List[float]],
        lesion_classes: List[int],
        H: int,
        W: int,
    ) -> Dict:
        """
        Process a single knee instance.

        Returns dict with context and lesion patches.
        """
        # Crop context (full knee region)
        ctx_x1, ctx_y1, ctx_x2, ctx_y2 = self._yolo_to_pascal(knee_box, W, H)

        # Add padding
        pad_x = int((ctx_x2 - ctx_x1) * self.pad_ratio)
        pad_y = int((ctx_y2 - ctx_y1) * self.pad_ratio)
        ctx_x1 = max(0, ctx_x1 - pad_x)
        ctx_y1 = max(0, ctx_y1 - pad_y)
        ctx_x2 = min(W, ctx_x2 + pad_x)
        ctx_y2 = min(H, ctx_y2 + pad_y)

        # Crop context patch
        ctx_patch = image[ctx_y1:ctx_y2, ctx_x1:ctx_x2]

        if ctx_patch.size == 0:
            return None

        ctx_patch = cv2.resize(ctx_patch, self.ctx_size, interpolation=cv2.INTER_CUBIC)

        # Compute context bbox in normalized coordinates
        ctx_bbox = self._pascal_to_yolo(ctx_x1, ctx_y1, ctx_x2, ctx_y2, W, H)

        # Apply photometric transforms to context
        if self.photometric_transform:
            transformed = self.photometric_transform(image=ctx_patch)
            ctx_patch = transformed["image"]

        # Normalize and convert to tensor
        ctx_patch = normalize_patch(ctx_patch)
        ctx_tensor = patch_to_tensor(ctx_patch)

        # Find lesions within this knee
        js_patches = []
        js_bboxes = []
        ost_patches = []
        ost_bboxes = []

        knee_area = (ctx_x2 - ctx_x1) * (ctx_y2 - ctx_y1)

        for lesion_box, lesion_cls in zip(lesion_boxes, lesion_classes):
            # Check if lesion intersects with knee
            lex1, ley1, lex2, ley2 = self._yolo_to_pascal(lesion_box, W, H)

            # Intersection check
            ix1 = max(ctx_x1, lex1)
            iy1 = max(ctx_y1, ley1)
            ix2 = min(ctx_x2, lex2)
            iy2 = min(ctx_y2, ley2)

            if ix1 >= ix2 or iy1 >= iy2:
                continue  # No intersection

            inter_area = (ix2 - ix1) * (iy2 - iy1)
            if inter_area < self.min_box_area_px:
                continue

            # Crop lesion patch with context
            pad_lesion = int(max(lex2 - lex1, ley2 - ley1) * self.pad_ratio)
            lex1 = max(0, lex1 - pad_lesion)
            ley1 = max(0, ley1 - pad_lesion)
            lex2 = min(W, lex2 + pad_lesion)
            ley2 = min(H, ley2 + pad_lesion)

            lesion_patch = image[ley1:ley2, lex1:lex2]

            # Skip if too small
            if lesion_patch.shape[0] < 10 or lesion_patch.shape[1] < 10:
                continue

            lesion_patch = cv2.resize(
                lesion_patch, self.patch_size, interpolation=cv2.INTER_CUBIC
            )

            # Compute lesion bbox in normalized coordinates
            lesion_bbox_norm = self._pascal_to_yolo(lex1, ley1, lex2, ley2, W, H)

            # Apply photometric transforms
            if self.photometric_transform:
                transformed = self.photometric_transform(image=lesion_patch)
                lesion_patch = transformed["image"]

            # Normalize and convert to tensor
            lesion_patch = normalize_patch(lesion_patch)
            lesion_tensor = patch_to_tensor(lesion_patch)

            # Classify by lesion type
            if lesion_cls in JS_CLASSES:
                js_patches.append(lesion_tensor)
                js_bboxes.append(torch.tensor(lesion_bbox_norm, dtype=torch.float32))
            elif lesion_cls in OST_CLASSES:
                ost_patches.append(lesion_tensor)
                ost_bboxes.append(torch.tensor(lesion_bbox_norm, dtype=torch.float32))

        # Stack patches and bboxes
        if js_patches:
            js_tensor = torch.stack(js_patches)
            js_bbox_tensor = torch.stack(js_bboxes)
        else:
            js_tensor = torch.empty(0, 3, *self.patch_size, dtype=torch.float32)
            js_bbox_tensor = torch.empty(0, 4, dtype=torch.float32)

        if ost_patches:
            ost_tensor = torch.stack(ost_patches)
            ost_bbox_tensor = torch.stack(ost_bboxes)
        else:
            ost_tensor = torch.empty(0, 3, *self.patch_size, dtype=torch.float32)
            ost_bbox_tensor = torch.empty(0, 4, dtype=torch.float32)

        return {
            "ctx": ctx_tensor,
            "ctx_bbox": torch.tensor(ctx_bbox, dtype=torch.float32),
            "js": js_tensor,
            "js_bboxes": js_bbox_tensor,
            "ost": ost_tensor,
            "ost_bboxes": ost_bbox_tensor,
        }


def collate_kiocmil_v3(batch: List[Dict]) -> Dict:
    """
    Custom collate function for KIOCMIL V3 dataset.

    Handles variable number of knees and lesions per image.
    """
    batch_data = []

    for item in batch:
        knees = item.get("knees", [])
        if knees:
            batch_data.append({"knees": knees, "label": item.get("label", 0)})

    return batch_data
