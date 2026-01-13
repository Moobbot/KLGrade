"""
KIOCMIL Dataset V2 with Fixed Augmentation Pipeline.

This version separates augmentation timing to fix the tensor/numpy incompatibility:
1. Load image (numpy)
2. Apply geometric transforms to FULL IMAGE (numpy)
3. Crop patches from transformed image (numpy)
4. Apply photometric transforms to EACH PATCH (numpy)
5. Normalize patches (numpy → float32)
6. Convert to tensor (LAST STEP!)

Date: 2026-01-13
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional
import warnings

# Import configurations
try:
    from src.config import OST_CLASSES, JS_CLASSES, KNEE_CLASS_ID, CLASSES_10_CLASS
except ImportError:
    # Fallback for standalone testing
    OST_CLASSES = {0, 1, 2, 3}
    JS_CLASSES = {4, 5}
    KNEE_CLASS_ID = 0
    CLASSES_10_CLASS = {i: str(i) for i in range(10)}

# Import v2 transforms
from src.datasets.kiocmil_transforms_v2 import normalize_patch, patch_to_tensor


class KiocmilDatasetV2(Dataset):
    """
    Dataset for Knee Instance Object-Context MIL Network (KIOCMIL) - Version 2.

    Fixed augmentation pipeline:
    1. Load Image (numpy)
    2. Apply Geometric Augmentation to full image (numpy)
    3. Load Knee & Lesion Bboxes
    4. Extract Knee Instances and crop patches (numpy)
    5. Apply Photometric Augmentation to each patch (numpy)
    6. Normalize patches (numpy → float32)
    7. Convert to tensors (final step)
    """

    def __init__(
        self,
        img_dir: str,
        knee_label_dir: str,
        lesion_label_dir: str,
        split_file: Optional[str] = None,
        geometric_transform=None,  # NEW: Separate geometric transform
        photometric_transform=None,  # NEW: Separate photometric transform
        ctx_size: Tuple[int, int] = (384, 384),
        patch_size: Tuple[int, int] = (224, 224),
        pad_ratio: float = 0.1,  # Pad lesions before crop
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

        # Load Labels (Max class in lesion file)
        self.labels_map = self._load_all_labels()

    def _load_image_list(self, split_file: Optional[str]) -> List[str]:
        if split_file and os.path.exists(split_file):
            with open(split_file, "r") as f:
                stems = [Path(line.strip()).stem for line in f if line.strip()]

            # Find images
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
                if not found:
                    # Try recursive search if needed, or warn
                    pass
            return files
        else:
            return [
                f.name
                for f in self.img_dir.iterdir()
                if f.suffix.lower() in {".jpg", ".png", ".jpeg"}
            ]

    def _load_all_labels(self) -> Dict[str, int]:
        """
        Pre-load image-level labels by parsing lesion files.
        Rule: Label = Max class ID found in the file. Defaults to 0 if empty.
        """
        labels = {}
        for filename in self.image_files:
            stem = Path(filename).stem
            path = self.lesion_label_dir / f"{stem}.txt"

            # Read classes
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
        """Convert YOLO format (cx, cy, w, h) to Pascal VOC format (x1, y1, x2, y2)"""
        cx, cy, bw, bh = box
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        return [max(0, x1), max(0, y1), min(w, x2), min(h, y2)]

    def _crop_patch_with_padding(self, img, box_pascal, target_size, padding_ratio=0.0):
        """Crop and resize a patch from image with optional padding"""
        h_img, w_img = img.shape[:2]
        x1, y1, x2, y2 = box_pascal
        bw, bh = x2 - x1, y2 - y1

        # Apply padding
        pad_w = int(bw * padding_ratio)
        pad_h = int(bh * padding_ratio)

        nx1 = max(0, x1 - pad_w)
        ny1 = max(0, y1 - pad_h)
        nx2 = min(w_img, x2 + pad_w)
        ny2 = min(h_img, y2 + pad_h)

        crop = img[ny1:ny2, nx1:nx2]
        if crop.size == 0:
            return np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)

        return cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        stem = Path(filename).stem

        # Paths
        img_path = self.img_dir / filename
        knee_path = self.knee_label_dir / f"{stem}.txt"
        lesion_path = self.lesion_label_dir / f"{stem}.txt"

        # 1. Load Image (numpy array)
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Failed to load {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # (H, W, 3) uint8
        h_img, w_img = image.shape[:2]

        # 2. Apply Geometric Augmentation to FULL IMAGE (if training)
        if self.geometric_transform is not None:
            try:
                augmented = self.geometric_transform(image=image)
                image = augmented["image"]  # Still numpy uint8
            except Exception as e:
                warnings.warn(f"Geometric augmentation failed: {e}")
                # Continue with original image

        # 3. Load Bboxes (YOLO format)
        knee_boxes, _ = self._load_yolo_boxes(knee_path)
        lesion_boxes, lesion_ids = self._load_yolo_boxes(lesion_path)

        # Convert to Pascal VOC format
        knee_boxes_px = [self._yolo_to_pascal(b, w_img, h_img) for b in knee_boxes]
        lesion_boxes_px = [self._yolo_to_pascal(b, w_img, h_img) for b in lesion_boxes]

        # 4. Extract Knee Instances and Generate Tokens
        knee_data = []

        for k_box in knee_boxes_px:
            kx1, ky1, kx2, ky2 = map(int, k_box)
            kw, kh = kx2 - kx1, ky2 - ky1
            k_center = ((kx1 + kx2) / 2, (ky1 + ky2) / 2)

            # --- 4a. Context Token (expand 2.0x) ---
            ctx_w, ctx_h = int(kw * 2.0), int(kh * 2.0)
            ctx_x1 = int(k_center[0] - ctx_w / 2)
            ctx_y1 = int(k_center[1] - ctx_h / 2)
            ctx_box = [
                max(0, ctx_x1),
                max(0, ctx_y1),
                min(w_img, ctx_x1 + ctx_w),
                min(h_img, ctx_y1 + ctx_h),
            ]

            # Crop context patch (numpy)
            ctx_patch = self._crop_patch_with_padding(
                image, ctx_box, self.ctx_size, padding_ratio=0
            )

            # --- 4b. Assign Lesions to this Knee ---
            my_js = []
            my_ost = []

            for l_box, l_cls in zip(lesion_boxes_px, lesion_ids):
                lx1, ly1, lx2, ly2 = map(int, l_box)
                l_center = ((lx1 + lx2) / 2, (ly1 + ly2) / 2)

                # Check if lesion center is inside knee box
                if kx1 <= l_center[0] <= kx2 and ky1 <= l_center[1] <= ky2:
                    if l_cls in JS_CLASSES:
                        my_js.append(l_box)
                    elif l_cls in OST_CLASSES:
                        my_ost.append(l_box)

            # --- 4c. Crop JS Patches ---
            js_patches = []
            for jb in my_js:
                patch = self._crop_patch_with_padding(
                    image, jb, self.patch_size, padding_ratio=self.pad_ratio
                )
                js_patches.append(patch)

            # Fallback if no JS detected: Horizontal band
            if not js_patches:
                band_h = int(kh * 0.22)
                by1 = int(k_center[1] - band_h / 2)
                by2 = int(k_center[1] + band_h / 2)
                bbox_fallback = [kx1, max(0, by1), kx2, min(h_img, by2)]
                patch = self._crop_patch_with_padding(
                    image, bbox_fallback, self.patch_size, padding_ratio=0
                )
                js_patches.append(patch)

            # --- 4d. Crop OST Patches ---
            ost_patches = []
            for ob in my_ost:
                patch = self._crop_patch_with_padding(
                    image, ob, self.patch_size, padding_ratio=self.pad_ratio
                )
                ost_patches.append(patch)

            # Fallback if no OST detected: Corner patches
            if not ost_patches:
                q_w, q_h = int(kw / 3), int(kh / 3)
                # Top-left corner
                tl_box = [kx1, ky1, kx1 + q_w, ky1 + q_h]
                # Top-right corner
                tr_box = [kx2 - q_w, ky1, kx2, ky1 + q_h]

                for fb in [tl_box, tr_box]:
                    patch = self._crop_patch_with_padding(
                        image, fb, self.patch_size, padding_ratio=0
                    )
                    ost_patches.append(patch)

            # 5. Apply Photometric Augmentation to EACH PATCH (if training)
            if self.photometric_transform is not None:
                try:
                    # Context patch
                    ctx_patch = self.photometric_transform(image=ctx_patch)["image"]

                    # JS patches
                    js_patches = [
                        self.photometric_transform(image=p)["image"] for p in js_patches
                    ]

                    # OST patches
                    ost_patches = [
                        self.photometric_transform(image=p)["image"]
                        for p in ost_patches
                    ]
                except Exception as e:
                    warnings.warn(f"Photometric augmentation failed: {e}")
                    # Continue with non-augmented patches

            # 6. Normalize patches (numpy → float32 with ImageNet stats)
            ctx_patch = normalize_patch(ctx_patch)
            js_patches = [normalize_patch(p) for p in js_patches]
            ost_patches = [normalize_patch(p) for p in ost_patches]

            # 7. Convert to tensors (LAST STEP!)
            ctx_token = patch_to_tensor(ctx_patch)  # (3, 384, 384)
            js_tokens = torch.stack(
                [patch_to_tensor(p) for p in js_patches]
            )  # (N, 3, 224, 224)
            ost_tokens = torch.stack(
                [patch_to_tensor(p) for p in ost_patches]
            )  # (M, 3, 224, 224)

            # Pack knee data
            knee_data.append(
                {
                    "ctx": ctx_token,
                    "js": js_tokens,
                    "ost": ost_tokens,
                }
            )

        return {
            "image_id": filename,
            "knees": knee_data,
            "label_10": self.labels_map.get(filename, 0),
            "grade": 0,  # Will be derived in trainer
            "type": 0,  # Will be derived in trainer
        }


def collate_kiocmil(batch):
    """
    Custom collator to handle variable knee counts and patch counts.
    Simply returns the batch as-is since we have variable-length sequences.
    """
    return batch
