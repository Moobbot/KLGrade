import os
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional
import warnings

# Import configurations
# Try generic import first, if fails (running as script), use local
try:
    from src.config import OST_CLASSES, JS_CLASSES, KNEE_CLASS_ID, CLASSES_10_CLASS
except ImportError:
    # Fallback for standalone testing
    OST_CLASSES = {0, 1, 2, 3}
    JS_CLASSES = {4, 5}
    KNEE_CLASS_ID = 0
    CLASSES_10_CLASS = {i: str(i) for i in range(10)}


class KiocmilDataset(Dataset):
    """
    Dataset for Knee Instance Object-Context MIL Network (KIOCMIL).

    Logic:
    1. Load Image
    2. Load Knee Bboxes (from labels-knee/)
    3. Load Lesion Bboxes (from labels/)
    4. Augment (Global)
    5. Extract Knee Instances
    6. Assign Lesions to Knees
    7. Generate Tokens (CTX, JS, OST) per Knee
    """

    def __init__(
        self,
        img_dir: str,
        knee_label_dir: str,
        lesion_label_dir: str,
        split_file: Optional[str] = None,
        transform: Optional[A.Compose] = None,
        ctx_size: Tuple[int, int] = (384, 384),
        patch_size: Tuple[int, int] = (224, 224),
        pad_ratio: float = 0.1,  # Pad lesions before crop
        min_box_area_px: int = 100,
    ):
        self.img_dir = Path(img_dir)
        self.knee_label_dir = Path(knee_label_dir)
        self.lesion_label_dir = Path(lesion_label_dir)
        self.transform = transform
        self.ctx_size = ctx_size
        self.patch_size = patch_size
        self.pad_ratio = pad_ratio
        self.min_box_area_px = min_box_area_px

        # Load image list
        self.image_files = self._load_image_list(split_file)

        # Load Labels (Max class in lesion file)
        self.labels_map = self._load_all_labels()

        # Assume label 10-class is inferred or provided.
        # For this implementation, we simulate it or read from a CSV if available.
        # Since the prompt says "label theo ảnh", usually implies a CSV.
        # BUT, if we reuse the YOLO file structure, maybe strict image-level labels
        # aren't in the txt.
        # TEMPORARY: using a placeholder label or parsing from filename if encoded.
        # If user has `train.txt` containing `image_path label`, I'd use that.
        # Here we just assume 0 for now or implement a CSV loader if asked.
        # Note: If reusing standard KL datasets, often grade is in filename or separate file.

    def _load_image_list(self, split_file: Optional[str]) -> List[str]:
        if split_file and os.path.exists(split_file):
            with open(split_file, "r") as f:
                stems = [line.strip() for line in f if line.strip()]

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
        cx, cy, bw, bh = box
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        return [max(0, x1), max(0, y1), min(w, x2), min(h, y2)]

    def _crop_patch_with_padding(self, img, box_pascal, target_size, padding_ratio=0.0):
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

        # Load Image
        image = cv2.imread(str(img_path))
        if image is None:
            # Return dummy if failed logic or raise
            raise ValueError(f"Failed to load {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_img, w_img = image.shape[:2]

        # Load Labels
        knee_boxes, _ = self._load_yolo_boxes(knee_path)  # Class is always 0
        lesion_boxes, lesion_ids = self._load_yolo_boxes(lesion_path)

        # Convert to Pascal for Augmentation
        # Combine: Knee boxes (Type=0), Lesion boxes (Type=1)
        # We store original index or ID to retrieve later
        # Albumentations list: [x1, y1, x2, y2, class_id]

        aug_boxes = []
        aug_cats = []  # Store tuple (is_lesion, original_class_id)

        # Knees
        for box in knee_boxes:
            pbox = self._yolo_to_pascal(box, w_img, h_img)
            aug_boxes.append(pbox + [0])  # label for box params, not used logic
            aug_cats.append((0, 0))  # Type 0 (Knee), Class 0

        # Lesions
        for box, cls in zip(lesion_boxes, lesion_ids):
            pbox = self._yolo_to_pascal(box, w_img, h_img)
            aug_boxes.append(pbox + [1])  # label 1
            aug_cats.append((1, cls))  # Type 1 (Lesion), Class ID

        # Transform
        if self.transform:
            # Albumentations expects boxes without extra info if not configured?
            # We use 'fields' to pass extra info
            # Or just pass simple list and map back if order is preserved (Albumentations usually preserves order if not filtering)
            # BUT filtering (min_area) might drop boxes.
            # Best: Assign unique ID to each box.

            # Simple approach: Assume order preserved or check
            # We'll use class_labels to store indices
            indices = list(range(len(aug_boxes)))

            try:
                # Remove extra dim for augment
                input_boxes = [b[:4] for b in aug_boxes]
                transformed = self.transform(
                    image=image,
                    bboxes=input_boxes,
                    class_labels=indices,  # Pass indices as labels
                )
                image = transformed["image"]
                t_boxes = transformed["bboxes"]
                t_indices = transformed[
                    "class_labels"
                ]  # Keeping indices of survived boxes

                # Reconstruct
                final_knee_boxes = []
                final_lesion_boxes = []
                final_lesion_ids = []

                for box, idx in zip(t_boxes, t_indices):
                    type_id, real_class = aug_cats[idx]
                    if type_id == 0:
                        final_knee_boxes.append(box)
                    else:
                        final_lesion_boxes.append(box)
                        final_lesion_ids.append(real_class)

            except Exception as e:
                warnings.warn(f"Augmentation failed: {e}")
                # Fallback to no augment
                final_knee_boxes = [
                    b[:4] for b in aug_boxes if aug_cats[aug_boxes.index(b)][0] == 0
                ]
                # ... Simplified fallback
                pass
        else:
            final_knee_boxes = [
                self._yolo_to_pascal(b, w_img, h_img) for b in knee_boxes
            ]
            le_pascal = [self._yolo_to_pascal(b, w_img, h_img) for b in lesion_boxes]
            final_lesion_boxes = le_pascal
            final_lesion_ids = lesion_ids
            # Manual normalization if image is tensor? No, keeping numpy for crop
            if not isinstance(image, np.ndarray):
                # Convert tensor back to numpy for slicing if transform made it tensor
                image = image.permute(1, 2, 0).cpu().numpy() * 255.0
                image = image.astype(np.uint8)

        # Ensure image is numpy for cropping
        if isinstance(image, torch.Tensor):
            image_np = image.permute(1, 2, 0).cpu().numpy()
            # If normalized, this might be float. We need uint8 for cv2 usually, or float is fine
            # Assuming standard Normalize was applied:
            # Reverse normalize? Or just crop on tensor?
            # Easier to crop on Tensor if it's already Tensor.
            is_tensor = True
        else:
            image_np = image
            is_tensor = False

        # --- Instance Extraction & Token Generation ---

        knee_data = []
        h_curr, w_curr = (
            (image.shape[1], image.shape[2]) if is_tensor else image.shape[:2]
        )

        for k_box in final_knee_boxes:
            kx1, ky1, kx2, ky2 = map(int, k_box)
            k_center = ((kx1 + kx2) / 2, (ky1 + ky2) / 2)

            # 1. Context Token (expand 2.0x)
            kw, kh = kx2 - kx1, ky2 - ky1
            # Expand
            ctx_w, ctx_h = int(kw * 2.0), int(kh * 2.0)
            ctx_x1 = int(k_center[0] - ctx_w / 2)
            ctx_y1 = int(k_center[1] - ctx_h / 2)
            ctx_box = [
                max(0, ctx_x1),
                max(0, ctx_y1),
                min(w_curr, ctx_x1 + ctx_w),
                min(h_curr, ctx_y1 + ctx_h),
            ]

            if is_tensor:
                # Crop tensor
                ctx_crop = image[:, ctx_box[1] : ctx_box[3], ctx_box[0] : ctx_box[2]]
                # Resize tensor? usually use F.interpolate
                ctx_token = torch.nn.functional.interpolate(
                    ctx_crop.unsqueeze(0),
                    size=self.ctx_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            else:
                ctx_crop = image_np[ctx_box[1] : ctx_box[3], ctx_box[0] : ctx_box[2]]
                if ctx_crop.size == 0:
                    continue
                ctx_token_np = cv2.resize(ctx_crop, self.ctx_size)
                ctx_token = (
                    torch.from_numpy(ctx_token_np).permute(2, 0, 1).float() / 255.0
                )

            # 2. Assign Lesions
            # Simplified Assignment: Center inside knee box
            my_js = []
            my_ost = []

            for l_box, l_cls in zip(final_lesion_boxes, final_lesion_ids):
                lx1, ly1, lx2, ly2 = map(int, l_box)
                l_center = ((lx1 + lx2) / 2, (ly1 + ly2) / 2)

                # Check inclusion
                if kx1 <= l_center[0] <= kx2 and ky1 <= l_center[1] <= ky2:
                    # Belong to this knee
                    if l_cls in JS_CLASSES:
                        my_js.append(l_box)
                    elif l_cls in OST_CLASSES:
                        my_ost.append(l_box)

            # 3. Crop Lesions
            js_tokens = []
            for jb in my_js:
                if is_tensor:
                    # Tensor crop
                    jx1, jy1, jx2, jy2 = map(int, jb)
                    # Pad
                    pw = int((jx2 - jx1) * self.pad_ratio)
                    ph = int((jy2 - jy1) * self.pad_ratio)
                    jx1, jy1 = max(0, jx1 - pw), max(0, jy1 - ph)
                    jx2, jy2 = min(w_curr, jx2 + pw), min(h_curr, jy2 + ph)

                    crop = image[:, jy1:jy2, jx1:jx2]
                    tok = torch.nn.functional.interpolate(
                        crop.unsqueeze(0), size=self.patch_size, mode="bilinear"
                    ).squeeze(0)
                    js_tokens.append(tok)
                else:
                    # Numpy crop
                    tok_np = self._crop_patch_with_padding(
                        image_np, jb, self.patch_size, self.pad_ratio
                    )
                    tok = torch.from_numpy(tok_np).permute(2, 0, 1).float() / 255.0
                    js_tokens.append(tok)

            # If no JS, use fallback (Horizontal band)
            if not js_tokens:
                # Band: center 22% of knee height?
                band_h = int(kh * 0.22)
                by1 = int(k_center[1] - band_h / 2)
                by2 = int(k_center[1] + band_h / 2)
                bx1, bx2 = kx1, kx2  # Full width

                bbox_fallback = [bx1, by1, bx2, by2]
                if is_tensor:
                    # Tensor logic (duplicate above ideally helper)
                    crop = image[
                        :,
                        max(0, by1) : min(h_curr, by2),
                        max(0, bx1) : min(w_curr, bx2),
                    ]
                    if crop.numel() > 0:
                        tok = torch.nn.functional.interpolate(
                            crop.unsqueeze(0), size=self.patch_size, mode="bilinear"
                        ).squeeze(0)
                        js_tokens.append(tok)
                    # Else ? Zero?
                else:
                    tok_np = self._crop_patch_with_padding(
                        image_np, bbox_fallback, self.patch_size, 0
                    )
                    tok = torch.from_numpy(tok_np).permute(2, 0, 1).float() / 255.0
                    js_tokens.append(tok)

            ost_tokens = []
            for ob in my_ost:
                if is_tensor:
                    ox1, oy1, ox2, oy2 = map(int, ob)
                    # Pad
                    pw = int((ox2 - ox1) * self.pad_ratio)
                    ph = int((oy2 - oy1) * self.pad_ratio)
                    ox1, oy1 = max(0, ox1 - pw), max(0, oy1 - ph)
                    ox2, oy2 = min(w_curr, ox2 + pw), min(h_curr, oy2 + ph)
                    crop = image[:, oy1:oy2, ox1:ox2]
                    tok = torch.nn.functional.interpolate(
                        crop.unsqueeze(0), size=self.patch_size, mode="bilinear"
                    ).squeeze(0)
                    ost_tokens.append(tok)
                else:
                    tok_np = self._crop_patch_with_padding(
                        image_np, ob, self.patch_size, self.pad_ratio
                    )
                    tok = torch.from_numpy(tok_np).permute(2, 0, 1).float() / 255.0
                    ost_tokens.append(tok)

            # If no OST, fallback (Corner patches)
            if not ost_tokens:
                # Top corners of tibia usually? Just generic top corners of knee box
                # or just reuse JS fallback to avoid crash
                # User spec: "Two corner patches... or patch top border of tibia"
                # Simplify: Top-Left and Top-Right quadrants of Knee Box
                q_w, q_h = int(kw / 3), int(kh / 3)

                # TL
                tl_box = [kx1, ky1, kx1 + q_w, ky1 + q_h]
                # TR
                tr_box = [kx2 - q_w, ky1, kx2, ky1 + q_h]

                for fb in [tl_box, tr_box]:
                    if is_tensor:
                        crop = image[
                            :,
                            max(0, fb[1]) : min(h_curr, fb[3]),
                            max(0, fb[0]) : min(w_curr, fb[2]),
                        ]
                        if crop.numel() > 0:
                            tok = torch.nn.functional.interpolate(
                                crop.unsqueeze(0), size=self.patch_size, mode="bilinear"
                            ).squeeze(0)
                            ost_tokens.append(tok)
                    else:
                        tok_np = self._crop_patch_with_padding(
                            image_np, fb, self.patch_size, 0
                        )
                        tok = torch.from_numpy(tok_np).permute(2, 0, 1).float() / 255.0
                        ost_tokens.append(tok)

            # Pack
            knee_data.append(
                {
                    "ctx": ctx_token,
                    "js": (
                        torch.stack(js_tokens) if js_tokens else torch.zeros(0)
                    ),  # Stack or List
                    "ost": torch.stack(ost_tokens) if ost_tokens else torch.zeros(0),
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
    """
    # Batch is list of dicts
    # return structure:
    # {
    #   'image_ids': [...],
    #   'knees': [ # Flattened list of all knees in batch? Or List of Lists
    #        [ # Image 1
    #           { 'ctx': Tensor, 'js': Tensor(N,C,H,W), 'ost': Tensor(M,C,H,W) }, ...
    #        ], ...
    #    ],
    #   'labels_10': Tensor
    # }

    # Actually models usually prefer flattened inputs for batching efficiency if possible
    # But varying patch counts (N, M) makes it hard.
    # We will return list structure and let Model loop or pack padded sequences.

    return batch
