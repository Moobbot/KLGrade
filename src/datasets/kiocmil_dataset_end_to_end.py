import os
import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


class KiocmilDatasetEndToEnd(Dataset):
    """
    Dataset for End-to-End Training (Detection + Classification).

    Returns:
    - image: (3, H, W) tensor, resized to target_size (default 640x640)
    - targets: Dict containing:
        - knee_boxes: (N, 4) [cx, cy, w, h] normalized
        - lesion_boxes: (M, 4) [cx, cy, w, h] normalized
        - lesion_classes: (M,) 0=JS, 1=OST
        - label: int (KL grade)
    """

    def __init__(
        self,
        img_dir: str,
        knee_label_dir: str,
        lesion_label_dir: str,
        split_file: Optional[str] = None,
        target_size: Tuple[int, int] = (640, 640),
        transform=None,
    ):
        self.img_dir = Path(img_dir)
        self.knee_label_dir = Path(knee_label_dir)
        self.lesion_label_dir = Path(lesion_label_dir)
        self.target_size = target_size
        self.split_file = split_file

        # Load image list
        self.image_files = self._load_image_list(split_file)

        # Load global labels (KL grades)
        self.labels_map = self._load_all_labels()

        # Default transform if none provided
        if transform is None:
            self.transform = A.Compose(
                [
                    A.Resize(height=target_size[0], width=target_size[1]),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ],
                bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
            )
        else:
            self.transform = transform

        # Define class mapping
        # Lesion file has classes: 0-3 (Ost KL0-3), 4-9 (JS KL0-4) typically
        # We simplify to: 0 = JS, 1 = OST (or vice versa based on logic)
        # Previous logic:
        #   OST_CLASSES = {0, 1, 2, 3} -> Class 1 (OST)
        #   JS_CLASSES = {4, 5, ...} -> Class 0 (JS)
        # We need to align with DetectionHead output (2 classes)
        self.OST_CLASSES = {0, 1, 2, 3}
        self.JS_CLASSES = {4, 5, 6, 7, 8, 9}

    def _load_image_list(self, split_file: Optional[str]) -> List[str]:
        if split_file and os.path.exists(split_file):
            with open(split_file, "r") as f:
                stems = [Path(line.strip()).stem for line in f if line.strip()]

            files = []
            img_exts = {".jpg", ".png", ".jpeg", ".bmp"}
            for stem in stems:
                for ext in img_exts:
                    p = self.img_dir / f"{stem}{ext}"
                    if p.exists():
                        files.append(p.name)
                        break
            return files
        else:
            return [
                f.name
                for f in self.img_dir.iterdir()
                if f.suffix.lower() in {".jpg", ".png"}
            ]

    def _load_all_labels(self) -> Dict[str, int]:
        labels = {}
        for filename in self.image_files:
            stem = Path(filename).stem
            # This logic mimics V3 dataset: max lesion class implies image label?
            # Actually dataset V3 read 'labels_10_class' or derived it.
            # Here we try to read from lesion label file as heuristic if no separate file
            # But normally we have a labels_10_class dir.
            # Using simple heuristic from V3 for now:
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

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        stem = Path(filename).stem

        # Load image
        img_path = self.img_dir / filename
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Cannot read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load boxes
        knee_path = self.knee_label_dir / f"{stem}.txt"
        lesion_path = self.lesion_label_dir / f"{stem}.txt"

        raw_knee_boxes, _ = self._load_yolo_boxes(knee_path)
        raw_lesion_boxes, raw_lesion_classes = self._load_yolo_boxes(lesion_path)

        # Prepare for transform
        # We need to merge all boxes, transform, then split back
        # Class mapping for transform:
        # 0: JS (Target 0)
        # 1: OST (Target 1)
        # 2: Knee (Target 0 for knee head)
        # Using 99 as Knee class temporary ID

        all_boxes = []
        all_cls_ids = []

        # Add knee boxes (ID 99)
        for box in raw_knee_boxes:
            # Clamp Yolo [cx, cy, w, h] to roughly [0,1] range to avoid float errors
            box = [max(0.001, min(0.999, v)) for v in box]
            all_boxes.append(box)
            all_cls_ids.append(99)

        # Add lesion boxes
        for box, cls in zip(raw_lesion_boxes, raw_lesion_classes):
            box = [max(0.001, min(0.999, v)) for v in box]
            all_boxes.append(box)
            # Map original class to JS(0)/OST(1)
            if cls in self.JS_CLASSES:
                all_cls_ids.append(0)  # JS
            elif cls in self.OST_CLASSES:
                all_cls_ids.append(1)  # OST
            else:
                # Fallback or ignore
                all_cls_ids.append(1)

        # Apply Transform
        if self.transform:
            # Sanitize boxes before transform to fix rounding errors
            sanitized_boxes = []
            sanitized_cls = []
            for box, cls in zip(all_boxes, all_cls_ids):
                sanitized_box = self._sanitize_box(box)
                # Filter out degenerate boxes (w or h <= 0)
                if sanitized_box[2] > 0.001 and sanitized_box[3] > 0.001:
                    sanitized_boxes.append(sanitized_box)
                    sanitized_cls.append(cls)

            try:
                transformed = self.transform(
                    image=image, bboxes=sanitized_boxes, class_labels=sanitized_cls
                )
                image_tensor = transformed["image"]
                boxes_trans = transformed["bboxes"]
                cls_trans = transformed["class_labels"]
            except Exception as e:
                # Fallback if transform fails (e.g. invalid box)
                print(f"Transform error on {filename}: {e}")
                image_tensor = A.Compose(
                    [
                        A.Resize(height=self.target_size[0], width=self.target_size[1]),
                        A.Normalize(),
                        ToTensorV2(),
                    ]
                )(image=image)["image"]
                boxes_trans = []
                cls_trans = []
        else:
            # Should not happen with default init
            pass

        # Separate boxes back
        knee_targets = []
        lesion_targets = []
        lesion_target_classes = []

        for box, cls in zip(boxes_trans, cls_trans):
            # Clamp box to [0, 1]
            bx, by, bw, bh = box
            bx = max(0.001, min(0.999, bx))
            by = max(0.001, min(0.999, by))
            bw = max(0.001, min(0.999, bw))
            bh = max(0.001, min(0.999, bh))

            # Ensure box is valid
            if bw <= 0.001 or bh <= 0.001:
                continue

            # Additional check: box edges within [0, 1]
            x1 = bx - bw / 2
            y1 = by - bh / 2
            x2 = bx + bw / 2
            y2 = by + bh / 2

            if x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1:
                # Clip validly?
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(1, x2)
                y2 = min(1, y2)
                bw = x2 - x1
                bh = y2 - y1
                bx = x1 + bw / 2
                by = y1 + bh / 2

            if bw <= 0.001 or bh <= 0.001:
                continue

            if cls == 99:
                knee_targets.append([bx, by, bw, bh])
            else:
                lesion_targets.append([bx, by, bw, bh])
                lesion_target_classes.append(int(cls))

        # Convert to tensors
        target_dict = {
            "knee_boxes": (
                torch.tensor(knee_targets, dtype=torch.float32)
                if knee_targets
                else torch.zeros((0, 4))
            ),
            "lesion_boxes": (
                torch.tensor(lesion_targets, dtype=torch.float32)
                if lesion_targets
                else torch.zeros((0, 4))
            ),
            "lesion_classes": (
                torch.tensor(lesion_target_classes, dtype=torch.long)
                if lesion_target_classes
                else torch.zeros((0,), dtype=torch.long)
            ),
            "label": torch.tensor(self.labels_map.get(filename, 0), dtype=torch.long),
        }

        return image_tensor, target_dict

    def _sanitize_box(self, box: List[float]) -> List[float]:
        """
        Sanitize YOLO box [cx, cy, w, h] to be strictly within [0, 1].
        """
        cx, cy, w, h = box

        # Convert to corners
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # Clip to [0, 1]
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))

        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))

        # Ensure x2 >= x1 and y2 >= y1
        x2 = max(x1, x2)
        y2 = max(y1, y2)

        # Recalculate YOLO
        new_w = x2 - x1
        new_h = y2 - y1
        new_cx = x1 + new_w / 2
        new_cy = y1 + new_h / 2

        return [new_cx, new_cy, new_w, new_h]


def collate_end_to_end(batch):
    images = []
    targets = []

    for img, target in batch:
        images.append(img)
        targets.append(target)

    images = torch.stack(images)
    return images, targets
