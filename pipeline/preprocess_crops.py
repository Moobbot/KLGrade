"""
Preprocess: crop knee regions from YOLO labels into a classification dataset.

Inputs
  - Images: dataset/images/*.jpg|png
  - Labels: dataset/labels/*.txt  (YOLO: class cx cy w h, normalized)

Outputs
  - processed/classification/images/*.jpg  (cropped and resized)
  - processed/classification/labels.csv   (columns: path,label)

Notes
  - Uses config.CLASSES for class space (0..4 KL grades).
  - Handles grayscale X-rays; converts to RGB before saving.
  - Skips empty or malformed labels.
"""
import os
import sys
import csv
import argparse
from typing import List, Tuple

import cv2
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import CLASSES, IMG_SIZE
from utils import yolo_to_xyxy_norm

IMG_DIR = os.path.join("dataset", "images")
LABEL_DIR = os.path.join("dataset", "labels")
OUT_IMG_DIR = os.path.join("processed", "classification", "images")
OUT_CSV = os.path.join("processed", "classification", "labels.csv")


def crop_and_save(image: np.ndarray, box_xyxy: Tuple[int, int, int, int], out_path: str, size: int):
    x1, y1, x2, y2 = box_xyxy
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return False
    crop = image[y1:y2, x1:x2]
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    # ensure RGB
    if len(crop.shape) == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    elif crop.shape[2] == 1:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(out_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    return True


def process_one(image_path: str, label_path: str, out_dir: str, size: int) -> List[Tuple[str, int]]:
    basename = os.path.splitext(os.path.basename(image_path))[0]
    # read image as grayscale then convert
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []
    H, W = img.shape
    items = []
    if not os.path.exists(label_path):
        return items
    with open(label_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                c = int(float(parts[0]))
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except Exception:
                continue
            # convert to absolute xyxy
            x1n, y1n, x2n, y2n = yolo_to_xyxy_norm(cx, cy, w, h)
            x1 = int(round(x1n * W))
            y1 = int(round(y1n * H))
            x2 = int(round(x2n * W))
            y2 = int(round(y2n * H))
            out_name = f"{basename}_obj{idx}_c{c}.jpg"
            out_path = os.path.join(out_dir, out_name)
            ok = crop_and_save(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), (x1, y1, x2, y2), out_path, size)
            if ok:
                items.append((out_path, c))
    return items


def parse_args():
    p = argparse.ArgumentParser(description="Crop YOLO boxes to classification patches")
    p.add_argument("--size", type=int, default=IMG_SIZE, help="Output crop size (square)")
    p.add_argument("--out_dir", default=OUT_IMG_DIR, help="Output directory for cropped images")
    p.add_argument("--out_csv", default=OUT_CSV, help="Output CSV path")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    all_images = [f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    all_images.sort()
    rows = []
    for img_name in all_images:
        stem = os.path.splitext(img_name)[0]
        img_path = os.path.join(IMG_DIR, img_name)
        label_path = os.path.join(LABEL_DIR, f"{stem}.txt")
        pairs = process_one(img_path, label_path, args.out_dir, args.size)
        rows.extend(pairs)

    # write CSV
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])  # label is class id (0..4)
        for p, c in rows:
            w.writerow([p, c])
    print(f"Saved {len(rows)} crops -> {args.out_csv}")


if __name__ == "__main__":
    main()
