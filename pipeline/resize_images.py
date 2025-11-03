"""
Resize existing images to a fixed size without cropping.

Usage (PowerShell):
  python pipeline/resize_images.py \
    --in_dir processed/classification/images \
    --out_dir processed/classification/images_512 \
    --size 512

Optional aspect-preserving letterbox (pad to square):
  python pipeline/resize_images.py --keep_aspect --size 512
"""
import os
import sys
import argparse
from typing import Tuple

import cv2
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import IMG_SIZE


def ensure_rgb(img: np.ndarray) -> np.ndarray:
    if img is None:
        return img
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_square(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def letterbox_square(img: np.ndarray, size: int, color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[:, :] = color
    x1 = (size - nw) // 2
    y1 = (size - nh) // 2
    canvas[y1 : y1 + nh, x1 : x1 + nw] = resized
    return canvas


def parse_args():
    p = argparse.ArgumentParser(description="Resize images without cropping")
    p.add_argument("--in_dir", default=os.path.join("processed", "classification", "images"))
    p.add_argument("--out_dir", default=os.path.join("processed", "classification", "images_512"))
    p.add_argument("--size", type=int, default=IMG_SIZE)
    p.add_argument("--keep_aspect", action="store_true", help="Preserve aspect by letterbox padding to square")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    count = 0
    for fn in os.listdir(args.in_dir):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        in_path = os.path.join(args.in_dir, fn)
        out_path = os.path.join(args.out_dir, fn)
        img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        img = ensure_rgb(img)
        if args.keep_aspect:
            out = letterbox_square(img, args.size)
        else:
            out = resize_square(img, args.size)
        cv2.imwrite(out_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        count += 1
    print(f"Resized {count} images -> {args.out_dir}")


if __name__ == "__main__":
    main()
