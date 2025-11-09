"""
Two-stage preprocessing (inherits logic from code-v2/visualize.py):

Stage 1 (Knee ROI):
- From full-leg images and knee detection labels, crop knee ROIs.
- Enforce a square crop centered on the knee box (like visualize.py), clamped within the image bounds.
- Reproject OA/KL boxes into the knee crop and write YOLO labels for the knee crop.

Stage 2 (Lesion classification):
- From the knee crop and its reprojected OA boxes, crop lesion patches and build a classification CSV (KL0..KL4).

Assumptions
- Knee labels and KL labels are in YOLO format: class cx cy w h (normalized).
- KL classes map to config.CLASSES (0..4).
- KL boxes are annotated on the same full image as knee boxes.

Outputs
- processed/knee/images/: knee crops (RGB, resized)
- processed/knee/labels/: YOLO labels reprojected on knee crops
- processed/classification/images/: lesion crops (RGB, resized)
- processed/classification/labels.csv: CSV with columns path,label

Notes
- OA/KL boxes are associated to a knee crop if they intersect with the knee square (first match, one-to-one as in visualize.py).
- Grayscale X-rays are converted to RGB before saving.
"""
import os
import sys
import csv
import argparse
from typing import List, Tuple, Dict

import cv2
import numpy as np


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import CLASSES, IMG_SIZE
from utils import yolo_to_xyxy_norm

# Extra margin (in pixels) to ensure KL boxes are not flush against the knee crop border
PADDING = 2


def ensure_rgb(img: np.ndarray) -> np.ndarray:
    if img is None:
        return img
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def clip_box(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int, int, int]:
    H, W = img.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)
    return x1, y1, x2, y2


def resize_and_save(rgb: np.ndarray, out_path: str, size: int) -> bool:
    if rgb is None:
        return False
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return True


def load_yolo_boxes(label_path: str) -> List[Tuple[int, float, float, float, float]]:
    """Return list of (cls, cx, cy, w, h) as floats (normalized)."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
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
                boxes.append((c, cx, cy, w, h))
            except Exception:
                continue
    return boxes


def point_in_box(px: int, py: int, box_xyxy: Tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = box_xyxy
    return (px >= x1) and (px <= x2) and (py >= y1) and (py <= y2)


def boxes_intersect(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    xx1 = max(ax1, bx1)
    yy1 = max(ay1, by1)
    xx2 = min(ax2, bx2)
    yy2 = min(ay2, by2)
    return max(0, xx2 - xx1) > 0 and max(0, yy2 - yy1) > 0


def parse_args():
    p = argparse.ArgumentParser(description="Two-stage preprocessing: knee crop then KL lesion crop")
    p.add_argument("--img_dir", default=os.path.join("dataset", "dataset_v0", "images"))
    p.add_argument("--knee_labels", default=os.path.join("dataset", "dataset_v0", "labels-knee"))
    p.add_argument("--kl_labels", default=os.path.join("dataset", "dataset_v0", "labels"))
    p.add_argument("--knee_size", type=int, default=IMG_SIZE, help="Output size for knee crops")
    p.add_argument("--lesion_size", type=int, default=IMG_SIZE, help="Output size for lesion crops")
    p.add_argument("--out_knee_dir", default=os.path.join("processed", "knee", "images"))
    p.add_argument("--out_knee_label_dir", default=os.path.join("processed", "knee", "labels"))
    p.add_argument("--out_cls_dir", default=os.path.join("processed", "classification", "images"))
    p.add_argument("--out_csv", default=os.path.join("processed", "classification", "labels.csv"))
    return p.parse_args()


def main():
    args = parse_args()

    img_names = [f for f in os.listdir(args.img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    img_names.sort()

    os.makedirs(args.out_knee_dir, exist_ok=True)
    os.makedirs(args.out_cls_dir, exist_ok=True)
    os.makedirs(args.out_knee_label_dir, exist_ok=True)
    rows: List[Tuple[str, int]] = []

    knee_count = 0
    lesion_count = 0

    for img_name in img_names:
        stem = os.path.splitext(img_name)[0]
        img_path = os.path.join(args.img_dir, img_name)
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            continue
        H, W = img_gray.shape
        img_rgb_full = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

        # Load knee boxes (normalized)
        knee_label_path = os.path.join(args.knee_labels, f"{stem}.txt")
        knee_boxes = load_yolo_boxes(knee_label_path)
        knee_xyxy_sq: List[Tuple[int, int, int, int]] = []

    # Prepare knee square proposals (do not save yet)
        for k_idx, (_, kcx, kcy, kw, kh) in enumerate(knee_boxes):
            # base knee rect in absolute pixels
            x1n, y1n, x2n, y2n = yolo_to_xyxy_norm(kcx, kcy, kw, kh)
            kx1 = int(round(x1n * W)); ky1 = int(round(y1n * H))
            kx2 = int(round(x2n * W)); ky2 = int(round(y2n * H))
            kx1, ky1, kx2, ky2 = clip_box(img_rgb_full, kx1, ky1, kx2, ky2)
            if kx2 <= kx1 or ky2 <= ky1:
                continue
            # make it square, centered
            width = kx2 - kx1
            height = ky2 - ky1
            side = max(width, height)
            cx = (kx1 + kx2) // 2
            cy = (ky1 + ky2) // 2
            half_side = side // 2
            sx1 = max(0, min(W - side, cx - half_side))
            sy1 = max(0, min(H - side, cy - half_side))
            sx2 = sx1 + side
            sy2 = sy1 + side
            sx1, sy1, sx2, sy2 = clip_box(img_rgb_full, sx1, sy1, sx2, sy2)
            if sx2 <= sx1 or sy2 <= sy1:
                continue
            knee_xyxy_sq.append((sx1, sy1, sx2, sy2))

        # Load KL lesion boxes (normalized)
        kl_label_path = os.path.join(args.kl_labels, f"{stem}.txt")
        kl_boxes = load_yolo_boxes(kl_label_path)

        # Convert KL boxes to absolute xyxy on full image
        kl_xyxy_abs: List[Tuple[int, int, int, int]] = []
        kl_classes: List[int] = []
        for (lc, lcx, lcy, lw, lh) in kl_boxes:
            x1n, y1n, x2n, y2n = yolo_to_xyxy_norm(lcx, lcy, lw, lh)
            bx1 = int(round(x1n * W)); by1 = int(round(y1n * H))
            bx2 = int(round(x2n * W)); by2 = int(round(y2n * H))
            bx1, by1, bx2, by2 = clip_box(img_rgb_full, bx1, by1, bx2, by2)
            if bx2 <= bx1 or by2 <= by1:
                continue
            kl_xyxy_abs.append((bx1, by1, bx2, by2))
            kl_classes.append(int(lc))

        # Group OA/KL boxes to knee squares by intersection (first match) for expansion (no strict filter here)
        groups_intersect: Dict[int, List[int]] = {i: [] for i in range(len(knee_xyxy_sq))}
        for i, box_oa in enumerate(kl_xyxy_abs):
            for j, box_knee in enumerate(knee_xyxy_sq):
                if boxes_intersect(box_oa, box_knee):
                    groups_intersect[j].append(i)
                    break  # one OA belongs to first matched knee

        # For each knee crop, adjust to fully cover its KL boxes, write labels, crop lesion patches, then save knee ROI
        for k_idx, knee_box in enumerate(knee_xyxy_sq):
            sx1, sy1, sx2, sy2 = knee_box
            # Expand/translate knee square to fully cover its associated KL boxes (if any)
            gi = groups_intersect.get(k_idx, [])
            if gi:
                ux1, uy1, ux2, uy2 = sx1, sy1, sx2, sy2
                for i in gi:
                    bx1, by1, bx2, by2 = kl_xyxy_abs[i]
                    ux1 = min(ux1, bx1); uy1 = min(uy1, by1)
                    ux2 = max(ux2, bx2); uy2 = max(uy2, by2)
                # add small padding so KL boxes are not tight to the crop border
                pux1 = max(0, ux1 - PADDING)
                puy1 = max(0, uy1 - PADDING)
                pux2 = min(W, ux2 + PADDING)
                puy2 = min(H, uy2 + PADDING)

                u_w = pux2 - pux1
                u_h = puy2 - puy1
                side = max(u_w, u_h)
                cx_u = (pux1 + pux2) // 2
                cy_u = (puy1 + puy2) // 2
                half = side // 2
                sx1 = max(0, min(W - side, cx_u - half))
                sy1 = max(0, min(H - side, cy_u - half))
                sx2 = sx1 + side
                sy2 = sy1 + side
                sx1, sy1, sx2, sy2 = clip_box(img_rgb_full, sx1, sy1, sx2, sy2)
                if sx2 <= sx1 or sy2 <= sy1:
                    sx1, sy1, sx2, sy2 = knee_box
            cw = sx2 - sx1
            ch = sy2 - sy1
            # Write knee labels file
            knee_label_path_out = os.path.join(args.out_knee_label_dir, f"{stem}_knee{k_idx}.txt")
            lines_out: List[str] = []
            for i in groups_intersect.get(k_idx, []):
                cls = kl_classes[i]
                bx1, by1, bx2, by2 = kl_xyxy_abs[i]
                # clamp OA to knee crop square
                nbx1 = max(bx1, sx1); nby1 = max(by1, sy1)
                nbx2 = min(bx2, sx2); nby2 = min(by2, sy2)
                if nbx1 >= nbx2 or nby1 >= nby2:
                    continue
                # to crop coordinates
                nbx1_c = max(0, min(cw, nbx1 - sx1))
                nby1_c = max(0, min(ch, nby1 - sy1))
                nbx2_c = max(0, min(cw, nbx2 - sx1))
                nby2_c = max(0, min(ch, nby2 - sy1))
                # YOLO normalized
                ncx = ((nbx1_c + nbx2_c) / 2) / cw
                ncy = ((nby1_c + nby2_c) / 2) / ch
                nbw = (nbx2_c - nbx1_c) / cw
                nbh = (nby2_c - nby1_c) / ch
                if not (0.0 <= ncx <= 1.0 and 0.0 <= ncy <= 1.0 and 0.0 < nbw <= 1.0 and 0.0 < nbh <= 1.0):
                    continue
                lines_out.append(f"{int(cls)} {ncx:.6f} {ncy:.6f} {nbw:.6f} {nbh:.6f}")
                # Also produce classification lesion crop from the knee crop region (before resize)
                lesion_roi = img_rgb_full[sy1:sy2, sx1:sx2]
                # crop lesion in knee-crop coordinates
                nbx1_c_i = int(round(nbx1_c)); nby1_c_i = int(round(nby1_c))
                nbx2_c_i = int(round(nbx2_c)); nby2_c_i = int(round(nby2_c))
                if nbx2_c_i > nbx1_c_i and nby2_c_i > nby1_c_i:
                    lesion_patch = lesion_roi[nby1_c_i:nby2_c_i, nbx1_c_i:nbx2_c_i]
                    out_name = f"{stem}_knee{k_idx}_lesion{i}_c{int(cls)}.jpg"
                    out_path = os.path.join(args.out_cls_dir, out_name)
                    if resize_and_save(lesion_patch, out_path, args.lesion_size):
                        rows.append((out_path, int(cls)))
                        lesion_count += 1

            # write YOLO label file for knee crop
            with open(knee_label_path_out, "w", encoding="utf-8") as f_ko:
                for ln in lines_out:
                    f_ko.write(ln + "\n")

            # Save the adjusted knee ROI
            knee_roi_final = img_rgb_full[sy1:sy2, sx1:sx2]
            knee_out = os.path.join(args.out_knee_dir, f"{stem}_knee{k_idx}.jpg")
            if resize_and_save(knee_roi_final, knee_out, args.knee_size):
                knee_count += 1

    # write CSV
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])  # label is 0..4
        for p, c in rows:
            w.writerow([p, c])

    print(f"Saved {knee_count} knee crops -> {args.out_knee_dir}")
    print(f"Wrote knee labels -> {args.out_knee_label_dir}")
    print(f"Saved {lesion_count} lesion crops -> {args.out_cls_dir}")
    print(f"CSV -> {args.out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
