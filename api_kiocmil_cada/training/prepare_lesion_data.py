"""
Prepare Lesion Detection Dataset:
1. Crop Knee regions from images using Ground Truth Knee Labels.
2. Extract Lesion Bounding Boxes (OST/JS) within the Knee region.
3. Adjust coordinates relative to the crop.
4. Map classes: {0,1,2,3} -> 0 (Osteophytes), {4,5} -> 1 (Joint Space).
5. Save as YOLO dataset.
"""

import os
import sys
import yaml
import shutil
import cv2
import random
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Constants
OST_CLASSES = {0, 1, 2, 3}
JS_CLASSES = {4, 5}
MIN_AREA = 50  # Minimum lesion area to keep


def yolo_to_bbox(yolo_box, w, h):
    """Convert xywh (normalized) to x1y1x2y2 (absolute)."""
    cx, cy, bw, bh = yolo_box
    cx, cy, bw, bh = float(cx), float(cy), float(bw), float(bh)
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def bbox_to_yolo(bbox, w, h):
    """Convert x1y1x2y2 (absolute) to xywh (normalized)."""
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return [cx, cy, bw, bh]


def process_dataset(img_dir, knee_label_dir, lesion_label_dir, output_dir, split_dir):
    """Process dataset and generate crops."""
    img_dir = Path(img_dir)
    knee_label_dir = Path(knee_label_dir)
    lesion_label_dir = Path(lesion_label_dir)
    output_dir = Path(output_dir)

    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)

    # Read splits from existing Knee splits to maintain consistency
    splits = ["train", "val", "test"]
    image_sets = {}

    for split in splits:
        split_file = Path(split_dir) / f"{split}.txt"
        if split_file.exists():
            with open(split_file, "r") as f:
                # Resolve paths relative to project root or absolute
                paths = [line.strip() for line in f if line.strip()]
                # Extract filenames
                image_sets[split] = {Path(p).name for p in paths}

    # If no splits found, just process all images
    all_images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    print(f"Found {len(all_images)} source images.")

    generated_files = {s: [] for s in splits}

    for img_path in tqdm(all_images, desc="Processing images"):
        stem = img_path.stem
        knee_path = knee_label_dir / f"{stem}.txt"
        lesion_path = lesion_label_dir / f"{stem}.txt"

        if not knee_path.exists() or not lesion_path.exists():
            continue

        # Determine split
        current_split = None
        for s in splits:
            if img_path.name in image_sets.get(s, set()):
                current_split = s
                break

        # If not in any split (and splits exist), skip
        if not current_split and image_sets:
            continue

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]

        # Load Knee Boxes
        knees = []
        with open(knee_path, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 5:
                    # class is irrelevant for knee (usually 0)
                    knees.append([float(x) for x in parts[1:5]])

        # Load Lesion Boxes
        lesions = []
        lesion_classes = []
        with open(lesion_path, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 5:
                    cls = int(float(parts[0]))
                    lesions.append([float(x) for x in parts[1:5]])
                    lesion_classes.append(cls)

        # Process each knee
        for i, knee_box in enumerate(knees):
            kx1, ky1, kx2, ky2 = yolo_to_bbox(knee_box, W, H)

            # Crop Image
            # Add small padding 10%
            pad_x = int((kx2 - kx1) * 0.1)
            pad_y = int((ky2 - ky1) * 0.1)

            cx1 = max(0, kx1 - pad_x)
            cy1 = max(0, ky1 - pad_y)
            cx2 = min(W, kx2 + pad_x)
            cy2 = min(H, ky2 + pad_y)

            crop_w = cx2 - cx1
            crop_h = cy2 - cy1

            if crop_w < 10 or crop_h < 10:
                continue

            crop_img = img[cy1:cy2, cx1:cx2]

            # Find lesions inside/intersecting crop
            msg_labels = []

            for l_box, l_cls in zip(lesions, lesion_classes):
                lx1, ly1, lx2, ly2 = yolo_to_bbox(l_box, W, H)

                # Intersection
                ix1 = max(cx1, lx1)
                iy1 = max(cy1, ly1)
                ix2 = min(cx2, lx2)
                iy2 = min(cy2, ly2)

                if ix1 < ix2 and iy1 < iy2:
                    area = (ix2 - ix1) * (iy2 - iy1)
                    if area < MIN_AREA:
                        continue

                    # Map Coordinates to Crop
                    # Clamp to crop
                    lx1_c = max(0, lx1 - cx1)
                    ly1_c = max(0, ly1 - cy1)
                    lx2_c = min(crop_w, lx2 - cx1)
                    ly2_c = min(crop_h, ly2 - cy1)

                    # Convert to YOLO format relative to crop
                    yolo_l = bbox_to_yolo([lx1_c, ly1_c, lx2_c, ly2_c], crop_w, crop_h)

                    # Map Class
                    new_cls = -1
                    if l_cls in OST_CLASSES:
                        new_cls = 0  # OST
                    elif l_cls in JS_CLASSES:
                        new_cls = 1  # JS

                    if new_cls != -1:
                        msg_labels.append(f"{new_cls} {' '.join(map(str, yolo_l))}")

            # Save files
            # Naming: original_image_name_idx.jpg
            new_name = f"{stem}_{i}"
            img_out_path = output_dir / "images" / f"{new_name}.jpg"
            label_out_path = output_dir / "labels" / f"{new_name}.txt"

            cv2.imwrite(str(img_out_path), crop_img)

            if msg_labels:
                with open(label_out_path, "w") as f:
                    f.write("\n".join(msg_labels))
            else:
                # Create empty label file for background
                with open(label_out_path, "w") as f:
                    pass

            if current_split:
                generated_files[current_split].append(str(img_out_path.absolute()))
            else:
                # Default to train if no split info
                generated_files["train"].append(str(img_out_path.absolute()))

    # Save splits
    for split, paths in generated_files.items():
        if paths:
            with open(output_dir / f"{split}.txt", "w") as f:
                f.write("\n".join(paths))

    # Create dataset.yaml
    yaml_data = {
        "path": str(output_dir.absolute()),
        "train": str(output_dir.absolute() / "train.txt"),
        "val": str(output_dir.absolute() / "val.txt"),
        "test": str(output_dir.absolute() / "test.txt"),
        "names": {0: "Osteophytes", 1: "Joint Space"},
    }

    with open(output_dir / "dataset.yaml", "w") as f:
        yaml.dump(yaml_data, f, sort_keys=False)

    print(f"\n✅ Lesion dataset prepared at {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default="datasets/dataset_v0/images")
    parser.add_argument("--knee_labels", default="datasets/dataset_v0/labels-knee")
    parser.add_argument(
        "--lesion_labels", default="datasets/dataset_v0/labels_10_class"
    )
    parser.add_argument("--split_dir", default="datasets/splits/knee")
    parser.add_argument("--out_dir", default="datasets/processed/lesion_detection")
    args = parser.parse_args()

    process_dataset(
        args.img_dir, args.knee_labels, args.lesion_labels, args.out_dir, args.split_dir
    )


if __name__ == "__main__":
    main()
