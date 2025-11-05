"""
Train a YOLO detector on knee crops with KL boxes (multi-box per image).

Assumptions
- Images: processed/knee/images/*.jpg
- Labels: processed/knee/labels/*.txt (YOLO fmt) under the same parent as images
  so that replacing '/images/' with '/labels/' yields matching label path.
- External splits list original stems in splits/train.txt and splits/val.txt
  (same stems as full images before cropping). We map crops back to original stems
  using split_utils.extract_orig_stem_from_crop_path().

This script generates YOLO dataset files and launches Ultralytics training:
- processed/det/train.txt, val.txt: absolute image paths
- processed/det/dataset.yaml: dataset config with class names from config.CLASSES

Usage (PowerShell):
  python pipeline/train_det_yolo.py `
    --img_dir processed/knee/images `
    --splits_dir splits `
    --model yolov8n.pt `
    --epochs 100 `
    --imgsz 640 `
    --batch 16
"""

import os
import sys
import argparse
from typing import List

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import CLASSES
from pipeline.utils.split_utils import (
    read_split_stems,
    extract_orig_stem_from_crop_path,
)


def list_images(img_dir: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in exts]
    files.sort()
    return files


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLO detector on knee crops")
    p.add_argument("--img_dir", default=os.path.join("processed", "knee", "images"))
    p.add_argument("--splits_dir", default="splits")
    p.add_argument("--out_dir", default=os.path.join("processed", "det"))
    p.add_argument("--model", default="yolov8n.pt", help="Ultralytics model weights, e.g., yolov8n.pt or yolo11n.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default=None, help="cuda, cpu, or cuda:0 ...")
    return p.parse_args()


def make_dataset_files(img_dir: str, splits_dir: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    imgs = list_images(img_dir)
    if not imgs:
        raise SystemExit(f"No images found in {img_dir}")

    train_list = os.path.join(splits_dir, "train.txt")
    val_list = os.path.join(splits_dir, "val.txt")
    if not (os.path.exists(train_list) and os.path.exists(val_list)):
        raise SystemExit("Missing splits/train.txt or splits/val.txt")

    train_stems = read_split_stems(train_list)
    val_stems = read_split_stems(val_list)
    # Some original image stems in splits end with _0/_1, while crop names drop this.
    # Build base-stem aliases without the trailing _<digit> so crops can map back.
    def base_stem(s: str) -> str:
        # remove trailing _<digits>
        if "_" in s and s.rsplit("_", 1)[-1].isdigit():
            return s.rsplit("_", 1)[0]
        return s
    train_base = {base_stem(s) for s in train_stems}
    val_base = {base_stem(s) for s in val_stems}

    abs_img_dir = os.path.abspath(img_dir)
    train_paths: List[str] = []
    val_paths: List[str] = []

    for f in imgs:
        p = os.path.join(abs_img_dir, f)
        orig_stem = extract_orig_stem_from_crop_path(p)
        if orig_stem in train_stems or orig_stem in train_base:
            train_paths.append(p.replace("\\", "/"))
        elif orig_stem in val_stems or orig_stem in val_base:
            val_paths.append(p.replace("\\", "/"))

    if not train_paths or not val_paths:
        raise SystemExit(
            f"Split mapping empty: train={len(train_paths)} val={len(val_paths)}. Check stems and crop naming."
        )

    train_txt = os.path.join(out_dir, "train.txt")
    val_txt = os.path.join(out_dir, "val.txt")
    with open(train_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(train_paths))
    with open(val_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(val_paths))

    names = [CLASSES[i] for i in sorted(CLASSES.keys())]
    yaml_path = os.path.join(out_dir, "dataset.yaml")
    # Use absolute paths in YAML to avoid double-prefix resolution issues
    train_txt_posix = os.path.abspath(train_txt).replace("\\", "/")
    val_txt_posix = os.path.abspath(val_txt).replace("\\", "/")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("names: \n")
        for i, n in enumerate(names):
            f.write(f"  {i}: {n}\n")
        f.write(f"nc: {len(names)}\n")
        f.write(f"train: {train_txt_posix}\n")
        f.write(f"val: {val_txt_posix}\n")

    return yaml_path


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Ultralytics not installed. Run: pip install ultralytics\n" + str(e)
        )

    yaml_path = make_dataset_files(args.img_dir, args.splits_dir, args.out_dir)
    print(f"Dataset YAML -> {yaml_path}")

    model = YOLO(args.model)
    # Auto-select device if not provided. Prefer CUDA when available, else CPU.
    device_arg = args.device
    if not device_arg:
        try:
            import torch  # type: ignore
            device_arg = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device_arg = "cpu"
    train_kwargs = dict(
        data=yaml_path.replace("\\", "/"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device_arg,
        project=os.path.join(args.out_dir, "runs"),
        name=os.path.splitext(os.path.basename(args.model))[0],
    )
    print("Training with:", train_kwargs)
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
