"""
Preview dữ liệu detection bằng dataset.py theo splits.

Chức năng:
- Đọc danh sách ảnh trong img_dir
- Map về stem gốc để lọc theo splits/train.txt, splits/val.txt
- Tạo YoloDataset cho train/val, in số lượng và duyệt N mẫu (không vẽ)

Ví dụ (PowerShell):
  python pipeline/preview_det_dataset.py `
    --img_dir processed/knee/images `
    --splits_dir splits `
    --n 5
"""

import os
import sys
import argparse
from typing import List
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pipeline.utils.split_utils import (
    read_split_stems,
    extract_orig_stem_from_crop_path,
)

try:
    from dataset import YoloDataset  # type: ignore
except Exception as e:
    raise SystemExit("Không import được dataset.YoloDataset: " + str(e))

try:
    from config import CLASSES  # type: ignore

    _CLASS_NAMES = [CLASSES[i] for i in sorted(CLASSES.keys())]
except Exception:
    _CLASS_NAMES = None


def list_images(img_dir: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in exts]
    files.sort()
    return files


def parse_args():
    p = argparse.ArgumentParser(
        description="Preview detection dataset via dataset.py (hiển thị hoặc lưu ảnh kèm nhãn)"
    )
    p.add_argument("--img_dir", default=os.path.join("processed", "knee", "images"))
    p.add_argument("--splits_dir", default="splits")
    p.add_argument("--n", type=int, default=5, help="Số mẫu duyệt từ train split")
    p.add_argument(
        "--out_dir",
        default=None,
        help="Nếu đặt, lưu ảnh đã vẽ bbox vào thư mục này thay vì chỉ hiển thị",
    )
    p.add_argument(
        "--split",
        choices=["train", "val"],
        default="train",
        help="Xem trước train hoặc val",
    )
    return p.parse_args()


def _visualize(
    image_tensor,
    boxes,
    labels,
    idx: int,
    out_dir: str | None,
    out_filename: str | None = None,
    img_name: str | None = None,
    label_name: str | None = None,
):
    # image_tensor: torch.Tensor CxHxW in [0,1]
    import torch  # lazy import for type

    if isinstance(image_tensor, torch.Tensor):
        image = image_tensor.permute(1, 2, 0).cpu().numpy()
    else:
        image = image_tensor
    # Tạo figure gồm 2 hàng: hàng trên dành cho text, hàng dưới là ảnh + bbox
    # Giảm chiều cao tổng thể và vùng header
    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[0.08, 0.92])
    gs.update(hspace=0.01)
    top_ax = fig.add_subplot(gs[0])
    img_ax = fig.add_subplot(gs[1])

    # Hàng trên: nền trắng, 2 dòng text (ảnh, label)
    top_ax.set_facecolor("white")
    top_ax.axis("off")
    line_y1 = 0.65
    line_y2 = 0.15
    if img_name:
        top_ax.text(
            0.01,
            line_y1,
            f"img: {img_name}",
            color="black",
            fontsize=10,
            ha="left",
            va="center",
        )
    if label_name:
        top_ax.text(
            0.01,
            line_y2,
            f"label: {label_name}",
            color="black",
            fontsize=10,
            ha="left",
            va="center",
        )

    # Hàng dưới: hiển thị ảnh và vẽ bbox
    img_ax.imshow(image)
    # convert to list for iteration
    try:
        boxes_iter = boxes.cpu().tolist()
    except Exception:
        boxes_iter = boxes
    try:
        labels_iter = labels.cpu().tolist()
    except Exception:
        labels_iter = labels
    for box, label in zip(boxes_iter, labels_iter):
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        rect = patches.Rectangle(
            (x1, y1), width, height, linewidth=2, edgecolor="lime", facecolor="none"
        )
        img_ax.add_patch(rect)
        lbl_txt = str(label)
        if _CLASS_NAMES and 0 <= int(label) < len(_CLASS_NAMES):
            lbl_txt = _CLASS_NAMES[int(label)]
        # Đặt nhãn ở giữa cạnh trên bbox, cao hơn để không che vùng khoanh
        label_x = x1 + width / 2
        label_y = max(y1 - 10, 0)
        img_ax.text(
            label_x,
            label_y,
            lbl_txt,
            color="yellow",
            fontsize=10,
            ha="center",
            va="bottom",
            bbox=dict(facecolor="black", alpha=0.9, boxstyle="round,pad=0.25"),
            zorder=5,
        )
    img_ax.axis("off")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        # nếu không cung cấp tên file, dùng chỉ số và .png
        save_name = out_filename or f"{idx}.png"
        out_path = os.path.join(out_dir, save_name)
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0.02, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def main():
    args = parse_args()

    train_list = os.path.join(args.splits_dir, "train.txt")
    val_list = os.path.join(args.splits_dir, "val.txt")
    if not (os.path.exists(train_list) and os.path.exists(val_list)):
        raise SystemExit("Missing splits/train.txt or splits/val.txt")

    train_stems = read_split_stems(train_list)
    val_stems = read_split_stems(val_list)

    def base_stem(s: str) -> str:
        if "_" in s and s.rsplit("_", 1)[-1].isdigit():
            return s.rsplit("_", 1)[0]
        return s

    train_base = {base_stem(s) for s in train_stems}
    val_base = {base_stem(s) for s in val_stems}

    labels_dir = args.img_dir.replace("\\", "/").replace("/images", "/labels")
    img_dir_norm = args.img_dir.replace("\\", "/")
    if not os.path.isdir(labels_dir):
        print(f"Warning: labels dir not found: {labels_dir}")

    train_ds = YoloDataset(img_dir_norm, labels_dir, transform=None)
    val_ds = YoloDataset(img_dir_norm, labels_dir, transform=None)

    # Lọc danh sách ảnh để khớp splits
    def stem_of_file(fname: str) -> str:
        full = os.path.join(img_dir_norm, fname)
        return extract_orig_stem_from_crop_path(full)

    train_files: List[str] = []
    val_files: List[str] = []
    for f in list_images(img_dir_norm):
        st = stem_of_file(f)
        if st in train_stems or base_stem(st) in train_base:
            train_files.append(f)
        elif st in val_stems or base_stem(st) in val_base:
            val_files.append(f)
    train_ds.image_files = sorted(train_files)
    val_ds.image_files = sorted(val_files)

    print(f"dataset.py -> train images: {len(train_ds)} | val images: {len(val_ds)}")

    # Chọn split để preview
    ds = train_ds if args.split == "train" else val_ds
    n = min(max(args.n, 0), len(ds))
    if n > 0:
        print(f"Preview {n} mẫu từ {args.split} bằng dataset.py (vẽ bbox)...")
        for i in range(n):
            # Lấy tên ảnh gốc và đặt tên file xuất giống hệt (không thêm tiền tố)
            fname = ds.image_files[i]
            stem = os.path.splitext(fname)[0]
            label_fname = stem + ".txt"
            img_t, boxes_t, labels_t = ds[i]
            # Lưu cùng tên ảnh gốc (không đè ảnh gốc vì out_dir khác)
            _visualize(
                img_t,
                boxes_t,
                labels_t,
                idx=i,
                out_dir=args.out_dir,
                out_filename=fname,
                img_name=fname,
                label_name=label_fname,
            )
        if args.out_dir:
            print(f"Đã lưu {n} ảnh preview vào: {args.out_dir}")
        else:
            print("Preview xong.")


if __name__ == "__main__":
    main()
