"""
# train_det.py
1) Load data (chuẩn bị file dataset YOLO + tuỳ chọn preview với dataset.py)
2) Load model (Ultralytics YOLO)
3) Train (gọi hàm train của Ultralytics)

Giả định thư mục dữ liệu:
- Ảnh crops: processed/knee/images/*.jpg
- Nhãn YOLO: processed/knee/labels/*.txt (cùng parent với images; thay '/images' -> '/labels')
- File chia tập: splits/train.txt, splits/val.txt (stem theo ảnh gốc). Ta sẽ map từ crop về stem gốc
  bằng pipeline.utils.split_utils.extract_orig_stem_from_crop_path().

Script sẽ tạo ra:
- processed/det/train.txt, processed/det/val.txt: đường dẫn ảnh tuyệt đối
- processed/det/dataset.yaml: file cấu hình YOLO (names, nc, train, val)

Cách chạy (PowerShell):
  python pipeline/train_det.py ^
    --img_dir processed/knee/images ^
    --splits_dir splits ^
    --out_dir processed/det ^
    --model yolov8n.pt ^
    --epochs 10 ^
    --imgsz 640 ^
    --batch 16 ^
    --use_dataset_py ^
    --preview_n 5
"""

# ============================
# QUY TRÌNH LOAD DATA (tóm tắt)
# ============================
# 1) Đầu vào cần có
#    - Ảnh crops: --img_dir (mặc định processed/knee/images)
#    - Splits: --splits_dir chứa train.txt, val.txt (theo STEM ẢNH GỐC)
#    - Lớp: lấy từ config.CLASSES
#
# 2) make_dataset_files:
#    - Liệt kê ảnh trong --img_dir
#    - Đọc stems gốc từ splits/train.txt, splits/val.txt
#    - Suy ra stem gốc của từng crop bằng extract_orig_stem_from_crop_path
#      (hỗ trợ tên file dạng '<stem>_obj...' hoặc '<stem>_knee...')
#    - Ánh xạ ảnh crops vào train/val theo danh sách stems gốc
#    - Ghi processed/det/train.txt, processed/det/val.txt (đường dẫn tuyệt đối)
#
# 3) Viết processed/det/dataset.yaml
#    - names, nc lấy từ config.CLASSES
#    - train/val trỏ tới train.txt/val.txt ở bước 2 (absolute posix path)
#
# 4) Trả về đường dẫn dataset.yaml -> dùng trực tiếp cho Ultralytics

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

# Preview dataset đã được tách ra script riêng: pipeline/preview_det_dataset.py


def list_images(img_dir: str) -> List[str]:
    """Liệt kê các file ảnh trong thư mục theo các đuôi phổ biến, đã sort."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in exts]
    files.sort()
    return files


def parse_args():
    """Parse tham số dòng lệnh cho huấn luyện YOLO."""
    p = argparse.ArgumentParser(description="Train YOLO detector on knee crops")
    p.add_argument("--img_dir", default=os.path.join("processed", "knee", "images"))
    p.add_argument("--splits_dir", default="splits")
    p.add_argument("--out_dir", default=os.path.join("processed", "det"))
    p.add_argument("--model", default="yolov8n.pt", help="Ultralytics model weights, e.g., yolov8n.pt or yolo11n.pt")
    p.add_argument("--backend", default=None, help="Backend detector: ví dụ 'ultralytics'. Nếu bỏ trống dùng mặc định.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default=None, help="cuda, cpu, hoặc cuda:0 ...")
    # wandb logging (tuỳ chọn)
    p.add_argument("--use_wandb", action="store_true", help="Bật logging Weights & Biases nếu đã cài wandb")
    p.add_argument("--wandb_project", default="KLGrade", help="Tên dự án wandb")
    p.add_argument("--wandb_name", default=None, help="Tên run wandb (mặc định dựa vào tên weights)")
    return p.parse_args()


def make_dataset_files(img_dir: str, splits_dir: str, out_dir: str) -> str:
    """
    BƯỚC 1A - CHUẨN BỊ DỮ LIỆU
    - Đọc danh sách ảnh crop trong img_dir
    - Map mỗi ảnh crop về stem gốc để chia train/val dựa vào splits/*.txt
    - Ghi processed/det/train.txt, val.txt (đường dẫn ảnh tuyệt đối, posix)
    - Tạo processed/det/dataset.yaml (names, nc, train, val)
    """
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

    # Xây alias base-stem: loại bỏ hậu tố _<digits> để linh hoạt khi map crop -> gốc
    def base_stem(s: str) -> str:
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
        if orig_stem in train_stems or base_stem(orig_stem) in train_base:
            train_paths.append(p.replace("\\", "/"))
        elif orig_stem in val_stems or base_stem(orig_stem) in val_base:
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
    # Dùng absolute posix path trong YAML để tránh lỗi đường dẫn trên Windows
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

    # ================================================================
    # BƯỚC 1 - LOAD DATA
    # 1A) Tạo file train.txt, val.txt và dataset.yaml cho Ultralytics
    # (Preview dữ liệu bằng dataset.py đã tách sang pipeline/preview_det_dataset.py)
    # ================================================================

    yaml_path = make_dataset_files(args.img_dir, args.splits_dir, args.out_dir)
    print(f"Dataset YAML -> {yaml_path}")

    # ================================================================
    # BƯỚC 2 - LOAD MODEL
    # Dùng Ultralytics YOLO; nếu chưa cài sẽ báo hướng dẫn cài
    # ================================================================

    # BƯỚC 2 - LOAD MODEL thông qua OOP wrapper
    from pipeline.model_det import build_detection_model

    # Khởi tạo model detector từ checkpoint (vd: yolov8n.pt, yolo11n.pt)
    detector = build_detection_model(weights=args.model, backend=args.backend)

    # Tự chọn device nếu không chỉ định: ưu tiên CUDA, nếu không có thì CPU
    device_arg = args.device
    if not device_arg:
        try:
            import torch  # type: ignore
            device_arg = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device_arg = "cpu"

    # ================================================================
    # BƯỚC 3 - TRAIN
    # Gọi hàm train của Ultralytics với YAML data vừa tạo
    # ================================================================

    train_kwargs = dict(
        data=yaml_path.replace("\\", "/"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device_arg,
        project=os.path.join(args.out_dir, "runs"),
        name=os.path.splitext(os.path.basename(args.model))[0],
    )
    # Optional: init wandb
    if args.use_wandb:
        try:
            import wandb  # type: ignore
            run_name = args.wandb_name or train_kwargs["name"]
            wandb.init(project=args.wandb_project, name=run_name, config=train_kwargs)
        except Exception as e:
            print("[WARN] wandb init failed or not installed:", e)
    print("Training with:", train_kwargs)
    detector.train(**train_kwargs)


if __name__ == "__main__":
    main()
