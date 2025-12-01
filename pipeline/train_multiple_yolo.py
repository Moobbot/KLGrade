"""
Script để huấn luyện nhiều phiên bản YOLO khác nhau (YOLOv8, YOLO 11, các kích thước khác nhau).
Chạy tuần tự từng model một.

Cách sử dụng:
  # Chạy tất cả các model mặc định
  python pipeline/train_multiple_yolo.py

  # Chỉ định danh sách model cụ thể
  python pipeline/train_multiple_yolo.py --models yolov8n.pt yolo11n.pt yolov8s.pt

  # Chạy với tham số tùy chỉnh
  python pipeline/train_multiple_yolo.py --epochs 50 --batch 32 --imgsz 640

  # Chỉ định device cụ thể
  python pipeline/train_multiple_yolo.py --device cuda:0
"""

import os
import sys
import argparse
import subprocess
from typing import Optional

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


# Danh sách các model YOLO phổ biến
DEFAULT_MODELS = [
    # YOLO 11 (mới nhất)
    "yolo11n.pt",  # nano
    "yolo11s.pt",  # small
    "yolo11m.pt",  # medium
    "yolo11l.pt",  # large
    "yolo11x.pt",  # extra large
    # YOLOv8 (phiên bản trước)
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
]


def run_training(
    model: str,
    img_dir: str,
    splits_dir: str,
    out_dir: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: Optional[str] = None,
    use_wandb: bool = False,
    wandb_project: str = "KLGrade",
    wandb_name: Optional[str] = None,
) -> tuple[str, bool, str]:
    """
    Chạy training cho một model cụ thể.
    
    Returns:
        (model_name, success, error_message)
    """
    print(f"\n{'='*60}")
    print(f"Training model: {model}")
    print(f"{'='*60}\n")
    
    # Xây dựng lệnh
    cmd = [
        sys.executable,
        os.path.join(ROOT_DIR, "pipeline", "train_det.py"),
        "--img_dir", img_dir,
        "--splits_dir", splits_dir,
        "--out_dir", out_dir,
        "--model", model,
        "--epochs", str(epochs),
        "--imgsz", str(imgsz),
        "--batch", str(batch),
    ]
    
    if device:
        cmd.extend(["--device", device])
    
    if use_wandb:
        cmd.append("--use_wandb")
        cmd.extend(["--wandb_project", wandb_project])
        if wandb_name:
            cmd.extend(["--wandb_name", wandb_name])
        else:
            # Tạo tên wandb dựa trên model
            model_name = os.path.splitext(model)[0]
            cmd.extend(["--wandb_name", f"{model_name}_epochs{epochs}_imgsz{imgsz}"])
    
    try:
        result = subprocess.run(
            cmd,
            cwd=ROOT_DIR,
            check=True,
            capture_output=False,  # Hiển thị output real-time
            text=True,
        )
        return (model, True, "")
    except subprocess.CalledProcessError as e:
        error_msg = f"Error training {model}: {e}"
        print(f"\n❌ {error_msg}\n")
        return (model, False, error_msg)
    except Exception as e:
        error_msg = f"Unexpected error with {model}: {e}"
        print(f"\n❌ {error_msg}\n")
        return (model, False, error_msg)


def parse_args():
    """Parse tham số dòng lệnh."""
    p = argparse.ArgumentParser(
        description="Train multiple YOLO model versions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Chạy tất cả model mặc định
  python pipeline/train_multiple_yolo.py

  # Chỉ định model cụ thể
  python pipeline/train_multiple_yolo.py --models yolov8n.pt yolo11n.pt

  # Chỉ định device cụ thể
  python pipeline/train_multiple_yolo.py --device cuda:0

  # Với wandb logging
  python pipeline/train_multiple_yolo.py --use_wandb --wandb_project MyProject
        """
    )
    
    p.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=f"Danh sách model để train (mặc định: {len(DEFAULT_MODELS)} models)"
    )
    p.add_argument("--img_dir", default=os.path.join("processed", "knee", "images"))
    p.add_argument("--splits_dir", default="splits")
    p.add_argument("--out_dir", default=os.path.join("processed", "det"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device để train (cuda:0, cuda:1, cpu). Nếu không chỉ định, tự động chọn."
    )
    p.add_argument("--use_wandb", action="store_true", help="Bật logging Weights & Biases")
    p.add_argument("--wandb_project", default="KLGrade", help="Tên dự án wandb")
    
    return p.parse_args()


def main():
    args = parse_args()
    
    # Chọn danh sách model
    models = args.models if args.models else DEFAULT_MODELS
    
    print(f"\n{'='*60}")
    print(f"Training {len(models)} YOLO models (tuần tự)")
    print(f"{'='*60}")
    print(f"Models: {', '.join(models)}")
    print(f"Epochs: {args.epochs}, Image size: {args.imgsz}, Batch: {args.batch}")
    if args.device:
        print(f"Device: {args.device}")
    else:
        print("Device: tự động chọn")
    print(f"Output directory: {args.out_dir}")
    print(f"{'='*60}\n")
    
    # Chạy training tuần tự
    results = []
    print(f"🔄 Chạy tuần tự {len(models)} models...\n")
    
    for i, model in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] Đang train model: {model}")
        result = run_training(
            model=model,
            img_dir=args.img_dir,
            splits_dir=args.splits_dir,
            out_dir=args.out_dir,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_name=None,
        )
        results.append(result)
    
    # Tóm tắt kết quả
    print(f"\n{'='*60}")
    print("TÓM TẮT KẾT QUẢ")
    print(f"{'='*60}\n")
    
    successful = [r[0] for r in results if r[1]]
    failed = [(r[0], r[2]) for r in results if not r[1]]
    
    print(f"✅ Thành công: {len(successful)}/{len(results)}")
    for model in successful:
        print(f"   - {model}")
    
    if failed:
        print(f"\n❌ Thất bại: {len(failed)}/{len(results)}")
        for model, error in failed:
            print(f"   - {model}: {error}")
    
    print(f"\n{'='*60}\n")
    
    # Trả về exit code
    if failed:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

