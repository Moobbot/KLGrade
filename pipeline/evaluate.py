"""
Script đánh giá model detection (YOLO) trên tập validation/test.

Tính toán các metrics: mAP, precision, recall, F1-score cho từng class và overall.
Có thể xuất kết quả visualization (vẽ cả bbox dự đoán và bbox thực tế trên cùng ảnh).

Cách sử dụng:
  # Đánh giá trên validation set
  python pipeline/evaluate.py --weights processed/det/runs/yolov8n/weights/best.pt --split val

  # Đánh giá trên test set
  python pipeline/evaluate.py --weights processed/det/runs/yolov8n/weights/best.pt --split test

  # Đánh giá với visualization (vẽ cả GT và predicted boxes)
  python pipeline/evaluate.py --weights processed/det/runs/yolov8n/weights/best.pt --split val --save_vis --vis_dir results/vis

  # Đánh giá với confidence threshold tùy chỉnh
  python pipeline/evaluate.py --weights processed/det/runs/yolov8n/weights/best.pt --conf 0.25 --iou 0.45

  # Visualization với chỉ định label directory
  python pipeline/evaluate.py --weights processed/det/runs/yolov8n/weights/best.pt --split val --save_vis --label_dir processed/knee/labels

Lưu ý về visualization:
  - GT boxes: màu xanh lá (green)
  - Predicted boxes: màu đỏ (red) với confidence score
  - Tự động tìm thư mục labels từ img_dir, hoặc dùng --label_dir để chỉ định
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import CLASSES
from pipeline.utils.split_utils import read_split_stems, extract_orig_stem_from_crop_path
from pipeline.model_det import build_detection_model


def list_images(img_dir: str) -> List[str]:
    """Liệt kê các file ảnh trong thư mục theo các đuôi phổ biến, đã sort."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in exts]
    files.sort()
    return files


def make_dataset_files_for_eval(
    img_dir: str, splits_dir: str, split: str, out_dir: str
) -> str:
    """
    Tạo file dataset YOLO cho evaluation (tương tự train_det.py nhưng chỉ cho một split).
    
    Args:
        img_dir: Thư mục chứa ảnh crops
        splits_dir: Thư mục chứa file splits
        split: Tên split ('val' hoặc 'test')
        out_dir: Thư mục output
    
    Returns:
        Đường dẫn đến file dataset.yaml
    """
    os.makedirs(out_dir, exist_ok=True)
    imgs = list_images(img_dir)
    if not imgs:
        raise SystemExit(f"No images found in {img_dir}")

    split_list = os.path.join(splits_dir, f"{split}.txt")
    if not os.path.exists(split_list):
        raise SystemExit(f"Missing {split_list}")

    split_stems = read_split_stems(split_list)
    print(f"Loaded {split} split: {len(split_stems)} stems")
    if len(split_stems) > 0:
        print(f"  Example stems: {list(split_stems)[:3]}")

    # Xây alias base-stem: loại bỏ hậu tố _<digits> để linh hoạt khi map crop -> gốc
    def base_stem(s: str) -> str:
        if "_" in s and s.rsplit("_", 1)[-1].isdigit():
            return s.rsplit("_", 1)[0]
        return s
    split_base = {base_stem(s) for s in split_stems}

    abs_img_dir = os.path.abspath(img_dir)
    split_paths: List[str] = []
    unmatched_stems = set()

    for f in imgs:
        p = os.path.join(abs_img_dir, f)
        orig_stem = extract_orig_stem_from_crop_path(p)
        if orig_stem in split_stems or base_stem(orig_stem) in split_base:
            split_paths.append(p.replace("\\", "/"))
        else:
            unmatched_stems.add(orig_stem)

    print(f"Found {len(imgs)} images total")
    print(f"Mapped to {split}: {len(split_paths)} images")
    if unmatched_stems and len(unmatched_stems) <= 10:
        print(f"  Unmatched stems (first 10): {list(unmatched_stems)[:10]}")
    elif unmatched_stems:
        print(f"  Unmatched stems count: {len(unmatched_stems)}")

    if not split_paths:
        raise SystemExit(f"No images mapped to {split} split. Check stems and crop naming.")

    split_txt = os.path.join(out_dir, f"{split}.txt")
    with open(split_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(split_paths))

    names = [CLASSES[i] for i in sorted(CLASSES.keys())]
    yaml_path = os.path.join(out_dir, f"dataset_{split}.yaml")
    split_txt_posix = os.path.abspath(split_txt).replace("\\", "/")
    
    # Cần tạo một file dataset.yaml tạm thời với train và val giống nhau (hoặc chỉ val)
    # Ultralytics yêu cầu cả train và val trong yaml, nhưng ta có thể dùng cùng file
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("names: \n")
        for i, n in enumerate(names):
            f.write(f"  {i}: {n}\n")
        f.write(f"nc: {len(names)}\n")
        f.write(f"train: {split_txt_posix}\n")
        f.write(f"val: {split_txt_posix}\n")

    return yaml_path


def parse_args():
    """Parse tham số dòng lệnh cho evaluation."""
    p = argparse.ArgumentParser(description="Evaluate YOLO detector on validation/test set")
    p.add_argument(
        "--weights",
        required=True,
        help="Đường dẫn đến file weights đã train (vd: processed/det/runs/yolov8n/weights/best.pt)"
    )
    p.add_argument(
        "--img_dir",
        default=os.path.join("processed", "knee", "images"),
        help="Thư mục chứa ảnh crops"
    )
    p.add_argument(
        "--splits_dir",
        default="splits",
        help="Thư mục chứa file splits"
    )
    p.add_argument(
        "--split",
        default="val",
        choices=["val", "test"],
        help="Split để đánh giá (val hoặc test)"
    )
    p.add_argument(
        "--out_dir",
        default=os.path.join("processed", "det", "eval"),
        help="Thư mục output cho kết quả evaluation"
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold cho detection (mặc định: 0.25)"
    )
    p.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold cho NMS (mặc định: 0.45)"
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Kích thước ảnh input (mặc định: 640)"
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device để chạy (cuda, cpu, hoặc cuda:0). Nếu không chỉ định, tự động chọn."
    )
    p.add_argument(
        "--save_vis",
        action="store_true",
        help="Lưu visualization (vẽ bbox dự đoán trên ảnh)"
    )
    p.add_argument(
        "--vis_dir",
        default=None,
        help="Thư mục lưu visualization (mặc định: {out_dir}/visualizations)"
    )
    p.add_argument(
        "--vis_n",
        type=int,
        default=50,
        help="Số lượng ảnh để visualize (mặc định: 50, -1 để visualize tất cả)"
    )
    p.add_argument(
        "--save_json",
        action="store_true",
        help="Lưu kết quả metrics dưới dạng JSON"
    )
    p.add_argument(
        "--backend",
        default=None,
        help="Backend detector (mặc định: ultralytics)"
    )
    p.add_argument(
        "--label_dir",
        default=None,
        help="Thư mục chứa labels GT (mặc định: tự động tìm từ img_dir)"
    )
    return p.parse_args()


def read_yolo_label(label_path: str) -> List[Tuple[int, float, float, float, float]]:
    """
    Đọc file label YOLO format.
    
    Returns:
        List of (class_id, x_center, y_center, width, height) - normalized coordinates
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    continue
                try:
                    cls_id = int(float(parts[0]))
                    x, y, w, h = map(float, parts[1:5])
                    # Validate normalized coordinates
                    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0:
                        boxes.append((cls_id, x, y, w, h))
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        print(f"  ⚠️  Lỗi đọc label {label_path}: {e}")
    
    return boxes


def draw_yolo_box(
    img: np.ndarray,
    cls_id: int,
    x: float,
    y: float,
    w: float,
    h: float,
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
    font_scale: float = 0.6,
    label_prefix: str = "",
    show_conf: bool = False,
    conf: float = 0.0,
) -> None:
    """
    Vẽ bounding box YOLO format (normalized coordinates) lên ảnh.
    
    Args:
        img: Ảnh numpy array (BGR format)
        cls_id: Class ID
        x, y, w, h: Normalized coordinates (center_x, center_y, width, height)
        color: Màu BGR
        thickness: Độ dày đường viền
        font_scale: Kích thước font
        label_prefix: Tiền tố cho label (vd: "GT" hoặc "Pred")
        show_conf: Có hiển thị confidence không
        conf: Confidence score
    """
    H, W = img.shape[:2]
    cx, cy = x * W, y * H
    bw, bh = w * W, h * H
    x1 = int(round(cx - bw / 2))
    y1 = int(round(cy - bh / 2))
    x2 = int(round(cx + bw / 2))
    y2 = int(round(cy + bh / 2))
    
    # Clamp to image boundaries
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W - 1, x2))
    y2 = max(0, min(H - 1, y2))
    
    if x2 <= x1 or y2 <= y1:
        return
    
    # Vẽ rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    # Tạo label text
    class_name = CLASSES.get(cls_id, f"Class{cls_id}")
    if label_prefix:
        label = f"{label_prefix}: {class_name}"
    else:
        label = class_name
    
    if show_conf and conf > 0:
        label += f" {conf:.2f}"
    
    # Vẽ label background và text
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
    cv2.putText(
        img, label, (x1 + 2, y1 - 2),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
        (255, 255, 255), 1, cv2.LINE_AA
    )


def visualize_predictions_with_gt(
    img_path: str,
    label_path: str,
    detector,
    conf_threshold: float,
    iou_threshold: float,
    imgsz: int,
    device: str,
    out_path: str,
) -> bool:
    """
    Vẽ cả GT boxes và predicted boxes lên cùng một ảnh.
    
    Args:
        img_path: Đường dẫn ảnh
        label_path: Đường dẫn file label GT
        detector: Model detector
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold
        imgsz: Image size
        device: Device
        out_path: Đường dẫn lưu ảnh kết quả
    
    Returns:
        True nếu thành công, False nếu có lỗi
    """
    try:
        # Đọc ảnh
        img = cv2.imread(img_path)
        if img is None:
            print(f"  ⚠️  Không thể đọc ảnh: {img_path}")
            return False
        
        # Đọc GT boxes
        gt_boxes = read_yolo_label(label_path)
        
        # Vẽ GT boxes (màu xanh lá)
        gt_color = (0, 255, 0)  # BGR: xanh lá
        for cls_id, x, y, w, h in gt_boxes:
            draw_yolo_box(img, cls_id, x, y, w, h, color=gt_color, thickness=2, label_prefix="GT")
        
        # Predict boxes
        pred_results = detector.predict(
            source=img_path,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        
        # Vẽ predicted boxes (màu đỏ)
        pred_color = (0, 0, 255)  # BGR: đỏ
        if pred_results and len(pred_results) > 0:
            result = pred_results[0]
            if hasattr(result, 'boxes'):
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    # Lấy thông tin boxes
                    xyxy = boxes.xyxy.cpu().numpy()  # Absolute coordinates
                    conf = boxes.conf.cpu().numpy()  # Confidence
                    cls = boxes.cls.cpu().numpy().astype(int)  # Class IDs
                    
                    H, W = img.shape[:2]
                    
                    # Convert từ absolute coordinates sang normalized YOLO format
                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = xyxy[i]
                        # Normalize
                        x_center = ((x1 + x2) / 2) / W
                        y_center = ((y1 + y2) / 2) / H
                        width = (x2 - x1) / W
                        height = (y2 - y1) / H
                        
                        cls_id = int(cls[i])
                        conf_score = float(conf[i])
                        
                        # Vẽ predicted box
                        draw_yolo_box(
                            img, cls_id, x_center, y_center, width, height,
                            color=pred_color, thickness=2,
                            label_prefix="Pred", show_conf=True, conf=conf_score
                        )
        
        # Lưu ảnh
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, img)
        return True
        
    except Exception as e:
        print(f"  ⚠️  Lỗi khi visualize {img_path}: {e}")
        return False


def format_metrics(metrics_dict: Dict) -> str:
    """Format metrics dictionary thành string dễ đọc."""
    lines = []
    lines.append("\n" + "="*60)
    lines.append("KẾT QUẢ ĐÁNH GIÁ")
    lines.append("="*60)
    
    # Overall metrics
    if "metrics/mAP50" in metrics_dict:
        lines.append(f"\n📊 Overall Metrics:")
        lines.append(f"  mAP50:     {metrics_dict.get('metrics/mAP50', 0):.4f}")
        lines.append(f"  mAP50-95:  {metrics_dict.get('metrics/mAP50-95', 0):.4f}")
        lines.append(f"  Precision: {metrics_dict.get('metrics/precision(B)', 0):.4f}")
        lines.append(f"  Recall:    {metrics_dict.get('metrics/recall(B)', 0):.4f}")
        lines.append(f"  F1-score:  {metrics_dict.get('metrics/f1(B)', 0):.4f}")
    
    # Per-class metrics
    names = [CLASSES[i] for i in sorted(CLASSES.keys())]
    if any(f"metrics/{name}/mAP50" in metrics_dict for name in names):
        lines.append(f"\n📈 Per-Class Metrics:")
        for i, name in enumerate(names):
            map50_key = f"metrics/{name}/mAP50"
            map50_95_key = f"metrics/{name}/mAP50-95"
            precision_key = f"metrics/{name}/precision"
            recall_key = f"metrics/{name}/recall"
            
            if map50_key in metrics_dict:
                lines.append(f"\n  {name}:")
                lines.append(f"    mAP50:     {metrics_dict.get(map50_key, 0):.4f}")
                lines.append(f"    mAP50-95:  {metrics_dict.get(map50_95_key, 0):.4f}")
                lines.append(f"    Precision: {metrics_dict.get(precision_key, 0):.4f}")
                lines.append(f"    Recall:    {metrics_dict.get(recall_key, 0):.4f}")
    
    lines.append("\n" + "="*60)
    return "\n".join(lines)


def main():
    args = parse_args()
    
    # Kiểm tra file weights tồn tại
    if not os.path.exists(args.weights):
        raise SystemExit(f"Weights file not found: {args.weights}")
    
    # Tự chọn device nếu không chỉ định
    device_arg = args.device
    if not device_arg:
        try:
            import torch
            device_arg = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device_arg = "cpu"
    
    print(f"\n{'='*60}")
    print(f"ĐÁNH GIÁ MODEL DETECTION")
    print(f"{'='*60}")
    print(f"Weights: {args.weights}")
    print(f"Split: {args.split}")
    print(f"Image directory: {args.img_dir}")
    print(f"Device: {device_arg}")
    print(f"Confidence threshold: {args.conf}")
    print(f"IoU threshold: {args.iou}")
    print(f"Image size: {args.imgsz}")
    print(f"{'='*60}\n")
    
    # ================================================================
    # BƯỚC 1 - CHUẨN BỊ DỮ LIỆU
    # ================================================================
    print("📁 Đang chuẩn bị dataset...")
    yaml_path = make_dataset_files_for_eval(
        img_dir=args.img_dir,
        splits_dir=args.splits_dir,
        split=args.split,
        out_dir=args.out_dir
    )
    print(f"Dataset YAML -> {yaml_path}\n")
    
    # ================================================================
    # BƯỚC 2 - LOAD MODEL
    # ================================================================
    print("🤖 Đang load model...")
    detector = build_detection_model(weights=args.weights, backend=args.backend)
    print(f"Model loaded: {args.weights}\n")
    
    # ================================================================
    # BƯỚC 3 - EVALUATION
    # ================================================================
    print("🔍 Đang đánh giá model...")
    val_kwargs = dict(
        data=yaml_path.replace("\\", "/"),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=device_arg,
        save_json=args.save_json,
        plots=True,  # Tạo confusion matrix và các plots
    )
    
    # Chạy validation
    results = detector.val(**val_kwargs)
    
    # ================================================================
    # BƯỚC 4 - XỬ LÝ KẾT QUẢ
    # ================================================================
    print("\n✅ Đánh giá hoàn tất!")
    
    # Lấy metrics từ kết quả
    metrics_dict = {}
    try:
        if hasattr(results, 'results_dict'):
            metrics_dict = results.results_dict
        elif hasattr(results, 'box'):
            # Ultralytics trả về results.box chứa metrics
            box = results.box
            if hasattr(box, 'map50'):
                metrics_dict['metrics/mAP50'] = float(box.map50)
            if hasattr(box, 'map'):
                metrics_dict['metrics/mAP50-95'] = float(box.map)
            if hasattr(box, 'mp'):
                metrics_dict['metrics/precision(B)'] = float(box.mp)
            if hasattr(box, 'mr'):
                metrics_dict['metrics/recall(B)'] = float(box.mr)
            if hasattr(box, 'f1'):
                metrics_dict['metrics/f1(B)'] = float(box.f1)
            
            # Per-class metrics
            names = [CLASSES[i] for i in sorted(CLASSES.keys())]
            if hasattr(box, 'maps'):
                maps = box.maps  # mAP50-95 per class
                if isinstance(maps, (list, np.ndarray)) and len(maps) >= len(names):
                    for i, name in enumerate(names):
                        if i < len(maps):
                            metrics_dict[f'metrics/{name}/mAP50-95'] = float(maps[i])
            if hasattr(box, 'maps50'):
                maps50 = box.maps50  # mAP50 per class
                if isinstance(maps50, (list, np.ndarray)) and len(maps50) >= len(names):
                    for i, name in enumerate(names):
                        if i < len(maps50):
                            metrics_dict[f'metrics/{name}/mAP50'] = float(maps50[i])
        
        # Thử lấy từ dict nếu có
        if not metrics_dict and isinstance(results, dict):
            metrics_dict = results
        
        # In thông tin debug nếu không tìm thấy metrics
        if not metrics_dict:
            print("⚠️  Không thể trích xuất metrics tự động. Kết quả có thể được lưu trong thư mục runs.")
            print(f"   Kiểm tra thư mục: {os.path.join(args.out_dir, 'runs')}")
    except Exception as e:
        print(f"⚠️  Lỗi khi trích xuất metrics: {e}")
        print("   Kết quả validation đã được chạy, kiểm tra thư mục runs để xem chi tiết.")
    
    # In kết quả
    print(format_metrics(metrics_dict))
    
    # Lưu JSON nếu được yêu cầu
    if args.save_json:
        json_path = os.path.join(args.out_dir, f"metrics_{args.split}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"\n💾 Metrics đã lưu vào: {json_path}")
    
    # ================================================================
    # BƯỚC 5 - VISUALIZATION (tùy chọn)
    # ================================================================
    if args.save_vis:
        print("\n🎨 Đang tạo visualization (GT + Predicted boxes)...")
        vis_dir = args.vis_dir or os.path.join(args.out_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Tìm thư mục labels tương ứng với img_dir
        if args.label_dir:
            label_dir = args.label_dir
        else:
            # Tự động tìm: giả định labels ở cùng parent directory, thay '/images' -> '/labels'
            label_dir = args.img_dir.replace("/images", "/labels").replace("\\images", "\\labels")
            if not os.path.exists(label_dir):
                # Thử tìm labels ở processed/knee/labels
                if "knee" in args.img_dir:
                    label_dir = os.path.join(os.path.dirname(os.path.dirname(args.img_dir)), "knee", "labels")
                else:
                    label_dir = os.path.join(os.path.dirname(args.img_dir), "labels")
        
        if not os.path.exists(label_dir):
            print(f"  ⚠️  Không tìm thấy thư mục labels: {label_dir}")
            print(f"  Chỉ vẽ predicted boxes (không có GT)...")
            label_dir = None
        
        # Đọc danh sách ảnh từ split
        split_txt = os.path.join(args.out_dir, f"{args.split}.txt")
        with open(split_txt, "r", encoding="utf-8") as f:
            image_paths = [line.strip() for line in f if line.strip()]
        
        # Giới hạn số lượng ảnh
        if args.vis_n > 0:
            image_paths = image_paths[:args.vis_n]
        
        print(f"  Visualizing {len(image_paths)} images...")
        if label_dir:
            print(f"  Label directory: {label_dir}")
        print(f"  GT boxes: màu xanh lá")
        print(f"  Predicted boxes: màu đỏ")
        
        success_count = 0
        for img_path in image_paths:
            # Tìm file label tương ứng
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = None
            if label_dir:
                label_path = os.path.join(label_dir, f"{img_name}.txt")
            
            # Đường dẫn output
            out_filename = f"{img_name}_vis.jpg"
            out_path = os.path.join(vis_dir, out_filename)
            
            # Vẽ cả GT và predicted boxes
            if visualize_predictions_with_gt(
                img_path=img_path,
                label_path=label_path or "",
                detector=detector,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
                imgsz=args.imgsz,
                device=device_arg,
                out_path=out_path,
            ):
                success_count += 1
        
        print(f"✅ Đã tạo visualization cho {success_count}/{len(image_paths)} ảnh")
        print(f"   Lưu tại: {vis_dir}")
    
    print(f"\n{'='*60}\n")
    print("✨ Hoàn tất đánh giá!")


if __name__ == "__main__":
    main()

