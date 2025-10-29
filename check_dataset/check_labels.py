import os
import math
import numpy as np
from utils import yolo_to_xyxy

labels_dir = "dataset/labels"
MIN_WH = 0.001  # Giá trị tối thiểu cho w/h


def is_valid_number(x):
    return not (math.isnan(x) or math.isinf(x))


def check_invalid_boxes(labels_dir):
    for fname in os.listdir(labels_dir):
        if not fname.endswith(".txt"):
            print(fname)
            continue
        fpath = os.path.join(labels_dir, fname)
        with open(fpath, "r") as f:
            for idx, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) < 5:
                    print(f"[{fname}][Line {idx}] Format error: {line.strip()}")
                    continue
                try:
                    width = float(parts[3])
                    height = float(parts[4])
                except ValueError:
                    print(f"[{fname}][Line {idx}] Value error: {line.strip()}")
                    continue
                if width <= 0 or height <= 0:
                    print(
                        f"[{fname}][Line {idx}] Invalid box: width={width}, height={height} | {line.strip()}"
                    )


def check_label_file(file_path):
    # print("check_label_file: ", file_path)
    with open(file_path, "r") as f:
        lines = f.readlines()
        if len(lines) == 0:
            print(f"CẢNH BÁO: {os.path.basename(file_path)} là file rỗng!")
        for i, line in enumerate(lines, 1):
            parts = line.strip().split()
            if len(parts) != 5:
                print(
                    f"LỖI: {os.path.basename(file_path)}, dòng {i}: Không đủ 5 giá trị ({line.strip()})")
                continue
            try:
                class_id, x, y, w, h = map(float, parts)
                if not (class_id == int(class_id) and class_id >= 0):
                    print(
                        f"LỖI: {os.path.basename(file_path)}, dòng {i}: class_id không hợp lệ ({class_id})")
                for v, name in zip([x, y, w, h], ['x', 'y', 'w', 'h']):
                    if not is_valid_number(v):
                        print(
                            f"LỖI: {os.path.basename(file_path)}, dòng {i}: {name} là NaN/Inf ({v})")
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    print(
                        f"LỖI: {os.path.basename(file_path)}, dòng {i}: Giá trị ngoài [0,1] hoặc w/h <= 0 ({line.strip()})")
                if w < MIN_WH or h < MIN_WH:
                    print(
                        f"CẢNH BÁO: {os.path.basename(file_path)}, dòng {i}: w/h quá nhỏ ({w}, {h})")
                # --- Check chuyển sang xyxy ---
                box_yolo = np.array([[x, y, w, h]], dtype=np.float32)
                box_xyxy = yolo_to_xyxy(box_yolo)
                xmin, ymin, xmax, ymax = box_xyxy[0]
                # Check xyxy hợp lệ
                if not (0.0 <= xmin <= 1.0 and 0.0 <= ymin <= 1.0 and 0.0 <= xmax <= 1.0 and 0.0 <= ymax <= 1.0):
                    print(f"LỖI: {os.path.basename(file_path)}, dòng {i}: xyxy ra ngoài [0,1] ({xmin:.4f}, {ymin:.4f}, {xmax:.4f}, {ymax:.4f}) | {line.strip()}")
                if xmax <= xmin or ymax <= ymin:
                    print(f"LỖI: {os.path.basename(file_path)}, dòng {i}: xyxy không hợp lệ (xmax <= xmin hoặc ymax <= ymin) | {line.strip()}")
            except Exception as e:
                print(
                    f"LỖI: {os.path.basename(file_path)}, dòng {i}: Không thể parse ({line.strip()}) - {e}")


def check_all_labels(labels_dir):
    for fname in os.listdir(labels_dir):
        if fname.endswith(".txt"):
            check_label_file(os.path.join(labels_dir, fname))


if __name__ == "__main__":
    check_all_labels(labels_dir)
    check_invalid_boxes(labels_dir)
