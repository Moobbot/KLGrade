import cv2
import os
import glob
import numpy as np
from utils import yolo_to_xyxy

# Đường dẫn file
img_path = 'dataset/dataset_v0/images/1.2.392.200036.9107.307.24972.20230201.143415.1043275.jpg'
label1_path = 'dataset/dataset_v0/labels/1.2.392.200036.9107.307.24972.20230201.143415.1043275.txt'
label2_path = 'dataset/dataset_v0/labels-knee/1.2.392.200036.9107.307.24972.20230201.143415.1043275.txt'

# Đường dẫn thư mục
img_dir = 'dataset/dataset_v0/images'
label1_dir = 'dataset/dataset_v0/labels'
label2_dir = 'dataset/dataset_v0/labels-knee'
cropped_dir = 'dataset/images'
cropped_label_dir = 'dataset/labels'

os.makedirs(cropped_dir, exist_ok=True)
os.makedirs(cropped_label_dir, exist_ok=True)

img_files = glob.glob(os.path.join(img_dir, '*.jpg'))

# Hàm kiểm tra định dạng YOLO


def is_valid_yolo_box(parts):
    if len(parts) != 5:
        return False
    try:
        cls, x, y, w, h = map(float, parts)
        # Kiểm tra các giá trị đều trong [0, 1]
        return all(0.0 <= v <= 1.0 for v in [x, y, w, h])
    except Exception:
        return False


for img_path in img_files:
    base = os.path.splitext(os.path.basename(img_path))[0]
    label1_path = os.path.join(label1_dir, base + '.txt')
    label2_path = os.path.join(label2_dir, base + '.txt')

    # Nếu thiếu label2 (labels-knee) thì bỏ qua
    if not os.path.exists(label2_path):
        continue

    # Đọc ảnh
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w, _ = img.shape

    # Đọc tất cả box từ labels-knee
    with open(label2_path) as f:
        label2_lines = f.readlines()

    invalid_image = False
    for idx, line in enumerate(label2_lines):
        parts = line.strip().split()
        if not is_valid_yolo_box(parts):
            print(
                f"[INVALID] {img_path}: Invalid YOLO box in {label2_path} line {idx+1}: {line.strip()}")
            invalid_image = True
            break
    if invalid_image:
        continue

    # Kiểm tra label1
    if not os.path.exists(label1_path):
        continue
    with open(label1_path) as f1:
        for lidx, l1 in enumerate(f1):
            parts1 = l1.strip().split()
            if not is_valid_yolo_box(parts1):
                print(
                    f"[INVALID] {img_path}: Invalid YOLO box in {label1_path} line {lidx+1}: {l1.strip()}")
                invalid_image = True
                break
    if invalid_image:
        continue

    # Nếu hợp lệ, mới thực hiện crop và ghi file
    for idx, line in enumerate(label2_lines):
        parts = line.strip().split()
        _, x, y, bw, bh = map(float, parts)
        cx, cy = x * w, y * h
        bw, bh = bw * w, bh * h
        # Lấy góc trái trên làm mốc, mở rộng thành hình vuông
        x1, y1 = int(cx - bw/2), int(cy - bh/2)
        side = int(max(bw, bh))
        x1 = max(0, x1)
        y1 = max(0, y1)
        if x1 + side > w:
            side = w - x1
        if y1 + side > h:
            side = h - y1
        x2, y2 = x1 + side, y1 + side

        # Cắt ảnh
        cropped_img = img[y1:y2, x1:x2]
        cropped_img_path = os.path.join(cropped_dir, f"{base}_{idx}.jpg")
        cv2.imwrite(cropped_img_path, cropped_img)
        ch, cw, _ = cropped_img.shape

        # Cập nhật lại label1 (labels) cho vùng ảnh đã cắt
        new_labels = []
        with open(label1_path) as f1:
            for lidx, l1 in enumerate(f1):
                parts1 = l1.strip().split()
                cls, x, y, bw, bh = map(float, parts1)
                bx, by = x * w, y * h
                bw_abs, bh_abs = bw * w, bh * h
                # Box gốc trong ảnh lớn (YOLO -> xyxy)
                box_yolo = np.array([[x, y, bw, bh]])
                box_xyxy = yolo_to_xyxy(box_yolo) * np.array([[w, h, w, h]])
                bx1, by1, bx2, by2 = box_xyxy[0]
                # Clamp box vào vùng crop
                nbx1 = max(bx1, x1)
                nby1 = max(by1, y1)
                nbx2 = min(bx2, x2)
                nby2 = min(by2, y2)
                # Nếu không giao thì bỏ qua
                if nbx1 >= nbx2 or nby1 >= nby2:
                    continue
                # Box mới trong crop (tọa độ so với crop)
                nbx1_crop = nbx1 - x1
                nby1_crop = nby1 - y1
                nbx2_crop = nbx2 - x1
                nby2_crop = nby2 - y1
                # Chuyển về format YOLO (center, width, height, normalized)
                ncx = (nbx1_crop + nbx2_crop) / 2 / cw
                ncy = (nby1_crop + nby2_crop) / 2 / ch
                nbw = (nbx2_crop - nbx1_crop) / cw
                nbh = (nby2_crop - nby1_crop) / ch
                # Clamp các giá trị về [0, 1]
                ncx = max(0.0, min(1.0, ncx))
                ncy = max(0.0, min(1.0, ncy))
                nbw = max(0.0, min(1.0, nbw))
                nbh = max(0.0, min(1.0, nbh))
                # Kiểm tra box mới hợp lệ
                if not (0.0 <= ncx <= 1.0 and 0.0 <= ncy <= 1.0 and 0.0 < nbw <= 1.0 and 0.0 < nbh <= 1.0):
                    print(
                        f"[INVALID] {img_path}: Cropped box out of range in {cropped_label_dir}/{base}_{idx}.txt from {label1_path} line {lidx+1}")
                    invalid_image = True
                    break
                if nbw <= 0 or nbh <= 0:
                    print(
                        f"[INVALID] {img_path}: Cropped box has non-positive size in {cropped_label_dir}/{base}_{idx}.txt from {label1_path} line {lidx+1}")
                    invalid_image = True
                    break
                new_labels.append(
                    f"{int(cls)} {ncx:.6f} {ncy:.6f} {nbw:.6f} {nbh:.6f}")
        if invalid_image:
            break
        cropped_label_path = os.path.join(
            cropped_label_dir, f"{base}_{idx}.txt")

        with open(cropped_label_path, 'w') as f2:
            for l in new_labels:
                f2.write(l + '\n')
        # Kiểm tra new_labels trước khi ghi file
        has_error = False
        for lidx, line in enumerate(new_labels):
            parts = line.split()
            if not is_valid_yolo_box(parts):
                print(
                    f"[CROP_LABEL_ERROR] {cropped_label_path}: invalid YOLO box at line {lidx+1}: {line} (skip write)")
                has_error = True
                break
        # Kiểm tra file label crop sau khi ghi
        # (Không xóa file, chỉ cảnh báo)

print('Done!')