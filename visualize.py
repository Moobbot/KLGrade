import cv2
import os
import glob
import numpy as np
from utils import yolo_to_xyxy

# Đường dẫn file
img_path = 'dataset/dataset_v0/images/1.2.392.200036.9107.307.24972.20230201.143415.1043275.jpg'
label_OA_path = 'dataset/dataset_v0/labels/1.2.392.200036.9107.307.24972.20230201.143415.1043275.txt'
labels_knee_path = 'dataset/dataset_v0/labels-knee/1.2.392.200036.9107.307.24972.20230201.143415.1043275.txt'

# Đường dẫn thư mục
img_dir = 'dataset/dataset_v0/images'
label_OA_dir = 'dataset/dataset_v0/labels'
labels_knee_dir = 'dataset/dataset_v0/labels-knee'
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
    label_OA_path = os.path.join(label_OA_dir, base + '.txt')
    labels_knee_path = os.path.join(labels_knee_dir, base + '.txt')

    # Nếu thiếu labels_knee (labels-knee) thì bỏ qua
    if not os.path.exists(labels_knee_path):
        continue

    # Đọc ảnh
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w, _ = img.shape

    # Đọc n box từ labels-knee (mỗi box là 1 vùng)
    with open(labels_knee_path) as f:
        labels_knee_lines = f.readlines()
    boxes_knee = []
    for idx, line in enumerate(labels_knee_lines):
        parts = line.strip().split()
        if not is_valid_yolo_box(parts):
            print(
                f"[INVALID] {img_path}: Invalid YOLO box in {labels_knee_path} line {idx+1}: {line.strip()}")
            boxes_knee = []
            break
        _, x, y, bw, bh = map(float, parts)
        boxes_knee.append([x, y, bw, bh])
    if len(boxes_knee) == 0:
        continue
    boxes_knee = np.array(boxes_knee)
    boxes_knee_xyxy = yolo_to_xyxy(boxes_knee) * np.array([w, h, w, h])

    # Kiểm tra label_OA
    if not os.path.exists(label_OA_path):
        continue
    with open(label_OA_path) as f1:
        label_OA_lines = f1.readlines()
    boxes_OA = []
    classes_OA = []
    for lidx, l1 in enumerate(label_OA_lines):
        parts1 = l1.strip().split()
        if not is_valid_yolo_box(parts1):
            print(
                f"[INVALID] {img_path}: Invalid YOLO box in {label_OA_path} line {lidx+1}: {l1.strip()}")
            boxes_OA = []
            break
        cls, x, y, bw, bh = map(float, parts1)
        boxes_OA.append([x, y, bw, bh])
        classes_OA.append(int(cls))
    if len(boxes_OA) == 0:
        continue
    boxes_OA = np.array(boxes_OA)
    boxes_OA_xyxy = yolo_to_xyxy(boxes_OA) * np.array([w, h, w, h])
    classes_OA = np.array(classes_OA)

    # Gom các box OA theo từng vùng (giao)
    groups = {i: [] for i in range(len(boxes_knee))}
    for i, box_OA in enumerate(boxes_OA_xyxy):
        for j, box_knee in enumerate(boxes_knee_xyxy):
            # Tính giao
            xx1 = max(box_OA[0], box_knee[0])
            yy1 = max(box_OA[1], box_knee[1])
            xx2 = min(box_OA[2], box_knee[2])
            yy2 = min(box_OA[3], box_knee[3])
            iw = max(0, xx2 - xx1)
            ih = max(0, yy2 - yy1)
            if iw > 0 and ih > 0:
                groups[j].append(i)
                break  # 1 box OA chỉ thuộc 1 vùng

    # Với mỗi vùng, crop và ghi file
    for idx in range(len(boxes_knee)):
        box_knee = boxes_knee[idx]
        box_knee_xyxy = boxes_knee_xyxy[idx]
        # Crop vùng box_knee
        x1, y1, x2, y2 = map(int, box_knee_xyxy)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        # ---- Căn chỉnh thành hình vuông, ưu tiên giữ tâm vùng crop cũ ----
        width = x2 - x1
        height = y2 - y1
        side = max(width, height)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        half_side = side // 2
        x1_sq = max(0, min(w - side, cx - half_side))
        y1_sq = max(0, min(h - side, cy - half_side))
        x2_sq = x1_sq + side
        y2_sq = y1_sq + side
        # Đảm bảo không vượt biên ảnh
        x1_sq = int(max(0, min(w - side, x1_sq)))
        y1_sq = int(max(0, min(h - side, y1_sq)))
        x2_sq = int(min(w, x1_sq + side))
        y2_sq = int(min(h, y1_sq + side))
        # --------------------------------------------------------------
        cropped_img = img[y1_sq:y2_sq, x1_sq:x2_sq]
        cropped_img_path = os.path.join(cropped_dir, f"{base}_{idx}.jpg")
        cv2.imwrite(cropped_img_path, cropped_img)
        ch, cw, _ = cropped_img.shape
        # Lấy các box OA thuộc vùng này
        indices = groups[idx]
        new_labels = []
        for i in indices:
            cls = classes_OA[i]
            # Box OA gốc (xyxy)
            bx1, by1, bx2, by2 = boxes_OA_xyxy[i]
            # Clamp vào crop
            nbx1 = max(bx1, x1_sq)
            nby1 = max(by1, y1_sq)
            nbx2 = min(bx2, x2_sq)
            nby2 = min(by2, y2_sq)
            if nbx1 >= nbx2 or nby1 >= nby2:
                continue
            # Box mới trong crop (tọa độ so với crop)
            nbx1_crop = max(0, min(cw, nbx1 - x1_sq))
            nby1_crop = max(0, min(ch, nby1 - y1_sq))
            nbx2_crop = max(0, min(cw, nbx2 - x1_sq))
            nby2_crop = max(0, min(ch, nby2 - y1_sq))
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
            if not (0.0 <= ncx <= 1.0 and 0.0 <= ncy <= 1.0 and 0.0 < nbw <= 1.0 and 0.0 < nbh <= 1.0):
                print(
                    f"[INVALID] {img_path}: Cropped box out of range in {cropped_label_dir}/{base}_{idx}.txt from OA idx {i}")
                continue
            if nbw <= 0 or nbh <= 0:
                print(
                    f"[INVALID] {img_path}: Cropped box has non-positive size in {cropped_label_dir}/{base}_{idx}.txt from OA idx {i}")
                continue
            new_labels.append(
                f"{int(cls)} {ncx:.6f} {ncy:.6f} {nbw:.6f} {nbh:.6f}")
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
