import cv2
import os
import glob

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

    for idx, line in enumerate(label2_lines):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
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
        if not os.path.exists(label1_path):
            continue
        new_labels = []
        with open(label1_path) as f1:
            for l1 in f1:
                parts1 = l1.strip().split()
                if len(parts1) < 5:
                    continue
                cls, x, y, bw, bh = map(float, parts1)
                bx, by = x * w, y * h
                bw_abs, bh_abs = bw * w, bh * h
                # Box gốc trong ảnh lớn
                bx1, by1 = bx - bw_abs/2, by - bh_abs/2
                bx2, by2 = bx + bw_abs/2, by + bh_abs/2
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
                new_labels.append(f"{int(cls)} {ncx:.6f} {ncy:.6f} {nbw:.6f} {nbh:.6f}")
        cropped_label_path = os.path.join(cropped_label_dir, f"{base}_{idx}.txt")
        with open(cropped_label_path, 'w') as f2:
            for l in new_labels:
                f2.write(l + '\n')

print('Done!')