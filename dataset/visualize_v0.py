import cv2
import matplotlib.pyplot as plt

img_name = "1.2.392.200036.9107.307.24972.20240613.83255.1076937"

# Đường dẫn file
img_path = f'dataset/dataset_v0/images/{img_name}.jpg'
label1_path = f'dataset/dataset_v0/labels/{img_name}.txt'
label2_path = f'dataset/dataset_v0/labels-knee/{img_name}.txt'

# Đọc ảnh
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, _ = img.shape

# Vẽ label1 (màu đỏ)
with open(label1_path) as f:
    for line in f:
        cls, x, y, bw, bh = map(float, line.strip().split())
        cx, cy = x * w, y * h
        bw, bh = bw * w, bh * h
        x1, y1 = int(cx - bw/2), int(cy - bh/2)
        x2, y2 = int(cx + bw/2), int(cy + bh/2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, f"L1-{int(cls)}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

# Vẽ label2 (màu xanh lá)
with open(label2_path) as f:
    for line in f:
        cls, x, y, bw, bh = map(float, line.strip().split())
        cx, cy = x * w, y * h
        bw, bh = bw * w, bh * h
        x1, y1 = int(cx - bw/2), int(cy - bh/2)
        x2, y2 = int(cx + bw/2), int(cy + bh/2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"L2-{int(cls)}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.axis('off')
plt.show()
