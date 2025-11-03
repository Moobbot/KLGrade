import cv2
import matplotlib.pyplot as plt
import os
import sys
import random

# Get image name from command line argument or use default
if len(sys.argv) > 1:
    img_name = sys.argv[1]
else:
    # Use default image name
    img_name = "1.2.392.200036.9107.307.24972.20220914.144601.1034072"

# Đường dẫn file
img_path = f'dataset_v0/images/{img_name}.jpg'
label1_path = f'dataset_v0/labels/{img_name}.txt'
label2_path = f'dataset_v0/labels-knee/{img_name}.txt'

# Đọc ảnh
img = cv2.imread(img_path)
if img is None:
    print(f"Error: Could not load image from {img_path}")
    print("Available images in the directory:")
    image_files = [f for f in os.listdir('dataset_v0/images') if f.endswith('.jpg')]
    print(f"Found {len(image_files)} images. First 5: {image_files[:5]}")
    
    # Try to use a random image if the specified one doesn't exist
    if image_files:
        random_img = random.choice(image_files)
        img_name = random_img.replace('.jpg', '')
        img_path = f'dataset_v0/images/{img_name}.jpg'
        label1_path = f'dataset_v0/labels/{img_name}.txt'
        label2_path = f'dataset_v0/labels-knee/{img_name}.txt'
        print(f"Using random image instead: {img_name}")
        img = cv2.imread(img_path)
        if img is None:
            print("Failed to load random image as well!")
            exit(1)
    else:
        print("No images found in the directory!")
        exit(1)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, _ = img.shape

# Vẽ label1 (màu đỏ)
if os.path.exists(label1_path):
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
else:
    print(f"Warning: Label file {label1_path} not found")

# Vẽ label2 (màu xanh lá)
if os.path.exists(label2_path):
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
else:
    print(f"Warning: Label file {label2_path} not found")

plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.axis('off')
plt.title(f'Image: {img_name}', fontsize=12, pad=20)
plt.show()
