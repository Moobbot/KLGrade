import cv2
import os
import glob
import numpy as np

# Đường dẫn thư mục
img_dir = 'dataset/dataset_v0/images'
label1_dir = 'dataset/dataset_v0/labels'
label2_dir = 'dataset/dataset_v0/labels-knee'
label2_2box_dir = 'dataset/dataset_v0/labels-knee-2box'

# Tạo thư mục nếu chưa tồn tại
os.makedirs(label2_2box_dir, exist_ok=True)

# Lấy danh sách file label
label_files = glob.glob(os.path.join(label2_dir, '*.txt'))

for file_path in label_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if len(lines) == 2:
        # Copy file sang thư mục mới
        base_name = os.path.basename(file_path)
        new_path = os.path.join(label2_2box_dir, base_name)
        with open(new_path, 'w', encoding='utf-8') as f_out:
            f_out.writelines(lines)
