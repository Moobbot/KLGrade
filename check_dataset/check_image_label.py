import os
import shutil

# Lọc các file label không có box và di chuyển sang folder khác
label_dir = 'dataset/labels'
empty_label_dir = 'dataset/labels-empty'
os.makedirs(empty_label_dir, exist_ok=True)

for label_file in os.listdir(label_dir):
    if not label_file.endswith('.txt'):
        continue
    path = os.path.join(label_dir, label_file)
    with open(path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    if len(lines) == 0:
        print(f"Di chuyển file: {path} -> {empty_label_dir}")
        shutil.move(path, os.path.join(empty_label_dir, label_file))

print('Đã di chuyển các file label không có box!')

# Di chuyển các ảnh không có label tương ứng
img_dir = 'dataset/images'
no_label_img_dir = 'dataset/images-no-label'
os.makedirs(no_label_img_dir, exist_ok=True)

for img_file in os.listdir(img_dir):
    if not (img_file.endswith('.jpg') or img_file.endswith('.png')):
        continue
    base = os.path.splitext(img_file)[0]
    # Có thể có hậu tố _0, _1... nên kiểm tra đúng tên label
    label_candidates = [f for f in os.listdir(label_dir) if f.startswith(base) and f.endswith('.txt')]
    if len(label_candidates) == 0:
        src = os.path.join(img_dir, img_file)
        dst = os.path.join(no_label_img_dir, img_file)
        print(f"Di chuyển ảnh không có label: {src} -> {dst}")
        shutil.move(src, dst)

print('Đã di chuyển các ảnh không có label!')
