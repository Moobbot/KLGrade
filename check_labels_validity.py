import os

labels_dir = "./dataset/labels"  # Đường dẫn tới thư mục labels

MIN_WH = 0.01  # Giá trị tối thiểu cho w/h

def check_label_file(file_path):
    with open(file_path, "r") as f:
        for i, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) == 5:
                class_id, x, y, w, h = map(float, parts)
                # Kiểm tra giá trị ngoài [0, 1] hoặc w/h <= 0
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    print(f"LỖI: {os.path.basename(file_path)}, dòng {i}: {line.strip()}")

def check_all_labels(labels_dir):
    for fname in os.listdir(labels_dir):
        if fname.endswith(".txt"):
            check_label_file(os.path.join(labels_dir, fname))

if __name__ == "__main__":
    check_all_labels(labels_dir)
