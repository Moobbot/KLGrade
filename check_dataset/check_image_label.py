import os
import shutil
import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Lọc nhãn rỗng và ảnh không có nhãn (tham số hoá thư mục)")
    p.add_argument("--label_dir", default="dataset/labels", help="Thư mục nhãn YOLO (.txt)")
    p.add_argument("--img_dir", default="dataset/images", help="Thư mục ảnh (.jpg/.png)")
    p.add_argument("--empty_label_dir", default=None, help="Nơi di chuyển nhãn rỗng (mặc định <label_dir>-empty)")
    p.add_argument("--no_label_img_dir", default=None, help="Nơi di chuyển ảnh không có nhãn (mặc định <img_dir>-no-label)")
    return p.parse_args()


def main():
    args = parse_args()
    label_dir = args.label_dir
    img_dir = args.img_dir
    empty_label_dir = args.empty_label_dir or (label_dir.rstrip("/\\") + "-empty")
    no_label_img_dir = args.no_label_img_dir or (img_dir.rstrip("/\\") + "-no-label")

    os.makedirs(empty_label_dir, exist_ok=True)
    os.makedirs(no_label_img_dir, exist_ok=True)

    # 1) Move empty label files
    moved_labels = 0
    for label_file in os.listdir(label_dir):
        if not label_file.lower().endswith(".txt"):
            continue
        path = os.path.join(label_dir, label_file)
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
        except Exception:
            lines = []
        if len(lines) == 0:
            print(f"Di chuyển file: {path} -> {empty_label_dir}")
            shutil.move(path, os.path.join(empty_label_dir, label_file))
            moved_labels += 1
    print(f"Đã di chuyển {moved_labels} file label không có box!")

    # 2) Move images without corresponding label files
    moved_images = 0
    label_files = set(os.listdir(label_dir))
    for img_file in os.listdir(img_dir):
        if not (img_file.lower().endswith(".jpg") or img_file.lower().endswith(".png") or img_file.lower().endswith(".jpeg")):
            continue
        base = os.path.splitext(img_file)[0]
        # Có thể có hậu tố trong tên; so khớp bắt đầu bằng base
        has_label = any(
            lf.lower().endswith(".txt") and lf.startswith(base)
            for lf in label_files
        )
        if not has_label:
            src = os.path.join(img_dir, img_file)
            dst = os.path.join(no_label_img_dir, img_file)
            print(f"Di chuyển ảnh không có label: {src} -> {dst}")
            shutil.move(src, dst)
            moved_images += 1
    print(f"Đã di chuyển {moved_images} ảnh không có label!")


if __name__ == "__main__":
    main()
