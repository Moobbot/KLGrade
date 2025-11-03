import os
import sys
import csv
from collections import Counter, defaultdict

# Ensure repository root is on path so we can import config.py
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import CLASSES


IMG_DIR = os.path.join("processed", "knee", "images")
LABEL_DIR = os.path.join("processed", "knee", "labels")
OUT_DIR = os.path.join("splits")


def list_images(img_dir: str):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in exts]
    files.sort()
    return files


def read_label_file(path: str):
    """Read a YOLO .txt label file and return a list of class_ids (ints).

    Each line: class cx cy w h
    """
    class_ids = []
    if not os.path.exists(path):
        return class_ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                cls_id = int(float(parts[0]))
            except Exception:
                # skip malformed line
                continue
            class_ids.append(cls_id)
    return class_ids


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    images = list_images(IMG_DIR)
    total_images = len(images)

    labels_per_image = {}
    classes_counter = Counter()
    images_with_labels = 0
    images_without_labels = []

    # Per-class distinct image counts (optional, useful)
    class_to_images = defaultdict(set)

    for img in images:
        stem, _ = os.path.splitext(img)
        label_path = os.path.join(LABEL_DIR, f"{stem}.txt")
        class_ids = read_label_file(label_path)
        labels_per_image[img] = len(class_ids)
        if class_ids:
            images_with_labels += 1
        else:
            images_without_labels.append(img)
        classes_counter.update(class_ids)
        for c in set(class_ids):
            class_to_images[c].add(img)

    # Also count orphan labels (label files that don't have a matching image)
    orphan_labels = []
    image_stems = {os.path.splitext(f)[0] for f in images}
    for fname in os.listdir(LABEL_DIR):
        if not fname.endswith(".txt"):
            continue
        if os.path.splitext(fname)[0] not in image_stems:
            orphan_labels.append(fname)

    # Write per-image stats to CSV
    csv_path = os.path.join(OUT_DIR, "labels_per_image.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "num_labels"])
        for img, n in labels_per_image.items():
            writer.writerow([img, n])

    # Pretty-print summary
    print("===== DATASET STATS =====")
    print(f"Images directory: {IMG_DIR}")
    print(f"Labels directory: {LABEL_DIR}")
    print(f"Total images: {total_images}")
    print(f"Images with labels: {images_with_labels}")
    print(f"Images without labels: {len(images_without_labels)}")
    if images_without_labels:
        print("  (saved list in check_dataset/labels_per_image.csv; you can filter num_labels==0)")

    total_labels = sum(labels_per_image.values())
    print(f"Total labels (all images): {total_labels}")

    # Distribution of labels per image
    dist = Counter(labels_per_image.values())
    print("\nLabels per image distribution (count -> num_images):")
    for k in sorted(dist):
        print(f"  {k}: {dist[k]}")

    # Per-class counts
    print("\nLabels per class:")
    for cid in sorted(CLASSES.keys()):
        name = CLASSES[cid]
        count = classes_counter.get(cid, 0)
        img_count = len(class_to_images.get(cid, set()))
        print(f"  {cid} ({name}): {count} labels in {img_count} images")

    # Orphan label files
    print("\nOrphan label files (no matching image):", len(orphan_labels))
    if orphan_labels[:10]:
        print("  e.g.", ", ".join(orphan_labels[:10]), ("..." if len(orphan_labels) > 10 else ""))

    print(f"\nPer-image counts saved to: {csv_path}")


if __name__ == "__main__":
    main()

# ===== DATASET STATS =====
# Images directory: dataset\images
# Labels directory: dataset\labels
# Total images: 1685
# Images with labels: 1685
# Images without labels: 0
# Total labels (all images): 3127

# Labels per image distribution (count -> num_images):
#   1: 677
#   2: 676
#   3: 239
#   4: 85
#   5: 7
#   6: 1

# Labels per class:
#   0 (KL0): 99 labels in 94 images
#   1 (KL1): 794 labels in 624 images
#   2 (KL2): 1357 labels in 857 images
#   3 (KL3): 580 labels in 383 images
#   4 (KL4): 297 labels in 224 images

# Orphan label files (no matching image): 0

# Per-image counts saved to: check_dataset\labels_per_image.csv