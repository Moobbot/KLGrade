"""
Multilabel, group-aware stratified split for YOLO-format datasets.

Overview
--------
- Reads images and labels from specified directories (YOLO txt format).
- Preserves multilabel class distribution across train/val/test using
    iterative-stratification (MultilabelStratifiedShuffleSplit).
- Groups images that belong to the same study/side (e.g., name_0, name_1)
    to avoid leakage across splits.
- Saves file lists to output directory (default: splits/train.txt, splits/val.txt, splits/test.txt).

Usage
-----
        python check_dataset/split_dataset.py --train 0.7 --val 0.15 --test 0.15 --seed 42
        python check_dataset/split_dataset.py --img_dir processed/knee/images --label_dir processed/knee/labels --out_dir splits

Requirements
------------
        pip install iterative-stratification
"""
import os
import sys
import argparse
from collections import defaultdict, Counter

import numpy as np

# Ensure repository root in path
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import CLASSES


try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
except Exception as e:
    raise SystemExit(
        "iterative-stratification is required. Install with: pip install iterative-stratification"
    )


# Default paths (can be overridden via CLI arguments)
DEFAULT_IMG_DIR = os.path.join("dataset", "knee", "images")
DEFAULT_LABEL_DIR = os.path.join("dataset", "knee", "labels")
DEFAULT_OUT_DIR = os.path.join("splits")


def list_images(img_dir: str):
    """List image files under img_dir (case-insensitive common extensions).

    Returns a sorted list of filenames (not full paths).
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in exts]
    files.sort()
    return files


def read_label_file(path: str):
    """Parse a YOLO .txt label file and return a list of class IDs per object.

    Each line is expected to be: class_id cx cy w h
    Lines that cannot be parsed are skipped.
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
                continue
            class_ids.append(cls_id)
    return class_ids


def group_key_from_stem(stem: str) -> str:
    """Return a grouping key from an image stem.

    Images that share the same prefix but differ in a trailing "_digit" suffix
    (e.g., 123_0, 123_1) are treated as a single group (same study/patient).
    This prevents leakage across splits.
    """
    # Group images like ..._0, ..._1 together
    if "_" in stem and stem.rsplit("_", 1)[1].isdigit():
        return stem.rsplit("_", 1)[0]
    return stem


def build_multilabel_matrix(images, label_dir: str):
    """Build a multilabel presence matrix (images × classes) and group IDs.

    For each image, mark presence (1) if the image contains at least one
    object of that class (based on its YOLO label file), else 0.
    Also derive a group identifier for group-aware splitting.
    """
    num_classes = len(CLASSES)
    X = np.zeros((len(images), num_classes), dtype=int)
    groups = []
    for i, img in enumerate(images):
        stem, _ = os.path.splitext(img)
        label_path = os.path.join(label_dir, f"{stem}.txt")
        cls_ids = read_label_file(label_path)
        for c in set(cls_ids):  # set -> multilabel presence per image (presence, not counts)
            if 0 <= c < num_classes:
                X[i, c] = 1
        groups.append(group_key_from_stem(stem))
    return X, np.array(groups)


def aggregate_by_group(images, X, groups):
    """Collapse image-level labels to group-level labels.

    Returns:
      - group_keys: list of group identifiers
      - G: (num_groups × num_classes) presence matrix at group level
      - group_sizes: number of images per group
      - group_to_indices: dict group_key -> list of image indices in that group
    """
    # Collapse image-level features to group-level
    group_to_indices = defaultdict(list)
    for idx, g in enumerate(groups):
        group_to_indices[g].append(idx)

    group_keys = []
    G = []
    group_sizes = []
    for g, idxs in group_to_indices.items():
        group_keys.append(g)
        group_sizes.append(len(idxs))
        # Union labels across images in group
        G.append(np.clip(X[idxs].sum(axis=0), 0, 1))
    G = np.array(G, dtype=int)
    group_sizes = np.array(group_sizes)
    return group_keys, G, group_sizes, group_to_indices


def stratified_group_split(group_keys, G, group_to_indices, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Perform two-stage multilabel stratified split at the group level.

    1) Split groups into train vs temp (val+test) with stratification on G.
    2) Split temp into val vs test with stratification on G[temp].
    Finally expand group indices back to image indices via group_to_indices.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=1 - train_ratio, random_state=seed)
    groups_arr = np.arange(len(group_keys))
    train_idx, temp_idx = next(msss.split(groups_arr, G))

    # Split temp into val+test
    if val_ratio + test_ratio == 0:
        val_idx = np.array([], dtype=int)
        test_idx = np.array([], dtype=int)
    else:
        # Proportion of the temp set allocated to test
        test_size = test_ratio / (val_ratio + test_ratio)
        msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        val_sub, test_sub = next(msss2.split(temp_idx, G[temp_idx]))
        val_idx = temp_idx[val_sub]
        test_idx = temp_idx[test_sub]

    # Expand to image indices
    split = {"train": [], "val": [], "test": []}
    for sname, sidx in ("train", train_idx), ("val", val_idx), ("test", test_idx):
        for gi in sidx:
            gkey = group_keys[gi]
            split[sname].extend(group_to_indices[gkey])
    return split


def summarize(images, X, split, label_dir: str):
    """Print summary statistics per split.

    Shows both per-class image presence counts and total object label counts
    (re-reading label files to accumulate exact object totals per class).
    """
    num_classes = len(CLASSES)
    for sname in ["train", "val", "test"]:
        idxs = split[sname]
        subX = X[idxs]
        img_count = len(idxs)
        label_counts = subX.sum(axis=0).astype(int)  # per-class image presence counts
        # Also count total OBJECT labels (not just presence) by re-reading files
        obj_counts = Counter()
        for i in idxs:
            stem, _ = os.path.splitext(images[i])
            class_ids = read_label_file(os.path.join(label_dir, f"{stem}.txt"))
            obj_counts.update(class_ids)

        print(f"\n=== {sname.upper()} ===")
        print(f"Images: {img_count}")
        print("Per-class (image contains class):")
        for cid in range(num_classes):
            name = CLASSES[cid]
            print(f"  {cid} ({name}): {label_counts[cid]}")
        print("Per-class (total object labels):")
        for cid in range(num_classes):
            name = CLASSES[cid]
            print(f"  {cid} ({name}): {obj_counts.get(cid, 0)}")


def extract_orig_stem_from_filename(filename: str) -> str:
    """Extract stem from filename (just remove extension).
    
    Since we no longer use _knee suffix, we just need to remove the file extension.
    The full stem (including any _<digit> suffixes) is preserved.
    
    Examples:
    - 'image_0_0.jpg' -> 'image_0_0' (keeps everything, just removes .jpg)
    - 'image_0.jpg' -> 'image_0' (keeps everything, just removes .jpg)
    """
    return os.path.splitext(filename)[0]


def save_lists(images, split, out_dir: str):
    """Write split file lists (original stems only) into out_dir.

    Files:
      - {out_dir}/train.txt
      - {out_dir}/val.txt
      - {out_dir}/test.txt
    
    Note: Extracts original stems from crop filenames (removes knee index _{k_idx})
    to ensure compatibility with train_det.py which maps crops back to original stems.
    """
    os.makedirs(out_dir, exist_ok=True)
    for sname in ["train", "val", "test"]:
        idxs = sorted(split[sname])
        out_path = os.path.join(out_dir, f"{sname}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            for i in idxs:
                # Extract original stem from filename (removes knee index _{k_idx})
                orig_stem = extract_orig_stem_from_filename(images[i])
                f.write(orig_stem + "\n")
        # Debug: show first few examples
        if len(idxs) > 0:
            sample_files = [images[i] for i in idxs[:3]]
            sample_stems = [extract_orig_stem_from_filename(f) for f in sample_files]
            print(f"  Sample: {sample_files[0]} -> {sample_stems[0]}")
        print(f"Saved {sname} list -> {out_path} ({len(idxs)} items)")


def parse_args():
    """Parse CLI arguments for split ratios, seed, and directory paths."""
    p = argparse.ArgumentParser(description="Multilabel group-aware stratified dataset split")
    p.add_argument("--img_dir", type=str, default=DEFAULT_IMG_DIR, help="Directory containing images")
    p.add_argument("--label_dir", type=str, default=DEFAULT_LABEL_DIR, help="Directory containing YOLO label files")
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR, help="Output directory for split lists")
    p.add_argument("--train", type=float, default=0.70, help="Train ratio")
    p.add_argument("--val", type=float, default=0.15, help="Val ratio")
    p.add_argument("--test", type=float, default=0.15, help="Test ratio")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def main():
    """Entry point: build matrices, perform group-aware stratified split, save lists."""
    args = parse_args()
    if abs(args.train + args.val + args.test - 1.0) > 1e-6:
        raise SystemExit("train + val + test must equal 1.0")

    images = list_images(args.img_dir)
    if not images:
        raise SystemExit(f"No images found in {args.img_dir}")

    X, groups = build_multilabel_matrix(images, args.label_dir)
    gkeys, G, gsz, g2idx = aggregate_by_group(images, X, groups)

    split = stratified_group_split(gkeys, G, g2idx, args.train, args.val, args.test, args.seed)
    summarize(images, X, split, args.label_dir)
    save_lists(images, split, args.out_dir)


if __name__ == "__main__":
    main()
