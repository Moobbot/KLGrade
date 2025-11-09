"""
Split utilities for classification CSVs.

Provides:
- Reading precomputed split lists (train.txt/val.txt/test.txt) and mapping to CSV rows.
- Stratified train/val(/test) indices using sklearn.model_selection.train_test_split.
"""
import os
from typing import Iterable, List, Sequence, Tuple, Set, Optional

import numpy as np
from sklearn.model_selection import train_test_split


def read_split_stems(path: str) -> Set[str]:
    """Read stems from split file.
    
    Split files contain stems (not full file paths). If a line looks like a file path
    (has known image extensions), we extract the stem. Otherwise, we use the line as-is.
    """
    stems: Set[str] = set()
    if not path or not os.path.exists(path):
        return stems
    # Known image extensions that should be removed
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            fn = line.strip()
            if not fn:
                continue
            # If it looks like a file path with a known extension, extract stem
            # Otherwise, treat the whole line as the stem
            basename = os.path.basename(fn)
            ext = os.path.splitext(basename)[1].lower()
            if ext in image_exts:
                stem = os.path.splitext(basename)[0]
            else:
                # No known extension, use the line as-is (it's already a stem)
                stem = basename
            stems.add(stem)
    return stems


def extract_orig_stem_from_crop_path(p: str) -> str:
    """Extract stem from crop path (just remove extension).
    
    Since we no longer use _knee suffix, we just need to remove the file extension.
    The full stem (including any _<digit> suffixes) is preserved to match with splits.
    
    Examples:
    - 'image_0_0.jpg' -> 'image_0_0' (keeps everything, just removes .jpg)
    - 'image_0.jpg' -> 'image_0' (keeps everything, just removes .jpg)
    """
    return os.path.splitext(os.path.basename(p))[0]


def build_subset_indices_from_splits(
    df_paths: Sequence[str], train_file: Optional[str], val_file: Optional[str], test_file: Optional[str] = None
) -> Tuple[List[int], List[int], List[int]]:
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    train_stems = read_split_stems(train_file) if train_file else set()
    val_stems = read_split_stems(val_file) if val_file else set()
    test_stems = read_split_stems(test_file) if test_file else set()

    for i, p in enumerate(df_paths):
        stem = extract_orig_stem_from_crop_path(p)
        if stem in train_stems:
            train_idx.append(i)
        elif stem in val_stems:
            val_idx.append(i)
        elif test_stems and stem in test_stems:
            test_idx.append(i)

    return train_idx, val_idx, test_idx


def stratified_train_val_indices(labels: Sequence[int], val_ratio: float = 0.15, seed: int = 42) -> Tuple[List[int], List[int]]:
    idx = np.arange(len(labels))
    y = np.array(labels)
    train_idx, val_idx = train_test_split(idx, test_size=val_ratio, random_state=seed, stratify=y)
    return train_idx.tolist(), val_idx.tolist()


def stratified_train_val_test_indices(
    labels: Sequence[int], val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    assert val_ratio + test_ratio < 1.0
    idx = np.arange(len(labels))
    y = np.array(labels)
    # train vs temp (val+test)
    temp_ratio = val_ratio + test_ratio
    train_idx, temp_idx, y_train, y_temp = train_test_split(idx, y, test_size=temp_ratio, random_state=seed, stratify=y)
    # split temp into val/test
    test_size = test_ratio / temp_ratio if temp_ratio > 0 else 0.0
    if temp_ratio > 0:
        val_idx, test_idx = train_test_split(temp_idx, test_size=test_size, random_state=seed, stratify=y_temp)
    else:
        val_idx, test_idx = np.array([], dtype=int), np.array([], dtype=int)
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()
