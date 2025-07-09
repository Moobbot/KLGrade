# How to Use `dataset.py`

This guide explains how to use the `dataset.py` utilities for loading, processing, and saving datasets in this project.

## 1. Import the utilities

```python
from dataset import ImageDataset, make_dataset_dataframe, save_dataset
from dataset import load_dataset_pt, load_dataset_npz, load_dataset_h5
import pandas as pd
```

## 2. Create a DataFrame from image and label folders

```python
# Create a DataFrame from folders and optionally save to CSV
# The DataFrame will have two columns: 'data' (image path), 'label' (label path)
df = make_dataset_dataframe("dataset/images", "dataset/labels", out_csv="dataset.csv")
```

## 3. Initialize the Dataset

You have three options:

### a) From a DataFrame

```python
dataset = ImageDataset(df=df)
```

### b) From a CSV file

```python
dataset = ImageDataset(df="dataset.csv")
```

### c) Directly from folders

```python
dataset = ImageDataset(images_dir="dataset/images", labels_dir="dataset/labels")
```

## 4. Access a sample

```python
item = dataset[0]
image = item["image"]      # Tensor [3, H, W]
target = item["target"]    # Tensor [N, 5] (YOLO: class_id, x, y, w, h)
```

## 5. Save the dataset to a file (.pt, .pth, .npz, .h5)

```python
save_dataset(dataset, "dataset/dataset.pt", fmt="pt")    # torch
save_dataset(dataset, "dataset/dataset.pth", fmt="pth")  # torch
save_dataset(dataset, "dataset/dataset.npz", fmt="npz")  # numpy
save_dataset(dataset, "dataset/dataset.h5", fmt="h5")    # hdf5 (requires h5py)
```

- `fmt` can be: `"pt"`, `"pth"`, `"npz"`, or `"h5"`.

## 6. Load a dataset from a saved file (.pt, .npz, .h5)

```python
# Load from .pt or .pth (PyTorch)
images, targets = load_dataset_pt("dataset/dataset.pt")

# Load from .npz (NumPy)
images, targets = load_dataset_npz("dataset/dataset.npz")

# Load from .h5 (HDF5, requires h5py)
images, targets = load_dataset_h5("dataset/dataset.h5")
```

- `images`: numpy array, shape (N, 3, H, W)
- `targets`: object array or list, each element is a [N_box, 5] array

## 7. Notes

- The dataset supports augmentation via albumentations. You can pass custom transforms when initializing.
- Labels must be in YOLO format: each line is `class_id x_center y_center width height` (float, normalized).
- When loading a saved file, you may need to convert arrays back to tensors as needed.
- For more advanced usage, see the code comments in `dataset.py`.

---

For more details or troubleshooting, see the code comments in `dataset.py` or contact the project maintainer.
