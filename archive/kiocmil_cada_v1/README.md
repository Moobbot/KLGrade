# KIOCMIL CADA - Reproduction Guide

This package contains the complete source code, trained weights, and scripts to reproduce the KIOCMIL CADA model results.

## 1. Prerequisites
- **OS**: Linux (Recommended)
- **Python**: 3.10+
- **GPU**: NVIDIA GPU with CUDA support (Recommended 12GB+ VRAM for training)

## 2. Installation

We recommend using a virtual environment (Conda or venv).

### Option A: Conda (Recommended)
```bash
conda create -n kiocmil_env python=3.10 -y
conda activate kiocmil_env
pip install -r requirements.txt
```

### Option B: Pip
```bash
pip install -r requirements.txt
```

## 3. Data Preparation

The model expects the data to be in a specific structure.

### 3.1 Input Structure
Make sure your raw knee images are located at:
- `processed/knee/images/`: Contains `.jpg` or `.png` images.
- `processed/knee/labels/`: Contains `.txt` YOLO-format labels.

### 3.2 preprocessing
Run the preprocessing script to apply CLAHE, Blur, and Balancing:

```bash
# From the root of this folder
python scripts/data_preparation/preprocess_knee_dataset.py
```
This will create a `processed/knee_balanced` directory.

## 4. Training

To train the model from scratch:

```bash
python src/training/train_kiocmil_cada.py \
    --img_dir processed/knee/images \
    --knee_label_dir processed/knee/labels \
    --lesion_label_dir processed/knee/labels \
    --epochs 100 \
    --batch_size 16 \
    --save_dir runs/kiocmil_cada_new
```

*Note: The script automatically handles train/val splitting logic or you can specify a split file.*

## 5. Evaluation

To evaluate the provided pretrained model (`weights/kiocmil_cada_best.pt`):

```bash
python src/training/evaluate_kiocmil_cada.py \
    --model_path weights/kiocmil_cada_best.pt \
    --img_dir processed/knee/images \
    --knee_label_dir processed/knee/labels \
    --lesion_label_dir processed/knee/labels \
    --split_file path/to/your/test_split.txt \
    --split_name test_run
```

### Output
Results will be saved in `analysis/plots/` (Confusion Matrices, ROC Curves, and Text Reports).

## 6. Project Structure

- **`src/models/`**: `KiocmilModelCADA` architecture (YOLO backbone + Fusion Transformer).
- **`src/datasets/`**: `KiocmilDatasetV3` loader logic.
- **`src/training/`**: Training and Evaluation loops.
- **`weights/`**: `kiocmil_cada_best.pt` (Best checkpoint).
- **`docs/`**: `KIOCMIL_CADA_REPORT.md` (Technical details).
