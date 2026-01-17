# KIOCMIL Pipeline (Full Image Version)

This document describes the complete pipeline for the **Knee Instance Object–Context MIL Network (KIOCMIL)**, using full X-ray images as input.

## 1. Data Preparation

Before training, you must check labels and generate dataset splits.

### 1.1 Generate 10-Class Labels
Parse original YOLO labels to create specific 10-class lesion labels in `dataset/dataset_v0/labels_10_class`.

```bash
python tools/check_dataset/class_split_report.py \
    --labels-dir dataset/dataset_v0/labels \
    --save-dir dataset/dataset_v0/labels_10_class \
    --limit 10
```

### 1.2 Split Dataset
Split the dataset into Train/Val/Test, outputting file lists to `splits/knee_full_10_class`.

```bash
python scripts/data_preparation/split_dataset.py \
    --img_dir dataset/dataset_v0/images \
    --label_dir dataset/dataset_v0/labels-knee \
    --out_dir splits/knee_full_10_class \
    --train 0.7 \
    --val 0.15 \
    --test 0.15 \
    --seed 42
```

**Output Files:**
-   `splits/knee_full_10_class/train.txt`
-   `splits/knee_full_10_class/val.txt`
-   `splits/knee_full_10_class/test.txt`

## 2. Training

Train the KIOCMIL model using the prepared splits and labels.

**Command:**
```bash
# Run in background (recommended)
nohup bash ./scripts/train_kiocmil.sh --use_sampler --use_weighted_loss > training.log 2>&1 &

# Or run directly
./scripts/train_kiocmil.sh --use_sampler --use_weighted_loss
```

**Configuration (Inside `scripts/train_kiocmil.sh`):**
-   **Images**: `dataset/dataset_v0/images`
-   **Knee Labels**: `dataset/dataset_v0/labels-knee`
-   **Lesion Labels**: `dataset/dataset_v0/labels_10_class`
-   **Splits**: `splits/knee_full_10_class/train.txt` / `val.txt`

## 3. Evaluation

Evaluate the trained model on the **Test Split**.

```bash
# Default evaluation (uses splits/knee_full_10_class/test.txt)
./scripts/eval_kiocmil.sh

# Or specify split and model explicitly
./scripts/eval_kiocmil.sh splits/knee_full_10_class/test.txt runs/kiocmil_v1/best_model.pth
```

## 4. Key Components

-   **Dataset Logic**: `src/datasets/kiocmil_dataset.py`
    -   Loads Full Image.
    -   Extracts Knee Instances (using `labels-knee`).
    -   Assigns Lesions (from `labels_10_class`) to knees.
    -   Generates Context Token (Knee + Context) and Object Tokens (Lesions).
-   **Model Architecture**: `src/models/kiocmil_model.py`
    -   Multi-head output: 10-Class (Softmax), KL Grade (0-4), Type (Osteophyte/Joint Space).
