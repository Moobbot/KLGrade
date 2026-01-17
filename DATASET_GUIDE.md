# Dataset Guide - KLGrade Project

Complete guide for all available datasets and how to use them with different loaders.

---

## 📊 Available Datasets

### 1. Cropped Knees (Unbalanced)
**Path**: `datasets/dataset_knees_cropped/`

**Structure**:
```
dataset_knees_cropped/
├── images/                    #  Cropped knee images (~1691 images)
├── labels/                    # 5-class KL labels (KL0-4)
├── labels_4class/             # 4-class labels (KL1-4, ~1783 labels)
├── labels_8class/             # 8-class with sub-grades (~1783 labels)
├── labels-knee/               # Knee detection boxes
├── labels_new/                # Lesion labels (~1783 labels)
├── images-no-labels/          # Filtered out (no labels)
├── labels-no-labels/          # Filtered out
└── dataset_statistics.txt     # Statistics report
```

**Class Distributions**:
- **5-class**: KL0, KL1, KL2, KL3, KL4
- **4-class**: KL1, KL2, KL3, KL4 (no KL0)
- **8-class**: KL0, KL1-a, KL1-b, KL2-a, KL2-b, KL3-a, KL3-b, KL4

**Use Case**: Standard YOLO training, KiocmilDataset training

---

### 2. Cropped Knees (Balanced)
**Path**: `datasets/dataset_knees_cropped_balanced/`

**Structure**: Same as unbalanced

**Features**:
- Balanced via oversampling with augmentation
- Target: ~500 samples per class
- See `balance_report.txt` for details

**Use Case**: Training with balanced class distribution

---

### 3. Full X-rays (Original - 10 Class)
**Path**: `datasets/dataset/dataset_v0/`

**Structure**:
```
dataset_v0/
├── images/                    # Full X-ray images
├── labels-knee/               # Knee detection boxes
├── labels_10_class/           # 10-class KL labels with sub-grades
└── labels_new/                # Lesion labels (Ost, JS)
```

**Split Files**: `splits/knee_full_10_class/`
- `train.txt` - Training split
- `val.txt` - Validation split  
- `test.txt` - Test split

**Class Distribution** (10-class):
- KL0, KL1-a, KL1-b, KL1-c
- KL2-a, KL2-b
- KL3-a, KL3-b, KL3-c
- KL4

**Use Case**: Full X-ray multi-instance learning, CADA architecture

---

## 🔧 Dataset Loaders

### 1. YOLODataset

**Purpose**: YOLO object detection training

**Supports**:
- Single-class or multi-class detection
- Flexible paths
- Multiple bbox formats (YOLO, Pascal VOC, COCO)
- Data augmentation via Albumentations

**Example - Cropped 5-class**:
```python
from src.datasets.yolo_dataset import YoloDataset, get_default_train_transform

dataset = YoloDataset(
    img_dir="datasets/dataset_knees_cropped/images",
    label_dir="datasets/dataset_knees_cropped/labels",
    split_file="datasets/dataset_knees_cropped/train.txt",  # After creating splits
    transform=get_default_train_transform(),
    filter_no_label=True,
    bbox_format="pascal_voc"
)
```

**Example - Balanced 4-class**:
```python
dataset = YoloDataset(
    img_dir="datasets/dataset_knees_cropped_balanced/images",
    label_dir="datasets/dataset_knees_cropped_balanced/labels_4class",
    split_file="datasets/dataset_knees_cropped_balanced/train.txt",
    transform=get_default_train_transform()
)
```

---

### 2. KiocmilDataset V3 (CADA Architecture)

**Purpose**: Multi-instance learning with context-aware deformable attention

**Features**:
- Dual labels: knee boxes + lesion boxes
- Context and patch extraction
- Bbox information for deformable attention
- Supports both cropped and full X-rays

**Example - Cropped Knees**:
```python
from src.datasets.kiocmil_dataset_v3 import KiocmilDatasetV3

dataset = KiocmilDatasetV3(
    img_dir="datasets/dataset_knees_cropped/images",
    knee_label_dir="datasets/dataset_knees_cropped/labels-knee",
    lesion_label_dir="datasets/dataset_knees_cropped/labels",
    split_file="datasets/dataset_knees_cropped/train.txt",
    ctx_size=(384, 384),
    patch_size=(224, 224)
)
```

**Example - Full X-rays (10-class)**:
```python
dataset = KiocmilDatasetV3(
    img_dir="datasets/dataset/dataset_v0/images",
    knee_label_dir="datasets/dataset/dataset_v0/labels-knee",
    lesion_label_dir="datasets/dataset/dataset_v0/labels_10_class",
    split_file="splits/knee_full_10_class/train.txt",
    ctx_size=(512, 512),
    patch_size=(256, 256)
)
```

---

### 3. COCODataset (DETR)

**Purpose**: DETR object detection training

**Status**: Documentation needed

---

## ⚙️ Creating Dataset Splits

### Automatic (All Datasets)
```bash
bash scripts/pipelines/regenerate_configs_and_splits.sh
```

This creates:
- `dataset_knees_cropped/train.txt` (70%)
- `dataset_knees_cropped/val.txt` (15%)
- `dataset_knees_cropped/test.txt` (15%)
- Plus splits for balanced dataset and all class variants

### Manual (Single Dataset)
```bash
python scripts/data_preparation/split_dataset.py \
    --img_dir datasets/dataset_knees_cropped/images \
    --label_dir datasets/dataset_knees_cropped/labels \
    --out_dir datasets/dataset_knees_cropped \
    --train 0.7 --val 0.15 --test 0.15 \
    --seed 42
```

---

## 📋 Configuration Matrix

| Model | Dataset | Img Dir | Label Dir | Split File | Notes |
|-------|---------|---------|-----------|------------|-------|
| YOLO 5-class | Cropped | `dataset_knees_cropped/images` | `labels` | `dataset_knees_cropped/train.txt` | Standard |
| YOLO 4-class | Cropped | `dataset_knees_cropped/images` | `labels_4class` | `dataset_knees_cropped/train.txt` | No KL0 |
| YOLO 8-class | Cropped | `dataset_knees_cropped/images` | `labels_8class` | `dataset_knees_cropped/train.txt` | Sub-grades |
| YOLO Balanced | Balanced | `dataset_knees_cropped_balanced/images` | `labels` | `dataset_knees_cropped_balanced/train.txt` | Oversampled |
| KIOCMIL Cropped | Cropped | `dataset_knees_cropped/images` | `labels-knee` + `labels` | `dataset_knees_cropped/train.txt` | MIL |
| KIOCMIL Full | Full X-ray | `dataset/dataset_v0/images` | `labels-knee` + `labels_10_class` | `splits/knee_full_10_class/train.txt` | CADA |

---

## 🚀 Quick Start

### 1. For YOLO Training (Cropped Knees)

```bash
# 1. Create splits (if not exists)
python scripts/data_preparation/split_dataset.py \
    --img_dir datasets/dataset_knees_cropped/images \
    --label_dir datasets/dataset_knees_cropped/labels \
    --out_dir datasets/dataset_knees_cropped

# 2. Train
yolo detect train \
    data=configs/yolo_5_class_baseline.yaml \
    epochs=100 \
    batch=16
```

### 2. For KIOCMIL Training (Full X-rays)

```bash
# Splits already exist at splits/knee_full_10_class/

# Train
bash scripts/training/train_kiocmil_v3_wandb.sh
```

### 3. For Balanced Dataset

```bash
# 1. Create balanced dataset
python scripts/data_preparation/balance_dataset.py \
    --input datasets/dataset_knees_cropped \
    --output datasets/dataset_knees_cropped_balanced

# 2. Create splits
python scripts/data_preparation/split_dataset.py \
    --img_dir datasets/dataset_knees_cropped_balanced/images \
    --label_dir datasets/dataset_knees_cropped_balanced/labels \
    --out_dir datasets/dataset_knees_cropped_balanced

# 3. Update config and train
```

---

## 🔍 Verification

### Check Dataset Statistics
```bash
python scripts/analyzes/analyze_knee_dataset.py \
    --dataset datasets/dataset_knees_cropped
```

### Verify Splits
```bash
# Check split files exist
ls -lh datasets/dataset_knees_cropped/*.txt

# Count images per split
wc -l datasets/dataset_knees_cropped/train.txt
wc -l datasets/dataset_knees_cropped/val.txt
wc -l datasets/dataset_knees_cropped/test.txt
```

### Test Loader
```python
from src.datasets.yolo_dataset import YoloDataset, visualize_dataset_sample

dataset = YoloDataset(
    img_dir="datasets/dataset_knees_cropped/images",
    label_dir="datasets/dataset_knees_cropped/labels",
    split_file="datasets/dataset_knees_cropped/train.txt"
)

print(f"Dataset size: {len(dataset)}")
visualize_dataset_sample(dataset, idx=0, save_path="test_sample.jpg")
```

---

## 📝 Notes

1. **Split files required**: All loaders need split files. Generate them before training.

2. **Label directories**: Different class counts use different label dirs:
   - 5-class: `labels/`
   - 4-class: `labels_4class/`
   - 8-class: `labels_8class/`
   - 10-class: `labels_10_class/`

3. **KIOCMIL needs dual labels**:
   - Knee boxes: `labels-knee/`
   - Lesion labels: `labels/` or `labels_10_class/`

4. **Balanced dataset**: Created via oversampling, use for imbalanced class scenarios

5. **Full X-rays vs Cropped**: 
   - Cropped: Pre-extracted knee regions, faster training
   - Full: Original X-rays, requires knee detection, supports multi-knee

---

**Last Updated**: 2026-01-17  
**Status**: Ready for use ✅
