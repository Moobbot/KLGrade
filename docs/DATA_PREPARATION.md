# Data preparation guide - KLGrade Object Detection

**Author**: Ngo Tam
**Date**: 20/12/2025
**Purpose**: Guide for preparing 3 datasets for training object detection

---

## 📋 Tổng Quan

Dự án này sử dụng **3 cấu hình dataset** khác nhau để training:

| Dataset       | Classes    | Mô tả                           | Use Case                   |
| ------------- | ---------- | ------------------------------- | -------------------------- |
| **Dataset 1** | 5 classes  | KL0-KL4 (baseline)              | So sánh performance cơ bản |
| **Dataset 2** | 10 classes | KL0-a/b đến KL4-a/b             | Fine-grained detection     |
| **Dataset 3** | 7 classes  | Filtered (loại bỏ rare classes) | Optimal for training       |

---

## 🗂️ Cấu Trúc Thư Mục

```
KLGrade/
├── dataset/
│   └── dataset_v0/
│       ├── images/                      # Raw images (1,685 images)
│       ├── labels/                      # Dataset 1: 5 classes (KL0-KL4)
│       └── labels_new/                  # Dataset 2: 10 classes (KL0-a/b -> KL4-a/b)
├── dataset/dataset_filtered/
│   ├── images/                         # Dataset 3: Filtered images (1,663 images)
│   └── labels/                         # Dataset 3: 7 classes (remapped 0-6)
├── splits/                             # Train/val/test splits cho Dataset 1 & 2
├── splits_filtered/                    # Train/val/test splits cho Dataset 3
├── dataset_analysis/                   # Analysis results cho Dataset 1
├── analysis_results_new/              # Analysis results cho Dataset 2
└── analysis_filtered/                 # Analysis results cho Dataset 3
```

---

## 🔧 Quy Trình Chuẩn Bị Dữ Liệu

### Bước 1: Tạo Labels_new (10 Classes)

**Script**: `check_dataset/class_split_report.py`

**Chức năng**: Chia mỗi KL class thành 2 subclasses (a/b) dựa trên:

- **"a" (gai xương/osteophyte)**: Bounding box hình vuông, nhỏ
- **"b" (khe khớp/joint space)**: Bounding box hình dài, lớn hơn

**Thuật toán**:

```python
def classify_box(w, h, area):
    ratio = w / h
    if ratio < 1.2 or area < 0.01:
        return "a"  # Gai xương
    if ratio > 2.0 or area > 0.03:
        return "b"  # Khe khớp
    return None  # Không rõ ràng -> fallback
```

**Câu lệnh**:

```powershell
python check_dataset\class_split_report.py `
    --labels-dir dataset\dataset_v0\labels `
    --save-dir dataset\dataset_v0\labels_new
```

**Output**:

- `dataset/dataset_v0/labels_new/` - 1,685 label files với class IDs 0-9

---

### Bước 2: Phân Tích Dataset

**Script**: `check_dataset/analyze_dataset.py`

**Chức năng**:

- Thống kê class distribution
- Phân tích bbox dimensions, aspect ratios
- Categorize object sizes (small/medium/large)
- Tạo visualizations và reports

**Câu lệnh**:

```powershell
# Dataset 1: 5 classes
python check_dataset/analyze_dataset.py `
    --img_dir dataset\dataset_v0\images `
    --label_dir dataset\dataset_v0\labels `
    --output_dir dataset_analysis `
    --class_names KL0 KL1 KL2 KL3 KL4

# Dataset 2: 10 classes
python check_dataset/analyze_dataset.py `
    --img_dir dataset\dataset_v0\images `
    --label_dir dataset\dataset_v0\labels_new `
    --output_dir analysis_results_new `
    --class_names KL0-a KL0-b KL1-a KL1-b KL2-a KL2-b KL3-a KL3-b KL4-a KL4-b
```

**Output**:

- `DATASET_ANALYSIS_REPORT.md` - Summary report
- `class_distribution.png` - Class distribution charts
- `bbox_analysis.png` - Comprehensive bbox analysis
- `dataset_statistics.json` - Raw statistics

**Kết quả quan trọng**:

- Dataset 1: Imbalance ratio **13.71:1**
- Dataset 2: Imbalance ratio **130.30:1** (rất cao!)
- 87.4% objects là **small objects** (<1% image area)

---

### Bước 3: Tạo Dataset Filtered (7 Classes)

**Script**: `filter_dataset_by_class.py`

**Chức năng**: Lọc bỏ rare classes (<1% threshold):

- Class 1 (KL0-b): 10 instances (0.32%)
- Class 3 (KL1-b): 23 instances (0.74%)
- Class 9 (KL4-b): 25 instances (0.80%)

**Câu lệnh**:

```powershell
python filter_dataset_by_class.py `
    --img_dir dataset\dataset_v0\images `
    --label_dir dataset\dataset_v0\labels_new `
    --output_img_dir dataset\dataset_filtered\images `
    --output_label_dir dataset\dataset_filtered\labels `
    --threshold 1.0 `
    --remove_rare_from_labels `
    --analysis_json analysis_results_new\dataset_statistics.json
```

**Options**:

- `--threshold 1.0`: Minimum percentage (1%)
- `--remove_rare_from_labels`: Keep images, chỉ xóa rare boxes
- Không dùng flag này: Skip toàn bộ images có rare boxes

**Output**:

- `dataset/dataset_filtered/images/` - 1,663 images (98.7% retention)
- `dataset/dataset_filtered/labels/` - 3,069 boxes (98.1% retention)
- `filter_report.json` & `FILTER_REPORT.md`

---

### Bước 4: Remap Class IDs (Dataset 3)

**Script**: `remap_filtered_labels.py`

**Chức năng**: Chuyển class IDs từ rời rạc {0,2,4,5,6,7,8} → continuous {0,1,2,3,4,5,6}

**Mapping**:

```python
CLASS_REMAP_FILTERED = {
    0: 0,  # KL0-a
    2: 1,  # KL1-a
    4: 2,  # KL2-a
    5: 3,  # KL2-b
    6: 4,  # KL3-a
    7: 5,  # KL3-b
    8: 6,  # KL4-a
}
```

**Câu lệnh**:

```powershell
# Dry run (kiểm tra mapping)
python remap_filtered_labels.py `
    --label_dir dataset\dataset_filtered\labels `
    --dry_run

# Thực hiện remap (overwrite)
python remap_filtered_labels.py `
    --label_dir dataset\dataset_filtered\labels
```

**Output**: Labels được remap in-place với class IDs 0-6

---

### Bước 5: Tạo Train/Val/Test Splits

**Script**: `split_dataset.py`

**Chức năng**:

- Stratified splitting (balanced class distribution)
- Multi-label support
- Rare class protection
- Reproducible (fixed seed)

**Câu lệnh**:

```powershell
# Dataset 1: 5 classes
python split_dataset.py `
    --img_dir dataset\dataset_v0\images `
    --label_dir dataset\dataset_v0\labels `
    --out_dir splits\base `
    --train 0.7 --val 0.15 --test 0.15 --seed 42

# Dataset 2: 10 classes
python split_dataset.py `
    --img_dir dataset\dataset_v0\images `
    --label_dir dataset\dataset_v0\labels_new `
    --out_dir splits\new `
    --train 0.7 --val 0.15 --test 0.15 --seed 42

# Dataset 3: 7 classes (filtered)
python split_dataset.py `
    --img_dir dataset\dataset_filtered\images `
    --label_dir dataset\dataset_filtered\labels `
    --out_dir splits\filtered `
    --train 0.7 --val 0.15 --test 0.15 --seed 42
```

**Output**:

- `train.txt` - ~70% images
- `val.txt` - ~15% images
- `test.txt` - ~15% images
- `split_info.json` - Detailed split statistics

---

### Bước 6: Phân Tích Dataset Filtered

```powershell
python check_dataset/analyze_dataset.py `
    --img_dir dataset\dataset_filtered\images `
    --label_dir dataset\dataset_filtered\labels `
    --output_dir analysis_filtered `
    --class_names KL0-a KL1-a KL2-a KL2-b KL3-a KL3-b KL4-a
```

**Kết quả**:

- Imbalance ratio giảm từ **130.30:1** xuống **24.13:1**
- Tổng instances: 3,069 (giảm 58 boxes)
- Small objects: 89.0% (tăng nhẹ do loại bỏ một số large rare boxes)

---

## 🎯 Kịch Bản Training - 3 Experiments

### **Experiment 1: Baseline (5 Classes)**

**Dataset**: `dataset/dataset_v0/` với `labels/`

**Class Mapping** (từ `config.py`):

```python
CLASSES = {
    0: "KL0", 1: "KL1", 2: "KL2", 3: "KL3", 4: "KL4"
}
```

**Training Commands**:

```powershell
# YOLO11
python examples/train_yolo11.py `
    --img_dir dataset\dataset_v0\images `
    --label_dir dataset\dataset_v0\labels `
    --model yolo11n.pt `
    --epochs 100 `
    --batch 16 `
    --img_size 640 `
    --name exp1_baseline_5_classes

# DETR
python examples/train_detr.py `
    --img_dir dataset\dataset_v0\images `
    --label_dir dataset\dataset_v0\labels `
    --model facebook/detr-resnet-50 `
    --epochs 50 `
    --batch 4 `
    --name exp1_baseline_5_classes
```

**Mục đích**: Baseline để so sánh performance

---

### **Experiment 2: Fine-grained (10 Classes)**

**Dataset**: `dataset/dataset_v0/` với `labels_new/`

**Class Mapping**:

```python
CLASSES_10_CLASS = {
    0: "KL0-a", 1: "KL0-b", 2: "KL1-a", 3: "KL1-b",
    4: "KL2-a", 5: "KL2-b", 6: "KL3-a", 7: "KL3-b",
    8: "KL4-a", 9: "KL4-b"
}
```

**Training Commands**:

```powershell
# YOLO11
python examples/train_yolo11.py `
    --img_dir dataset\dataset_v0\images `
    --label_dir dataset\dataset_v0\labels_new `
    --use_labels_new `
    --model yolo11n.pt `
    --epochs 100 `
    --batch 16 `
    --name exp2_finegrained_10_classes

# DETR
python examples/train_detr.py `
    --img_dir dataset\dataset_v0\images `
    --label_dir dataset\dataset_v0\labels_new `
    --use_labels_new `
    --model facebook/detr-resnet-50 `
    --epochs 50 `
    --batch 4 `
    --name exp2_finegrained_10_classes
```

**Challenges**:

- ⚠️ Imbalance ratio cao (130:1)
- ⚠️ 3 rare classes (<1%)
- Cần weighted loss hoặc focal loss

---

### **Experiment 3: Filtered Optimal (7 Classes)**

**Dataset**: `dataset/dataset_filtered/`

**Class Mapping**:

```python
CLASSES_FILTERED = {
    0: "KL0-a", 1: "KL1-a", 2: "KL2-a", 3: "KL2-b",
    4: "KL3-a", 5: "KL3-b", 6: "KL4-a"
}
```

**Training Commands**:

```powershell
# YOLO11
python examples/train_yolo11.py `
    --img_dir dataset\dataset_filtered\images `
    --label_dir dataset\dataset_filtered\labels `
    --model yolo11n.pt `
    --epochs 100 `
    --batch 16 `
    --name exp3_filtered_7classes

# DETR
python examples/train_detr.py `
    --img_dir dataset\dataset_filtered\images `
    --label_dir dataset\dataset_filtered\labels `
    --model facebook/detr-resnet-50 `
    --epochs 50 `
    --batch 4 `
    --name exp3_filtered_7classes
```

**Advantages**:

- ✅ Imbalance ratio thấp hơn (24:1)
- ✅ Không có rare classes
- ✅ Stable training
- ✅ 98.7% data retention

---

## 📊 So Sánh 3 Datasets

| Metric          | Dataset 1 (5 cls) | Dataset 2 (10 cls) | Dataset 3 (7 cls) |
| --------------- | ----------------- | ------------------ | ----------------- |
| **Classes**     | 5                 | 10                 | 7                 |
| **Images**      | 1,685             | 1,685              | 1,663 (-1.3%)     |
| **Boxes**       | 3,127             | 3,127              | 3,069 (-1.9%)     |
| **Imbalance**   | 13.71:1           | 130.30:1           | 24.13:1           |
| **Min class**   | 99 (KL0)          | 10 (KL0-b)         | 54 (KL2-b)        |
| **Max class**   | 1,357 (KL2)       | 1,303 (KL2-a)      | 1,303 (KL2-a)     |
| **Small obj %** | 87.4%             | 87.4%              | 89.0%             |

---

## ⚙️ Config.py - Class Mappings

```python
# Dataset 1: Baseline
CLASSES = {
    0: "KL0", 1: "KL1", 2: "KL2", 3: "KL3", 4: "KL4"
}

# Dataset 2: Fine-grained
CLASSES_10_CLASS = {
    0: "KL0-a", 1: "KL0-b", 2: "KL1-a", 3: "KL1-b",
    4: "KL2-a", 5: "KL2-b", 6: "KL3-a", 7: "KL3-b",
    8: "KL4-a", 9: "KL4-b"
}

# Dataset 3: Filtered
CLASSES_FILTERED = {
    0: "KL0-a", 1: "KL1-a", 2: "KL2-a", 3: "KL2-b",
    4: "KL3-a", 5: "KL3-b", 6: "KL4-a"
}

# Remap từ labels_new -> filtered
CLASS_REMAP_FILTERED = {
    0: 0, 1: None, 2: 1, 3: None, 4: 2,
    5: 3, 6: 4, 7: 5, 8: 6, 9: None
}
```

---

## 📝 Recommendations

### Training Tips:

1. **Dataset 1 (Baseline)**:

   - Good starting point
   - Moderate imbalance → use weighted loss
   - Nhiều small objects → use FPN

2. **Dataset 2 (Fine-grained)**:

   - ⚠️ Rất imbalanced → **MUST use Focal Loss**
   - Consider oversampling rare classes
   - Hoặc train hierarchical: coarse→fine

3. **Dataset 3 (Filtered - Recommended)**:
   - Best balance
   - Stable training
   - Optimal cho production

### Augmentation Strategy:

```python
# For small objects (89% of dataset)
augmentations = [
    A.RandomScale(scale_limit=0.3),  # Zoom in
    A.RandomCrop(height=640, width=640),
    A.HorizontalFlip(p=0.5),
    A.CLAHE(clip_limit=2.0),  # Enhance contrast
    A.ShiftScaleRotate(rotate_limit=5),
]
```

### Anchor Box Optimization:

```powershell
# Chạy k-means clustering trên bbox dimensions
python optimize_anchors.py `
    --label_dir dataset\dataset_filtered\labels `
    --num_anchors 9 `
    --img_size 640
```

---

## 🛠️ Complete Pipeline Script

```powershell
# ============================================
# FULL DATA PREPARATION PIPELINE
# ============================================

# 1. Tạo labels_new (10 classes)
python check_dataset\class_split_report.py `
    --labels-dir dataset\dataset_v0\labels `
    --save-dir dataset\dataset_v0\labels_new

# 2. Phân tích datasets
python check_dataset/analyze_dataset.py `
    --img_dir dataset\dataset_v0\images `
    --label_dir dataset\dataset_v0\labels `
    --output_dir dataset_analysis `
    --class_names KL0 KL1 KL2 KL3 KL4

python check_dataset/analyze_dataset.py `
    --img_dir dataset\dataset_v0\images `
    --label_dir dataset\dataset_v0\labels_new `
    --output_dir analysis_results_new `
    --class_names KL0-a KL0-b KL1-a KL1-b KL2-a KL2-b KL3-a KL3-b KL4-a KL4-b

# 3. Filter rare classes
python filter_dataset_by_class.py `
    --img_dir dataset\dataset_v0\images `
    --label_dir dataset\dataset_v0\labels_new `
    --output_img_dir dataset\dataset_filtered\images `
    --output_label_dir dataset\dataset_filtered\labels `
    --threshold 1.0 `
    --remove_rare_from_labels `
    --analysis_json analysis_results_new\dataset_statistics.json

# 4. Remap class IDs
python remap_filtered_labels.py `
    --label_dir dataset\dataset_filtered\labels

# 5. Tạo splits
python split_dataset.py `
    --img_dir dataset\dataset_v0\images `
    --label_dir dataset\dataset_v0\labels `
    --out_dir splits `
    --train 0.7 --val 0.15 --test 0.15 --seed 42

python split_dataset.py `
    --img_dir dataset\dataset_filtered\images `
    --label_dir dataset\dataset_filtered\labels `
    --out_dir splits_filtered `
    --train 0.7 --val 0.15 --test 0.15 --seed 42

# 6. Phân tích filtered dataset
python check_dataset/analyze_dataset.py `
    --img_dir dataset\dataset_filtered\images `
    --label_dir dataset\dataset_filtered\labels `
    --output_dir analysis_filtered `
    --class_names KL0-a KL1-a KL2-a KL2-b KL3-a KL3-b KL4-a

Write-Host "✅ Data preparation completed!" -ForegroundColor Green
```

---

## 📚 References

- **Stratified Splitting**: https://scikit-learn.org/stable/modules/cross_validation.html
- **Object Detection Metrics**: https://mAP calculation
- **YOLO Anchor Optimization**: https://github.com/ultralytics/yolov5/discussions/6795
- **Class Imbalance**: Focal Loss paper (Lin et al., 2017)
- **Small Object Detection**: Feature Pyramid Networks (FPN)

---

## ✅ Checklist

- [ ] Đã chạy `class_split_report.py` để tạo labels_new
- [ ] Đã analyze cả 3 datasets
- [ ] Đã filter và remap filtered dataset
- [ ] Đã tạo splits cho tất cả datasets
- [ ] Đã review analysis reports
- [ ] Đã chuẩn bị training configs cho 3 experiments
- [ ] Đã test load datasets với YoloDataset/CocoDataset
- [ ] Ready to train! 🚀
