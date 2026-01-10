# KLGrade - Data Preprocessing Pipeline

Complete pipeline for processing knee osteoarthritis X-ray dataset from raw images to training-ready crops.

---

## 📋 Overview

**Input:** Full X-ray images with multiple label types
**Output:** Square knee region crops with KL grade labels

**Pipeline Stages:**

1. Dataset Analysis
2. Knee Region Cropping
3. Label Filtering
4. Visualization & Quality Check ✨
5. Image Resize ✨
6. Validation & Verification
7. Final Analysis
8. Create 10-class Dataset
9. Create Filtered Datasets (4-class, 8-class)
10. **Stratified Train/Val/Test Split** ✨

---

## 🔄 Stage 1: Initial Dataset Analysis

**Purpose:** Understand the raw dataset structure and statistics

**Command:**

```powershell
.venv\Scripts\python.exe tools\check_dataset\comprehensive_analysis.py `
    --dataset_dir dataset\dataset_v0 `
    --output analysis\dataset_v0
```

**Output:**

- `analysis/dataset_v0/ANALYSIS_REPORT.md` - Summary
- `analysis/dataset_v0/DETAILED_ANALYSIS_REPORT.md` - Full statistics
- `analysis/dataset_v0/analysis_report.json` - Raw data
- `analysis/dataset_v0/class_distribution.png` - Visualization

**Key Findings:**

- 1,473 full X-ray images
- 3 label types: `labels/`, `labels-knee/`, `labels-knee-2box/`
- 319 images have 2 knee boxes (need special handling)

---

## 🔄 Stage 2: Knee Region Cropping

**Purpose:** Extract square knee regions with KL grade labels

**Command:**

```powershell
.venv\Scripts\python.exe scripts\preprocessing\crop_knee_regions.py `
    --dataset_dir dataset\dataset_v0 `
    --output_dir processed\knee `
    --margin 0.15 `
    --min_size 300
```

**Parameters:**

- `--margin`: Expansion around knee box (0.15 = 15%)
- `--min_size`: Minimum crop size in pixels (300px)

**Features:**

- ✅ **Proximity-based expansion:** Knee box mở rộng để bao KL labels gần đó (trong 0.75× knee size)
- ✅ **Smart label retention:** Chỉ drop labels hoàn toàn xa khỏi crop region
- ✅ **Boundary clamping:** Clamp box coordinates để tránh out-of-range errors
- ✅ **Multi-knee support:** Processes ALL knee boxes (not just first one)
- ✅ **Square crops with margin:** 15% expansion for context
- ✅ **Indexed naming:** Single knee → `name.jpg`, multiple → `name_knee0.jpg`, `name_knee1.jpg`
- ✅ **Comprehensive logging:** Skipped files and dropped labels tracked

**Strategy Details:**

1. **Load knee detection box** from `labels-knee/`
2. **Proximity check:** Find KL labels within 0.75× knee size distance
3. **Expand knee box** to include nearby KL labels (prevents dropping valid labels)
4. **Square expansion:** Make square + 15% margin
5. **Transform labels:** Convert coordinates from full image → crop space
6. **Clamp boundaries:** Ensure all coordinates in valid range [0,1]
7. **Save:** Cropped image + transformed labels

**Output:**

```
processed/knee/
├── images/              # All cropped knees
├── labels/              # Transformed KL labels
├── crop_stats.json      # Statistics
├── skipped_files.json   # Files that couldn't be cropped
└── dropped_labels.json  # Labels dropped (center outside crop)
```

**Expected Results:**

- **Total knee crops:** ~1,780 (from 1,473 images)
  - ~1,141 images with 1 knee → 1,141 crops
  - ~319 images with 2 knees → 638 crops
  - ~1 image with 0 knees (skipped)
- **With labels:** ~1,688 crops (94.8%)
- **Without labels:** ~92 crops (5.2%) → moved to `images-no-labels/`
- **Label retention:** ~100.4% (3,141/3,127 boxes)
- **Dropped labels:** ~1,062 (labels far from knee region)
- **Skipped files:** ~14 total
  - 12 no knee box
  - 2 crop too small (<300px)

---

## 🔄 Stage 3: Filter Crops Without Labels

**Purpose:** Separate crops that don't contain any KL grade labels

**Command:**

```powershell
.venv\Scripts\python.exe scripts\preprocessing\filter_no_labels.py `
    --input processed\knee
```

**Output:**

```
processed/knee/
├── images/              # Crops WITH labels (ready for training)
├── labels/              # KL grade labels
├── images-no-labels/    # Crops WITHOUT labels (excluded from training)
├── labels-no-labels/    # Empty label files
└── no_label_files.json  # List of filtered files
```

**Expected Results:**

- ~1,688 crops with labels (94.8%)
- ~92 crops without labels (5.2%)

---

## 🔄 Stage 4: Visualize Cropped Samples

**Purpose:** Visual quality check of cropped knee regions before proceeding

**Command:**

```powershell
.venv\Scripts\python.exe tools\check_dataset\visualize_samples.py `
    --img_dir processed\knee\images `
    --label_dir processed\knee\labels `
    --out_dir analysis\knee\visualizations `
    --color blue `
    --thickness 2
```

**What to check:**

- ✅ Knee regions properly centered
- ✅ KL grade boxes visible and correct
- ✅ No excessive cropping of lesion areas
- ✅ Multi-knee images correctly split

**Output:**

- Visualization images with bounding boxes drawn
- Sample from all classes for quality assessment

---

## 🔄 Stage 5: Resize Images to 640x640

**Purpose:** Standardize all images to uniform size for training

**Command:**

```powershell
.venv\Scripts\python.exe tools\check_dataset\resize_images.py `
    --in_dir processed\knee\images `
    --out_dir processed\knee_5_class\images `
    --size 640
```

**Features:**

- Resizes all images to 640x640 (IMG_SIZE from config.py)
- Maintains aspect ratio with letterbox padding if needed
- Uses high-quality interpolation

**Note:** Labels remain unchanged (already in normalized YOLO format)

**Output:**

```
processed/knee_5_class/images/  (1,688 images at 640x640)
```

---

## 🔄 Stage 6: Validate Processed Dataset

**Purpose:** Verify data quality and label integrity

**Command:**

```powershell
.venv\Scripts\python.exe tools\check_dataset\validate_dataset.py `
    --img_dir processed\knee\images `
    --label_dir processed\knee\labels
```

**Checks:**

- Label format correctness
- Bounding box valid ranges
- Image-label pairing
- No missing files

---

## 🔄 Stage 7: Final Analysis

**Purpose:** Understand the final 5-class training dataset

**Command:**

```powershell
.venv\Scripts\python.exe tools\check_dataset\comprehensive_analysis.py `
    --dataset_dir processed\knee `
    --output analysis\knee
```

**Output:**

- Detailed statistics on cropped dataset
- Class distribution analysis (5 classes: KL0-4)
- Recommendations for training

---

## 🔄 Stage 8: Create 10-Class Dataset (A/B Split)

**Purpose:** Split KL grades into osteophyte (a) and joint space (b) for detailed classification

**Command:**

```powershell
# Run class split
.venv\Scripts\python.exe tools\check_dataset\class_split_report.py `
    --labels-dir processed\knee\labels `
    --save-dir processed\knee\labels_10_class `
    --limit 10

# Create separate 10-class dataset (Copy images for reuse, Move labels)
New-Item -ItemType Directory -Path "processed\knee_10_class\images", "processed\knee_10_class\labels" -Force
Copy-Item "processed\knee_5_class\images\*" "processed\knee_10_class\images\" -Force
Move-Item "processed\knee\labels_10_class\*" "processed\knee_10_class\labels\" -Force

# Analyze 10-class dataset
.venv\Scripts\python.exe tools\check_dataset\comprehensive_analysis.py `
    --dataset_dir processed\knee_10_class `
    --output analysis\knee_10_class
```

**Features:**

- Splits each KL grade into 2 sub-classes based on geometry:
  - **a = Osteophyte** (gai xương): Small, near-square boxes
  - **b = Joint space** (khe khớp): Long, thin boxes
- Outputs 10 classes: KL0-a, KL0-b, KL1-a, KL1-b, ..., KL4-a, KL4-b
- Uses YOLO standard format (class_id 0-9)

**Output:**

```
processed/knee_10_class/
├── images/    (1,688 - same as 5-class)
└── labels/    (10 classes)
```

**Expected Results:**

Total: 1,688 images, 3,141 boxes

**Class Distribution:**

- Class 0 (KL0-a): 89 boxes (2.8%) - Osteophyte
- Class 1 (KL0-b): 10 boxes (0.3%) - Joint space
- Class 2 (KL1-a): 777 boxes (24.7%) - Osteophyte
- Class 3 (KL1-b): 22 boxes (0.7%) - Joint space
- Class 4 (KL2-a): 1,315 boxes (41.9%) - **Most common**
- Class 5 (KL2-b): 47 boxes (1.5%) - Joint space
- Class 6 (KL3-a): 536 boxes (17.1%) - Osteophyte
- Class 7 (KL3-b): 46 boxes (1.5%) - Joint space
- Class 8 (KL4-a): 279 boxes (8.9%) - Osteophyte
- Class 9 (KL4-b): 20 boxes (0.6%) - Joint space

**Insight:** "a" classes (osteophyte) >> "b" classes (joint space)

---

## 🔄 Stage 9: Create Filtered Datasets (No KL0)

**Purpose:** Create variants without KL0 for training comparison

**Commands:**

```powershell
# Create 4-class dataset (KL1-4 only)
.venv\Scripts\python.exe scripts\preprocessing\filter_kl0.py `
    --input processed\knee `
    --output processed\knee_4_class `
    --num_classes 5

# Create 8-class dataset (KL1-a through KL4-b only)
.venv\Scripts\python.exe scripts\preprocessing\filter_kl0.py `
    --input processed\knee_10_class `
    --output processed\knee_8_class `
    --num_classes 10

# Analyze filtered datasets
.venv\Scripts\python.exe tools\check_dataset\comprehensive_analysis.py `
    --dataset_dir processed\knee_4_class `
    --output analysis\knee_4_class

.venv\Scripts\python.exe tools\check_dataset\comprehensive_analysis.py `
    --dataset_dir processed\knee_8_class `
    --output analysis\knee_8_class
```

**Features:**

- Removes images with only KL0 labels
- Remaps remaining class IDs starting from 0
- 4-class: KL1→0, KL2→1, KL3→2, KL4→3
- 8-class: KL1-a→0, KL1-b→1, ..., KL4-a→6, KL4-b→7

**Expected Results:**

- ~1,603 images (95% retention)
- ~85 images filtered (only had KL0)

**Output:**

```
processed/knee_4_class/     # 4 classes, 1,603 images
processed/knee_8_class/     # 8 classes, 1,603 images
```

**Actual Results (from filtering):**

**4-class dataset:**

- ✅ Kept: 1,603 images (95.0%)
- 🗑️ Filtered: 85 images (5.0%)
- Total boxes: 3,042 (96.8% retention)
- Class distribution:
  - Class 0 (KL1): 799 boxes (26.3%)
  - Class 1 (KL2): 1,362 boxes (44.8%) - **Most common**
  - Class 2 (KL3): 582 boxes (19.1%)
  - Class 3 (KL4): 299 boxes (9.8%)

**8-class dataset:**

- ✅ Kept: 1,603 images (95.0%)
- 🗑️ Filtered: 85 images (5.0%)
- Total boxes: 3,042 (96.8% retention)
- Same distribution as 10-class but excluding KL0-a/b

---

## 🔄 Stage 10: Stratified Train/Val/Test Split

**Purpose:** Create stratified splits for all 4 dataset variants for model training

**Strategy:**

- Split ratio: 70% train / 15% val / 15% test
- Stratified sampling to balance class distribution
- Same seed (42) for reproducibility

**Commands:**

```powershell
# Split 5-class dataset
.venv\Scripts\python.exe scripts\data_preparation\split_dataset.py `
    --img_dir processed\knee\images `
    --label_dir processed\knee\labels `
    --out_dir processed\splits\knee_5_class `
    --train 0.7 --val 0.15 --test 0.15 --seed 42

# Split 10-class dataset
.venv\Scripts\python.exe scripts\data_preparation\split_dataset.py `
    --img_dir processed\knee_10_class\images `
    --label_dir processed\knee_10_class\labels `
    --out_dir processed\splits\knee_10_class `
    --train 0.7 --val 0.15 --test 0.15 --seed 42

# Split 4-class dataset
.venv\Scripts\python.exe scripts\data_preparation\split_dataset.py `
    --img_dir processed\knee_4_class\images `
    --label_dir processed\knee_4_class\labels `
    --out_dir processed\splits\knee_4_class `
    --train 0.7 --val 0.15 --test 0.15 --seed 42

# Split 8-class dataset
.venv\Scripts\python.exe scripts\data_preparation\split_dataset.py `
    --img_dir processed\knee_8_class\images `
    --label_dir processed\knee_8_class\labels `
    --out_dir processed\splits\knee_8_class `
    --train 0.7 --val 0.15 --test 0.15 --seed 42
```

**Output:**

```
processed/splits/
├── knee_5_class/
│   ├── train.txt (1,181 images)
│   ├── val.txt (253 images)
│   ├── test.txt (254 images)
│   └── split_info.json
├── knee_10_class/
│   ├── train.txt (1,181 images)
│   ├── val.txt (253 images)
│   ├── test.txt (254 images)
│   └── split_info.json
├── knee_4_class/
│   ├── train.txt (1,122 images)
│   ├── val.txt (240 images)
│   ├── test.txt (241 images)
│   └── split_info.json
└── knee_8_class/
    ├── train.txt (1,122 images)
    ├── val.txt (240 images)
    ├── test.txt (241 images)
    └── split_info.json
```

**Actual Results:**

All datasets split successfully with stratified class distribution:

| Dataset  | Train (70%) | Val (15%) | Test (15%) | Total |
| -------- | ----------- | --------- | ---------- | ----- |
| 5-class  | 1,181       | 253       | 254        | 1,688 |
| 10-class | 1,181       | 253       | 254        | 1,688 |
| 4-class  | 1,122       | 240       | 241        | 1,603 |
| 8-class  | 1,122       | 240       | 241        | 1,603 |

**Validation:**

- ✅ All classes present in each split
- ✅ Balanced distribution maintained
- ✅ Rare classes preserved (min 1-2 samples per class per split)
- ✅ No overlap between splits

---

## 📊 Complete Workflow Script

Run all stages sequentially to create all 4 dataset variants:

```powershell
# Stage 1: Analyze original dataset
.venv\Scripts\python.exe tools\check_dataset\comprehensive_analysis.py `
    --dataset_dir dataset\dataset_v0 `
    --output analysis\dataset_v0

# Stage 2: Crop knee regions
.venv\Scripts\python.exe scripts\preprocessing\crop_knee_regions.py `
    --dataset_dir dataset\dataset_v0 `
    --output_dir processed\knee `
    --margin 0.15 `
    --min_size 300

# Stage 3: Filter no-labels
.venv\Scripts\python.exe scripts\preprocessing\filter_no_labels.py `
    --input processed\knee

# Stage 4: Visualize samples (quality check)
.venv\Scripts\python.exe tools\check_dataset\visualize_samples.py `
    --img_dir processed\knee\images `
    --label_dir processed\knee\labels `
    --out_dir analysis\knee\visualizations `
    --color blue `
    --thickness 2

# Stage 5: Resize to 640x640
.venv\Scripts\python.exe tools\check_dataset\resize_images.py `
    --in_dir processed\knee\images `
    --out_dir processed\knee_5_class\images `
    --size 640

# Stage 6: Validate
.venv\Scripts\python.exe tools\check_dataset\validate_dataset.py `
    --img_dir processed\knee_5_class\images `
    --label_dir processed\knee_5_class\labels

# Stage 7: Final analysis
.venv\Scripts\python.exe tools\check_dataset\comprehensive_analysis.py `
    --dataset_dir processed\knee `
    --output analysis\knee
```

---

## 📁 File Structure After Pipeline

```
KLGrade/
├── dataset/dataset_v0/              # Original data
│   ├── images/                       # 1,473 full X-rays
│   ├── labels/                       # KL 0-4 (3,157 boxes)
│   ├── labels-knee/                  # Knee detection
│   └── labels-knee-2box/             # Left/Right knee
│
├── processed/                        # All processed datasets
│   ├── knee/                         # 5-class (KL0-4)
│   │   ├── images/                   # 1,688 crops WITH labels (training-ready)
│   │   ├── labels/                   # 5 classes, 3,141 boxes
│   │   ├── images-no-labels/         # 92 crops WITHOUT labels
│   │   ├── labels-no-labels/         # Empty labels
│   │   ├── crop_stats.json           # Cropping statistics
│   │   ├── skipped_files.json        # 14 skipped files (12 no knee, 2 too small)
│   │   ├── dropped_labels.json       # 1,062 dropped labels (far from knee)
│   │   └── no_label_files.json       # 92 filtered files list
│   │
│   ├── knee_5_class/                 # Resized 5-class dataset
│   │   ├── images/                   # 1,688 images at 640x640
│   │   └── labels/                   # Copied from knee/labels
│   │
│   ├── knee_10_class/                 # 10-class (KL0-a → KL4-b)
│   │   ├── images/                   # 1,688 crops (same as knee)
│   │   └── labels/                   # 10 classes
│   │
│   ├── knee_4_class/                  # 4-class (KL1-4, no KL0)
│   │   ├── images/                   # 1,603 crops
│   │   ├── labels/                   # 4 classes (remapped)
│   │   └── filter_stats.json
│   │
│   └── knee_8_class/                  # 8-class (KL1-a → KL4-b, no KL0)
│       ├── images/                   # 1,603 crops
│       ├── labels/                   # 8 classes (remapped)
│       └── filter_stats.json
│
├── analysis/                         # All analysis outputs (gitignored)
│   ├── dataset_v0/                   # Original dataset analysis
│   ├── knee/                         # 5-class analysis
│   ├── knee_10_class/                 # 10-class analysis
│   ├── knee_4_class/                  # 4-class analysis
│   └── knee_8_class/                  # 8-class analysis
│
├── scripts/preprocessing/
│   ├── crop_knee_regions.py         # Multi-knee cropping
│   ├── filter_no_labels.py          # Post-crop filtering
│   └── filter_kl0.py                # Remove KL0 classes
│
└── tools/check_dataset/
    ├── comprehensive_analysis.py
    ├── validate_dataset.py
    ├── visualize_samples.py
    ├── class_split_report.py        # A/B splitting
    ├── check_augment.py
    └── resize_images.py
```

---

## 🎯 Key Implementation Details

### Crop Naming Convention

- **Single knee:** `original_filename.jpg`
- **Multiple knees:** `original_filename_knee0.jpg`, `original_filename_knee1.jpg`

### Label Transformation

- **Keep criterion:** Box center within cropped region
- **Drop criterion:** Box center outside crop → logged
- **Coordinate transform:** Full image normalized → Crop normalized

### Quality Assurance

- All operations logged with detailed reasons
- JSON outputs for programmatic analysis
- Skipped files tracked with metadata
- Dropped labels documented with context

---

## 🔍 Verification Checklist

After running pipeline, verify:

- [ ] Number of crops matches expectation (~1,780 total)
- [ ] Crops with labels (~1,688, ~94.8%)
- [ ] Crops without labels separated to `images-no-labels/`
- [ ] Label retention rate ~84-100%
- [ ] No data quality errors in validation
- [ ] Class distribution maintained (check for severe imbalance)

---

## ⚠️ Troubleshooting

### Issue: Fewer crops than expected

**Check:** `skipped_files.json` for reasons
**Common causes:**

- Missing knee detection boxes
- Crop size too small (< min_size)

### Issue: Low label retention rate

**Check:** `dropped_labels.json` for dropped boxes
**Common causes:**

- KL boxes center outside knee region
- Knee crop doesn't contain lesion areas

### Issue: Many crops in no-labels folder

**Check:** `no_label_files.json` for affected files
**Possible reasons:**

- Healthy knees (no KL grades)
- Lesions outside detected knee region
- Annotation gaps in original dataset

---

## 📝 Notes

- **Reproducible:** All scripts use fixed random seeds where applicable
- **Idempotent:** Running pipeline multiple times produces same results
- **Logged:** Every decision tracked in JSON logs
- **Configurable:** All parameters exposed via CLI arguments

---

## 🚀 Next Steps After Pipeline

1. **Optional: Class Split (a/b):**

   ```powershell
   python tools/check_dataset/class_split_report.py `
       --labels-dir processed/knee/labels `
       --save-dir processed/knee/labels_new
   ```

2. **Data Splitting:**

   ```powershell
   python scripts/data_preparation/split_dataset.py `
       --image_dir processed/knee/images `
       --label_dir processed/knee/labels `
       --output_dir processed/knee/splits
   ```

3. **Training:** Use processed dataset for YOLO/DETR training

---

**Last Updated:** 2026-01-07
**Pipeline Version:** 1.0
