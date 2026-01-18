# Data Processing Log - KLGrade Project
Date: 2026-01-14
Session: Preprocessing Modular Refactoring & Dataset Preparation

## Summary
Successfully refactored preprocessing into modular architecture and prepared multiple dataset variants with different preprocessing methods.

---

## Stage 0: Generate 10-Class Labels for Full X-rays

**Date**: 2026-01-14 05:50
**Command**:
```bash
python tools/check_dataset/class_split_report.py \
    --labels-dir datasets/dataset_v0/labels \
    --save-dir datasets/dataset_v0/labels_10_class \
    --limit 10
```

**Input**:
- `datasets/dataset_v0/labels/` (1,473 files, 5-class)

**Output**:
- `datasets/dataset_v0/labels_10_class/` (1,473 files, 10-class)

**Result**: ✅ Success
- Generated 10-class labels from 5-class using shape classification
- Mapping: class 0→0,1 | 1→2,3 | 2→4,5 | 3→6,7 | 4→8,9

---

## Stage 1: Knee Cropping from Full X-rays

**Date**: 2026-01-14 04:42
**Command**:
```bash
python scripts/prepare_knee_crops.py \
    --input datasets/dataset_v0 \
    --output datasets/dataset_knees_cropped \
    --margin 0.15
```

**Input**:
- `datasets/dataset_v0/images/` (1,473 full X-ray images)
- `datasets/dataset_v0/labels/` (5-class KL labels)
- `datasets/dataset_v0/labels-knee/` (knee bounding boxes)

**Output**:
- `datasets/dataset_knees_cropped/images/` (1,783 cropped knee images)
- `datasets/dataset_knees_cropped/labels/` (5-class)
- `datasets/dataset_knees_cropped/labels_10_class/` (10-class, auto-generated)
- `datasets/dataset_knees_cropped/labels_4_class/` (4-class, filtered KL0)
- `datasets/dataset_knees_cropped/labels_8_class/` (8-class, filtered KL0-a/b)
- `datasets/dataset_knees_cropped/labels-knee/` (knee boxes)

**Processing Details**:
- Total images: 1,473 → 1,783 knee crops
- Images with knees: 1,461
- Images without knees: 12
- Multi-knee images split (e.g., `file_knee0.jpg`, `file_knee1.jpg`)
- 10-class labels generated using shape classification:
  - -a (bone spike): w/h < 1.2 or area < 0.01
  - -b (joint space): w/h > 2.0 or area > 0.03

**Result**: ✅ Success

---

## Stage 1.5: Analyze Cropped Dataset

**Date**: 2026-01-14 04:45
**Command**:
```bash
PYTHONPATH=/home/ngoductam/KLGrade \
python scripts/analyzes/analyze_knee_dataset.py \
    --dataset datasets/dataset_knees_cropped
```

**Output**:
- `datasets/dataset_knees_cropped/dataset_statistics.txt`

**Statistics**:
- Total images: 1,783
- 5-class distribution: KL0 (3.2%), KL1 (25.4%), KL2 (43.3%), KL3 (18.5%), KL4 (9.6%)
- 10-class distribution: Full 0-9 with -a/-b variants
  - KL0-a: 90 (2.9%), KL0-b: 10 (0.3%)
  - KL1-a: 778 (24.7%), KL1-b: 21 (0.7%)
  - KL2-a: 1315 (41.8%), KL2-b: 48 (1.5%)
  - KL3-a: 537 (17.1%), KL3-b: 45 (1.4%)
  - KL4-a: 283 (9.0%), KL4-b: 20 (0.6%)

**Result**: ✅ Success

---

## Stage 2: Preprocessing Full X-rays

**Date**: 2026-01-14 05:44-05:45
**Command**:
```bash
echo "all" | PYTHONPATH=/home/ngoductam/KLGrade \
python scripts/preprocess_production.py
```

**Input**:
- `datasets/dataset_v0/images/` (1,473 images)

**Output**: 4 preprocessed datasets
1. `datasets/data_processed/resize_only/` - Resize to 640x640 only
2. `datasets/data_processed/blur_clahe2/` - Blur + CLAHE 2.0 (standard)
3. `datasets/data_processed/sharp_clahe4/` - No Blur + CLAHE 4.0 (legacy sharp)
4. `datasets/data_processed/blur_clahe2_notebook/` - Blur + CLAHE 2.0 (notebook method)

**Processing Details**:
- Total images processed: 5,892 (1,473 × 4 presets)
- Processing time: ~1 minute
- Each preset includes: images/ + labels/ + labels-knee/

**Result**: ✅ Success

---

## Stage 2.1: Preprocessing Cropped Knees

**Date**: 2026-01-14 05:51-05:52
**Command**:
```bash
bash scripts/preprocess_knees_cropped.sh
```

**Input**:
- `datasets/dataset_knees_cropped/images/` (1,783 cropped knee images)

**Output**: 4 preprocessed datasets
1. `datasets/data_processed_knees/resize_only/`
2. `datasets/data_processed_knees/blur_clahe2/`
3. `datasets/data_processed_knees/sharp_clahe4/`
4. `datasets/data_processed_knees/blur_clahe2_notebook/`

**Processing Details**:
- Total images processed: 7,132 (1,783 × 4 presets)
- Processing time: ~1.5 minutes
- Each preset includes: images/ + labels/ + labels_10_class/ + labels_4_class/ + labels_8_class/ + labels-knee/

**Result**: ✅ Success

---

## Stage 2.5: Analyze All Preprocessed Datasets

**Date**: 2026-01-14 05:54
**Commands**:
```bash
# Analyze preprocessed cropped knees
for preset in resize_only blur_clahe2 sharp_clahe4 blur_clahe2_notebook; do
    PYTHONPATH=/home/ngoductam/KLGrade \
    python scripts/analyzes/analyze_knee_dataset.py \
        --dataset datasets/data_processed_knees/$preset
done
```

**Output**:
- `datasets/data_processed_knees/{preset}/dataset_statistics.txt` (4 files)

**Result**: ✅ Success
- All statistics consistent across presets (labels unchanged)

---

## Stage 3: Visualization Examples

**Date**: 2026-01-14 03:31-03:33, 07:36

### 3.1 Custom Preprocessing Examples
**Command**:
```bash
PYTHONPATH=/home/ngoductam/KLGrade \
python examples/preprocessing_custom.py
```

**Output** (`datasets/data_examples/`):
- `basic_example.png`
- `v0_example.png`
- `custom_example.png`
- `fully_custom_example.png`
- `augmented_example.png`
- `comparison/` (4 preset comparisons)

**Result**: ✅ Success

### 3.2 Preprocessing Comparison - Full X-rays
**Date**: 2026-01-14 07:36
**Command**:
```bash
PYTHONPATH=/home/ngoductam/KLGrade \
python examples/preprocessing_comparison.py
```

**Output** (`datasets/data_examples/`):
- `comparison_raw_vs_processed.png` (2.2MB, 3 samples × 5 methods)
- `comparison_detailed.png` (714KB, with histograms and statistics)

**Processing Details**:
- 3 sample full X-rays processed
- 5 preprocessing methods compared: Raw, Basic, v0, v3 Legacy, Notebook
- Includes pixel intensity histograms and statistics (mean, std, range)

**Result**: ✅ Success

### 3.3 Preprocessing Comparison - Cropped Knees
**Date**: 2026-01-14 07:36
**Command**:
```bash
PYTHONPATH=/home/ngoductam/KLGrade \
python examples/preprocessing_comparison_knees.py
```

**Output** (`datasets/data_examples/knees_cropped/`):
- `comparison_raw_vs_processed.png` (2.2MB, 3 knee crops × 5 methods)
- `comparison_detailed.png` (758KB, with histograms and statistics)

**Processing Details**:
- 3 sample cropped knee images processed
- 5 preprocessing methods compared: Raw, Basic, v0, v3 Legacy, Notebook
- Knee-specific preprocessing analysis

---

## Stage 4: Data Balancing & Filtering

**Date**: 2026-01-17

### 4.1: Filter Empty Labels from Cropped Dataset
**Command**:
```bash
PYTHONPATH=/home/ngoductam/KLGrade \
python scripts/data_preparation/filter_no_labels.py \
    --input datasets/dataset_knees_cropped
```

**Input**:
- `datasets/dataset_knees_cropped/` (1,783 crops)

**Output**:
- `datasets/dataset_knees_cropped/images/` (1,691 clean images)
- `datasets/dataset_knees_cropped/images-no-labels/` (92 filtered)
- `datasets/dataset_knees_cropped/labels-no-labels/` (92 empty labels)
- `datasets/dataset_knees_cropped/no_label_files.json` (filter log)

**Result**: ✅ Success
- Filtered 92 images without KL grade labels
- Clean dataset: 1,691 images with 3,147 instances

---

### 4.2: Balance Dataset via Oversampling
**Command**:
```bash
PYTHONPATH=/home/ngoductam/KLGrade \
python scripts/balance_dataset.py \
    --input-images datasets/dataset_knees_cropped/images \
    --input-labels datasets/dataset_knees_cropped/labels \
    --output-dir datasets/dataset_knees_cropped_balanced \
    --num-classes 5 \
    --aux-labels datasets/dataset_knees_cropped/labels_10_class \
                 datasets/dataset_knees_cropped/labels-knee
```

**Input**:
- Unbalanced dataset: 1,691 images, 3,147 instances
- Class distribution: KL0 (3.2%), KL1 (25.4%), KL2 (43.3%), KL3 (18.5%), KL4 (9.6%)

**Output**:
- `datasets/dataset_knees_cropped_balanced/` (4,737 images, 6,814 instances)
- All label variants synced: labels/, labels_10_class/, labels-knee/
- `balance_report.txt`

**Balancing Strategy**:
- Oversampling via horizontal flip augmentation
- Target: 20% per class (1,363 instances each)
- Minority classes augmented to match majority

**Result**: ✅ Success
- Balanced dataset: 4,737 images (4,645 after filtering)
- Perfect balance: KL0-4 @ 20% each

---

### 4.3: Filter Empty Labels from Balanced Dataset
**Command**:
```bash
PYTHONPATH=/home/ngoductam/KLGrade \
python scripts/data_preparation/filter_no_labels.py \
    --input datasets/dataset_knees_cropped_balanced
```

**Output**:
- Clean balanced dataset: 4,645 images
- Filtered: 92 empty labels (same as original)

**Result**: ✅ Success

---

### 4.4: Preprocess Balanced Dataset
**Command**:
```bash
bash scripts/preprocess_knees_balanced.sh
```

**Input**:
- `datasets/dataset_knees_cropped_balanced/` (4,645 images)

**Output**: 4 balanced preprocessed datasets
1. `datasets/data_processed_balanced/resize_only/`
2. `datasets/data_processed_balanced/blur_clahe2/`
3. `datasets/data_processed_balanced/sharp_clahe4/`
4. `datasets/data_processed_balanced/blur_clahe2_notebook/`

**Processing Details**:
- Total images processed: 18,580 (4,645 × 4 presets)
- Each preset includes all label variants
- Processing time: ~3 minutes

**Result**: ✅ Success

---

### 4.5: Filter Processed Datasets
**Command**:
```bash
bash scripts/filter_all_processed_datasets.sh
```

**Datasets Filtered**:
- `data_processed_balanced/*` (4 presets)
- Filtered 92 empty labels from each

**Result**: ✅ Success
- All preprocessed datasets now clean

---

## Stage 5: Comprehensive Analysis

**Date**: 2026-01-17

### 5.1: Add Bbox Analysis Visualization
**Tool Enhanced**: `tools/check_dataset/comprehensive_analysis.py`

**New Visualizations**:
- Width/Height/Area distributions
- Aspect ratio histogram (for anchor box optimization)
- Single/Multiple objects per image
- Width vs Height scatter (colored by area)
- Object center heatmap

**Result**: ✅ Success

---

### 5.2: Analyze All Datasets
**Command**:
```bash
bash scripts/run_comprehensive_analysis_all.sh
```

**Datasets Analyzed** (11 total):
1. `datasets/dataset_v0` (full X-rays)
2. `datasets/dataset_knees_cropped` (unbalanced)
3. `datasets/dataset_knees_cropped_balanced` (balanced)
4-7. `datasets/data_processed/*` (4 presets, unbalanced)
8-11. `datasets/data_processed_balanced/*` (4 presets, balanced)

**Output for Each Dataset**:
- `analysis/{dataset}/bbox_analysis.png` ⭐ NEW!
- `analysis/{dataset}/class_distribution.png`
- `analysis/{dataset}/analysis_report.json`
- `analysis/{dataset}/ANALYSIS_REPORT.md` (summary)
- `analysis/{dataset}/DETAILED_ANALYSIS_REPORT.md`

**Result**: ✅ Success

---

## Stage 6: Integration & Automation

**Date**: 2026-01-17

### 6.1: Integrate Filter into prepare_knee_crops.py
**Enhancement**:
- Added `filter_empty_labels()` function
- Filtering runs by default after cropping
- Use `--skip-filter` flag to disable

**Result**: ✅ Success
- Cleaner workflow
- No separate filter step needed
- Direct function call (no subprocess)

---

## Code Changes Made (Stages 4-6)

### New Modules Created
1. **`src/data/balancing/`** - Data balancing pipeline
   - `validators.py` - Data validation utilities
   - `sampler.py` - Class balancing and oversampling
   - `augmentor.py` - Augmentation with label adjustment
   - `yolo_utils.py` - YOLO-specific transformations
   - `__init__.py` - Package exports

2. **Scripts**:
   - `scripts/balance_dataset.py` - Balance dataset via oversampling
   - `scripts/validate_dataset.py` - Validate dataset integrity
   - `scripts/data_preparation/filter_no_labels.py` - Filter empty labels
   - `scripts/preprocess_knees_balanced.sh` - Preprocess balanced dataset
   - `scripts/filter_all_processed_datasets.sh` - Batch filter processed datasets
   - `scripts/run_comprehensive_analysis_all.sh` - Analyze all datasets

### Enhanced Tools
1. **`tools/check_dataset/comprehensive_analysis.py`**
   - Added bbox analysis visualization (8-panel analysis)
   - Image statistics analysis
   - Multiple label directory support

### Modified Files
1. **`scripts/prepare_knee_crops.py`**
   - Integrated `filter_empty_labels()` function
   - Auto-filtering by default
   - Added `--skip-filter` option
   - Saves crop_report.txt

2. **Documentation**:
   - `PREPROCESSING_WORKFLOW.md` - Added Stage 1.2 (filtering)
   - `DATA_BALANCING_ANALYSIS.md` - Complete rewrite (implementation summary)

---

## Final Dataset Structure (Updated)

```
datasets/
├── dataset/
│   ├── dataset_v0/                 # Original full X-rays
│   │   ├── images/                 # 1,473 full X-rays
│   │   ├── labels/                 # 5-class KL labels
│   │   ├── labels_10_class/        # 10-class (Stage 0)
│   │   ├── labels-knee/            # Knee bounding boxes
│   │   └── labels_10_class/             # 10-class (original)
│   │
│   ├── dataset_knees_cropped/      # Cropped knees (Stage 1)
│   │   ├── images/                 # 1,691 clean crops
│   │   ├── images-no-labels/       # 92 filtered (Stage 4.1) ⭐ NEW
│   │   ├── labels/                 # 5-class
│   │   ├── labels-no-labels/       # 92 empty labels ⭐ NEW
│   │   ├── labels_10_class/             # 10-class
│   │   ├── labels_4_class/          # 4-class (filtered)
│   │   ├── labels_8_class/          # 8-class (filtered)
│   │   ├── labels-knee/            # Knee boxes
│   │   ├── crop_report.txt
│   │   ├── dataset_statistics.txt
│   │   └── no_label_files.json     ⭐ NEW
│   │
│   └── dataset_knees_cropped_balanced/  # Balanced (Stage 4.2) ⭐ NEW
│       ├── images/                     # 4,645 clean
│       ├── images-no-labels/           # 92 filtered
│       ├── labels/                     # Balanced 5-class
│       ├── labels-no-labels/           # Empty labels
│       ├── labels_10_class/                 # Balanced 10-class
│       ├── labels-knee/                # Synced knee boxes
│       ├── balance_report.txt
│       ├── dataset_statistics.txt
│       └── no_label_files.json
│
├── data_processed/                 # Preprocessed full X-rays (Stage 2)
│   ├── resize_only/
│   ├── blur_clahe2/
│   ├── sharp_clahe4/
│   └── blur_clahe2_notebook/
│       └── (each has images/ + labels/)
│
├── data_processed_knees/           # Preprocessed cropped knees (Stage 2.1)
│   ├── resize_only/
│   ├── blur_clahe2/
│   ├── sharp_clahe4/
│   └── blur_clahe2_notebook/
│       └── (each has images/ + 5 label variants)
│
├── data_examples/                  # Visualization examples
│   ├── *.png (example outputs)
│   ├── knees_cropped/
│   │   ├── comparison_raw_vs_processed.png
│   │   └── comparison_detailed.png
│   └── comparison/ (preset comparisons)
└── analysis/                      # Comprehensive analysis (Stage 5) ⭐ NEW
    ├── dataset_v0/
    ├── knees_cropped/
    ├── knees_cropped_balanced/
    ├── processed/
    │   ├── resize_only/
    │   ├── blur_clahe2/
    │   ├── sharp_clahe4/
    │   └── blur_clahe2_notebook/
    └── processed_balanced/
        └── (4 presets)
                └── (each: bbox_analysis.png, class_distribution.png, reports)
```

---

## Summary Statistics (Updated)

### Dataset Sizes
- Original full X-rays: 1,473 images
- **Cropped knees (clean)**: 1,691 images (filtered from 1,783)
- **Balanced knees (clean)**: 4,645 images (filtered from 4,737)
- Preprocessed unbalanced: 6,764 images (1,691 × 4 presets)
- **Preprocessed balanced**: 18,580 images (4,645 × 4 presets) ⭐ NEW
- **Total training-ready images**: 25,344

### Label Variants
- 5-class: KL0-4 (standard)
- 10-class: KL0-a/b to KL4-a/b (shape-based split)
- 4-class: KL1-4 (filtered KL0)
- 8-class: KL1-a/b to KL4-a/b (filtered KL0)

### Class Balance
**Unbalanced** (1,691 images):
- KL0: 3.2%, KL1: 25.4%, KL2: 43.3%, KL3: 18.5%, KL4: 9.6%

**Balanced** (4,645 images):
- KL0-4: 20% each (~1,363 instances per class)

---
