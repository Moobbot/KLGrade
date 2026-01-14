# Preprocessing Workflow - KLGrade Project

This document outlines the complete preprocessing workflow for the KLGrade project, from raw X-ray images to training-ready datasets.

## Overview

The preprocessing pipeline consists of 3 main stages:
1. **Data Preparation**: Knee cropping from full X-rays
2. **Preprocessing**: Apply blur, CLAHE, and other transformations
3. **Verification**: Analysis and visualization

**Alternative Workflow**: Generate 10-class labels for full X-rays (no cropping) - See Stage 0.

---

## Stage 0 (Alternative): Generate 10-Class Labels Without Cropping

### Purpose
Generate 10-class labels for the original full X-ray dataset without cropping.

### Command
```bash
python tools/check_dataset/class_split_report.py \
    --labels-dir datasets/dataset/dataset_v0/labels \
    --save-dir datasets/dataset/dataset_v0/labels_10_class \
    --limit 10
```

### Inputs
- `datasets/dataset/dataset_v0/labels/` - Original 5-class labels (KL0-4)

### Outputs
- `datasets/dataset/dataset_v0/labels_10_class/` - 10-class labels (0-9)

### What It Does
Splits each 5-class label into a/b variants based on shape:
- **-a (bone spike)**: w/h < 1.2 or area < 0.01 → class_id = base × 2
- **-b (joint space)**: w/h > 2.0 or area > 0.03 → class_id = base × 2 + 1

Mapping:
- Class 0 → 0 (KL0-a) or 1 (KL0-b)
- Class 1 → 2 (KL1-a) or 3 (KL1-b)
- Class 2 → 4 (KL2-a) or 5 (KL2-b)
- Class 3 → 6 (KL3-a) or 7 (KL3-b)
- Class 4 → 8 (KL4-a) or 9 (KL4-b)

### Use Case
When you want to train on full X-rays without knee cropping, but still need 10-class labels.

---

## Stage 1: Data Preparation - Knee Cropping

### Purpose
Crop knee regions from full X-ray images using existing knee bounding box labels.

### Command
```bash
python scripts/prepare_knee_crops.py \
    --input datasets/dataset/dataset_v0 \
    --output datasets/dataset/knees_cropped \
    --margin 0.15
```

### Inputs
- `datasets/dataset/dataset_v0/images/` - Full X-ray images
- `datasets/dataset/dataset_v0/labels/` - KL labels (5-class: KL0-4)
- `datasets/dataset/dataset_v0/labels-knee/` - Knee bounding boxes

### Outputs
- `datasets/dataset/knees_cropped/images/` - Cropped knee images (1783 images)
- `datasets/dataset/knees_cropped/labels/` - **5-class** labels (KL0-4)
- `datasets/dataset/knees_cropped/labels_new/` - **10-class** labels (KL0-a/b to KL4-a/b)
- `datasets/dataset/knees_cropped/labels_4class/` - **4-class** labels (KL1-4, filtered KL0)
- `datasets/dataset/knees_cropped/labels_8class/` - **8-class** labels (KL1-a/b to KL4-a/b, filtered KL0)
- `datasets/dataset/knees_cropped/labels-knee/` - Knee boxes (full crop)

### What It Does
1. Loads knee bounding boxes from `labels-knee/`
2. Crops knee regions with 15% margin
3. Transforms KL labels to crop space
4. **Automatically generates 10-class labels** from 5-class using shape classification:
   - **-a (bone spike)**: w/h < 1.2 or area < 0.01 → class_id = base × 2
   - **-b (joint space)**: w/h > 2.0 or area > 0.03 → class_id = base × 2 + 1
5. Creates filtered 4-class and 8-class variants

### Statistics
- Total images: 1,473 → 1,783 cropped knees
- Multi-knee images split into separate crops (e.g., `file_knee0.jpg`, `file_knee1.jpg`)

---

## Stage 1.5: Analyze Cropped Dataset

### Purpose
Generate statistical report for the cropped knee dataset.

### Command
```bash
PYTHONPATH=/home/ngoductam/KLGrade \
/home/ngoductam/miniconda3/envs/klgrade/bin/python \
scripts/analyzes/analyze_knee_dataset.py \
    --dataset datasets/dataset/knees_cropped
```

### Output
- `datasets/dataset/knees_cropped/dataset_statistics.txt` - Comprehensive statistics report

### Report Contents
- Image count and distribution
- Class distribution for all 4 label variants (5/10/4/8-class)
- Bounding box statistics
- Label retention rates

---

## Stage 2: Preprocessing - Apply Transformations

### Purpose
Apply various preprocessing methods (resize, blur, CLAHE) to cropped knee images.

### Command
```bash
echo "all" | PYTHONPATH=/home/ngoductam/KLGrade \
/home/ngoductam/miniconda3/envs/klgrade/bin/python \
scripts/preprocess_production.py
```

Or interactively select a preset when prompted.

### Preprocessing Presets

1. **`resize_only/`** - Basic (Resize to 640x640 only)
2. **`blur_clahe2/`** - Standard (Blur + CLAHE 2.0)
3. **`sharp_clahe4/`** - Legacy Sharp (No Blur + CLAHE 4.0)
4. **`blur_clahe2_notebook/`** - Notebook Method (Blur + CLAHE 2.0)

### Inputs
- `datasets/dataset/knees_cropped/` (any cropped dataset)

### Outputs
Each preset creates a complete dataset:
```
datasets/data_processed/{preset_name}/
├── images/          # Preprocessed images
├── labels/          # 5-class labels (copied)
├── labels_new/      # 10-class labels (copied)
├── labels_4class/   # 4-class labels (copied)
├── labels_8class/   # 8-class labels (copied)
└── labels-knee/     # Knee boxes (copied)
```

### Processing Time
- ~1 minute for 1,783 images × 4 presets

---

## Stage 2.5: Analyze Preprocessed Datasets

### Purpose
Generate statistics for each preprocessed dataset variant.

### Commands
```bash
# Analyze each preprocessing preset (full X-rays)
for preset in resize_only blur_clahe2 sharp_clahe4 blur_clahe2_notebook; do
    PYTHONPATH=/home/ngoductam/KLGrade \
    /home/ngoductam/miniconda3/envs/klgrade/bin/python \
    scripts/analyzes/analyze_knee_dataset.py \
        --dataset datasets/data_processed/$preset
done

# Analyze each preprocessing preset (cropped knees)
for preset in resize_only blur_clahe2 sharp_clahe4 blur_clahe2_notebook; do
    PYTHONPATH=/home/ngoductam/KLGrade \
    /home/ngoductam/miniconda3/envs/klgrade/bin/python \
    scripts/analyzes/analyze_knee_dataset.py \
        --dataset datasets/data_processed_knees/$preset
done
```

### Output
- `datasets/data_processed/{preset}/dataset_statistics.txt` for each preset (full X-rays)
- `datasets/data_processed_knees/{preset}/dataset_statistics.txt` for each preset (cropped knees)

---

## Stage 3: Visualization & Examples

### 3.1 Custom Preprocessing Examples

### Purpose
Demonstrate modular preprocessing capabilities on sample images.

### Command
```bash
PYTHONPATH=/home/ngoductam/KLGrade \
/home/ngoductam/miniconda3/envs/klgrade/bin/python \
examples/preprocessing_custom.py
```

### Input
- Sample images from `datasets/dataset/dataset_v0/images/`

### Outputs (saved to `datasets/data_examples/`)
- `basic_example.png` - Resize only
- `v0_example.png` - Blur + CLAHE 2.0
- `custom_example.png` - Custom pipeline demo
- `fully_custom_example.png` - Fully custom composition
- `augmented_example.png` - With augmentation
- `comparison/` - All presets side-by-side

### What It Does
Demonstrates 6 different preprocessing pipelines:
1. Basic (resize only)
2. v0 (standard blur + CLAHE)
3. Custom (configurable)
4. Fully custom composition
5. With augmentation
6. Comparison of all presets

---

### 3.2 Preprocessing Comparison - Full X-rays

### Purpose
Create before/after comparison visualizations for full X-ray dataset.

### Command
```bash
PYTHONPATH=/home/ngoductam/KLGrade \
/home/ngoductam/miniconda3/envs/klgrade/bin/python \
examples/preprocessing_comparison.py
```

### Input
- Sample images from `datasets/dataset/dataset_v0/images/`

### Outputs (saved to `datasets/data_examples/`)
- `comparison_raw_vs_processed.png` (2.2MB, 3 samples × 5 methods)
- `comparison_detailed.png` (714KB, with histograms and statistics)

### What It Shows
- Raw image vs all preprocessing methods side-by-side
- Pixel intensity histograms
- Statistical metrics (mean, std, range)

---

### 3.3 Preprocessing Comparison - Cropped Knees

### Purpose
Create before/after comparison visualizations for cropped knee dataset.

### Command
```bash
PYTHONPATH=/home/ngoductam/KLGrade \
/home/ngoductam/miniconda3/envs/klgrade/bin/python \
examples/preprocessing_comparison_knees.py
```

### Input
- Sample images from `datasets/dataset_knees_cropped/images/`

### Outputs (saved to `datasets/data_examples/knees_cropped/`)
- `comparison_raw_vs_processed.png` (2.2MB, 3 knee crops × 5 methods)
- `comparison_detailed.png` (758KB, with histograms and statistics)

### What It Shows
- Raw knee crop vs all preprocessing methods
- Pixel intensity histograms
- Statistical metrics for knee-specific data

---

## Complete Workflow Example

```bash
#!/bin/bash
# Complete preprocessing workflow

# === OPTION A: With Knee Cropping ===

# 0. (Optional) Generate 10-class for original dataset
python tools/check_dataset/class_split_report.py \
    --labels-dir datasets/dataset/dataset_v0/labels \
    --save-dir datasets/dataset/dataset_v0/labels_10_class \
    --limit 10

# 1. Crop knees from full X-rays
python scripts/prepare_knee_crops.py \
    --input datasets/dataset/dataset_v0 \
    --output datasets/dataset/knees_cropped \
    --margin 0.15

# 2. Analyze cropped dataset
PYTHONPATH=/home/ngoductam/KLGrade \
python scripts/analyzes/analyze_knee_dataset.py \
    --dataset datasets/dataset/knees_cropped

# 3. Apply all preprocessing presets
echo "all" | PYTHONPATH=/home/ngoductam/KLGrade \
python scripts/preprocess_production.py

# 3.1. Preprocess cropped knees
bash scripts/preprocess_knees_cropped.sh

# 4. Analyze each preprocessed dataset
for preset in resize_only blur_clahe2 sharp_clahe4 blur_clahe2_notebook; do
    PYTHONPATH=/home/ngoductam/KLGrade \
    python scripts/analyzes/analyze_knee_dataset.py \
        --dataset datasets/data_processed/$preset
    
    # Also analyze cropped knees
    PYTHONPATH=/home/ngoductam/KLGrade \
    python scripts/analyzes/analyze_knee_dataset.py \
        --dataset datasets/data_processed_knees/$preset
done

# 5. Generate visualization examples
PYTHONPATH=/home/ngoductam/KLGrade \
python examples/preprocessing_custom.py

# Generate comparisons for full X-rays
PYTHONPATH=/home/ngoductam/KLGrade \
python examples/preprocessing_comparison.py

# Generate comparisons for cropped knees
PYTHONPATH=/home/ngoductam/KLGrade \
python examples/preprocessing_comparison_knees.py

echo "✅ Preprocessing workflow complete!"
```

### Alternative Workflow (Full X-rays, No Cropping)

```bash
#!/bin/bash
# Preprocessing for full X-rays without knee cropping

# 1. Generate 10-class labels
python tools/check_dataset/class_split_report.py \
    --labels-dir datasets/dataset/dataset_v0/labels \
    --save-dir datasets/dataset/dataset_v0/labels_10_class \
    --limit 10

# 2. Preprocess full X-rays with selected preset
# (Use preprocess_production.py on dataset_v0 directly)

echo "✅ 10-class labels generated for full X-rays!"
```

---

## Output Directory Structure

```
KLGrade/
├── datasets/
│   ├── dataset/
│   │   └── dataset_v0/              # Original full X-rays
│   │       ├── images/              # 1473 full X-rays
│   │       ├── labels/              # 5-class KL labels
│   │       ├── labels_10_class/     # 10-class (Stage 0 - optional)
│   │       ├── labels-knee/         # Knee bounding boxes
│   │       └── labels_new/          # 10-class (original, may be incomplete)
│   │   
│   │   └── knees_cropped/           # Cropped knees (Stage 1)
│   │       ├── images/              # 1783 knee crops
│   │       ├── labels/              # 5-class
│   │       ├── labels_new/          # 10-class (auto-generated)
│   │       ├── labels_4class/       # 4-class (filtered)
│   │       ├── labels_8class/       # 8-class (filtered)
│   │       ├── labels-knee/         # Knee boxes
│   │       └── dataset_statistics.txt
│   │
│   ├── data_processed/              # Preprocessed full X-rays (Stage 2)
│   │   ├── resize_only/
│   │   ├── blur_clahe2/
│   │   ├── sharp_clahe4/
│   │   └── blur_clahe2_notebook/
│   │       ├── images/
│   │       ├── labels/
│   │       ├── labels_new/
│   │       ├── labels_4class/
│   │       ├── labels_8class/
│   │       ├── labels-knee/
│   │       └── dataset_statistics.txt
│   │
│   ├── data_processed_knees/        # Preprocessed cropped knees (Stage 2.1)
│   │   ├── resize_only/
│   │   ├── blur_clahe2/
│   │   ├── sharp_clahe4/
│   │   └── blur_clahe2_notebook/
│   │       ├── images/ (1783 knee crops)
│   │       ├── labels/
│   │       ├── labels_new/
│   │       ├── labels_4class/
│   │       ├── labels_8class/
│   │       ├── labels-knee/
│   │       └── dataset_statistics.txt
│   │
│   └── data_examples/               # Visualization examples (Stage 3)
│       ├── basic_example.png
│       ├── v0_example.png
│       ├── comparison_raw_vs_processed.png  # Full X-rays
│       ├── comparison_detailed.png          # Full X-rays
│       ├── knees_cropped/
│       │   ├── comparison_raw_vs_processed.png  # Cropped knees
│       │   └── comparison_detailed.png          # Cropped knees
│       └── comparison/
│           ├── basic.png
│           ├── v0.png
│           ├── v3_legacy.png
│           └── notebook.png
```

---

## Next Steps After Preprocessing

1. **Create train/val/test splits** for each dataset variant
2. **Choose preprocessing preset** based on analysis results
3. **Start training** with selected preprocessed dataset

## Notes

- **Environment**: Requires `klgrade` conda environment
- **PYTHONPATH**: Must be set to project root for imports to work
- **Class Variants**: All 4 class variants (5/10/4/8) are maintained through the entire pipeline
- **10-Class Generation**: Automatically performed during knee cropping (no need for separate tool)
- **Flexibility**: Can mix and match preprocessing presets with different class variants
