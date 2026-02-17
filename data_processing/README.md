# Data Processing Pipeline

This directory contains all scripts for data preparation, preprocessing, and augmentation for the KLGrade project.

## Directory Structure

```
data_processing/
├── preparation/        # Raw data → YOLO format conversion
├── preprocessing/      # Image preprocessing (resize, CLAHE, etc.)
├── pipelines/          # End-to-end automated pipelines
├── validation/         # Data quality checks
└── examples/           # Example usage and comparisons
```

---

## 📁 preparation/

**Purpose**: Convert raw data to YOLO format, create splits, balance classes

### Key Scripts

- **`balance_dataset.py`** - Balance class distribution
- **`crop_knee_regions.py`** - Extract knee regions from full X-rays
- **`prepare_knee_crops.py`** - Prepare cropped knee dataset
- **`split_dataset.py`** - Create train/val/test splits
- **`create_splits_v2.py`** - Alternative splitting strategy
- **`filter_dataset.py`** - Filter dataset by criteria
- **`filter_kl0.py`** - Filter KL0 class
- **`filter_no_labels.py`** - Remove images without labels
- **`remap_labels.py`** - Remap class labels
- **`validate_dataset.py`** - Validate dataset integrity
- **`setup_balanced_training.py`** - Setup balanced training datasets
- **`run_balanced_training.sh`** - Run balanced training pipeline

### Usage Example
```bash
# Balance dataset
python data_processing/preparation/balance_dataset.py \
    --input datasets/raw/ \
    --output datasets/balanced/

# Create splits
python data_processing/preparation/split_dataset.py \
    --dataset datasets/balanced/ \
    --split 70/20/10
```

---

## 📁 preprocessing/

**Purpose**: Image preprocessing and augmentation

### Key Scripts

- **`preprocess_dataset.py`** - Preprocess full dataset
- **`preprocess_balanced.py`** - Preprocess balanced datasets
- **`preprocess_production.py`** - Production preprocessing pipeline

### Preprocessing Methods
- **Resize**: Resize to target dimensions
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization
- **Blur**: Gaussian blur (optional)
- **Sharpen**: Image sharpening (optional)

### Usage Example
```bash
# Preprocess balanced dataset
python data_processing/preprocessing/preprocess_balanced.py \
    --input datasets/balanced/ \
    --output datasets/processed_balanced/ \
    --method clahe
```

---

## 📁 pipelines/

**Purpose**: End-to-end automated data processing pipelines

### Pipeline Steps

1. **Step 1: Crop Knees** - `step_1_crop_knees.sh`
2. **Step 2: Generate Labels** - `step_2_generate_labels.sh`
3. **Step 3: Balance Classes** - `step_3_balance.sh`
4. **Step 4: Preprocess** - `step_4_preprocess_balanced.sh`
5. **Step 5: Create Splits** - `step_5_create_splits.sh`

### Full Pipeline
```bash
# Run complete pipeline
bash data_processing/pipelines/run_full_pipeline.sh
```

### Individual Steps
```bash
# Run specific step
bash data_processing/pipelines/step_1_crop_knees.sh
bash data_processing/pipelines/step_2_generate_labels.sh
# ... etc
```

---

## 📁 examples/

**Purpose**: Example scripts and preprocessing comparisons

### Scripts

- **`preprocessing_comparison.py`** - Compare different preprocessing methods
- **`preprocessing_comparison_knees.py`** - Compare preprocessing on knee crops
- **`preprocessing_custom.py`** - Custom preprocessing examples

### Usage
```bash
# Compare preprocessing methods
python data_processing/examples/preprocessing_comparison.py \
    --image sample.jpg \
    --output comparisons/
```

---

## 📁 validation/

**Purpose**: Data quality checks and validation

*(To be populated with validation scripts)*

---

## Common Workflows

### 1. Prepare New Dataset
```bash
# 1. Crop knees from full X-rays
python data_processing/preparation/crop_knee_regions.py \
    --input datasets/raw_xrays/ \
    --output datasets/knees_cropped/

# 2. Balance classes
python data_processing/preparation/balance_dataset.py \
    --input datasets/knees_cropped/ \
    --output datasets/balanced/

# 3. Create splits
python data_processing/preparation/split_dataset.py \
    --dataset datasets/balanced/ \
    --split 70/20/10

# 4. Preprocess
python data_processing/preprocessing/preprocess_balanced.py \
    --input datasets/balanced/ \
    --output datasets/processed_balanced/
```

### 2. Run Full Automated Pipeline
```bash
# One command to rule them all
bash data_processing/pipelines/run_full_pipeline.sh
```

---

## Notes & Important Information

### Environment Setup
```bash
# Always activate conda environment before running scripts
conda activate klgrade

# Verify environment
which python  # Should point to klgrade environment
python --version  # Should be Python 3.10+
```

### Running Scripts

#### 1. From Project Root
**IMPORTANT**: All scripts must be run from the project root directory:
```bash
cd /home/ngoductam/KLGrade

# ✅ Correct
python data_processing/preparation/balance_dataset.py --help

# ❌ Wrong (will cause import errors)
cd data_processing/preparation
python balance_dataset.py --help
```

#### 2. Script Help
All scripts support `--help` flag for detailed usage:
```bash
python data_processing/preparation/crop_knee_regions.py --help
python data_processing/preprocessing/preprocess_balanced.py --help
```

#### 3. Path Conventions
- **Input paths**: Can be relative or absolute
- **Output paths**: Will be created if they don't exist
- **Dataset paths**: Usually relative to project root (e.g., `datasets/...`)

### Common Issues & Solutions

#### Issue 1: Import Errors
```
ModuleNotFoundError: No module named 'src'
```
**Solution**: Run from project root, not from subdirectory

#### Issue 2: File Not Found
```
FileNotFoundError: [Errno 2] No such file or directory: 'datasets/...'
```
**Solution**: Check that you're in project root and path is correct

#### Issue 3: Permission Denied
```
PermissionError: [Errno 13] Permission denied
```
**Solution**: 
```bash
# For shell scripts
chmod +x data_processing/pipelines/*.sh

# For output directories
mkdir -p datasets/output_dir
```

### Pipeline Execution Order

For complete data processing, follow this order:

1. **Preparation** (Raw → YOLO format)
   ```bash
   # Crop knees, balance classes, create splits
   bash data_processing/pipelines/step_1_crop_knees.sh
   bash data_processing/pipelines/step_2_generate_labels.sh
   bash data_processing/pipelines/step_3_balance.sh
   ```

2. **Preprocessing** (Apply transformations)
   ```bash
   bash data_processing/pipelines/step_4_preprocess_balanced.sh
   ```

3. **Splitting** (Train/Val/Test)
   ```bash
   bash data_processing/pipelines/step_5_create_splits.sh
   ```

**OR** run all at once:
```bash
bash data_processing/pipelines/run_full_pipeline.sh
```

### Script Parameters

#### Common Parameters
- `--input` / `-i`: Input directory or file
- `--output` / `-o`: Output directory
- `--dataset` / `-d`: Dataset path
- `--split`: Split ratio (e.g., `70/20/10`)
- `--method`: Processing method (e.g., `resize`, `clahe`, `blur`)
- `--help` / `-h`: Show help message

#### Example with All Parameters
```bash
python data_processing/preprocessing/preprocess_balanced.py \
    --input datasets/balanced/knees_cropped/ \
    --output datasets/processed_balanced/knees_cropped/ \
    --method clahe \
    --size 640 \
    --verbose
```

### Monitoring Progress

Most scripts show progress bars and logs:
```bash
# Example output
Processing images: 100%|████████████| 1000/1000 [02:15<00:00,  7.38it/s]
✅ Processed 1000 images
📊 Saved to: datasets/processed_balanced/
```

### Data Quality Checks

After processing, always validate:
```bash
# Check dataset integrity
python data_processing/preparation/validate_dataset.py \
    --dataset datasets/processed_balanced/

# Check splits
ls -lh datasets/processed_balanced/train.txt
ls -lh datasets/processed_balanced/val.txt
ls -lh datasets/processed_balanced/test.txt
```

### Performance Tips

- **Parallel Processing**: Some scripts support `--workers` parameter
- **GPU Acceleration**: Use `--device cuda` if available
- **Batch Processing**: Process in batches for large datasets
- **Disk Space**: Ensure sufficient space (preprocessing can double dataset size)

---

## Related Documentation

- [Main README](../README.md)
- [Training Documentation](../docs/TRAINING.md)
- [Dataset Documentation](../docs/DATASETS.md)
