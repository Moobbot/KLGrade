# Full Pipeline Execution Guide

## Overview

This script regenerates the entire KLGrade dataset from raw data to ready-to-train YOLO configs.

## Prerequisites

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Ensure raw data exists
ls dataset/dataset_v1/images  # Should have .jpg files
ls dataset/dataset_v1/labels  # Should have .txt files
```

## Quick Start

```bash
# Make script executable
chmod +x scripts/run_full_pipeline.sh

# Run the entire pipeline
./scripts/run_full_pipeline.sh
```

## Pipeline Stages

### Stage 1: Initial Analysis

- Analyzes raw dataset statistics
- Generates class distribution reports

### Stage 2: Knee Cropping

- Crops knee regions from full X-ray images
- Preserves bounding box annotations

### Stage 3: Image Resizing

- Resizes all images to 640x640
- Adjusts bounding box coordinates

### Stage 4: Standard Structure

- Creates YOLO-compatible directory structure
- `dataset_yolo/images/` and `dataset_yolo/labels/`

### Stage 5: Stratified Splitting (5-class)

- Train: 70% (1181 images)
- Val: 15% (253 images)
- Test: 15% (254 images)

### Stage 6: 10-class Dataset

- Splits each KL grade into A (osteophyte) and B (joint space)
- Maintains same images, different labels

### Stage 7: 4-class Dataset

- Filters out KL0 (healthy) cases
- Only KL1-4 remain

### Stage 8: 8-class Dataset

- 10-class without KL0-a and KL0-b
- Only diseased cases with A/B split

### Stage 9: Fix Split Paths

- Converts relative paths to absolute paths
- Ensures cross-platform compatibility

### Stage 10: Verification

- Checks dataset integrity
- Verifies label format
- Counts images per split

### Stage 11: Analysis Reports

- Generates final statistics
- Creates visualization plots

## Output Structure

```
processed/
├── knee/
│   ├── images/              # Raw cropped images
│   ├── labels/              # 5-class labels
│   ├── images_640/          # Resized images
│   ├── labels_640/          # Adjusted labels
│   └── dataset_yolo/        # Standard YOLO structure
│       ├── images/
│       └── labels/
├── knee_10_class/
│   ├── images/
│   └── labels/
├── knee_4_class/
│   ├── images/
│   └── labels/
└── knee_8_class/
    ├── images/
    └── labels/

splits/
├── knee_5_class/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
├── knee_10_class/
├── knee_4_class/
└── knee_8_class/

configs/
├── yolo_5_class_baseline.yaml
├── yolo_5_class_conservative.yaml
├── yolo_10_class_baseline.yaml
├── yolo_10_class_conservative.yaml
├── yolo_4_class_baseline.yaml
├── yolo_4_class_conservative.yaml
├── yolo_8_class_baseline.yaml
└── yolo_8_class_conservative.yaml
```

## Execution Time

- **Stage 1-3**: ~5-10 minutes (image processing)
- **Stage 4-8**: ~2-5 minutes (dataset creation)
- **Stage 9-11**: ~1-2 minutes (verification)
- **Total**: ~10-20 minutes depending on hardware

## Troubleshooting

### Error: "Image directory not found"

```bash
# Check raw data location
ls dataset/dataset_v1/images
# If missing, extract your raw dataset first
```

### Error: "Module not found"

```bash
# Ensure virtual environment is activated
source .venv/bin/activate
# Install dependencies
pip install -r requirements.txt
```

### Error: "Permission denied"

```bash
# Make script executable
chmod +x scripts/run_full_pipeline.sh
```

## Partial Re-runs

If you only need to regenerate specific parts:

```bash
# Only fix split paths
python tools/fix_splits_paths.py

# Only verify datasets
python scripts/analysis/check_dataset.py --split_dir splits/knee_5_class

# Only create 10-class variant
# (Run stages 6, 9-11 manually)
```

## Notes

- The script uses `set -e` to stop on first error
- All intermediate files are preserved for debugging
- Logs are printed to stdout (redirect to file if needed: `./run_full_pipeline.sh > pipeline.log 2>&1`)
- Safe to re-run (will overwrite existing processed data)
