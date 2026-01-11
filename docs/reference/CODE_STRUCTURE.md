# Codebase Structure

This document outlines the refactored structure of the KLGrade project, designed to adhere to DRY (Don't Repeat Yourself) and SOLID principles.

## Directory Layout

```
KLGrade/
├── src/                      # Core Library Code
│   ├── api/                  # Inference API Logic
│   │   ├── inference.py      # YOLO Model Wrapper
│   │   ├── server.py         # FastAPI Server Definition
│   │   └── utils.py          # API Utilities (e.g., Image Reading)
│   ├── data/                 # Data Management
│   │   ├── filter.py         # Dataset Class Filtering Logic
│   │   ├── loader.py         # Dataset Loading Logic
│   │   ├── preprocessing.py  # Image Preprocessing (CLAHE, Blur, etc.)
│   │   ├── splitter.py       # Stratified Splitter Logic
│   │   └── utils.py          # Shared Data Utilities
│   ├── evaluation/           # Evaluation Logic
│   │   └── evaluator.py      # YOLO Evaluator Class
│   ├── training/             # Training Utilities
│   │   ├── trainer.py        # Common Training Setup
│   │   └── detr_trainer.py   # DETR Specific Training Logic
│   ├── config.py             # Global Configuration (Class Names, etc.)
│   └── datasets/             # (Legacy) PyTorch Dataset definitions
│
├── scripts/                  # Executable Scripts (Entry Points)
│   ├── api_server.py         # Start Inference Server (Use src.api)
│   ├── evaluate_all_models.py # Evaluate all trained models
│   ├── evaluate_yolo_standalone.py # Evaluate single model
│   ├── inference_api.py      # Run inference on single image
│   ├── run_full_pipeline.sh  # End-to-end data preparation
│   ├── data_preparation/     # Data Prep Scripts
│   │   ├── crop_knee_regions.py
│   │   ├── filter_dataset.py
│   │   ├── filter_kl0.py
│   │   ├── filter_no_labels.py
│   │   └── split_dataset.py
│   ├── training/             # Training Scripts
│   │   ├── train_yolo.py     # Simple Training
│   │   ├── train_yolo_enhanced.py # Enhanced Training (Preproc)
│   │   └── train_detr.py     # DETR Training Wrapper
│   └── legacy/               # Deprecated Scripts
│
├── docs/                     # Documentation
├── configs/                  # YOLO Training Configs (YAML)
├── dataset/                  # Raw Data
├── processed/                # Processed Data
├── splits/                   # Split Files (train.txt/val.txt/test.txt)
└── runs/                     # Training/Evaluation Results
```

## Key Modules

### `src.data`
Centralizes all data manipulation logic.
- **`StratifiedSplitter`**: Used by `scripts/data_preparation/split_dataset.py` to create balanced train/val/test splits.
- **`DatasetFilter`**: Used by `scripts/data_preparation/filter_dataset.py` to remove rare classes or filter specific labels.
- **`preprocessing`**: Contains `preprocess_image_clahe` and `balance_dataset_with_flip` used by `train_yolo_enhanced.py`.

### `src.evaluation`
- **`YOLOEvaluator`**: Wraps `ultralytics.YOLO`, handles temporary config creation for evaluation, and metric extraction. Used by `evaluate_yolo_standalone.py`.

### `src.api`
- **`YOLOModel`**: Wrapper around `ultralytics.YOLO` that handles prediction and class name mapping (from `src.config`).
- **`read_image_file`**: Robust image reading (supports DICOM, PNG, JPG).

## Usage Guide

### Data Preparation
Use scripts in `scripts/data_preparation/` which now utilize `src.data` for heavy lifting.
```bash
python scripts/data_preparation/split_dataset.py --help
```

### Training
Use scripts in `scripts/training/`.
```bash
python scripts/training/train_yolo.py --help
```

### Evaluation
Use `scripts/evaluate_all_models.py` to evaluate everything in `runs/`.
```bash
python scripts/evaluate_all_models.py
```
