# KLGrade Training Workflow

Complete step-by-step guide from data preprocessing to API deployment.

---

## 📚 Related Documentation

- **[PREPROCESSING_WORKFLOW.md](PREPROCESSING_WORKFLOW.md)** - Detailed preprocessing guide with examples
- **[DATA_BALANCING_ANALYSIS.md](DATA_BALANCING_ANALYSIS.md)** - Data balancing strategies and analysis
- **[DATA_PROCESSING_LOG.md](DATA_PROCESSING_LOG.md)** - Processing history and dataset statistics
- **[README.md](README.md)** - Project overview and quick start

---

## Overview

```
Raw Data → Preprocessing → Balancing → Training → Evaluation → API
```

---

## Prerequisites

```bash
# Activate conda environment
conda activate klgrade

# Verify GPU
python scripts/testing/test_gpu.py
```

---

## Stage 1: Data Preprocessing

> 📖 **See also**: [PREPROCESSING_WORKFLOW.md](PREPROCESSING_WORKFLOW.md) for detailed preprocessing guide

### 1.1 Prepare Knee Crops

Extract knee regions from full X-ray images using YOLO detection model.

```bash
python scripts/preprocessing/prepare_knee_crops.py \
    --input datasets/dataset/dataset_v0 \
    --output datasets/dataset_knees_cropped \
    --model models/knee_detector.pt \
    --conf 0.25
```

**Output**:
- `datasets/dataset_knees_cropped/images/` - Cropped knee images
- `datasets/dataset_knees_cropped/labels/` - 5-class labels (KL0-4)
- `datasets/dataset_knees_cropped/labels_4_class/` - 4-class labels (KL1-4)
- `datasets/dataset_knees_cropped/labels_8_class/` - 8-class labels
- `crop_report.txt` - Statistics

**Verify**:
```bash
ls -lh datasets/dataset_knees_cropped/
```

---

### 1.2 Filter Empty Labels (Auto-applied)

Empty label filtering is automatically integrated in `prepare_knee_crops.py`.

To skip filtering:
```bash
python scripts/preprocessing/prepare_knee_crops.py --skip-filter ...
```

**Manual filtering** (if needed):
```bash
python scripts/data_preparation/filter_no_labels.py \
    --dataset datasets/dataset_knees_cropped
```

---

## Stage 2: Data Balancing (Optional)

> 📖 **See also**: [DATA_BALANCING_ANALYSIS.md](DATA_BALANCING_ANALYSIS.md) for balancing strategies and analysis

Balance dataset using oversampling with augmentation.

```bash
python scripts/data_preparation/balance_dataset.py \
    --input datasets/dataset_knees_cropped \
    --output datasets/dataset_knees_cropped_balanced \
    --target-samples 500 \
    --augment
```

**Output**:
- `datasets/dataset_knees_cropped_balanced/` - Balanced dataset
- `balance_report.txt` - Class distribution before/after

**Verify balance**:
```bash
python scripts/analyzes/analyze_knee_dataset.py \
    --dataset datasets/dataset_knees_cropped_balanced
```

---

## Stage 3: Dataset Splits & Configs

### 3.1 Generate Train/Val/Test Splits

```bash
# Run automated script to create all splits and update configs
bash scripts/pipelines/regenerate_configs_and_splits.sh
```

This creates:
- `datasets/dataset_knees_cropped/train.txt` (70%)
- `datasets/dataset_knees_cropped/val.txt` (15%)
- `datasets/dataset_knees_cropped/test.txt` (15%)
- Plus splits for 4-class, 8-class, 10-class variants

**Manual split** (if needed):
```bash
python scripts/data_preparation/split_dataset.py \
    --img_dir datasets/dataset_knees_cropped/images \
    --label_dir datasets/dataset_knees_cropped/labels \
    --out_dir datasets/dataset_knees_cropped \
    --train 0.7 --val 0.15 --test 0.15 \
    --seed 42
```

### 3.2 Verify Configs

```bash
# Check if all configs point to correct datasets
for config in configs/yolo_*_class_*.yaml; do
    echo "Config: $(basename $config)"
    grep "^val:" "$config"
done
```

---

## Stage 4: Model Training

### 4.1 Quick Test (1 config, 1 epoch)

```bash
# Test a single config to verify setup
python scripts/testing/test_configs.py \
    --configs configs/yolo_5_class_baseline.yaml \
    --epochs 1 \
    --batch 8
```

### 4.2 Full Training - Single Config

```bash
# Train 5-class baseline
yolo detect train \
    data=configs/yolo_5_class_baseline.yaml \
    epochs=100 \
    batch=16 \
    imgsz=640 \
    device=0 \
    project=runs/detect \
    name=knee_5_class_baseline
```

### 4.3 Full Training - All Configs

```bash
# Test all 10 configs (quick 2-epoch test)
python scripts/testing/test_configs.py --preset all --epochs 2

# Or train all configs properly (100 epochs each)
bash scripts/training/train_all_configs.sh
```

**Available configs**:
- `yolo_4_class_baseline.yaml` - 4 classes, no augmentation
- `yolo_4_class_conservative.yaml` - 4 classes, light augmentation
- `yolo_5_class_baseline.yaml` - 5 classes (KL0-4), no augmentation
- `yolo_5_class_conservative.yaml` - 5 classes, light augmentation
- `yolo_8_class_baseline.yaml` - 8 classes (sub-grades)
- `yolo_8_class_conservative.yaml` - 8 classes, light augmentation
- `yolo_10_class_baseline.yaml` - 10 classes (all sub-grades)
- `yolo_10_class_conservative.yaml` - 10 classes, light augmentation

**Monitor training**:
```bash
# View logs
tail -f runs/detect/knee_5_class_baseline/train.log

# TensorBoard (if enabled)
tensorboard --logdir runs/detect
```

---

## Stage 5: Model Evaluation

### 5.1 Evaluate Single Model

```bash
python scripts/evaluation/evaluate_yolo_standalone.py \
    --model runs/detect/knee_5_class_baseline/weights/best.pt \
    --data configs/yolo_5_class_baseline.yaml \
    --split test \
    --batch 16
```

### 5.2 Evaluate All Models

```bash
python scripts/evaluation/evaluate_all_models.py
```

**Output**:
- `runs/evaluate/<model_name>/` - Evaluation results
- Confusion matrices, PR curves, metrics JSON

### 5.3 Compare Results

```bash
python scripts/evaluation/summarize_experiments.py
```

Creates summary table comparing all models:
- mAP@50, mAP@50-95
- Precision, Recall
- Per-class metrics

---

## Stage 6: Visualization & Analysis

### 6.1 Comprehensive Dataset Analysis

```bash
# Analyze all datasets
bash scripts/analyzes/run_comprehensive_analysis_all.sh
```

### 6.2 GradCAM Visualization

```bash
python scripts/visualization/visualize_gradcam.py \
    --source datasets/dataset/dataset_v0/images/sample.jpg \
    --knee-model models/knee_detector.pt \
    --grade-model runs/detect/knee_5_class_baseline/weights/best.pt
```

### 6.3 Prediction Visualization

```bash
python scripts/visualization/visualize_manual.py \
    --prediction-json prediction.json \
    --image-dir datasets/dataset/dataset_v0/images \
    --output prediction_vis.jpg
```

---

## Stage 7: API Deployment

### 7.1 Test API Locally

```bash
# Start API server
python src/api/app.py

# Test in another terminal
curl -X POST http://localhost:5000/predict \
    -F "image=@test_image.jpg"
```

### 7.2 Deploy to Production

```bash
# Build Docker image
docker build -t klgrade-api .

# Run container
docker run -p 5000:5000 --gpus all klgrade-api
```

**API Endpoints**:
- `POST /predict` - Predict KL grade from X-ray image
- `POST /predict/batch` - Batch prediction
- `GET /health` - Health check
- `GET /models` - List available models

---

## Troubleshooting

### Issue: "No labels found"

**Solution**:
```bash
# Verify label files exist
ls -lh datasets/dataset_knees_cropped/labels/ | head -10

# Check label format
head -5 datasets/dataset_knees_cropped/labels/*.txt
```

### Issue: "CUDA out of memory"

**Solution**:
```bash
# Reduce batch size
yolo detect train data=config.yaml batch=8  # instead of 16

# Or use smaller image size
yolo detect train data=config.yaml imgsz=416  # instead of 640
```

### Issue: Config files not found

**Solution**:
```bash
# Regenerate configs and splits
bash scripts/pipelines/regenerate_configs_and_splits.sh
```

---

## Quick Reference

### Important Directories

```
datasets/
  ├── dataset_knees_cropped/          # Main preprocessed dataset
  ├── dataset_knees_cropped_balanced/ # Balanced dataset (optional)
  └── data_processed/                 # Alternative processing variants

configs/                              # YAML training configs
runs/
  ├── detect/                          # Training outputs
  └── evaluate/                        # Evaluation outputs

scripts/
  ├── preprocessing/                   # Data preprocessing
  ├── data_preparation/                # Splitting, balancing, filtering
  ├── training/                        # Training scripts
  ├── evaluation/                      # Evaluation scripts
  ├── visualization/                   # Visualization tools
  └── pipelines/                       # End-to-end workflows
```

### Key Scripts

| Task | Script |
|------|--------|
| Knee cropping | `scripts/preprocessing/prepare_knee_crops.py` |
| Data balancing | `scripts/data_preparation/balance_dataset.py` |
| Create splits | `scripts/data_preparation/split_dataset.py` |
| Regenerate configs | `scripts/pipelines/regenerate_configs_and_splits.sh` |
| Test configs | `scripts/testing/test_configs.py` |
| Evaluate model | `scripts/evaluation/evaluate_yolo_standalone.py` |
| Compare results | `scripts/evaluation/summarize_experiments.py` |

---

## Complete Workflow Example

```bash
# 1. Activate environment
conda activate klgrade

# 2. Preprocess data
python scripts/preprocessing/prepare_knee_crops.py \
    --input datasets/dataset/dataset_v0 \
    --output datasets/dataset_knees_cropped \
    --model models/knee_detector.pt

# 3. Balance dataset (optional)
python scripts/data_preparation/balance_dataset.py \
    --input datasets/dataset_knees_cropped \
    --output datasets/dataset_knees_cropped_balanced

# 4. Create splits & update configs
bash scripts/pipelines/regenerate_configs_and_splits.sh

# 5. Test setup
python scripts/testing/test_configs.py \
    --configs configs/yolo_5_class_baseline.yaml \
    --epochs 1

# 6. Train model
yolo detect train \
    data=configs/yolo_5_class_baseline.yaml \
    epochs=100 \
    batch=16 \
    device=0

# 7. Evaluate
python scripts/evaluation/evaluate_yolo_standalone.py \
    --model runs/detect/knee_5_class_baseline/weights/best.pt \
    --data configs/yolo_5_class_baseline.yaml

# 8. Deploy API
python src/api/app.py
```

---

**Last Updated**: 2026-01-17  
**Status**: Ready for training ✅
