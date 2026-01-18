# Training and Evaluation Guide

This document covers **Part 2** of the KLGrade pipeline: **Model Training and Evaluation**.
It assumes you have already generated datasets using the [Data Pipeline](DATA_PIPELINE.md).

---

## 1. Quick Reference

### Train a Baseline YOLO Model
```bash
yolo detect train \
    data=configs/yolo_5_class_baseline.yaml \
    epochs=100 \
    batch=16 \
    imgsz=640 \
    device=0 \
    project=runs/detect \
    name=knee_5_class_baseline
```

### Train KIOCMIL (Full X-rays)
```bash
bash scripts/training/train_kiocmil_v3_wandb.sh
```

---

## 2. Training Workflow

### Step 1: Verify Configs
Ensure that your YAML configuration files point to the correct datasets and split files.
```bash
# Check validation paths in all configs
grep "^val:" configs/yolo_*.yaml
```

### Step 2: Run a Quick Test
Run a 1-epoch test to ensure the environment, GPU, and data loading work correctly.
```bash
python scripts/testing/test_configs.py --configs configs/yolo_5_class_baseline.yaml --epochs 1
```

### Step 3: Full Training
You can train a single model or multiple configurations.

**Single Config:**
See specific command in Quick Reference.

**All Configs (Batch Training):**
```bash
bash scripts/training/train_all_configs.sh
```

### Available Configurations
*   `yolo_5_class_baseline`: Standard 5-class (KL0-4).
*   `yolo_4_class_baseline`: 4-class (No KL0).
*   `yolo_8_class_baseline`: 8-class (Sub-grades).
*   `yolo_10_class_baseline`: 10-class (All sub-grades).
*   `*_conservative`: Variants with lighter augmentation.

---

## 3. Evaluation

### Evaluate a Trained Model
```bash
python scripts/evaluation/evaluate_yolo_standalone.py \
    --model runs/detect/knee_5_class_baseline/weights/best.pt \
    --data configs/yolo_5_class_baseline.yaml \
    --split test
```

### Compare All Experiments
Generates a summary table of metrics (mAP, Precision, Recall) for all models in `runs/detect`.
```bash
python scripts/evaluation/summarize_experiments.py
```

### Visualization
**GradCAM Analysis** (Heatmaps):
```bash
python scripts/visualization/visualize_gradcam.py \
    --source datasets/dataset_v0/images/sample.jpg \
    --grade-model runs/detect/knee_5_class_baseline/weights/best.pt
```

---

## 4. Troubleshooting

*   **CUDA Out of Memory**: Reduce `batch` size (e.g., to 8) or `imgsz` (e.g., to 416).
*   **Config Not Found**: Run `bash scripts/pipelines/regenerate_configs_and_splits.sh` to recreate them.
