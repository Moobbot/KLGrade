#!/bin/bash
# YOLO Training Commands with WandB Integration
# KLGrade - Knee OA Detection Project

set -e

# ============================================================================
# Environment Setup (Conda)
# ============================================================================

echo "Activating conda environment..."
conda activate klgrade

# Verify Python is from conda env
which python
python --version

# ============================================================================
# WandB Setup
# ============================================================================

echo "Setting up WandB..."
export WANDB_API_KEY="wandb_v1_hello"
export WANDB_PROJECT="KLGrade-Knee-OA"

# Login to WandB
wandb login $WANDB_API_KEY

echo "✅ WandB configured - Experiments will be logged to: https://wandb.ai"

# ============================================================================
# PHASE 1: Baseline Training (No Augmentation)
# ============================================================================

echo "Starting Baseline Training Phase..."

# E001: 5-class Baseline
echo "Training E001: 5-class Baseline..."
yolo detect train \
  data=configs/yolo_5_class_baseline.yaml \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  device=0 \
  project=runs/detect \
  name=E001_5class_baseline \
  exist_ok=False

# E004: 10-class Baseline  
echo "Training E004: 10-class Baseline..."
yolo detect train \
  data=configs/yolo_10_class_baseline.yaml \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  device=0 \
  project=runs/detect \
  name=E004_10class_baseline \
  exist_ok=False

# E005: 4-class Baseline
echo "Training E005: 4-class Baseline..."
yolo detect train \
  data=configs/yolo_4_class_baseline.yaml \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  device=0 \
  project=runs/detect \
  name=E005_4class_baseline \
  exist_ok=False

# E006: 8-class Baseline
echo "Training E006: 8-class Baseline..."
yolo detect train \
  data=configs/yolo_8_class_baseline.yaml \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  device=0 \
  project=runs/detect \
  name=E006_8class_baseline \
  exist_ok=False

echo "Baseline training phase complete!"

# ============================================================================
# PHASE 2: Conservative Augmentation
# ============================================================================

echo "Starting Conservative Augmentation Phase..."

# E002: 5-class + Conservative Augmentation
echo "Training E002: 5-class + Conservative..."
yolo detect train \
  data=configs/yolo_5_class_conservative.yaml \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  device=0 \
  project=runs/detect \
  name=E002_5class_conservative \
  exist_ok=False

# E007: 10-class + Conservative Augmentation
echo "Training E007: 10-class + Conservative..."
yolo detect train \
  data=configs/yolo_10_class_conservative.yaml \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  device=0 \
  project=runs/detect \
  name=E007_10class_conservative \
  exist_ok=False

# E008: 4-class + Conservative Augmentation
echo "Training E008: 4-class + Conservative..."
yolo detect train \
  data=configs/yolo_4_class_conservative.yaml \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  device=0 \
  project=runs/detect \
  name=E008_4class_conservative \
  exist_ok=False

# E009: 8-class + Conservative Augmentation
echo "Training E009: 8-class + Conservative..."
yolo detect train \
  data=configs/yolo_8_class_conservative.yaml \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  device=0 \
  project=runs/detect \
  name=E009_8class_conservative \
  exist_ok=False

echo "Conservative augmentation phase complete!"
echo "✅ All experiments logged to WandB: https://wandb.ai/your-username/KLGrade-Knee-OA"

# ============================================================================
# Usage Instructions
# ============================================================================

# Run in background with logging:
# nohup bash docs/TRAINING_COMMANDS_WANDB.sh > training.log 2>&1 &

# Monitor progress:
# - Local: tail -f training.log
# - WandB: https://wandb.ai
# - GPU: watch -n 1 nvidia-smi
