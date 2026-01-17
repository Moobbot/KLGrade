#!/bin/bash
# YOLO Training Commands with WandB Integration
# KLGrade - Knee OA Detection Project
# chmod +x docs/TRAINING_COMMANDS_YOLO_WANDB.sh
set -e

# ============================================================================
# Environment Setup (Conda)
# ============================================================================

echo "Activating conda environment..."
# Initialize conda for bash (if not already done)
eval "$(conda shell.bash hook)"
conda activate klgrade

# Verify Python is from conda env
which python
python --version

# ============================================================================
# WandB Setup
# ============================================================================

echo "Setting up WandB..."
# Load WandB credentials from .wandb.env
if [ -f ".wandb.env" ]; then
    set -a
    source .wandb.env
    set +a
    echo "✅ Loaded WandB credentials from .wandb.env"
else
    echo "❌ Error: .wandb.env file not found"
    exit 1
fi

# Verify credentials are loaded
if [ -z "$WANDB_API_KEY" ]; then
    echo "❌ Error: WANDB_API_KEY not set"
    exit 1
fi

# Login to WandB
wandb login $WANDB_API_KEY

# Enable WandB in Ultralytics YOLO settings
echo "Enabling WandB integration in Ultralytics YOLO..."
yolo settings wandb=True

echo "✅ WandB configured - Experiments will be logged to: https://wandb.ai"

# ============================================================================
# PHASE 1: Baseline Training (YOLO11)
# ============================================================================

echo "Starting YOLO Baseline Training Phase..."

# E001: 5-class Baseline
echo "Training E001: YOLO 5-class Baseline..."
yolo detect train \
  data=configs/yolo_5_class_baseline.yaml \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  device=0 \
  project=runs/detect \
  name=E001_5_class_baseline \
  exist_ok=False \
  patience=50 \
  save_period=10 \
  plots=True

echo "✅ E001 complete!"
echo ""

# E004: 10-class Baseline  
echo "Training E004: YOLO 10-class Baseline..."
yolo detect train \
  data=configs/yolo_10_class_baseline.yaml \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  device=0 \
  project=runs/detect \
  name=E004_10_class_baseline \
  exist_ok=False \
  patience=50 \
  save_period=10 \
  plots=True

echo "✅ E004 complete!"
echo ""

# E005: 4-class Baseline
echo "Training E005: YOLO 4-class Baseline..."
yolo detect train \
  data=configs/yolo_4_class_baseline.yaml \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  device=0 \
  project=runs/detect \
  name=E005_4_class_baseline \
  exist_ok=False \
  patience=50 \
  save_period=10 \
  plots=True

echo "✅ E005 complete!"
echo ""

# E006: 8-class Baseline
echo "Training E006: YOLO 8-class Baseline..."
yolo detect train \
  data=configs/yolo_8_class_baseline.yaml \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  device=0 \
  project=runs/detect \
  name=E006_8_class_baseline \
  exist_ok=False \
  patience=50 \
  save_period=10 \
  plots=True

echo "✅ E006 complete!"
echo ""

echo "YOLO baseline training phase complete!"

# ============================================================================
# PHASE 2: Conservative Augmentation
# ============================================================================

echo ""
echo "Starting Conservative Augmentation Phase..."

# E002: 5-class + Conservative Augmentation
echo "Training E002: YOLO 5-class + Conservative..."
yolo detect train \
  data=configs/yolo_5_class_conservative.yaml \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  device=0 \
  project=runs/detect \
  name=E002_5_class_conservative \
  exist_ok=False \
  patience=50 \
  save_period=10 \
  plots=True

echo "✅ E002 complete!"
echo ""

# E007: 10-class + Conservative Augmentation
echo "Training E007: YOLO 10-class + Conservative..."
yolo detect train \
  data=configs/yolo_10_class_conservative.yaml \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  device=0 \
  project=runs/detect \
  name=E007_10_class_conservative \
  exist_ok=False \
  patience=50 \
  save_period=10 \
  plots=True

echo "✅ E007 complete!"
echo ""

# E008: 4-class + Conservative Augmentation
echo "Training E008: YOLO 4-class + Conservative..."
yolo detect train \
  data=configs/yolo_4_class_conservative.yaml \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  device=0 \
  project=runs/detect \
  name=E008_4_class_conservative \
  exist_ok=False \
  patience=50 \
  save_period=10 \
  plots=True

echo "✅ E008 complete!"
echo ""

# E009: 8-class + Conservative Augmentation
echo "Training E009: YOLO 8-class + Conservative..."
yolo detect train \
  data=configs/yolo_8_class_conservative.yaml \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  device=0 \
  project=runs/detect \
  name=E009_8_class_conservative \
  exist_ok=False \
  patience=50 \
  save_period=10 \
  plots=True

echo "✅ E009 complete!"
echo ""

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "========================================================================"
echo "✅ ALL YOLO TRAINING EXPERIMENTS COMPLETE!"
echo "========================================================================"
echo ""
echo "Results saved to: runs/detect/"
echo "  - E001_5_class_baseline/"
echo "  - E004_10_class_baseline/"
echo "  - E005_4_class_baseline/"
echo "  - E006_8_class_baseline/"
echo "  - E002_5_class_conservative/"
echo "  - E007_10_class_conservative/"
echo "  - E008_4_class_conservative/"
echo "  - E009_8_class_conservative/"
echo ""
echo "WandB Dashboard:"
echo "  https://wandb.ai/ngotam2k1-thuyloi-university/KLGrade-Knee-OA"
echo ""
echo "========================================================================"

# ============================================================================
# Usage Instructions
# ============================================================================

# Run all experiments:
# bash docs/TRAINING_COMMANDS_YOLO_WANDB.sh

# Run in background with logging:
# nohup bash docs/TRAINING_COMMANDS_YOLO_WANDB.sh > training_yolo.log 2>&1 &

# Monitor progress:
# - Local: tail -f training_yolo.log
# - WandB: https://wandb.ai
# - GPU: watch -n 1 nvidia-smi

# Run single baseline experiment:
# source this file and run individual commands, or:
# yolo detect train data=configs/yolo_5_class_baseline.yaml epochs=100 batch=16 device=0 project=runs/detect name=E001_5_class_baseline
