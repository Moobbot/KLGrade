#!/bin/bash
# YOLO Training Commands for Dataset V0 (Original Dataset)
# KLGrade - Knee OA Detection Project
# chmod +x docs/TRAINING_COMMANDS_YOLO_DATASET_V0.sh
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
# Dataset V0 Training Experiments
# ============================================================================

echo ""
echo "========================================================================"
echo "Training YOLO on Original Dataset V0"
echo "Dataset: dataset/dataset_v0/"
echo "  - Images: 1473"
echo "  - Classes: 5 (KL0-4)"
echo "  - Splits: 70/15/15"
echo "========================================================================"
echo ""

# E_V0_001: Dataset V0 Baseline
echo "Training E_V0_001: YOLO Dataset V0 Baseline..."
yolo detect train \
  data=configs/yolo_dataset_v0_baseline.yaml \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  device=0 \
  project=runs/detect \
  name=E_V0_001_baseline \
  exist_ok=False \
  patience=50 \
  save_period=10 \
  plots=True

echo "✅ E_V0_001 complete!"
echo ""

# E_V0_002: Dataset V0 Conservative Augmentation
echo "Training E_V0_002: YOLO Dataset V0 Conservative..."
yolo detect train \
  data=configs/yolo_dataset_v0_conservative.yaml \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  device=0 \
  project=runs/detect \
  name=E_V0_002_conservative \
  exist_ok=False \
  patience=50 \
  save_period=10 \
  plots=True

echo "✅ E_V0_002 complete!"
echo ""

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "========================================================================"
echo "✅ DATASET V0 TRAINING EXPERIMENTS COMPLETE!"
echo "========================================================================"
echo ""
echo "Results saved to: runs/detect/"
echo "  - E_V0_001_baseline/"
echo "  - E_V0_002_conservative/"
echo ""
echo "Dataset Statistics:"
echo "  - Total images: 1473"
echo "  - Train: 1031 images (70%)"
echo "  - Val: 220 images (15%)"
echo "  - Test: 222 images (15%)"
echo ""
echo "Class Distribution:"
echo "  - KL0: 93 images"
echo "  - KL1: 572 images"
echo "  - KL2: 771 images"
echo "  - KL3: 342 images"
echo "  - KL4: 207 images"
echo ""
echo "WandB Dashboard:"
echo "  https://wandb.ai/ngotam2k1-thuyloi-university/KLGrade-Knee-OA"
echo ""
echo "========================================================================"

# ============================================================================
# Usage Instructions
# ============================================================================

# Run all experiments:
# bash docs/TRAINING_COMMANDS_YOLO_DATASET_V0.sh

# Run in background with logging:
# nohup bash docs/TRAINING_COMMANDS_YOLO_DATASET_V0.sh > training_dataset_v0.log 2>&1 &

# Monitor progress:
# - Local: tail -f training_dataset_v0.log
# - WandB: https://wandb.ai
# - GPU: watch -n 1 nvidia-smi

# Run single experiment:
# yolo detect train data=configs/yolo_dataset_v0_baseline.yaml epochs=100 batch=16 device=0 project=runs/detect name=E_V0_001_baseline
