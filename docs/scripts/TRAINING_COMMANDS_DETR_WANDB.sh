#!/bin/bash
# DETR Training Commands with WandB Integration
# KLGrade - Knee OA Detection Project
# chmod +x docs/TRAINING_COMMANDS_DETR_WANDB.sh
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

echo "✅ WandB configured - Experiments will be logged to: https://wandb.ai"

# ============================================================================
# PHASE 1: Baseline Training (DETR)
# ============================================================================

echo "Starting DETR Baseline Training Phase..."

# E010: 5-class Baseline
echo "Training E010: DETR 5-class Baseline..."
python scripts/training/train_detr.py \
  --img_dir processed/knee_5_class/images \
  --label_dir processed/knee_5_class/labels \
  --split_dir processed/splits/knee_5_class \
  --num_classes 5 \
  --model facebook/detr-resnet-50 \
  --epochs 50 \
  --batch 4 \
  --lr 1e-4 \
  --device cuda \
  --output runs/detr/E010_5class_baseline

# E011: 10-class Baseline  
echo "Training E011: DETR 10-class Baseline..."
python scripts/training/train_detr.py \
  --img_dir processed/knee_10_class/images \
  --label_dir processed/knee_10_class/labels \
  --split_dir processed/splits/knee_10_class \
  --num_classes 10 \
  --model facebook/detr-resnet-50 \
  --epochs 50 \
  --batch 4 \
  --lr 1e-4 \
  --device cuda \
  --output runs/detr/E011_10class_baseline

# E012: 4-class Baseline
echo "Training E012: DETR 4-class Baseline..."
python scripts/training/train_detr.py \
  --img_dir processed/knee_4_class/images \
  --label_dir processed/knee_4_class/labels \
  --split_dir processed/splits/knee_4_class \
  --num_classes 4 \
  --model facebook/detr-resnet-50 \
  --epochs 50 \
  --batch 4 \
  --lr 1e-4 \
  --device cuda \
  --output runs/detr/E012_4class_baseline

# E013: 8-class Baseline
echo "Training E013: DETR 8-class Baseline..."
python scripts/training/train_detr.py \
  --img_dir processed/knee_8_class/images \
  --label_dir processed/knee_8_class/labels \
  --split_dir processed/splits/knee_8_class \
  --num_classes 8 \
  --model facebook/detr-resnet-50 \
  --epochs 50 \
  --batch 4 \
  --lr 1e-4 \
  --device cuda \
  --output runs/detr/E013_8class_baseline

echo "DETR baseline training phase complete!"
echo "✅ All experiments logged to WandB: https://wandb.ai/your-username/KLGrade-Knee-OA"

# ============================================================================
# Usage Instructions
# ============================================================================

# Run in background with logging:
# nohup bash docs/TRAINING_COMMANDS_DETR_WANDB.sh > training_detr.log 2>&1 &

# Monitor progress:
# - Local: tail -f training_detr.log
# - WandB: https://wandb.ai
# - GPU: watch -n 1 nvidia-smi
