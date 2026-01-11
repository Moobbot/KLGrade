#!/bin/bash
# DETR Model Evaluation Commands
# KLGrade - Knee OA Detection Project
# chmod +x docs/EVALUATION_COMMANDS_DETR.sh
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
# WandB Setup (Optional - for logging eval results)
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

echo "✅ WandB configured"

# ============================================================================
# DETR Model Evaluation
# ============================================================================

echo ""
echo "========================================================================"
echo "Starting DETR Model Evaluation"
echo "========================================================================"
echo ""

# ----------------------------------------------------------------------------
# E010: 5-class Baseline Evaluation
# ----------------------------------------------------------------------------
echo "Evaluating E010: DETR 5-class Baseline..."
python scripts/evaluation/evaluate_detr.py \
  --model_path runs/detr/E010_5class_baseline/best_model.pt \
  --img_dir processed/knee/dataset_yolo/images \
  --label_dir processed/knee/labels \
  --split_dir splits/knee_5_class \
  --conf_threshold 0.01 \
  --device cuda \
  --output runs/detr/E010_5class_baseline/evaluation

echo "✅ E010 evaluation complete!"
echo ""

# ----------------------------------------------------------------------------
# E011: 10-class Baseline Evaluation
# ----------------------------------------------------------------------------
echo "Evaluating E011: DETR 10-class Baseline..."
python scripts/evaluation/evaluate_detr.py \
  --model_path runs/detr/E011_10class_baseline/best_model.pt \
  --img_dir processed/knee_10_class/images \
  --label_dir processed/knee_10_class/labels \
  --split_dir splits/knee_10_class \
  --use_10_class \
  --conf_threshold 0.01 \
  --device cuda \
  --output runs/detr/E011_10class_baseline/evaluation

echo "✅ E011 evaluation complete!"
echo ""

# ----------------------------------------------------------------------------
# E012: 4-class Baseline Evaluation
# ----------------------------------------------------------------------------
echo "Evaluating E012: DETR 4-class Baseline..."
python scripts/evaluation/evaluate_detr.py \
  --model_path runs/detr/E012_4class_baseline/best_model.pt \
  --img_dir processed/knee_4_class/images \
  --label_dir processed/knee_4_class/labels \
  --split_dir splits/knee_4_class \
  --use_4_class \
  --conf_threshold 0.01 \
  --device cuda \
  --output runs/detr/E012_4class_baseline/evaluation

echo "✅ E012 evaluation complete!"
echo ""

# ----------------------------------------------------------------------------
# E013: 8-class Baseline Evaluation
# ----------------------------------------------------------------------------
echo "Evaluating E013: DETR 8-class Baseline..."
python scripts/evaluation/evaluate_detr.py \
  --model_path runs/detr/E013_8class_baseline/best_model.pt \
  --img_dir processed/knee_8_class/images \
  --label_dir processed/knee_8_class/labels \
  --split_dir splits/knee_8_class \
  --use_8_class \
  --conf_threshold 0.01 \
  --device cuda \
  --output runs/detr/E013_8class_baseline/evaluation

echo "✅ E013 evaluation complete!"
echo ""

# ============================================================================
# Summary Report
# ============================================================================

echo ""
echo "========================================================================"
echo "✅ ALL EVALUATIONS COMPLETE!"
echo "========================================================================"
echo ""
echo "Results saved to:"
echo "  - runs/detr/E010_5class_baseline/evaluation/"
echo "  - runs/detr/E011_10class_baseline/evaluation/"
echo "  - runs/detr/E012_4class_baseline/evaluation/"
echo "  - runs/detr/E013_8class_baseline/evaluation/"
echo ""
echo "Each directory contains:"
echo "  - metrics.json: Detailed COCO metrics in JSON format"
echo "  - results.txt: Human-readable summary"
echo "  - predictions.json: All model predictions for error analysis"
echo ""
echo "View results with:"
echo "  cat runs/detr/E010_5class_baseline/evaluation/results.txt"
echo "  cat runs/detr/E011_10class_baseline/evaluation/results.txt"
echo "  cat runs/detr/E012_4class_baseline/evaluation/results.txt"
echo "  cat runs/detr/E013_8class_baseline/evaluation/results.txt"
echo ""

# ============================================================================
# Quick Summary Display
# ============================================================================

echo "Quick mAP Summary:"
echo "=================="

if [ -f runs/detr/E010_5class_baseline/evaluation/metrics.json ]; then
    echo -n "E010 (5-class):  mAP50="
    cat runs/detr/E010_5class_baseline/evaluation/metrics.json | grep -o '"mAP50": [0-9.]*' | head -1 | grep -o '[0-9.]*'
fi

if [ -f runs/detr/E011_10class_baseline/evaluation/metrics.json ]; then
    echo -n "E011 (10-class): mAP50="
    cat runs/detr/E011_10class_baseline/evaluation/metrics.json | grep -o '"mAP50": [0-9.]*' | head -1 | grep -o '[0-9.]*'
fi

if [ -f runs/detr/E012_4class_baseline/evaluation/metrics.json ]; then
    echo -n "E012 (4-class):  mAP50="
    cat runs/detr/E012_4class_baseline/evaluation/metrics.json | grep -o '"mAP50": [0-9.]*' | head -1 | grep -o '[0-9.]*'
fi

if [ -f runs/detr/E013_8class_baseline/evaluation/metrics.json ]; then
    echo -n "E013 (8-class):  mAP50="
    cat runs/detr/E013_8class_baseline/evaluation/metrics.json | grep -o '"mAP50": [0-9.]*' | head -1 | grep -o '[0-9.]*'
fi

echo ""
echo "========================================================================"

# ============================================================================
# Usage Instructions
# ============================================================================

# Run all evaluations:
# bash docs/EVALUATION_COMMANDS_DETR.sh

# Run in background with logging:
# nohup bash docs/EVALUATION_COMMANDS_DETR.sh > evaluation_detr.log 2>&1 &

# Monitor progress:
# - Local: tail -f evaluation_detr.log
# - GPU: watch -n 1 nvidia-smi
