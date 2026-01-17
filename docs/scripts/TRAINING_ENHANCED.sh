#!/bin/bash
# Enhanced YOLO Training with CLAHE + Data Balancing
# Based on Yolo_Detection_XuongKhop_v2.ipynb techniques
# chmod +x docs/scripts/TRAINING_ENHANCED.sh
set -e

# ============================================================================
# Environment Setup
# ============================================================================

echo "Activating conda environment..."
# Initialize conda for bash (if not already done)
eval "$(conda shell.bash hook)"
conda activate klgrade

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

wandb login $WANDB_API_KEY
yolo settings wandb=True

echo "✅ WandB configured"

# ============================================================================
# Enhanced Training Experiments
# ============================================================================

echo ""
echo "========================================================================"
echo "Enhanced YOLO Training with Advanced Preprocessing"
echo "========================================================================"
echo ""
echo "Techniques applied:"
echo "  ✅ CLAHE (Contrast Limited Adaptive Histogram Equalization)"
echo "  ✅ Gaussian Blur (noise reduction)"
echo "  ✅ Data Balancing (flip augmentation for minority classes)"
echo "  ✅ Label Scaling (proper bbox adjustment after resize)"
echo "  ✅ Advanced Augmentation (Mosaic, Mixup, HSV, Geometric)"
echo "  ✅ Cosine Annealing Learning Rate"
echo ""
echo "========================================================================"
echo ""

# E_ENHANCED_001: 5-class Baseline (Matches E001)
echo "Training E_ENHANCED_001: YOLO 5-class + CLAHE + Balancing..."
python scripts/training/train_yolo_enhanced.py \
  --img_dir processed/knee/images \
  --label_dir processed/knee/labels \
  --split_dir processed/splits/knee_5_class \
  --num_classes 5 \
  --model yolo11l.pt \
  --epochs 100 \
  --batch 8 \
  --imgsz 640 \
  --device 0 \
  --project runs/detect \
  --name E_ENHANCED_001_5_class_full

echo "✅ E_ENHANCED_001 complete!"
echo ""

# E_ENHANCED_004: 10-class Baseline (Matches E004)
echo "Training E_ENHANCED_004: YOLO 10-class + CLAHE + Balancing..."
python scripts/training/train_yolo_enhanced.py \
  --img_dir processed/knee_10_class/images \
  --label_dir processed/knee_10_class/labels \
  --split_dir processed/splits/knee_10_class \
  --num_classes 10 \
  --model yolo11l.pt \
  --epochs 100 \
  --batch 8 \
  --imgsz 640 \
  --device 0 \
  --project runs/detect \
  --name E_ENHANCED_004_10_class_full

echo "✅ E_ENHANCED_004 complete!"
echo ""

# E_ENHANCED_005: 4-class Baseline (Matches E005)
echo "Training E_ENHANCED_005: YOLO 4-class + CLAHE + Balancing..."
python scripts/training/train_yolo_enhanced.py \
  --img_dir processed/knee_4_class/images \
  --label_dir processed/knee_4_class/labels \
  --split_dir processed/splits/knee_4_class \
  --num_classes 4 \
  --model yolo11l.pt \
  --epochs 100 \
  --batch 8 \
  --imgsz 640 \
  --device 0 \
  --project runs/detect \
  --name E_ENHANCED_005_4_class_full

echo "✅ E_ENHANCED_005 complete!"
echo ""

# E_ENHANCED_006: 8-class Baseline (Matches E006)
echo "Training E_ENHANCED_006: YOLO 8-class + CLAHE + Balancing..."
python scripts/training/train_yolo_enhanced.py \
  --img_dir processed/knee_8_class/images \
  --label_dir processed/knee_8_class/labels \
  --split_dir processed/splits/knee_8_class \
  --num_classes 8 \
  --model yolo11l.pt \
  --epochs 100 \
  --batch 8 \
  --imgsz 640 \
  --device 0 \
  --project runs/detect \
  --name E_ENHANCED_006_8_class_full

echo "✅ E_ENHANCED_006 complete!"
echo ""

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "========================================================================"
echo "✅ ENHANCED TRAINING EXPERIMENTS COMPLETE!"
echo "========================================================================"
echo ""
echo "Results saved to: runs/detect/"
echo "  - E_ENHANCED_001_5_class_full/"
echo "  - E_ENHANCED_004_10_class_full/"
echo "  - E_ENHANCED_005_4_class_full/"
echo "  - E_ENHANCED_006_8_class_full/"
echo ""
echo "Enhanced Features Applied:"
echo "  📊 CLAHE - Better contrast for medical images"
echo "  🔍 Gaussian Blur - Noise reduction"
echo "  ⚖️  Data Balancing - Flip augmentation for minority classes"
echo "  📏 Label Scaling - Proper bbox adjustment"
echo ""
echo "WandB Dashboard:"
echo "  https://wandb.ai/ngotam20082001/KLGrade-Knee-OA"
echo ""
echo "========================================================================"

# ============================================================================
# Usage Instructions
# ============================================================================

# Run all experiments:
# bash docs/scripts/TRAINING_ENHANCED.sh

# Run in background:
# nohup bash docs/scripts/TRAINING_ENHANCED.sh > training_enhanced.log 2>&1 &
