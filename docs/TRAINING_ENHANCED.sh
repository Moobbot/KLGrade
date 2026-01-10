#!/bin/bash
# Enhanced YOLO Training with CLAHE + Data Balancing
# Based on Yolo_Detection_XuongKhop_v2.ipynb techniques
# chmod +x docs/TRAINING_ENHANCED.sh
set -e

# ============================================================================
# Environment Setup
# ============================================================================

echo "Activating conda environment..."
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
echo ""
echo "========================================================================"
echo ""

# E_ENHANCED_001: Dataset V0 with Full Enhancement
echo "Training E_ENHANCED_001: Dataset V0 + CLAHE + Balancing..."
python scripts/training/train_yolo_enhanced.py \
  --img_dir dataset/dataset_v0/images \
  --label_dir dataset/dataset_v0/labels \
  --split_dir processed/splits/dataset_v0 \
  --num_classes 5 \
  --model yolo11l.pt \
  --epochs 100 \
  --batch 8 \
  --imgsz 640 \
  --device 0 \
  --project runs/detect \
  --name yolo11l_E_ENHANCED_001_v0_full

echo "✅ E_ENHANCED_001 complete!"
echo ""

# E_ENHANCED_002: Dataset V0 with Preprocessing Only (no balancing)
echo "Training E_ENHANCED_002: Dataset V0 + CLAHE (no balancing)..."
python scripts/training/train_yolo_enhanced.py \
  --img_dir dataset/dataset_v0/images \
  --label_dir dataset/dataset_v0/labels \
  --split_dir processed/splits/dataset_v0 \
  --num_classes 5 \
  --model yolo11l.pt \
  --epochs 100 \
  --batch 8 \
  --imgsz 640 \
  --device 0 \
  --project runs/detect \
  --name yolo11l_E_ENHANCED_002_v0_clahe_only \
  --no-balancing

echo "✅ E_ENHANCED_002 complete!"
echo ""

# E_ENHANCED_003: Processed knee_5_class with Enhancement
echo "Training E_ENHANCED_003: knee_5_class + CLAHE + Balancing..."
python scripts/training/train_yolo_enhanced.py \
  --img_dir processed/knee/dataset_yolo/images \
  --label_dir processed/knee/dataset_yolo/labels \
  --split_dir processed/splits/knee_5_class \
  --num_classes 5 \
  --model yolo11l.pt \
  --epochs 100 \
  --batch 8 \
  --imgsz 640 \
  --device 0 \
  --project runs/detect \
  --name yolo11l_E_ENHANCED_003_knee5_full

echo "✅ E_ENHANCED_003 complete!"
echo ""


# ============================================================================
# Additional Splits Experiments (10-class, 4-class, 8-class)
# ============================================================================

# E_ENHANCED_004: 10-class Split
echo "Training E_ENHANCED_004: knee_10_class + CLAHE + Balancing..."
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
  --name yolo11l_E_ENHANCED_004_knee10_full

echo "✅ E_ENHANCED_004 complete!"
echo ""

# E_ENHANCED_005: 4-class Split
echo "Training E_ENHANCED_005: knee_4_class + CLAHE + Balancing..."
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
  --name yolo11l_E_ENHANCED_005_knee4_full

echo "✅ E_ENHANCED_005 complete!"
echo ""

# E_ENHANCED_006: 8-class Split
echo "Training E_ENHANCED_006: knee_8_class + CLAHE + Balancing..."
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
  --name yolo11l_E_ENHANCED_006_knee8_full

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
echo "  - yolo11l_E_ENHANCED_001_v0_full/"
echo "  - yolo11l_E_ENHANCED_002_v0_clahe_only/"
echo "  - yolo11l_E_ENHANCED_003_knee5_full/"
echo ""
echo "Enhanced Features Applied:"
echo "  📊 CLAHE - Better contrast for medical images"
echo "  🔍 Gaussian Blur - Noise reduction"
echo "  ⚖️  Data Balancing - Flip augmentation for minority classes"
echo "  📏 Label Scaling - Proper bbox adjustment"
echo ""
echo "Processed Data Location:"
echo "  - processed/enhanced/images/"
echo "  - processed/enhanced/labels/"
echo ""
echo "WandB Dashboard:"
echo "  https://wandb.ai/ngotam2k1-thuyloi-university/KLGrade-Knee-OA"
echo ""
echo "========================================================================"

# ============================================================================
# Usage Instructions
# ============================================================================

# Run all experiments:
# bash docs/TRAINING_ENHANCED.sh

# Run in background:
# nohup bash docs/TRAINING_ENHANCED.sh > training_enhanced.log 2>&1 &

# Run single experiment:
# python scripts/training/train_yolo_enhanced.py \
#   --img_dir dataset/dataset_v0/images \
#   --label_dir dataset/dataset_v0/labels \
#   --split_dir processed/splits/dataset_v0 \
#   --num_classes 5 \
#   --epochs 100 \
#   --batch 16 \
#   --device 0 \
#   --name test_enhanced
