#!/bin/bash
# Sequential Training Script: 5-class then 10-class Lesion Detectors
# This script will run automatically after being started

set -e  # Exit on error

echo "=========================================="
echo "Sequential Lesion Detector Training"
echo "=========================================="
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate klgrade

# ==========================================
# Step 1: Train 5-class Lesion Detector
# ==========================================
echo "Step 1/2: Training 5-class Lesion Detector (KL0-KL4)"
echo "Dataset: datasets/dataset_knees_cropped"
echo "Epochs: 100, Batch: 16, Patience: 20"
echo "------------------------------------------"

python api_two_step_yolo/training/train_lesion_detector.py \
    --data datasets/splits/dataset_knees_cropped_70_20_10/dataset.yaml \
    --epochs 100 \
    --batch 16 \
    --device 0

echo ""
echo "✅ 5-class training completed!"
echo ""

# ==========================================
# Step 2: Train 10-class Lesion Detector
# ==========================================
echo "Step 2/2: Training 10-class Lesion Detector (KL0-a to KL4-b)"
echo "Dataset: datasets/balanced/knees_cropped_10_class"
echo "Epochs: 100, Batch: 16, Patience: 20"
echo "------------------------------------------"

python api_two_step_yolo/training/train_lesion_detector.py \
    --data datasets/splits/dataset_knees_cropped_10_class_60_20_20/dataset.yaml \
    --epochs 100 \
    --batch 16 \
    --device 0

echo ""
echo "=========================================="
echo "✅ All training completed successfully!"
echo "=========================================="
echo ""
echo "Results:"
echo "  5-class:  runs/detect/train_lesion_detector/weights/best.pt"
echo "  10-class: runs/detect/train_lesion_detector2/weights/best.pt"
echo ""
