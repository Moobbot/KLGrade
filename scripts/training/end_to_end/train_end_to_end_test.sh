#!/bin/bash
# Quick test training script for end-to-end model
# Runs 5 epochs to verify everything works

echo "Starting 5-epoch test training for End-to-End Model"
echo "===================================================="

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate klgrade

# Run training with memory-optimized settings
python scripts/training/train_end_to_end.py \
  --backbone yolo11l \
  --num-classes 10 \
  --no-pretrained \
  --train-img-dir datasets/dataset_v0/images \
  --train-knee-label-dir datasets/dataset_v0/labels \
  --train-lesion-label-dir datasets/dataset_v0/labels \
  --train-split-file datasets/dataset_v0/train.txt \
  --val-split-file datasets/dataset_v0/val.txt \
  --epochs 5 \
  --batch-size 1 \
  --lr 0.001 \
  --device cuda \
  --save-dir runs/end_to_end/test_5epochs

echo ""
echo "===================================================="
echo "Training completed! Check runs/end_to_end/test_5epochs for results"
