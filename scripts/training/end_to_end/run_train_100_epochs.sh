#!/bin/bash
# Train End-to-End Model for 100 Epochs

echo "Starting End-to-End Training (100 Epochs)"
echo "=========================================="

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate klgrade

python scripts/training/end_to_end/train_end_to_end.py \
  --backbone yolo11s \
  --epochs 100 \
  --batch-size 1 \
  --save-dir runs/end_to_end/train_100epochs \
  --train-img-dir datasets/dataset_v0/images \
  --train-knee-label-dir datasets/dataset_v0/labels-knee \
  --train-lesion-label-dir datasets/dataset_v0/labels \
  --train-split-file datasets/dataset_v0/train.txt

echo "Training complete."
