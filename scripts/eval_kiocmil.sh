#!/bin/bash

# Evaluation script for KIOCMIL
# Usage: ./scripts/eval_kiocmil.sh [split_file] [model_path]
# Example: ./scripts/eval_kiocmil.sh splits/val.txt runs/kiocmil_v1/best_model.pth

# Explicitly use the conda environment python
PYTHON_EXEC="/home/ngoductam/miniconda3/envs/klgrade/bin/python"

SPLIT_FILE=${1:-"splits/val.txt"}
MODEL_PATH=${2:-"runs/kiocmil_v1/best_model.pth"}

# Check if we are in the project root (src exists)
if [ ! -d "src" ]; then
    echo "Error: Please run this script from the project root directory."
    exit 1
fi

echo "Starting KIOCMIL Evaluation..."
echo "Split: $SPLIT_FILE"
echo "Model: $MODEL_PATH"

$PYTHON_EXEC src/training/evaluate_kiocmil.py \
    --img_dir dataset/dataset_v0/images \
    --knee_labels dataset/dataset_v0/labels-knee \
    --lesion_labels dataset/dataset_v0/labels_new \
    --split_file "$SPLIT_FILE" \
    --model_path "$MODEL_PATH" \
    --backbone resnet18 \
    --batch_size 16
