#!/bin/bash

# Training script for KIOCMIL V3 with WandB
# Usage: ./scripts/train_kiocmil_v3_wandb.sh [extra_args]
# Example: ./scripts/train_kiocmil_v3_wandb.sh --epochs 50
# chmod +x scripts/train_kiocmil_v3_wandb.sh
set -e

# Explicitly use the conda environment
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate klgrade

PYTHON_EXEC="python"

# Check if we are in the project root (src exists)
if [ ! -d "src" ]; then
    echo "Error: Please run this script from the project root directory."
    exit 1
fi

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
    echo "⚠️  Warning: .wandb.env file not found. Assuming environment variables are set or not using WandB."
fi

# Verify credentials are loaded (optional, but good practice if WandB is expected)
if [ -n "$WANDB_API_KEY" ]; then
    echo "Logging in to WandB..."
    wandb login $WANDB_API_KEY
    echo "✅ WandB configured"
else
    echo "ℹ️  WANDB_API_KEY not set. Continuing without explicit login (WandB might be disabled or already logged in)."
fi

# ============================================================================
# Training
# ============================================================================

echo "Starting KIOCMIL V3 Training (YOLO11L Backbone)..."
echo "Python: $PYTHON_EXEC"

$PYTHON_EXEC src/training/train_kiocmil_v3.py \
    --img_dir dataset/dataset_v0/images \
    --knee_labels dataset/dataset_v0/labels-knee \
    --lesion_labels dataset/dataset_v0/labels_new \
    --train_split splits/knee_full_10_class/train.txt \
    --val_split splits/knee_full_10_class/val.txt \
    --backbone yolo11l \
    --save_dir runs/kiocmil_v3_yolo11l \
    --batch_size 2 \
    --epochs 30 \
    --lr 1e-4 \
    --wandb_project "KIOCMIL_KLGrade" \
    --wandb_name "v3_yolo11l_wandb" \
    --augmentation_level strong \
    --use_clahe \
    --use_v2_dataset \
    "$@"
