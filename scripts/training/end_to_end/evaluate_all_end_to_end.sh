#!/bin/bash

# Script to evaluate all End-to-End KIOCMIL Training Experiments
# Usage: bash scripts/training/end_to_end/evaluate_all_end_to_end.sh

set -e  # Exit on error

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate klgrade || echo "⚠️  Warning: Failed to activate klgrade env"

echo "========================================================"
echo "STARTING ALL END-TO-END EVALUATIONS"
echo "========================================================"

# Function to run evaluation
evaluate_experiment() {
    local EXP_NAME=$1
    local NUM_CLASSES=$2
    local IMG_DIR=$3
    local SPLIT_DIR=$4
    local KNEE_LABEL_DIR=$5
    local LESION_LABEL_DIR=$6
    
    local CHECKPOINT="runs/end_to_end/$EXP_NAME/best.pt"
    local OUTPUT_FILE="runs/end_to_end/$EXP_NAME/evaluation_results.json"

    if [ ! -f "$CHECKPOINT" ]; then
        echo "⚠️  Skipping $EXP_NAME: Checkpoint not found at $CHECKPOINT"
        return
    fi

    echo ""
    echo "--------------------------------------------------------"
    echo "Evaluating Experiment: $EXP_NAME"
    echo "Checkpoint: $CHECKPOINT"
    echo "--------------------------------------------------------"

    python scripts/training/end_to_end/evaluate_end_to_end.py \
        --checkpoint "$CHECKPOINT" \
        --img-dir "$IMG_DIR" \
        --knee-label-dir "$KNEE_LABEL_DIR" \
        --lesion-label-dir "$LESION_LABEL_DIR" \
        --split-file "$SPLIT_DIR/val.txt" \
        --num_classes $NUM_CLASSES \
        --output "$OUTPUT_FILE" \
        --batch-size 32

    echo "✅ Completed: $EXP_NAME"
}

# ============================================================
# 1. 10-Class Experiments
# ============================================================
# Baseline (Unbalanced)
evaluate_experiment \
    "e2e_10class_unbalanced" 10 \
    "datasets/dataset_v0/images" \
    "datasets/splits/knee_full_10_class" \
    "datasets/dataset_v0/labels-knee" \
    "datasets/dataset_v0/labels_10_class"

# Balanced
evaluate_experiment \
    "e2e_10class_balanced" 10 \
    "datasets/balanced/full_xray_10_class/images" \
    "datasets/splits/balanced_full_xray_10_class" \
    "datasets/balanced/full_xray_10_class/labels-knee" \
    "datasets/balanced/full_xray_10_class/labels"

# Balanced + Resize Only
evaluate_experiment \
    "e2e_10class_balanced_resize" 10 \
    "datasets/processed_balanced/full_xray/resize_only/images" \
    "datasets/splits/balanced_full_xray_10_class" \
    "datasets/processed_balanced/full_xray/resize_only/labels-knee" \
    "datasets/processed_balanced/full_xray/resize_only/labels_10_class"

# Balanced + Blur CLAHE
evaluate_experiment \
    "e2e_10class_balanced_blur" 10 \
    "datasets/processed_balanced/full_xray/blur_clahe2/images" \
    "datasets/splits/balanced_full_xray_10_class" \
    "datasets/processed_balanced/full_xray/blur_clahe2/labels-knee" \
    "datasets/processed_balanced/full_xray/blur_clahe2/labels_10_class"

# Balanced + Sharp CLAHE
evaluate_experiment \
    "e2e_10class_balanced_sharp" 10 \
    "datasets/processed_balanced/full_xray/sharp_clahe4/images" \
    "datasets/splits/balanced_full_xray_10_class" \
    "datasets/processed_balanced/full_xray/sharp_clahe4/labels-knee" \
    "datasets/processed_balanced/full_xray/sharp_clahe4/labels_10_class"


# ============================================================
# 2. 5-Class Experiments
# ============================================================
# Baseline (Unbalanced)
evaluate_experiment \
    "e2e_5class_unbalanced" 5 \
    "datasets/dataset_v0/images" \
    "datasets/splits/knee_full_10_class" \
    "datasets/dataset_v0/labels-knee" \
    "datasets/dataset_v0/labels"

# Balanced
evaluate_experiment \
    "e2e_5class_balanced" 5 \
    "datasets/balanced/full_xray/images" \
    "datasets/splits/balanced_full_xray_10_class" \
    "datasets/balanced/full_xray/labels-knee" \
    "datasets/balanced/full_xray/labels"

# ... (Add others if needed, but these are the main ones running)

echo "========================================================"
echo "ALL END-TO-END EVALUATIONS COMPLETED"
echo "========================================================"
