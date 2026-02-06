#!/bin/bash

# Script to run all End-to-End KIOCMIL Training Experiments
# Usage: bash scripts/training/end_to_end/run_all_end_to_end_experiments.sh

set -e  # Exit on error

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate klgrade || echo "⚠️  Warning: Failed to activate klgrade env"

# Common settings
EPOCHS=100
BATCH_SIZE=4  # Conservative batch size to prevent OOM
PATIENCE=15

echo "========================================================"
echo "STARTING ALL END-TO-END EXPERIMENTS"
echo "========================================================"

# Function to run experiment
run_experiment() {
    local EXP_NAME=$1
    local NUM_CLASSES=$2
    local IMG_DIR=$3
    local SPLIT_DIR=$4
    local KNEE_LABEL_DIR=$5
    local LESION_LABEL_DIR=$6
    
    local SAVE_DIR="runs/end_to_end/$EXP_NAME"

    echo ""
    echo "--------------------------------------------------------"
    echo "Running Experiment: $EXP_NAME"
    echo "Classes: $NUM_CLASSES"
    echo "Images: $IMG_DIR"
    echo "Splits: $SPLIT_DIR"
    echo "Save Dir: $SAVE_DIR"
    echo "--------------------------------------------------------"

    python scripts/training/end_to_end/train_end_to_end.py \
        --backbone yolo11l \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --save-dir "$SAVE_DIR" \
        --num_classes $NUM_CLASSES \
        --patience $PATIENCE \
        --train-img-dir "$IMG_DIR" \
        --train-knee-label-dir "$KNEE_LABEL_DIR" \
        --train-lesion-label-dir "$LESION_LABEL_DIR" \
        --train-split-file "$SPLIT_DIR/train.txt" \
        --val-img-dir "$IMG_DIR" \
        --val-knee-label-dir "$KNEE_LABEL_DIR" \
        --val-lesion-label-dir "$LESION_LABEL_DIR" \
        --val-split-file "$SPLIT_DIR/val.txt"

    echo "✅ Completed: $EXP_NAME"
}

# ============================================================
# 1. 10-Class Experiments
# ============================================================
# Baseline (Unbalanced)
run_experiment \
    "e2e_10class_unbalanced" 10 \
    "datasets/dataset_v0/images" \
    "datasets/splits/knee_full_10_class" \
    "datasets/dataset_v0/labels-knee" \
    "datasets/dataset_v0/labels_10_class"

# Balanced
run_experiment \
    "e2e_10class_balanced" 10 \
    "datasets/balanced/full_xray_10_class/images" \
    "datasets/splits/balanced_full_xray_10_class" \
    "datasets/balanced/full_xray_10_class/labels-knee" \
    "datasets/balanced/full_xray_10_class/labels"

# Balanced + Resize Only
run_experiment \
    "e2e_10class_balanced_resize" 10 \
    "datasets/processed_balanced/full_xray/resize_only/images" \
    "datasets/splits/balanced_full_xray_10_class" \
    "datasets/processed_balanced/full_xray/resize_only/labels-knee" \
    "datasets/processed_balanced/full_xray/resize_only/labels_10_class"

# Balanced + Blur CLAHE
run_experiment \
    "e2e_10class_balanced_blur" 10 \
    "datasets/processed_balanced/full_xray/blur_clahe2/images" \
    "datasets/splits/balanced_full_xray_10_class" \
    "datasets/processed_balanced/full_xray/blur_clahe2/labels-knee" \
    "datasets/processed_balanced/full_xray/blur_clahe2/labels_10_class"

# Balanced + Sharp CLAHE
run_experiment \
    "e2e_10class_balanced_sharp" 10 \
    "datasets/processed_balanced/full_xray/sharp_clahe4/images" \
    "datasets/splits/balanced_full_xray_10_class" \
    "datasets/processed_balanced/full_xray/sharp_clahe4/labels-knee" \
    "datasets/processed_balanced/full_xray/sharp_clahe4/labels_10_class"


# ============================================================
# 2. 5-Class Experiments
# ============================================================
# Baseline (Unbalanced)
run_experiment \
    "e2e_5class_unbalanced" 5 \
    "datasets/dataset_v0/images" \
    "datasets/splits/knee_full_10_class" \
    "datasets/dataset_v0/labels-knee" \
    "datasets/dataset_v0/labels"

# Balanced
run_experiment \
    "e2e_5class_balanced" 5 \
    "datasets/balanced/full_xray/images" \
    "datasets/splits/balanced_full_xray_10_class" \
    "datasets/balanced/full_xray/labels-knee" \
    "datasets/balanced/full_xray/labels"

# Balanced + Resize Only
run_experiment \
    "e2e_5class_balanced_resize" 5 \
    "datasets/processed_balanced/full_xray/resize_only/images" \
    "datasets/splits/balanced_full_xray_10_class" \
    "datasets/processed_balanced/full_xray/resize_only/labels-knee" \
    "datasets/processed_balanced/full_xray/resize_only/labels"

# Balanced + Blur CLAHE
run_experiment \
    "e2e_5class_balanced_blur" 5 \
    "datasets/processed_balanced/full_xray/blur_clahe2/images" \
    "datasets/splits/balanced_full_xray_10_class" \
    "datasets/processed_balanced/full_xray/blur_clahe2/labels-knee" \
    "datasets/processed_balanced/full_xray/blur_clahe2/labels"

# Balanced + Sharp CLAHE
run_experiment \
    "e2e_5class_balanced_sharp" 5 \
    "datasets/processed_balanced/full_xray/sharp_clahe4/images" \
    "datasets/splits/balanced_full_xray_10_class" \
    "datasets/processed_balanced/full_xray/sharp_clahe4/labels-knee" \
    "datasets/processed_balanced/full_xray/sharp_clahe4/labels"


# ============================================================
# 3. 4-Class Experiments
# ============================================================
# Baseline (Unbalanced)
run_experiment \
    "e2e_4class_unbalanced" 4 \
    "datasets/dataset_v0_4_class/images" \
    "datasets/splits/knee_full_4_class" \
    "datasets/dataset_v0/labels-knee" \
    "datasets/dataset_v0_4_class/labels"

# Balanced
run_experiment \
    "e2e_4class_balanced" 4 \
    "datasets/balanced/full_xray_4_class/images" \
    "datasets/splits/balanced_full_xray_4_class" \
    "datasets/balanced/full_xray_4_class/labels-knee" \
    "datasets/balanced/full_xray_4_class/labels"

# Balanced + Resize Only
run_experiment \
    "e2e_4class_balanced_resize" 4 \
    "datasets/processed_balanced/full_xray_4_class/resize_only/images" \
    "datasets/splits/balanced_full_xray_4_class" \
    "datasets/processed_balanced/full_xray_4_class/resize_only/labels-knee" \
    "datasets/processed_balanced/full_xray_4_class/resize_only/labels"

# Balanced + Blur CLAHE
run_experiment \
    "e2e_4class_balanced_blur" 4 \
    "datasets/processed_balanced/full_xray_4_class/blur_clahe2/images" \
    "datasets/splits/balanced_full_xray_4_class" \
    "datasets/processed_balanced/full_xray_4_class/blur_clahe2/labels-knee" \
    "datasets/processed_balanced/full_xray_4_class/blur_clahe2/labels"

# Balanced + Sharp CLAHE
run_experiment \
    "e2e_4class_balanced_sharp" 4 \
    "datasets/processed_balanced/full_xray_4_class/sharp_clahe4/images" \
    "datasets/splits/balanced_full_xray_4_class" \
    "datasets/processed_balanced/full_xray_4_class/sharp_clahe4/labels-knee" \
    "datasets/processed_balanced/full_xray_4_class/sharp_clahe4/labels"


# ============================================================
# 4. 8-Class Experiments
# ============================================================
# Baseline (Unbalanced)
run_experiment \
    "e2e_8class_unbalanced" 8 \
    "datasets/dataset_v0_4_class/images" \
    "datasets/splits/knee_full_8_class" \
    "datasets/dataset_v0/labels-knee" \
    "datasets/dataset_v0_4_class/labels_8_class"

# Balanced
run_experiment \
    "e2e_8class_balanced" 8 \
    "datasets/balanced/full_xray_8_class/images" \
    "datasets/splits/balanced_full_xray_8_class" \
    "datasets/balanced/full_xray_8_class/labels-knee" \
    "datasets/balanced/full_xray_8_class/labels"

# Balanced + Resize Only
run_experiment \
    "e2e_8class_balanced_resize" 8 \
    "datasets/processed_balanced/full_xray_8_class/resize_only/images" \
    "datasets/splits/balanced_full_xray_8_class" \
    "datasets/processed_balanced/full_xray_8_class/resize_only/labels-knee" \
    "datasets/processed_balanced/full_xray_8_class/resize_only/labels"

# Balanced + Blur CLAHE
run_experiment \
    "e2e_8class_balanced_blur" 8 \
    "datasets/processed_balanced/full_xray_8_class/blur_clahe2/images" \
    "datasets/splits/balanced_full_xray_8_class" \
    "datasets/processed_balanced/full_xray_8_class/blur_clahe2/labels-knee" \
    "datasets/processed_balanced/full_xray_8_class/blur_clahe2/labels"

# Balanced + Sharp CLAHE
run_experiment \
    "e2e_8class_balanced_sharp" 8 \
    "datasets/processed_balanced/full_xray_8_class/sharp_clahe4/images" \
    "datasets/splits/balanced_full_xray_8_class" \
    "datasets/processed_balanced/full_xray_8_class/sharp_clahe4/labels-knee" \
    "datasets/processed_balanced/full_xray_8_class/sharp_clahe4/labels"

echo "========================================================"
echo "ALL END-TO-END EXPERIMENTS COMPLETED"
echo "========================================================"
