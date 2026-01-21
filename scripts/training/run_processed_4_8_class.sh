#!/bin/bash

# Script to run only the processed 4-class and 8-class experiments
# Usage: bash scripts/training/run_processed_4_8_class.sh

set -e  # Exit on error

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate klgrade || echo "⚠️  Warning: Failed to activate klgrade env"

# Common settings
EPOCHS=100
BATCH_SIZE=16
PROJECT_NAME="klgrade-kiocmil-cada"

echo "========================================================"
echo "STARTING PROCESSED 4-CLASS & 8-CLASS EXPERIMENTS"
echo "========================================================"

# Function to run experiment
run_experiment() {
    local EXP_NAME=$1
    local NUM_CLASSES=$2
    local IMG_DIR=$3
    local SPLIT_DIR=$4
    local KNEE_LABEL_DIR=$5
    local LESION_LABEL_DIR=$6
    
    echo ""
    echo "--------------------------------------------------------"
    echo "Running Experiment: $EXP_NAME"
    echo "Classes: $NUM_CLASSES"
    echo "Images: $IMG_DIR"
    echo "Splits: $SPLIT_DIR"
    echo "--------------------------------------------------------"

    python src/training/train_kiocmil_cada.py \
        --wandb_project $PROJECT_NAME \
        --wandb_name $EXP_NAME \
        --num_classes $NUM_CLASSES \
        --train_img_dir "$IMG_DIR" \
        --val_img_dir "$IMG_DIR" \
        --train_split_file "$SPLIT_DIR/train.txt" \
        --val_split_file "$SPLIT_DIR/val.txt" \
        --train_knee_label_dir "$KNEE_LABEL_DIR" \
        --val_knee_label_dir "$KNEE_LABEL_DIR" \
        --train_lesion_label_dir "$LESION_LABEL_DIR" \
        --val_lesion_label_dir "$LESION_LABEL_DIR" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --save_dir "runs/kiocmil_cada/$EXP_NAME"
}

# 4-Class Processed
# Balanced + Resize Only
run_experiment \
    "cada_4class_balanced_resize" 4 \
    "datasets/processed_balanced/full_xray_4_class/resize_only/images" \
    "datasets/splits/balanced_full_xray_4_class" \
    "datasets/processed_balanced/full_xray_4_class/resize_only/labels-knee" \
    "datasets/processed_balanced/full_xray_4_class/resize_only/labels"

# Balanced + Blur CLAHE
run_experiment \
    "cada_4class_balanced_blur" 4 \
    "datasets/processed_balanced/full_xray_4_class/blur_clahe2/images" \
    "datasets/splits/balanced_full_xray_4_class" \
    "datasets/processed_balanced/full_xray_4_class/blur_clahe2/labels-knee" \
    "datasets/processed_balanced/full_xray_4_class/blur_clahe2/labels"

# Balanced + Sharp CLAHE
run_experiment \
    "cada_4class_balanced_sharp" 4 \
    "datasets/processed_balanced/full_xray_4_class/sharp_clahe4/images" \
    "datasets/splits/balanced_full_xray_4_class" \
    "datasets/processed_balanced/full_xray_4_class/sharp_clahe4/labels-knee" \
    "datasets/processed_balanced/full_xray_4_class/sharp_clahe4/labels"

# 8-Class Processed
# Balanced + Resize Only
run_experiment \
    "cada_8class_balanced_resize" 8 \
    "datasets/processed_balanced/full_xray_8_class/resize_only/images" \
    "datasets/splits/balanced_full_xray_8_class" \
    "datasets/processed_balanced/full_xray_8_class/resize_only/labels-knee" \
    "datasets/processed_balanced/full_xray_8_class/resize_only/labels"

# Balanced + Blur CLAHE
run_experiment \
    "cada_8class_balanced_blur" 8 \
    "datasets/processed_balanced/full_xray_8_class/blur_clahe2/images" \
    "datasets/splits/balanced_full_xray_8_class" \
    "datasets/processed_balanced/full_xray_8_class/blur_clahe2/labels-knee" \
    "datasets/processed_balanced/full_xray_8_class/blur_clahe2/labels"

# Balanced + Sharp CLAHE
run_experiment \
    "cada_8class_balanced_sharp" 8 \
    "datasets/processed_balanced/full_xray_8_class/sharp_clahe4/images" \
    "datasets/splits/balanced_full_xray_8_class" \
    "datasets/processed_balanced/full_xray_8_class/sharp_clahe4/labels-knee" \
    "datasets/processed_balanced/full_xray_8_class/sharp_clahe4/labels"

echo "========================================================"
echo "ALL MISSING EXPERIMENTS COMPLETED"
echo "========================================================"
