#!/bin/bash

# Train preprocessed 5-class and 10-class KIOCMIL CADA models
# Includes resize, blur+CLAHE, sharp+CLAHE variations
# Usage: bash scripts/training/run_processed_5_10_class.sh

set -e

eval "$(conda shell.bash hook)"
conda activate klgrade || echo "⚠️  Warning: Failed to activate klgrade env"

EPOCHS=100
BATCH_SIZE=16
PROJECT_NAME="klgrade-kiocmil-cada"

echo "========================================================"
echo "TRAINING PREPROCESSED 5-CLASS AND 10-CLASS MODELS"
echo "========================================================"

run_experiment() {
    local EXP_NAME=$1
    local NUM_CLASSES=$2
    local IMG_DIR=$3
    local SPLIT_DIR=$4
    local KNEE_LABEL_DIR=$5
    local LESION_LABEL_DIR=$6
    
    echo ""
    echo "--------------------------------------------------------"
    echo "Training: $EXP_NAME"
    echo "Classes: $NUM_CLASSES"
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

# 5-Class Preprocessed
echo ""
echo "========================================================"
echo "PHASE 1: 5-CLASS PREPROCESSED MODELS"
echo "========================================================"

run_experiment \
    "cada_5class_balanced_resize" 5 \
    "datasets/processed_balanced/full_xray/resize_only/images" \
    "datasets/splits/balanced_full_xray_10_class" \
    "datasets/processed_balanced/full_xray/resize_only/labels-knee" \
    "datasets/processed_balanced/full_xray/resize_only/labels"

run_experiment \
    "cada_5class_balanced_blur" 5 \
    "datasets/processed_balanced/full_xray/blur_clahe2/images" \
    "datasets/splits/balanced_full_xray_10_class" \
    "datasets/processed_balanced/full_xray/blur_clahe2/labels-knee" \
    "datasets/processed_balanced/full_xray/blur_clahe2/labels"

run_experiment \
    "cada_5class_balanced_sharp" 5 \
    "datasets/processed_balanced/full_xray/sharp_clahe4/images" \
    "datasets/splits/balanced_full_xray_10_class" \
    "datasets/processed_balanced/full_xray/sharp_clahe4/labels-knee" \
    "datasets/processed_balanced/full_xray/sharp_clahe4/labels"

# 10-Class Preprocessed
echo ""
echo "========================================================"
echo "PHASE 2: 10-CLASS PREPROCESSED MODELS"
echo "========================================================"

run_experiment \
    "cada_10class_balanced_resize" 10 \
    "datasets/processed_balanced/full_xray/resize_only/images" \
    "datasets/splits/balanced_full_xray_10_class" \
    "datasets/processed_balanced/full_xray/resize_only/labels-knee" \
    "datasets/processed_balanced/full_xray/resize_only/labels_10_class"

run_experiment \
    "cada_10class_balanced_blur" 10 \
    "datasets/processed_balanced/full_xray/blur_clahe2/images" \
    "datasets/splits/balanced_full_xray_10_class" \
    "datasets/processed_balanced/full_xray/blur_clahe2/labels-knee" \
    "datasets/processed_balanced/full_xray/blur_clahe2/labels_10_class"

run_experiment \
    "cada_10class_balanced_sharp" 10 \
    "datasets/processed_balanced/full_xray/sharp_clahe4/images" \
    "datasets/splits/balanced_full_xray_10_class" \
    "datasets/processed_balanced/full_xray/sharp_clahe4/labels-knee" \
    "datasets/processed_balanced/full_xray/sharp_clahe4/labels_10_class"

echo "========================================================"
echo "PREPROCESSED 5-CLASS AND 10-CLASS TRAINING COMPLETED"
echo "========================================================"
