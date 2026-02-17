#!/bin/bash

# Train 4-class and 5-class KIOCMIL CADA models
# Usage: bash scripts/training/run_train_4_5_class.sh

set -e

eval "$(conda shell.bash hook)"
conda activate klgrade || echo "⚠️  Warning: Failed to activate klgrade env"

EPOCHS=100
BATCH_SIZE=16
PROJECT_NAME="klgrade-kiocmil-cada"

echo "========================================================"
echo "TRAINING 4-CLASS AND 5-CLASS KIOCMIL CADA MODELS"
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

# 5-Class Models
echo ""
echo "========================================================"
echo "PHASE 1: 5-CLASS MODELS"
echo "========================================================"

run_experiment \
    "cada_5class_unbalanced" 5 \
    "datasets/dataset_v0/images" \
    "datasets/splits/knee_full_10_class" \
    "datasets/dataset_v0/labels-knee" \
    "datasets/dataset_v0/labels"

run_experiment \
    "cada_5class_balanced" 5 \
    "datasets/balanced/full_xray/images" \
    "datasets/splits/balanced_full_xray_10_class" \
    "datasets/balanced/full_xray/labels-knee" \
    "datasets/balanced/full_xray/labels"

# 4-Class Models
echo ""
echo "========================================================"
echo "PHASE 2: 4-CLASS MODELS"
echo "========================================================"

run_experiment \
    "cada_4class_unbalanced" 4 \
    "datasets/dataset_v0_4_class/images" \
    "datasets/splits/knee_full_4_class" \
    "datasets/dataset_v0/labels-knee" \
    "datasets/dataset_v0_4_class/labels"

run_experiment \
    "cada_4class_balanced" 4 \
    "datasets/balanced/full_xray_4_class/images" \
    "datasets/splits/balanced_full_xray_4_class" \
    "datasets/balanced/full_xray_4_class/labels-knee" \
    "datasets/balanced/full_xray_4_class/labels"

echo "========================================================"
echo "4-CLASS AND 5-CLASS TRAINING COMPLETED"
echo "========================================================"
