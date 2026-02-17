#!/bin/bash

# Train 8-class and 10-class KIOCMIL CADA models  
# Usage: bash scripts/training/run_train_8_10_class.sh

set -e

eval "$(conda shell.bash hook)"
conda activate klgrade || echo "⚠️  Warning: Failed to activate klgrade env"

EPOCHS=100
BATCH_SIZE=16
PROJECT_NAME="klgrade-kiocmil-cada"

echo "========================================================"
echo "TRAINING 8-CLASS AND 10-CLASS KIOCMIL CADA MODELS"
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

# 10-Class Models
echo ""
echo "========================================================"
echo "PHASE 1: 10-CLASS MODELS"
echo "========================================================"

run_experiment \
    "cada_10class_unbalanced" 10 \
    "datasets/dataset_v0/images" \
    "datasets/splits/knee_full_10_class" \
    "datasets/dataset_v0/labels-knee" \
    "datasets/dataset_v0/labels_10_class"

run_experiment \
    "cada_10class_balanced" 10 \
    "datasets/balanced/full_xray_10_class/images" \
    "datasets/splits/balanced_full_xray_10_class" \
    "datasets/balanced/full_xray_10_class/labels-knee" \
    "datasets/balanced/full_xray_10_class/labels"

# 8-Class Models
echo ""
echo "========================================================"
echo "PHASE 2: 8-CLASS MODELS"
echo "========================================================"

run_experiment \
    "cada_8class_unbalanced" 8 \
    "datasets/dataset_v0_4_class/images" \
    "datasets/splits/knee_full_8_class" \
    "datasets/dataset_v0/labels-knee" \
    "datasets/dataset_v0_4_class/labels_8_class"

run_experiment \
    "cada_8class_balanced" 8 \
    "datasets/balanced/full_xray_8_class/images" \
    "datasets/splits/balanced_full_xray_8_class" \
    "datasets/balanced/full_xray_8_class/labels-knee" \
    "datasets/balanced/full_xray_8_class/labels"

echo "========================================================"
echo "8-CLASS AND 10-CLASS TRAINING COMPLETED"
echo "========================================================"
