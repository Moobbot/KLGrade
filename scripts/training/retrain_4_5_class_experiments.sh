#!/bin/bash

# Script to re-train ONLY 4-class and 5-class KIOCMIL CADA experiments
# These models were trained with wrong num_classes (10 instead of 4/5)
# This script re-trains them with the correct configuration

set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate klgrade || echo "⚠️  Warning: Failed to activate klgrade env"

# Common settings
EPOCHS=100
BATCH_SIZE=16
PROJECT_NAME="klgrade-kiocmil-cada"

echo "=========================================================="
echo "RE-TRAINING 4-CLASS AND 5-CLASS KIOCMIL CADA EXPERIMENTS"
echo "=========================================================="
echo ""
echo "These models were previously trained with wrong num_classes"
echo "Re-training with FIXED training script..."
echo ""

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
    echo "Re-training Experiment: $EXP_NAME"
    echo "Classes: $NUM_CLASSES (CORRECTED)"
    echo "Images: $IMG_DIR"
    echo "Splits: $SPLIT_DIR"
    echo "--------------------------------------------------------"

    python src/training/train_kiocmil_cada.py \
        --wandb_project $PROJECT_NAME \
        --wandb_name "${EXP_NAME}_v2" \
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
        --save_dir "runs/kiocmil_cada/${EXP_NAME}_corrected"
}

# ============================================================
# 5-Class Experiments (5 experiments)
# ============================================================
echo ""
echo "=========================================================="
echo "PHASE 1: RE-TRAINING 5-CLASS EXPERIMENTS"
echo "=========================================================="

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

# ============================================================
# 4-Class Experiments (5 experiments)
# ============================================================
echo ""
echo "=========================================================="
echo "PHASE 2: RE-TRAINING 4-CLASS EXPERIMENTS"
echo "=========================================================="

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

run_experiment \
    "cada_4class_balanced_resize" 4 \
    "datasets/processed_balanced/full_xray_4_class/resize_only/images" \
    "datasets/splits/balanced_full_xray_4_class" \
    "datasets/processed_balanced/full_xray_4_class/resize_only/labels-knee" \
    "datasets/processed_balanced/full_xray_4_class/resize_only/labels"

run_experiment \
    "cada_4class_balanced_blur" 4 \
    "datasets/processed_balanced/full_xray_4_class/blur_clahe2/images" \
    "datasets/splits/balanced_full_xray_4_class" \
    "datasets/processed_balanced/full_xray_4_class/blur_clahe2/labels-knee" \
    "datasets/processed_balanced/full_xray_4_class/blur_clahe2/labels"

run_experiment \
    "cada_4class_balanced_sharp" 4 \
    "datasets/processed_balanced/full_xray_4_class/sharp_clahe4/images" \
    "datasets/splits/balanced_full_xray_4_class" \
    "datasets/processed_balanced/full_xray_4_class/sharp_clahe4/labels-knee" \
    "datasets/processed_balanced/full_xray_4_class/sharp_clahe4/labels"

echo "=========================================================="
echo "ALL 10 RE-TRAINING EXPERIMENTS COMPLETED"
echo "=========================================================="
echo ""
echo "Summary:"
echo "- 5-class experiments: 5 models re-trained"
echo "- 4-class experiments: 5 models re-trained"
echo "- Total: 10 models corrected"
echo ""
echo "New models saved to: runs/kiocmil_cada/*_corrected/"
echo ""
