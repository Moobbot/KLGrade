#!/bin/bash

# Script to run all KIOCMIL CADA training experiments
# Usage: bash scripts/training/run_all_cada_experiments.sh

set -e  # Exit on error

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate klgrade || echo "⚠️  Warning: Failed to activate klgrade env"

# Common settings
EPOCHS=100
BATCH_SIZE=16
PROJECT_NAME="klgrade-kiocmil-cada"

echo "========================================================"
echo "STARTING ALL KIOCMIL CADA EXPERIMENTS"
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

# ============================================================
# 1. 10-Class Experiments
# ============================================================
# Baseline (Unbalanced)
run_experiment \
    "cada_10class_unbalanced" 10 \
    "datasets/dataset_v0/images" \
    "datasets/splits/knee_full_10_class" \
    "datasets/dataset_v0/labels-knee" \
    "datasets/dataset_v0/labels_10_class"

# Balanced
run_experiment \
    "cada_10class_balanced" 10 \
    "datasets/balanced/full_xray_10_class/images" \
    "datasets/splits/balanced_full_xray_10_class" \
    "datasets/balanced/full_xray_10_class/labels-knee" \
    "datasets/balanced/full_xray_10_class/labels"

# Balanced + Resize Only
run_experiment \
    "cada_10class_balanced_resize" 10 \
    "datasets/processed_balanced/full_xray/resize_only/images" \
    "datasets/splits/balanced_full_xray_10_class" \
    "datasets/processed_balanced/full_xray/resize_only/labels-knee" \
    "datasets/processed_balanced/full_xray/resize_only/labels_10_class"

# Balanced + Blur CLAHE
run_experiment \
    "cada_10class_balanced_blur" 10 \
    "datasets/processed_balanced/full_xray/blur_clahe2/images" \
    "datasets/splits/balanced_full_xray_10_class" \
    "datasets/processed_balanced/full_xray/blur_clahe2/labels-knee" \
    "datasets/processed_balanced/full_xray/blur_clahe2/labels_10_class"

# Balanced + Sharp CLAHE
run_experiment \
    "cada_10class_balanced_sharp" 10 \
    "datasets/processed_balanced/full_xray/sharp_clahe4/images" \
    "datasets/splits/balanced_full_xray_10_class" \
    "datasets/processed_balanced/full_xray/sharp_clahe4/labels-knee" \
    "datasets/processed_balanced/full_xray/sharp_clahe4/labels_10_class"


# ============================================================
# 2. 5-Class Experiments
# ============================================================
# Baseline (Unbalanced)
# Note: Using knee_full_10_class splits as proxy if specific 5-class full splits don't exist, 
# or assuming a 5-class split folder exists. Let's start with standard splits.
# Step 5 likely created splits/dataset_v0 which is 5-class?
# Let's check splits below. Assuming splits/dataset_v0/ exists. 
# actually step 5 creates 'knee_full_10_class'. 
# For 5-class, we can use the SAME split file as 10-class because images are the same.
# Just the labels differ.

run_experiment \
    "cada_5class_unbalanced" 5 \
    "datasets/dataset_v0/images" \
    "datasets/splits/knee_full_10_class" \
    "datasets/dataset_v0/labels-knee" \
    "datasets/dataset_v0/labels"

# Balanced
run_experiment \
    "cada_5class_balanced" 5 \
    "datasets/balanced/full_xray/images" \
    "datasets/splits/balanced_full_xray_10_class" \
    "datasets/balanced/full_xray/labels-knee" \
    "datasets/balanced/full_xray/labels"

# Balanced + Resize Only
run_experiment \
    "cada_5class_balanced_resize" 5 \
    "datasets/processed_balanced/full_xray/resize_only/images" \
    "datasets/splits/balanced_full_xray_10_class" \
    "datasets/processed_balanced/full_xray/resize_only/labels-knee" \
    "datasets/processed_balanced/full_xray/resize_only/labels"

# Balanced + Blur CLAHE
run_experiment \
    "cada_5class_balanced_blur" 5 \
    "datasets/processed_balanced/full_xray/blur_clahe2/images" \
    "datasets/splits/balanced_full_xray_10_class" \
    "datasets/processed_balanced/full_xray/blur_clahe2/labels-knee" \
    "datasets/processed_balanced/full_xray/blur_clahe2/labels"

# Balanced + Sharp CLAHE
run_experiment \
    "cada_5class_balanced_sharp" 5 \
    "datasets/processed_balanced/full_xray/sharp_clahe4/images" \
    "datasets/splits/balanced_full_xray_10_class" \
    "datasets/processed_balanced/full_xray/sharp_clahe4/labels-knee" \
    "datasets/processed_balanced/full_xray/sharp_clahe4/labels"

# ============================================================
# 3. 4-Class Experiments
# ============================================================
# Baseline (Unbalanced)
run_experiment \
    "cada_4class_unbalanced" 4 \
    "datasets/dataset_v0_4_class/images" \
    "datasets/splits/knee_full_4_class" \
    "datasets/dataset_v0/labels-knee" \
    "datasets/dataset_v0_4_class/labels"

# Balanced
run_experiment \
    "cada_4class_balanced" 4 \
    "datasets/balanced/full_xray_4_class/images" \
    "datasets/splits/balanced_full_xray_4_class" \
    "datasets/balanced/full_xray_4_class/labels-knee" \
    "datasets/balanced/full_xray_4_class/labels"

# ============================================================
# 4. 8-Class Experiments
# ============================================================
# Baseline (Unbalanced)
run_experiment \
    "cada_8class_unbalanced" 8 \
    "datasets/dataset_v0_4_class/images" \
    "datasets/splits/knee_full_8_class" \
    "datasets/dataset_v0/labels-knee" \
    "datasets/dataset_v0_4_class/labels_8_class"

# Balanced
run_experiment \
    "cada_8class_balanced" 8 \
    "datasets/balanced/full_xray_8_class/images" \
    "datasets/splits/balanced_full_xray_8_class" \
    "datasets/balanced/full_xray_8_class/labels-knee" \
    "datasets/balanced/full_xray_8_class/labels"

echo "========================================================"
echo "ALL 8 EXPERIMENTS COMPLETED"
echo "========================================================"
