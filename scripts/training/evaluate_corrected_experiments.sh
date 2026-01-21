#!/bin/bash

# Script to evaluate CORRECTED 4-class and 5-class KIOCMIL CADA experiments
# These models were re-trained with correct num_classes

set -e

# Activate env
eval "$(conda shell.bash hook)"
conda activate klgrade || echo "⚠️  Warning: Failed to activate klgrade env"

echo "=========================================================="
echo "EVALUATING CORRECTED 4-CLASS AND 5-CLASS EXPERIMENTS"
echo "=========================================================="

# Function to run evaluation
evaluate_experiment() {
    local EXP_NAME=$1
    local NUM_CLASSES=$2
    local IMG_DIR=$3
    local SPLIT_DIR=$4
    local KNEE_LABEL_DIR=$5
    local LESION_LABEL_DIR=$6
    
    MODEL_PATH="runs/kiocmil_cada/${EXP_NAME}_corrected/best_acc_model.pt"
    
    if [ ! -f "$MODEL_PATH" ]; then
        echo "❌ Model not found: $MODEL_PATH"
        return
    fi
    
    echo ""
    echo "--------------------------------------------------------"
    echo "Evaluating: ${EXP_NAME}_corrected"
    echo "Classes: $NUM_CLASSES (CORRECTED)"
    echo "Model: $MODEL_PATH"
    echo "--------------------------------------------------------"

    python src/training/evaluate_kiocmil_cada.py \
        --model_path "$MODEL_PATH" \
        --num_classes $NUM_CLASSES \
        --img_dir "$IMG_DIR" \
        --split_file "$SPLIT_DIR/val.txt" \
        --knee_label_dir "$KNEE_LABEL_DIR" \
        --lesion_label_dir "$LESION_LABEL_DIR" \
        --save_dir "runs/kiocmil_cada/${EXP_NAME}_corrected/evaluation" \
        --batch_size 16
}

# ============================================================
# 5-Class Experiments
# ============================================================
echo ""
echo "==========================================================
PHASE 1: EVALUATING 5-CLASS CORRECTED MODELS"
echo "=========================================================="

evaluate_experiment \
    "cada_5class_unbalanced" 5 \
    "datasets/dataset_v0/images" \
    "datasets/splits/knee_full_10_class" \
    "datasets/dataset_v0/labels-knee" \
    "datasets/dataset_v0/labels"

evaluate_experiment \
    "cada_5class_balanced" 5 \
    "datasets/balanced/full_xray/images" \
    "datasets/splits/balanced_full_xray_10_class" \
    "datasets/balanced/full_xray/labels-knee" \
    "datasets/balanced/full_xray/labels"

evaluate_experiment \
    "cada_5class_balanced_resize" 5 \
    "datasets/processed_balanced/full_xray/resize_only/images" \
    "datasets/splits/balanced_full_xray_10_class" \
    "datasets/processed_balanced/full_xray/resize_only/labels-knee" \
    "datasets/processed_balanced/full_xray/resize_only/labels"

evaluate_experiment \
    "cada_5class_balanced_blur" 5 \
    "datasets/processed_balanced/full_xray/blur_clahe2/images" \
    "datasets/splits/balanced_full_xray_10_class" \
    "datasets/processed_balanced/full_xray/blur_clahe2/labels-knee" \
    "datasets/processed_balanced/full_xray/blur_clahe2/labels"

evaluate_experiment \
    "cada_5class_balanced_sharp" 5 \
    "datasets/processed_balanced/full_xray/sharp_clahe4/images" \
    "datasets/splits/balanced_full_xray_10_class" \
    "datasets/processed_balanced/full_xray/sharp_clahe4/labels-knee" \
    "datasets/processed_balanced/full_xray/sharp_clahe4/labels"

# ============================================================
# 4-Class Experiments
# ============================================================
echo ""
echo "=========================================================="
echo "PHASE 2: EVALUATING 4-CLASS CORRECTED MODELS"
echo "=========================================================="

evaluate_experiment \
    "cada_4class_unbalanced" 4 \
    "datasets/dataset_v0_4_class/images" \
    "datasets/splits/knee_full_4_class" \
    "datasets/dataset_v0/labels-knee" \
    "datasets/dataset_v0_4_class/labels"

evaluate_experiment \
    "cada_4class_balanced" 4 \
    "datasets/balanced/full_xray_4_class/images" \
    "datasets/splits/balanced_full_xray_4_class" \
    "datasets/balanced/full_xray_4_class/labels-knee" \
    "datasets/balanced/full_xray_4_class/labels"

evaluate_experiment \
    "cada_4class_balanced_resize" 4 \
    "datasets/processed_balanced/full_xray_4_class/resize_only/images" \
    "datasets/splits/balanced_full_xray_4_class" \
    "datasets/processed_balanced/full_xray_4_class/resize_only/labels-knee" \
    "datasets/processed_balanced/full_xray_4_class/resize_only/labels"

evaluate_experiment \
    "cada_4class_balanced_blur" 4 \
    "datasets/processed_balanced/full_xray_4_class/blur_clahe2/images" \
    "datasets/splits/balanced_full_xray_4_class" \
    "datasets/processed_balanced/full_xray_4_class/blur_clahe2/labels-knee" \
    "datasets/processed_balanced/full_xray_4_class/blur_clahe2/labels"

evaluate_experiment \
    "cada_4class_balanced_sharp" 4 \
    "datasets/processed_balanced/full_xray_4_class/sharp_clahe4/images" \
    "datasets/splits/balanced_full_xray_4_class" \
    "datasets/processed_balanced/full_xray_4_class/sharp_clahe4/labels-knee" \
    "datasets/processed_balanced/full_xray_4_class/sharp_clahe4/labels"

echo "=========================================================="
echo "ALL CORRECTED MODEL EVALUATIONS COMPLETED"
echo "=========================================================="
