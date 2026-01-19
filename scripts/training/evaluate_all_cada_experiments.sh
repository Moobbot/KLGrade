#!/bin/bash

# Script to evaluate all KIOCMIL CADA experiments
# Usage: bash scripts/training/evaluate_all_cada_experiments.sh

set -e

# Activate env
eval "$(conda shell.bash hook)"
conda activate klgrade || echo "⚠️  Warning: Failed to activate klgrade env"

echo "========================================================"
echo "STARTING EVALUATION OF ALL KIOCMIL CADA EXPERIMENTS"
echo "========================================================"

# Function to run evaluation
evaluate_experiment() {
    local EXP_NAME=$1
    local NUM_CLASSES=$2
    local IMG_DIR=$3
    local SPLIT_DIR=$4
    local KNEE_LABEL_DIR=$5
    local LESION_LABEL_DIR=$6
    
    MODEL_PATH="runs/kiocmil_cada/$EXP_NAME/best_acc_model.pt"
    
    # Check if best_acc_model exists, fallback to best_model if not
    if [ ! -f "$MODEL_PATH" ]; then
        MODEL_PATH="runs/kiocmil_cada/$EXP_NAME/best_model.pt"
    fi

    if [ ! -f "$MODEL_PATH" ]; then
        echo "❌ Model not found for $EXP_NAME: $MODEL_PATH"
        return
    fi
    
    echo ""
    echo "--------------------------------------------------------"
    echo "Evaluating Experiment: $EXP_NAME"
    echo "Model: $MODEL_PATH"
    echo "Classes: $NUM_CLASSES"
    echo "--------------------------------------------------------"

    python src/training/evaluate_kiocmil_cada.py \
        --model_path "$MODEL_PATH" \
        --num_classes $NUM_CLASSES \
        --img_dir "$IMG_DIR" \
        --split_file "$SPLIT_DIR/val.txt" \
        --knee_label_dir "$KNEE_LABEL_DIR" \
        --lesion_label_dir "$LESION_LABEL_DIR" \
        --save_dir "runs/kiocmil_cada/$EXP_NAME/evaluation" \
        --batch_size 16
}

# ============================================================
# 1. 10-Class Experiments
# ============================================================
evaluate_experiment \
    "cada_10class_unbalanced" 10 \
    "datasets/dataset_v0/images" \
    "datasets/splits/knee_full_10_class" \
    "datasets/dataset_v0/labels-knee" \
    "datasets/dataset_v0/labels_10_class"

evaluate_experiment \
    "cada_10class_balanced" 10 \
    "datasets/balanced/full_xray_10_class/images" \
    "datasets/splits/balanced_full_xray_10_class" \
    "datasets/balanced/full_xray_10_class/labels-knee" \
    "datasets/balanced/full_xray_10_class/labels"

# Balanced + Resize Only
evaluate_experiment \
    "cada_10class_balanced_resize" 10 \
    "datasets/processed_balanced/full_xray/resize_only/images" \
    "datasets/splits/balanced_full_xray_10_class" \
    "datasets/processed_balanced/full_xray/resize_only/labels-knee" \
    "datasets/processed_balanced/full_xray/resize_only/labels_10_class"

# Balanced + Blur CLAHE
evaluate_experiment \
    "cada_10class_balanced_blur" 10 \
    "datasets/processed_balanced/full_xray/blur_clahe2/images" \
    "datasets/splits/balanced_full_xray_10_class" \
    "datasets/processed_balanced/full_xray/blur_clahe2/labels-knee" \
    "datasets/processed_balanced/full_xray/blur_clahe2/labels_10_class"

# Balanced + Sharp CLAHE
evaluate_experiment \
    "cada_10class_balanced_sharp" 10 \
    "datasets/processed_balanced/full_xray/sharp_clahe4/images" \
    "datasets/splits/balanced_full_xray_10_class" \
    "datasets/processed_balanced/full_xray/sharp_clahe4/labels-knee" \
    "datasets/processed_balanced/full_xray/sharp_clahe4/labels_10_class"

# ============================================================
# 2. 5-Class Experiments
# ============================================================
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

# Balanced + Resize Only
evaluate_experiment \
    "cada_5class_balanced_resize" 5 \
    "datasets/processed_balanced/full_xray/resize_only/images" \
    "datasets/splits/balanced_full_xray_10_class" \
    "datasets/processed_balanced/full_xray/resize_only/labels-knee" \
    "datasets/processed_balanced/full_xray/resize_only/labels"

# Balanced + Blur CLAHE
evaluate_experiment \
    "cada_5class_balanced_blur" 5 \
    "datasets/processed_balanced/full_xray/blur_clahe2/images" \
    "datasets/splits/balanced_full_xray_10_class" \
    "datasets/processed_balanced/full_xray/blur_clahe2/labels-knee" \
    "datasets/processed_balanced/full_xray/blur_clahe2/labels"

# Balanced + Sharp CLAHE
evaluate_experiment \
    "cada_5class_balanced_sharp" 5 \
    "datasets/processed_balanced/full_xray/sharp_clahe4/images" \
    "datasets/splits/balanced_full_xray_10_class" \
    "datasets/processed_balanced/full_xray/sharp_clahe4/labels-knee" \
    "datasets/processed_balanced/full_xray/sharp_clahe4/labels"

# ============================================================
# 3. 4-Class Experiments
# ============================================================
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

# ============================================================
# 4. 8-Class Experiments
# ============================================================
evaluate_experiment \
    "cada_8class_unbalanced" 8 \
    "datasets/dataset_v0_4_class/images" \
    "datasets/splits/knee_full_8_class" \
    "datasets/dataset_v0/labels-knee" \
    "datasets/dataset_v0_4_class/labels_8_class"

evaluate_experiment \
    "cada_8class_balanced" 8 \
    "datasets/balanced/full_xray_8_class/images" \
    "datasets/splits/balanced_full_xray_8_class" \
    "datasets/balanced/full_xray_8_class/labels-knee" \
    "datasets/balanced/full_xray_8_class/labels"

echo "========================================================"
echo "ALL EVALUATIONS COMPLETED"
echo "========================================================"
