#!/bin/bash

# Script to run training experiments on different data splits
# Usages: 
#   bash scripts/training/run_split_experiments.sh          # Run all splits
#   bash scripts/training/run_split_experiments.sh _80_10_10   # Run specific split

set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate klgrade || echo "⚠️  Warning: Failed to activate klgrade env"

# Settings
EPOCHS=50  # Reduced epochs for experiments
BATCH_SIZE=16
PROJECT_NAME="klgrade-kiocmil-cada-splits"

# Define splits to run if not provided
if [ -z "$1" ]; then
    SPLIT_SUFFIXES=("_80_10_10" "_60_20_20" "_70_20_10")
else
    SPLIT_SUFFIXES=("$1")
fi

echo "========================================================"
echo "STARTING SPLIT EXPERIMENTS"
echo "Split Suffixes: ${SPLIT_SUFFIXES[*]}"
echo "Epochs: $EPOCHS"
echo "========================================================"

run_experiment() {
    local EXP_NAME=$1
    local NUM_CLASSES=$2
    local IMG_DIR=$3
    local BASE_SPLIT_DIR=$4  # The base split directory name without suffix
    local SUFFIX=$5          # The suffix to append
    local KNEE_LABEL_DIR=$6
    local LESION_LABEL_DIR=$7
    
    # Construct the actual split path
    local SPLIT_DIR="${BASE_SPLIT_DIR}${SUFFIX}"
    
    # Construct unique experiment name
    local RUN_ID="${EXP_NAME}${SUFFIX}"

    echo ""
    echo "--------------------------------------------------------"
    echo "Running Experiment: $RUN_ID"
    echo "Split Dir: $SPLIT_DIR"
    echo "--------------------------------------------------------"

    if [ ! -d "$SPLIT_DIR" ]; then
        echo "⚠️  Skipping: Split directory not found: $SPLIT_DIR"
        return
    fi
    
    # Check if files exist
    if [ ! -f "$SPLIT_DIR/train.txt" ]; then
         echo "⚠️  Skipping: train.txt not found in $SPLIT_DIR"
         return
    fi

    python src/training/train_kiocmil_cada.py \
        --wandb_project $PROJECT_NAME \
        --wandb_name $RUN_ID \
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
        --save_dir "runs/kiocmil_cada/$RUN_ID"
}

# Iterate over each split configuration
for SUFFIX in "${SPLIT_SUFFIXES[@]}"; do
    echo ""
    echo "########################################################"
    echo "PROCESSING EXPERIMENTS FOR SPLIT: $SUFFIX"
    echo "########################################################"
    
    # ============================================================
    # 10-Class Experiments
    # ============================================================
    
    # 1. Baseline (Unbalanced)
    run_experiment \
        "cada_10class_unbalanced" 10 \
        "datasets/dataset_v0/images" \
        "datasets/splits/knee_full_10_class" \
        "$SUFFIX" \
        "datasets/dataset_v0/labels-knee" \
        "datasets/dataset_v0/labels_10_class"

    # 2. Balanced
    run_experiment \
        "cada_10class_balanced" 10 \
        "datasets/balanced/full_xray_10_class/images" \
        "datasets/splits/balanced_full_xray_10_class" \
        "$SUFFIX" \
        "datasets/balanced/full_xray_10_class/labels-knee" \
        "datasets/balanced/full_xray_10_class/labels"

    # 3. Balanced + Resize Only
    run_experiment \
        "cada_10class_balanced_resize" 10 \
        "datasets/processed_balanced/full_xray/resize_only/images" \
        "datasets/splits/balanced_full_xray_10_class" \
        "$SUFFIX" \
        "datasets/processed_balanced/full_xray/resize_only/labels-knee" \
        "datasets/processed_balanced/full_xray/resize_only/labels_10_class"

    # 4. Balanced + Blur CLAHE 2
    run_experiment \
        "cada_10class_balanced_blur" 10 \
        "datasets/processed_balanced/full_xray/blur_clahe2/images" \
        "datasets/splits/balanced_full_xray_10_class" \
        "$SUFFIX" \
        "datasets/processed_balanced/full_xray/blur_clahe2/labels-knee" \
        "datasets/processed_balanced/full_xray/blur_clahe2/labels_10_class"

    # 5. Balanced + Sharp CLAHE 4
    run_experiment \
        "cada_10class_balanced_sharp" 10 \
        "datasets/processed_balanced/full_xray/sharp_clahe4/images" \
        "datasets/splits/balanced_full_xray_10_class" \
        "$SUFFIX" \
        "datasets/processed_balanced/full_xray/sharp_clahe4/labels-knee" \
        "datasets/processed_balanced/full_xray/sharp_clahe4/labels_10_class"

    # ============================================================
    # 5-Class Experiments
    # ============================================================
    
    # 1. Baseline (Unbalanced) - Using 10-class splits as proxy if same images
    run_experiment \
        "cada_5class_unbalanced" 5 \
        "datasets/dataset_v0/images" \
        "datasets/splits/knee_full_10_class" \
        "$SUFFIX" \
        "datasets/dataset_v0/labels-knee" \
        "datasets/dataset_v0/labels"

    # 2. Balanced
    run_experiment \
        "cada_5class_balanced" 5 \
        "datasets/balanced/full_xray/images" \
        "datasets/splits/balanced_full_xray_10_class" \
        "$SUFFIX" \
        "datasets/balanced/full_xray/labels-knee" \
        "datasets/balanced/full_xray/labels"

    # 3. Balanced + Resize Only
    run_experiment \
        "cada_5class_balanced_resize" 5 \
        "datasets/processed_balanced/full_xray/resize_only/images" \
        "datasets/splits/balanced_full_xray_10_class" \
        "$SUFFIX" \
        "datasets/processed_balanced/full_xray/resize_only/labels-knee" \
        "datasets/processed_balanced/full_xray/resize_only/labels"

    # 4. Balanced + Blur CLAHE
    run_experiment \
        "cada_5class_balanced_blur" 5 \
        "datasets/processed_balanced/full_xray/blur_clahe2/images" \
        "datasets/splits/balanced_full_xray_10_class" \
        "$SUFFIX" \
        "datasets/processed_balanced/full_xray/blur_clahe2/labels-knee" \
        "datasets/processed_balanced/full_xray/blur_clahe2/labels"

    # 5. Balanced + Sharp CLAHE
    run_experiment \
        "cada_5class_balanced_sharp" 5 \
        "datasets/processed_balanced/full_xray/sharp_clahe4/images" \
        "datasets/splits/balanced_full_xray_10_class" \
        "$SUFFIX" \
        "datasets/processed_balanced/full_xray/sharp_clahe4/labels-knee" \
        "datasets/processed_balanced/full_xray/sharp_clahe4/labels"

    # ============================================================
    # 4-Class Experiments
    # ============================================================

    # 1. Baseline (Unbalanced)
    run_experiment \
        "cada_4class_unbalanced" 4 \
        "datasets/dataset_v0_4_class/images" \
        "datasets/splits/knee_full_4_class" \
        "$SUFFIX" \
        "datasets/dataset_v0/labels-knee" \
        "datasets/dataset_v0_4_class/labels"

    # 2. Balanced
    run_experiment \
        "cada_4class_balanced" 4 \
        "datasets/balanced/full_xray_4_class/images" \
        "datasets/splits/balanced_full_xray_4_class" \
        "$SUFFIX" \
        "datasets/balanced/full_xray_4_class/labels-knee" \
        "datasets/balanced/full_xray_4_class/labels"

    # 3. Balanced + Resize Only
    run_experiment \
        "cada_4class_balanced_resize" 4 \
        "datasets/processed_balanced/full_xray_4_class/resize_only/images" \
        "datasets/splits/balanced_full_xray_4_class" \
        "$SUFFIX" \
        "datasets/processed_balanced/full_xray_4_class/resize_only/labels-knee" \
        "datasets/processed_balanced/full_xray_4_class/resize_only/labels"

    # 4. Balanced + Blur CLAHE
    run_experiment \
        "cada_4class_balanced_blur" 4 \
        "datasets/processed_balanced/full_xray_4_class/blur_clahe2/images" \
        "datasets/splits/balanced_full_xray_4_class" \
        "$SUFFIX" \
        "datasets/processed_balanced/full_xray_4_class/blur_clahe2/labels-knee" \
        "datasets/processed_balanced/full_xray_4_class/blur_clahe2/labels"

    # 5. Balanced + Sharp CLAHE
    run_experiment \
        "cada_4class_balanced_sharp" 4 \
        "datasets/processed_balanced/full_xray_4_class/sharp_clahe4/images" \
        "datasets/splits/balanced_full_xray_4_class" \
        "$SUFFIX" \
        "datasets/processed_balanced/full_xray_4_class/sharp_clahe4/labels-knee" \
        "datasets/processed_balanced/full_xray_4_class/sharp_clahe4/labels"

    # ============================================================
    # 8-Class Experiments
    # ============================================================

    # 1. Baseline (Unbalanced)
    run_experiment \
        "cada_8class_unbalanced" 8 \
        "datasets/dataset_v0_4_class/images" \
        "datasets/splits/knee_full_8_class" \
        "$SUFFIX" \
        "datasets/dataset_v0/labels-knee" \
        "datasets/dataset_v0_4_class/labels_8_class"

    # 2. Balanced
    run_experiment \
        "cada_8class_balanced" 8 \
        "datasets/balanced/full_xray_8_class/images" \
        "datasets/splits/balanced_full_xray_8_class" \
        "$SUFFIX" \
        "datasets/balanced/full_xray_8_class/labels-knee" \
        "datasets/balanced/full_xray_8_class/labels"

    # 3. Balanced + Resize Only
    run_experiment \
        "cada_8class_balanced_resize" 8 \
        "datasets/processed_balanced/full_xray_8_class/resize_only/images" \
        "datasets/splits/balanced_full_xray_8_class" \
        "$SUFFIX" \
        "datasets/processed_balanced/full_xray_8_class/resize_only/labels-knee" \
        "datasets/processed_balanced/full_xray_8_class/resize_only/labels"

    # 4. Balanced + Blur CLAHE
    run_experiment \
        "cada_8class_balanced_blur" 8 \
        "datasets/processed_balanced/full_xray_8_class/blur_clahe2/images" \
        "datasets/splits/balanced_full_xray_8_class" \
        "$SUFFIX" \
        "datasets/processed_balanced/full_xray_8_class/blur_clahe2/labels-knee" \
        "datasets/processed_balanced/full_xray_8_class/blur_clahe2/labels"

    # 5. Balanced + Sharp CLAHE
    run_experiment \
        "cada_8class_balanced_sharp" 8 \
        "datasets/processed_balanced/full_xray_8_class/sharp_clahe4/images" \
        "datasets/splits/balanced_full_xray_8_class" \
        "$SUFFIX" \
        "datasets/processed_balanced/full_xray_8_class/sharp_clahe4/labels-knee" \
        "datasets/processed_balanced/full_xray_8_class/sharp_clahe4/labels"

done

echo ""
echo "========================================================"
echo "✅ ALL SPLIT EXPERIMENTS COMPLETED!"
echo "========================================================"
