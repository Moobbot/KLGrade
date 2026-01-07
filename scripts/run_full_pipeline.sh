#!/bin/bash
# Master Pipeline Script - KLGrade Dataset Preparation
# Regenerates everything from raw data to YOLO configs
# Author: AI Assistant
# Date: 2026-01-08

set -e  # Exit on error

echo "🚀 Starting KLGrade Dataset Preparation Pipeline"
echo "=================================================="

# ============================================================================
# Environment Setup (Conda)
# ============================================================================
echo "Activating conda environment..."
conda activate klgrade

# ============================================================================
# STAGE 1: Initial Dataset Analysis
# ============================================================================
echo ""
echo "📊 STAGE 1: Analyzing raw dataset..."
python scripts/analysis/analyze_dataset.py \
    --img_dir dataset/dataset_v0/images \
    --label_dir dataset/dataset_v0/labels \
    --output docs/dataset_analysis

# ============================================================================
# STAGE 2: Knee Cropping & Preprocessing
# ============================================================================
echo ""
echo "✂️  STAGE 2: Cropping knee regions..."
python scripts/preprocessing/crop_knee_roi.py \
    --input_images dataset/dataset_v0/images \
    --input_labels dataset/dataset_v0/labels \
    --output_images processed/knee/images \
    --output_labels processed/knee/labels \
    --crop_padding 50

# ============================================================================
# STAGE 3: Image Resizing (640x640)
# ============================================================================
echo ""
echo "🔄 STAGE 3: Resizing images to 640x640..."
python scripts/preprocessing/resize_images.py \
    --input_images processed/knee/images \
    --input_labels processed/knee/labels \
    --output_images processed/knee/images_640 \
    --output_labels processed/knee/labels_640 \
    --target_size 640

# ============================================================================
# STAGE 4: Create Standard Dataset Structure
# ============================================================================
echo ""
echo "📁 STAGE 4: Creating standard YOLO structure..."
mkdir -p processed/knee/dataset_yolo/images
mkdir -p processed/knee/dataset_yolo/labels

cp -r processed/knee/images_640/* processed/knee/dataset_yolo/images/
cp -r processed/knee/labels_640/* processed/knee/dataset_yolo/labels/

# ============================================================================
# STAGE 5: Stratified Data Splitting (5-class)
# ============================================================================
echo ""
echo "🎲 STAGE 5: Creating stratified train/val/test splits (5-class)..."
python scripts/preprocessing/split_dataset.py \
    --img_dir processed/knee/dataset_yolo/images \
    --label_dir processed/knee/dataset_yolo/labels \
    --output_dir splits/knee_5_class \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --seed 42

# ============================================================================
# STAGE 6: Create 10-class Dataset (with A/B split)
# ============================================================================
echo ""
echo "🔟 STAGE 6: Creating 10-class dataset..."
mkdir -p processed/knee_10_class/images
mkdir -p processed/knee_10_class/labels

# Copy images (shared)
cp -r processed/knee/images_640/* processed/knee_10_class/images/

# Generate 10-class labels
python scripts/preprocessing/create_10class_labels.py \
    --input_labels processed/knee/labels_640 \
    --output_labels processed/knee_10_class/labels

# Create splits for 10-class
python scripts/preprocessing/split_dataset.py \
    --img_dir processed/knee_10_class/images \
    --label_dir processed/knee_10_class/labels \
    --output_dir splits/knee_10_class \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --seed 42

# ============================================================================
# STAGE 7: Create 4-class Dataset (no KL0)
# ============================================================================
echo ""
echo "4️⃣  STAGE 7: Creating 4-class dataset (filtering out KL0)..."
mkdir -p processed/knee_4_class/images
mkdir -p processed/knee_4_class/labels

python scripts/preprocessing/filter_classes.py \
    --input_images processed/knee/images_640 \
    --input_labels processed/knee/labels_640 \
    --output_images processed/knee_4_class/images \
    --output_labels processed/knee_4_class/labels \
    --exclude_classes 0

# Create splits for 4-class
python scripts/preprocessing/split_dataset.py \
    --img_dir processed/knee_4_class/images \
    --label_dir processed/knee_4_class/labels \
    --output_dir splits/knee_4_class \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --seed 42

# ============================================================================
# STAGE 8: Create 8-class Dataset (no KL0, with A/B)
# ============================================================================
echo ""
echo "8️⃣  STAGE 8: Creating 8-class dataset..."
mkdir -p processed/knee_8_class/images
mkdir -p processed/knee_8_class/labels

python scripts/preprocessing/create_8class_labels.py \
    --input_images processed/knee/images_640 \
    --input_labels processed/knee_10_class/labels \
    --output_images processed/knee_8_class/images \
    --output_labels processed/knee_8_class/labels \
    --exclude_classes 0,1  # Exclude KL0-a and KL0-b

# Create splits for 8-class
python scripts/preprocessing/split_dataset.py \
    --img_dir processed/knee_8_class/images \
    --label_dir processed/knee_8_class/labels \
    --output_dir splits/knee_8_class \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --seed 42

# ============================================================================
# STAGE 9: Fix Split File Paths (Add absolute paths)
# ============================================================================
echo ""
echo "🔧 STAGE 9: Fixing split file paths..."
python tools/fix_splits_paths.py

# ============================================================================
# STAGE 10: Verify Dataset Integrity
# ============================================================================
echo ""
echo "✅ STAGE 10: Verifying dataset integrity..."

# Verify 5-class
python scripts/analysis/check_dataset.py \
    --split_dir splits/knee_5_class \
    --label_dir processed/knee/dataset_yolo/labels

# Verify 10-class
python scripts/analysis/check_dataset.py \
    --split_dir splits/knee_10_class \
    --label_dir processed/knee_10_class/labels

# Verify 4-class
python scripts/analysis/check_dataset.py \
    --split_dir splits/knee_4_class \
    --label_dir processed/knee_4_class/labels

# Verify 8-class
python scripts/analysis/check_dataset.py \
    --split_dir splits/knee_8_class \
    --label_dir processed/knee_8_class/labels

# ============================================================================
# STAGE 11: Generate Dataset Analysis Reports
# ============================================================================
echo ""
echo "📈 STAGE 11: Generating analysis reports..."

python scripts/analysis/analyze_splits.py \
    --splits_dir splits \
    --output_dir docs/dataset_analysis

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "=================================================="
echo "✨ Pipeline Complete!"
echo "=================================================="
echo ""
echo "📊 Generated Datasets:"
echo "  - 5-class:  processed/knee/dataset_yolo/ (1688 images)"
echo "  - 10-class: processed/knee_10_class/ (1688 images)"
echo "  - 4-class:  processed/knee_4_class/ (~1600 images, no KL0)"
echo "  - 8-class:  processed/knee_8_class/ (~1600 images, no KL0-a/b)"
echo ""
echo "📁 Stratified Splits:"
echo "  - splits/knee_5_class/  (train/val/test)"
echo "  - splits/knee_10_class/ (train/val/test)"
echo "  - splits/knee_4_class/  (train/val/test)"
echo "  - splits/knee_8_class/  (train/val/test)"
echo ""
echo "⚙️  YOLO Configs:"
echo "  All configs are in: configs/"
echo "  - yolo_5_class_baseline.yaml"
echo "  - yolo_5_class_conservative.yaml"
echo "  - yolo_10_class_baseline.yaml"
echo "  - yolo_10_class_conservative.yaml"
echo "  - yolo_4_class_baseline.yaml"
echo "  - yolo_4_class_conservative.yaml"
echo "  - yolo_8_class_baseline.yaml"
echo "  - yolo_8_class_conservative.yaml"
echo ""
echo "🎯 Next Steps:"
echo "  1. Review analysis reports in docs/dataset_analysis/"
echo "  2. Start training with: bash docs/TRAINING_COMMANDS_WANDB.sh"
echo ""
