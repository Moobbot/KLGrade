#!/bin/bash
# Complete Pipeline - From Raw Data (dataset_v0) to Ready YAMLs
# KLGrade - Knee OA Detection
# Author: AI Assistant

set -e

echo "🚀 KLGrade - Complete Data Preparation Pipeline"
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
echo "   Input: dataset/dataset_v0/"

if [ -f "tools/check_dataset/comprehensive_analysis.py" ]; then
    python tools/check_dataset/comprehensive_analysis.py \
        --dataset_dir dataset/dataset_v0 \
        --output analysis/dataset_v0
    echo "   ✅ Analysis complete: analysis/dataset_v0/"
else
    echo "   ⏭️  Skipping analysis (script not found)"
fi

# ============================================================================
# STAGE 2: Crop Knee Regions
# ============================================================================
echo ""
echo "✂️  STAGE 2: Cropping knee regions from full X-rays..."
echo "   Input: dataset/dataset_v0/"
echo "   Output: processed/knee/"

python scripts/preprocessing/crop_knee_regions.py \
    --dataset_dir dataset/dataset_v0 \
    --output_dir processed/knee \
    --margin 0.15 \
    --min_size 300

echo "   ✅ Knee cropping complete"

# ============================================================================
# STAGE 3: Filter Images Without Labels
# ============================================================================
echo ""
echo "🗑️  STAGE 3: Filtering images without labels..."

python scripts/preprocessing/filter_no_labels.py \
    --input processed/knee

echo "   ✅ Filtered: images WITH labels remain in processed/knee/"

# ============================================================================
# STAGE 4: Resize to 640x640
# ============================================================================
echo ""
echo "🔄 STAGE 4: Resizing images to 640x640..."
echo "   Input: processed/knee/images"
echo "   Output: processed/knee/images_640"

if [ -f "tools/check_dataset/resize_images.py" ]; then
    python tools/check_dataset/resize_images.py \
        --in_dir processed/knee/images \
        --out_dir processed/knee/images_640 \
        --size 640
    
    # Also resize labels directory (copy, since labels are normalized)
    mkdir -p processed/knee/labels_640
    cp -r processed/knee/labels/* processed/knee/labels_640/
    
    echo "   ✅ Images resized to 640x640"
else
    echo "   ❌ resize_images.py not found!"
    exit 1
fi

# ============================================================================
# STAGE 5: Create Standard YOLO Structure (5-class)
# ============================================================================
echo ""
echo "📁 STAGE 5: Creating standard YOLO structure..."

mkdir -p processed/knee/dataset_yolo/images
mkdir -p processed/knee/dataset_yolo/labels

cp -r processed/knee/images_640/* processed/knee/dataset_yolo/images/
cp -r processed/knee/labels_640/* processed/knee/dataset_yolo/labels/

echo "   ✅ Created: processed/knee/dataset_yolo/"

# ============================================================================
# STAGE 6: Create Stratified Splits (5-class)
# ============================================================================
echo ""
echo "🎲 STAGE 6: Creating stratified train/val/test splits (5-class)..."

python scripts/data_preparation/split_dataset.py \
    --img_dir processed/knee/dataset_yolo/images \
    --label_dir processed/knee/dataset_yolo/labels \
    --output_dir splits/knee_5_class \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --seed 42

echo "   ✅ Splits created: splits/knee_5_class/"

# ============================================================================
# STAGE 7: Create 10-class Dataset
# ============================================================================
echo ""
echo "🔟 STAGE 7: Creating 10-class dataset..."

mkdir -p processed/knee_10_class/images
mkdir -p processed/knee_10_class/labels

# Copy images (shared)
cp -r processed/knee/images_640/* processed/knee_10_class/images/

# Check if labels_new exists in raw data
if [ -d "dataset/dataset_v0/labels_new" ]; then
    echo "   Using existing labels_new from dataset_v0"
    cp -r dataset/dataset_v0/labels_new/* processed/knee_10_class/labels/
else
    echo "   ⚠️  labels_new not found, using 5-class as placeholder"
    cp -r processed/knee/labels_640/* processed/knee_10_class/labels/
fi

# Create splits for 10-class
python scripts/data_preparation/split_dataset.py \
    --img_dir processed/knee_10_class/images \
    --label_dir processed/knee_10_class/labels \
    --output_dir splits/knee_10_class \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --seed 42

echo "   ✅ 10-class dataset created"

# ============================================================================
# STAGE 8: Create 4-class Dataset (filter KL0)
# ============================================================================
echo ""
echo "4️⃣  STAGE 8: Creating 4-class dataset (excluding KL0)..."

mkdir -p processed/knee_4_class/images
mkdir -p processed/knee_4_class/labels

python scripts/data_preparation/filter_dataset.py \
    --input_images processed/knee/images_640 \
    --input_labels processed/knee/labels_640 \
    --output_images processed/knee_4_class/images \
    --output_labels processed/knee_4_class/labels \
    --exclude_classes 0

# Remap labels (0,1,2,3,4 -> remove 0 -> 1,2,3,4 -> remap to 0,1,2,3)
python scripts/data_preparation/remap_labels.py \
    --label_dir processed/knee_4_class/labels \
    --output_dir processed/knee_4_class/labels \
    --class_mapping "1:0,2:1,3:2,4:3"

# Create splits
python scripts/data_preparation/split_dataset.py \
    --img_dir processed/knee_4_class/images \
    --label_dir processed/knee_4_class/labels \
    --output_dir splits/knee_4_class \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --seed 42

echo "   ✅ 4-class dataset created"

# ============================================================================
# STAGE 9: Create 8-class Dataset (10-class without KL0-a/b)
# ============================================================================
echo ""
echo "8️⃣  STAGE 9: Creating 8-class dataset..."

mkdir -p processed/knee_8_class/images
mkdir -p processed/knee_8_class/labels

# Filter out classes 0 and 1 (KL0-a and KL0-b from 10-class)
python scripts/data_preparation/filter_dataset.py \
    --input_images processed/knee_10_class/images \
    --input_labels processed/knee_10_class/labels \
    --output_images processed/knee_8_class/images \
    --output_labels processed/knee_8_class/labels \
    --exclude_classes 0,1

# Remap (remove 0,1 then remap remaining)
python scripts/data_preparation/remap_labels.py \
    --label_dir processed/knee_8_class/labels \
    --output_dir processed/knee_8_class/labels \
    --class_mapping "2:0,3:1,4:2,5:3,6:4,7:5,8:6,9:7"

# Create splits
python scripts/data_preparation/split_dataset.py \
    --img_dir processed/knee_8_class/images \
    --label_dir processed/knee_8_class/labels \
    --output_dir splits/knee_8_class \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --seed 42

echo "   ✅ 8-class dataset created"

# ============================================================================
# STAGE 10: Fix Split Paths (Add Absolute Paths)
# ============================================================================
echo ""
echo "🔧 STAGE 10: Fixing split file paths..."

python tools/fix_splits_paths.py

echo "   ✅ All split files updated with absolute paths"

# ============================================================================
# STAGE 11: Verify All Configs
# ============================================================================
echo ""
echo "✅ STAGE 11: Verifying YOLO configs..."

configs=(
    "configs/yolo_5_class_baseline.yaml"
    "configs/yolo_5_class_conservative.yaml"
    "configs/yolo_10_class_baseline.yaml"
    "configs/yolo_10_class_conservative.yaml"
    "configs/yolo_4_class_baseline.yaml"
    "configs/yolo_4_class_conservative.yaml"
    "configs/yolo_8_class_baseline.yaml"
    "configs/yolo_8_class_conservative.yaml"
)

echo "   Checking config files:"
for config in "${configs[@]}"; do
    if [ -f "$config" ]; then
        echo "      ✅ $config"
    else
        echo "      ❌ $config (MISSING - should exist!)"
    fi
done

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "===================================================="
echo "✨ Complete Pipeline Finished!"
echo "===================================================="
echo ""
echo "📊 Generated Datasets:"
echo "  - 5-class:  processed/knee/dataset_yolo/ (~1688 images)"
echo "  - 10-class: processed/knee_10_class/"
echo "  - 4-class:  processed/knee_4_class/ (no KL0)"
echo "  - 8-class:  processed/knee_8_class/ (10-class, no KL0)"
echo ""
echo "📁 Stratified Splits (70/15/15):"
echo "  - splits/knee_5_class/"
echo "  - splits/knee_10_class/"
echo "  - splits/knee_4_class/"
echo "  - splits/knee_8_class/"
echo ""
echo "⚙️  YOLO Configs: configs/*.yaml"
echo ""
echo "🎯 Next Steps:"
echo "  1. Test GPU: python scripts/test_yolo_gpu.py"
echo "  2. Quick 2-epoch test: python scripts/test_all_configs.py"
echo "  3. Start training: bash docs/TRAINING_COMMANDS_WANDB.sh"
echo ""
echo "📖 Full documentation: PIPELINE.md"
echo ""
