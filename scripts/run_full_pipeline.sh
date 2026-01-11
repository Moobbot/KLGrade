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
# run this in cmd: conda activate klgrade
# conda activate klgrade

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

python scripts/data_preparation/crop_knee_regions.py \
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

python scripts/data_preparation/filter_no_labels.py \
    --input processed/knee

echo "   ✅ Filtered: images WITH labels remain in processed/knee/"

# ============================================================================
# STAGE 4: Resize to 640x640
# ============================================================================
echo ""
echo "🔄 STAGE 4: Resizing images to 640x640..."
echo "   Input: processed/knee/images"
echo "   Output: processed/knee_5_class/images"

if [ -f "tools/check_dataset/resize_images.py" ]; then
    python tools/check_dataset/resize_images.py \
        --in_dir processed/knee/images \
        --out_dir processed/knee_5_class/images \
        --size 640
    
    # Copy labels (they're already normalized, no resize needed)
    mkdir -p processed/knee_5_class/labels
    cp -r processed/knee/labels/* processed/knee_5_class/labels/
    
    echo "   ✅ Images resized to 640x640 (saved to knee_5_class/images)"
    echo "   ✅ Labels copied to knee_5_class/labels/"
else
    echo "   ❌ resize_images.py not found!"
    exit 1
fi

# ============================================================================
# STAGE 5: Create Stratified Splits (5-class)
# ============================================================================
echo ""
echo "🎲 STAGE 5: Creating stratified train/val/test splits (5-class)..."

python scripts/data_preparation/split_dataset.py \
    --img_dir processed/knee_5_class/images \
    --label_dir processed/knee_5_class/labels \
    --out_dir splits/knee_5_class \
    --train 0.7 \
    --val 0.15 \
    --test 0.15 \
    --seed 42

echo "   ✅ Splits created: splits/knee_5_class/"

# ============================================================================
# STAGE 6: Create 10-class Dataset
# ============================================================================
echo ""
echo "🔟 STAGE 6: Creating 10-class dataset..."

mkdir -p processed/knee_10_class/images
mkdir -p processed/knee_10_class/labels

# Copy images (shared)
cp -r processed/knee_5_class/images/* processed/knee_10_class/images/

# Generate 10-class labels (split 5 classes into a/b)
echo "   Generating 10-class labels (splitting by shape)..."
python tools/check_dataset/class_split_report.py \
    --labels-dir processed/knee_5_class/labels \
    --save-dir processed/knee_10_class/labels \
    --limit 10

# Create splits for 10-class
python scripts/data_preparation/split_dataset.py \
    --img_dir processed/knee_10_class/images \
    --label_dir processed/knee_10_class/labels \
    --out_dir splits/knee_10_class \
    --train 0.7 \
    --val 0.15 \
    --test 0.15 \
    --seed 42

echo "   ✅ 10-class dataset created"

# Analyze 10-class dataset
if [ -f "tools/check_dataset/comprehensive_analysis.py" ]; then
    echo "   📊 Analyzing 10-class dataset..."
    python tools/check_dataset/comprehensive_analysis.py \
        --dataset_dir processed/knee_10_class \
        --output analysis/results_10_class
    echo "   ✅ Analysis saved: analysis/results_10_class/"
fi

# ============================================================================
# STAGE 7: Create 4-class Dataset (filter KL0)
# ============================================================================
echo ""
echo "4️⃣  STAGE 7: Creating 4-class dataset (excluding KL0)..."

mkdir -p processed/knee_4_class/images
mkdir -p processed/knee_4_class/labels

# filter_kl0.py expects input to have images/ and labels/ subdirectories
# So we create a temp structure or use processed/knee directly
# Since we have images_640 and labels_640, we'll create temp links

mkdir -p processed/knee_temp/images
mkdir -p processed/knee_temp/labels
cp -r processed/knee_5_class/images/* processed/knee_temp/images/
cp -r processed/knee_5_class/labels/* processed/knee_temp/labels/

# Use filter_kl0.py to remove KL0 class (auto-remaps KL1-4 to 0-3)
python scripts/data_preparation/filter_kl0.py \
    --input processed/knee_temp \
    --output processed/knee_4_class \
    --num_classes 5

# Clean up temp
rm -rf processed/knee_temp

# Create splits
python scripts/data_preparation/split_dataset.py \
    --img_dir processed/knee_4_class/images \
    --label_dir processed/knee_4_class/labels \
    --out_dir splits/knee_4_class \
    --train 0.7 \
    --val 0.15 \
    --test 0.15 \
    --seed 42

echo "   ✅ 4-class dataset created"

# Analyze 4-class dataset
if [ -f "tools/check_dataset/comprehensive_analysis.py" ]; then
    echo "   📊 Analyzing 4-class dataset..."
    python tools/check_dataset/comprehensive_analysis.py \
        --dataset_dir processed/knee_4_class \
        --output analysis/results_4_class
    echo "   ✅ Analysis saved: analysis/results_4_class/"
fi

# ============================================================================
# STAGE 8: Create 8-class Dataset (10-class without KL0-a/b)
# ============================================================================
echo ""
echo "8️⃣  STAGE 8: Creating 8-class dataset..."

# Check if 10-class dataset was created
if [ -d "processed/knee_10_class/labels" ] && [ "$(ls -A processed/knee_10_class/labels)" ]; then
    echo "   Creating 8-class from 10-class dataset..."
    
    # Use filter_kl0.py to remove KL0-a and KL0-b (classes 0,1)
    python scripts/data_preparation/filter_kl0.py \
        --input processed/knee_10_class \
        --output processed/knee_8_class \
        --num_classes 10
    
    # Create splits
    python scripts/data_preparation/split_dataset.py \
        --img_dir processed/knee_8_class/images \
        --label_dir processed/knee_8_class/labels \
        --out_dir splits/knee_8_class \
        --train 0.7 \
        --val 0.15 \
        --test 0.15 \
        --seed 42
    
    echo "   ✅ 8-class dataset created"
    
    # Analyze 8-class dataset
    if [ -f "tools/check_dataset/comprehensive_analysis.py" ]; then
        echo "   📊 Analyzing 8-class dataset..."
        python tools/check_dataset/comprehensive_analysis.py \
            --dataset_dir processed/knee_8_class \
            --output analysis/results_8_class
        echo "   ✅ Analysis saved: analysis/results_8_class/"
    fi
else
    echo "   ⚠️  10-class labels not found, skipping 8-class"
    echo "   (This is expected if labels_new doesn't exist in dataset_v0)"
fi

# ============================================================================
# STAGE 9: Fix Split Paths (DEPRECATED - handled by split_dataset.py)
# ============================================================================
# echo ""
# echo "🔧 STAGE 9: Fixing split file paths..."
#
# python tools/fix_splits_paths.py
#
# echo "   ✅ All split files updated with absolute paths"

# ============================================================================
# STAGE 10: Verify All Configs
# ============================================================================
echo ""
echo "✅ STAGE 10: Verifying YOLO configs..."

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
echo "  - 5-class:  processed/knee_5_class/ (~1688 images)"
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
echo "  3. Start training: bash TRAINING_COMMANDS_WANDB.sh"
echo ""
echo "📖 Full documentation: PIPELINE.md"
echo ""
echo "===================================================="