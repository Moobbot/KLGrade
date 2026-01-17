#!/bin/bash
#
# Step 2: Generate Label Variants for Both Datasets
#
# Input:  datasets/dataset_v0/labels/              (full X-rays, 5-class)
#         datasets/dataset_knees_cropped/labels/   (cropped knees, 5-class)
# Output: 10-class, 4-class, 8-class labels for BOTH datasets
#
# Logic:
#   For each dataset:
#     1. Generate 10-class labels from 5-class
#     2. Filter KL0 to create 4-class labels
#     3. Generate 8-class labels from 4-class labels
#

set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate klgrade || echo "⚠️  Warning: Failed to activate klgrade env"

echo "════════════════════════════════════════════════════════"
echo "Step 2: Generate Label Variants"
echo "════════════════════════════════════════════════════════"
echo ""
echo "This step generates label variants for 2 datasets:"
echo "  1. dataset_v0            (full X-rays)"
echo "  2. dataset_knees_cropped (cropped knees)"
echo ""

# ============================================================
# 2.1: Generate labels for FULL X-RAYS (dataset_v0)
# ============================================================
echo "─────────────────────────────────────────────────────────"
echo "2.1 Generating labels for FULL X-RAYS (dataset_v0)..."
echo "─────────────────────────────────────────────────────────"
echo ""

# 2.1.1 Generate 10-class
echo "  2.1.1 Generating 10-class labels..."
python tools/check_dataset/class_split_report.py \
    --labels-dir datasets/dataset_v0/labels \
    --save-dir datasets/dataset_v0/labels_10_class \
    --limit 10

echo "  ✅ 10-class: datasets/dataset_v0/labels_10_class/"

# 2.1.2 Generate 4-class
echo ""
echo "  2.1.2 Generating 4-class labels (filter KL0)..."
python scripts/data_preparation/filter_kl0.py \
    --input datasets/dataset_v0 \
    --output datasets/dataset_v0_4_class \
    --num_classes 5 \
    --classes 0

echo "  ✅ 4-class: datasets/dataset_v0_4_class/labels/"

# 2.1.3 Generate 8-class
echo ""
echo "  2.1.3 Generating 8-class labels (from 4-class)..."
python tools/check_dataset/class_split_report.py \
    --labels-dir datasets/dataset_v0_4_class/labels \
    --save-dir datasets/dataset_v0_4_class/labels_8_class \
    --limit 8

echo "  ✅ 8-class: datasets/dataset_v0_4_class/labels_8_class/"

echo ""
echo "✅ Full X-rays labels complete"
echo ""

# ============================================================
# 2.2: Generate labels for CROPPED KNEES
# ============================================================
echo "─────────────────────────────────────────────────────────"
echo "2.2 Generating labels for CROPPED KNEES..."
echo "─────────────────────────────────────────────────────────"
echo ""

if [ -d "datasets/dataset_knees_cropped/labels" ]; then
    # 2.2.1 Generate 10-class
    echo "  2.2.1 Generating 10-class labels..."
    python tools/check_dataset/class_split_report.py \
        --labels-dir datasets/dataset_knees_cropped/labels \
        --save-dir datasets/dataset_knees_cropped/labels_10_class \
        --limit 10
    
    echo "  ✅ 10-class: datasets/dataset_knees_cropped/labels_10_class/"
    
    # 2.2.2 Generate 4-class
    echo ""
    echo "  2.2.2 Generating 4-class labels (filter KL0)..."
    python scripts/data_preparation/filter_kl0.py \
        --input datasets/dataset_knees_cropped \
        --output datasets/dataset_knees_cropped_4_class \
        --num_classes 5 \
        --classes 0
    
    echo "  ✅ 4-class: datasets/dataset_knees_cropped_4_class/labels/"
    
    # 2.2.3 Generate 8-class
    echo ""
    echo "  2.2.3 Generating 8-class labels (from 4-class)..."
    python tools/check_dataset/class_split_report.py \
        --labels-dir datasets/dataset_knees_cropped_4_class/labels \
        --save-dir datasets/dataset_knees_cropped_4_class/labels_8_class \
        --limit 8
    
    echo "  ✅ 8-class: datasets/dataset_knees_cropped_4_class/labels_8_class/"
    
    echo ""
    echo "✅ Cropped knees labels complete"
else
    echo "⚠️  Skipping: datasets/dataset_knees_cropped not found"
    echo "   Run step_1_crop_knees.sh first"
fi

echo ""

# ============================================================
# Summary
# ============================================================
echo "════════════════════════════════════════════════════════"
echo "✅ Step 2 Complete!"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Summary:"
echo ""
echo "  Full X-rays:"
echo "    ✅ datasets/dataset_v0/labels_10_class/          (10 classes)"
echo "    ✅ datasets/dataset_v0_4_class/labels/           (4 classes)"
echo "    ✅ datasets/dataset_v0_4_class/labels_8_class/   (8 classes)"
echo ""
echo "  Cropped Knees:"
echo "    ✅ datasets/dataset_knees_cropped/labels_10_class/         (10 classes)"
echo "    ✅ datasets/dataset_knees_cropped_4_class/labels/          (4 classes)"
echo "    ✅ datasets/dataset_knees_cropped_4_class/labels_8_class/  (8 classes)"
echo ""
echo "Benefits:"
echo "  ✓ Consistent label variants across both datasets"
echo "  ✓ Can train on full X-rays OR cropped knees"
echo "  ✓ All class configurations available (5, 10, 4, 8)"
echo ""
echo "Next: Run step_4_balance.sh"
echo "      (step_3_preprocess.sh is optional for ablation studies)"

