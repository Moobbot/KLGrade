#!/bin/bash
#
# Step 1: Generate Label Variants (10-class, 4-class dataset, 8-class)
#
# Input:  datasets/dataset/dataset_v0/
# Output: datasets/dataset/dataset_v0/labels_10_class/
#         datasets/dataset_v0_4_class/          (complete dataset, KL0 filtered)
#         datasets/dataset_v0_4_class/labels_8_class/
#
# Logic:
#   1. Generate 10-class labels from 5-class
#   2. Create 4-class dataset (filter KL0 → separate images + labels)
#   3. Generate 8-class labels from 4-class dataset (no need to filter KL0 again)
#

set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate klgrade || echo "⚠️  Warning: Failed to activate klgrade env"

echo "════════════════════════════════════════════════════════"
echo "Step 1: Generate Label Variants"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Input:  datasets/dataset/dataset_v0/"
echo "Output: labels_10_class/, dataset_v0_4_class/, labels_8_class/"
echo ""

# 1.1 Generate 10-class labels
echo "─────────────────────────────────────────────────────────"
echo "1.1 Generating 10-class labels (from 5-class)..."
echo "─────────────────────────────────────────────────────────"
python tools/check_dataset/class_split_report.py \
    --labels-dir datasets/dataset/dataset_v0/labels \
    --save-dir datasets/dataset/dataset_v0/labels_10_class \
    --limit 10

echo ""
echo "✅ 10-class labels created"
echo "   Location: datasets/dataset/dataset_v0/labels_10_class/"
echo "   Classes: KL0-a/b, KL1-a/b, KL2-a/b, KL3-a/b, KL4-a/b"
echo ""

# 1.2 Create 4-class dataset (filter KL0 → complete dataset)
echo "─────────────────────────────────────────────────────────"
echo "1.2 Creating 4-class dataset (filter KL0)..."
echo "─────────────────────────────────────────────────────────"
echo "    This creates a separate dataset with KL0 removed"
echo ""

python scripts/data_preparation/filter_kl0.py \
    --input datasets/dataset/dataset_v0 \
    --output datasets/dataset_v0_4_class \
    --num_classes 5

echo ""
echo "✅ 4-class dataset created"
echo "   Location: datasets/dataset_v0_4_class/"
echo "   Images: datasets/dataset_v0_4_class/images/"
echo "   Labels: datasets/dataset_v0_4_class/labels/"
echo "   Classes: KL1, KL2, KL3, KL4 (remapped to 0-3)"
echo ""

# 1.3 Generate 8-class labels (from 4-class dataset)
echo "─────────────────────────────────────────────────────────"
echo "1.3 Generating 8-class labels (from 4-class dataset)..."
echo "─────────────────────────────────────────────────────────"
echo "    Using 4-class dataset (KL0 already filtered)"
echo ""

python tools/check_dataset/class_split_report.py \
    --labels-dir datasets/dataset_v0_4_class/labels \
    --save-dir datasets/dataset_v0_4_class/labels_8_class \
    --limit 8

echo ""
echo "✅ 8-class labels created"
echo "   Location: datasets/dataset_v0_4_class/labels_8_class/"
echo "   Classes: KL1-a/b, KL2-a/b, KL3-a/b, KL4-a/b"
echo "   Note: No KL0 (already filtered in step 1.2)"
echo ""

echo "════════════════════════════════════════════════════════"
echo "✅ Step 1 Complete!"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Summary:"
echo "  ✅ 10-class: KL0-a/b through KL4-a/b"
echo "     → datasets/dataset/dataset_v0/labels_10_class/"
echo ""
echo "  ✅ 4-class dataset: KL1-4 only (KL0 filtered)"
echo "     → datasets/dataset_v0_4_class/"
echo "     → Complete dataset with images and labels"
echo ""
echo "  ✅ 8-class: KL1-a/b through KL4-a/b (from 4-class)"
echo "     → datasets/dataset_v0_4_class/labels_8_class/"
echo "     → Uses 4-class images (no duplicate processing)"
echo ""
echo "Next: Run step_2_crop_knees.sh"
