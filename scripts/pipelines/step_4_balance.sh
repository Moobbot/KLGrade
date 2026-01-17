#!/bin/bash
#
# Step 4: Balance Dataset via Oversampling
#
# Input:  datasets/dataset_knees_cropped/
# Output: datasets/balanced/knees_cropped/
#

set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate klgrade || echo "⚠️  Warning: Failed to activate klgrade env"

echo "════════════════════════════════════════════════════════"
echo "Step 4: Balance Datasets (Comprehensive)"
echo "════════════════════════════════════════════════════════"
echo ""
echo "This step balances ALL dataset variants:"
echo "  1. Cropped knees (5-class)"
echo "  2. Cropped knees (4-class)"
echo "  3. Full X-rays (5-class)"
echo "  4. Full X-rays (4-class)"
echo ""
echo "Strategy: Oversample minority classes to max class count"
echo "Method:   Horizontal/vertical flips with label adjustment"
echo ""

echo "─────────────────────────────────────────────────────────"
echo "4.1 Balancing cropped knees (5-class)..."
echo "─────────────────────────────────────────────────────────"

python scripts/data_preparation/balance_dataset.py \
    --input-images datasets/dataset_knees_cropped/images \
    --input-labels datasets/dataset_knees_cropped/labels \
    --output-dir datasets/balanced/knees_cropped \
    --num-classes 5 \
    --aux-labels datasets/dataset_knees_cropped/labels_10_class \
                 datasets/dataset_knees_cropped/labels-knee

echo ""
echo "─────────────────────────────────────────────────────────"
echo "4.2 Balancing cropped knees (4-class)..."
echo "─────────────────────────────────────────────────────────"

python scripts/data_preparation/balance_dataset.py \
    --input-images datasets/dataset_knees_cropped_4_class/images \
    --input-labels datasets/dataset_knees_cropped_4_class/labels \
    --output-dir datasets/balanced/knees_cropped_4class \
    --num-classes 4 \
    --aux-labels datasets/dataset_knees_cropped_4_class/labels_8_class \
                 datasets/dataset_knees_cropped_4_class/labels-knee

echo ""
echo "─────────────────────────────────────────────────────────"
echo "4.3 Balancing full X-rays (5-class)..."
echo "─────────────────────────────────────────────────────────"

python scripts/data_preparation/balance_dataset.py \
    --input-images datasets/dataset_v0/images \
    --input-labels datasets/dataset_v0/labels \
    --output-dir datasets/balanced/full_xray \
    --num-classes 5 \
    --aux-labels datasets/dataset_v0/labels_10_class \
                 datasets/dataset_v0/labels-knee

echo ""
echo "─────────────────────────────────────────────────────────"
echo "4.4 Balancing full X-rays (4-class)..."
echo "─────────────────────────────────────────────────────────"

python scripts/data_preparation/balance_dataset.py \
    --input-images datasets/dataset_v0_4_class/images \
    --input-labels datasets/dataset_v0_4_class/labels \
    --output-dir datasets/balanced/full_xray_4class \
    --num-classes 4 \
    --aux-labels datasets/dataset_v0_4_class/labels_8_class \
                 datasets/dataset_v0_4_class/labels-knee

echo ""
echo "✅ Balancing complete for all datasets"
echo ""

echo "─────────────────────────────────────────────────────────"
echo "Checking balanced dataset statistics..."
echo "─────────────────────────────────────────────────────────"

# Summary function
show_balance_summary() {
    local name=$1
    local dir=$2
    
    if [ -d "$dir/images" ]; then
        local count=$(find "$dir/images" -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l)
        echo ""
        echo "  $name: $count images"
        
        if [ -f "$dir/balance_report.txt" ]; then
            echo "    $(grep "augmented)" "$dir/balance_report.txt" | tail -1)"
        fi
    fi
}

echo ""
show_balance_summary "Cropped knees (5-class)" "datasets/balanced/knees_cropped"
show_balance_summary "Cropped knees (4-class)" "datasets/balanced/knees_cropped_4class"
show_balance_summary "Full X-rays (5-class)" "datasets/balanced/full_xray"
show_balance_summary "Full X-rays (4-class)" "datasets/balanced/full_xray_4class"

echo ""
echo "════════════════════════════════════════════════════════"
echo "✅ Step 4 Complete!"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Summary:"
echo "  ✅ 4 datasets balanced via flip augmentation"
echo "  ✅ All label variants copied and adjusted"
echo "  ✅ Balance reports generated"
echo ""
echo "Output structure:"
echo "  datasets/balanced/"
echo "    ├── knees_cropped/       # 5-class cropped knees"
echo "    ├── knees_cropped_4class/ # 4-class cropped knees"
echo "    ├── full_xray/           # 5-class full X-rays"
echo "    └── full_xray_4class/    # 4-class full X-rays"
echo ""
echo "Next: Run step_5_create_splits.sh"
