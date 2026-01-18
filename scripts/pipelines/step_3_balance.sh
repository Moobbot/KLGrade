#!/bin/bash
#
# Step 3: Balance Datasets via Oversampling
#
# Creates balanced versions of all dataset variants by oversampling minority classes.
# Balanced datasets enable training models with better class representation.
#

set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate klgrade || echo "⚠️  Warning: Failed to activate klgrade env"

echo "════════════════════════════════════════════════════════"
echo "Step 3: Balance Datasets (Comprehensive)"
echo "════════════════════════════════════════════════════════"
echo ""
echo "This step balances ALL dataset variants:"
echo "  1. Cropped knees (5-class)"
echo "  2. Cropped knees (10-class)"
echo "  3. Cropped knees (4-class)"
echo "  4. Cropped knees (8-class)"
echo "  5. Full X-rays (5-class)"
echo "  6. Full X-rays (10-class)"
echo "  7. Full X-rays (4-class)"
echo "  8. Full X-rays (8-class)"
echo ""
echo "Strategy: Oversample minority classes to max class count"
echo "Method:   Horizontal/vertical flips with label adjustment"
echo ""
echo "⚠️  This will create 8 balanced datasets"
echo ""

# Prompt user to continue
read -p "Continue with dataset balancing? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Skipping dataset balancing."
    echo ""
    echo "Next: bash scripts/pipelines/step_4_preprocess_balanced.sh"
    exit 0
fi

echo ""
echo "─────────────────────────────────────────────────────────"
echo "3.1 Balancing cropped knees (5-class)..."
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
echo "3.2 Balancing cropped knees (10-class)..."
echo "─────────────────────────────────────────────────────────"

python scripts/data_preparation/balance_dataset.py \
    --input-images datasets/dataset_knees_cropped/images \
    --input-labels datasets/dataset_knees_cropped/labels_10_class \
    --output-dir datasets/balanced/knees_cropped_10_class \
    --num-classes 10 \
    --aux-labels datasets/dataset_knees_cropped/labels \
                 datasets/dataset_knees_cropped/labels-knee

echo ""
echo "─────────────────────────────────────────────────────────"
echo "3.3 Balancing cropped knees (4-class)..."
echo "─────────────────────────────────────────────────────────"

python scripts/data_preparation/balance_dataset.py \
    --input-images datasets/dataset_knees_cropped_4_class/images \
    --input-labels datasets/dataset_knees_cropped_4_class/labels \
    --output-dir datasets/balanced/knees_cropped_4_class \
    --num-classes 4 \
    --aux-labels datasets/dataset_knees_cropped_4_class/labels_8_class \
                 datasets/dataset_knees_cropped_4_class/labels-knee

echo ""
echo "─────────────────────────────────────────────────────────"
echo "3.4 Balancing cropped knees (8-class)..."
echo "─────────────────────────────────────────────────────────"

python scripts/data_preparation/balance_dataset.py \
    --input-images datasets/dataset_knees_cropped_4_class/images \
    --input-labels datasets/dataset_knees_cropped_4_class/labels_8_class \
    --output-dir datasets/balanced/knees_cropped_8_class \
    --num-classes 8 \
    --aux-labels datasets/dataset_knees_cropped_4_class/labels \
                 datasets/dataset_knees_cropped_4_class/labels-knee

echo ""
echo "─────────────────────────────────────────────────────────"
echo "3.5 Balancing full X-rays (5-class)..."
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
echo "3.6 Balancing full X-rays (10-class)..."
echo "─────────────────────────────────────────────────────────"

python scripts/data_preparation/balance_dataset.py \
    --input-images datasets/dataset_v0/images \
    --input-labels datasets/dataset_v0/labels_10_class \
    --output-dir datasets/balanced/full_xray_10_class \
    --num-classes 10 \
    --aux-labels datasets/dataset_v0/labels \
                 datasets/dataset_v0/labels-knee

echo ""
echo "─────────────────────────────────────────────────────────"
echo "3.7 Balancing full X-rays (4-class)..."
echo "─────────────────────────────────────────────────────────"

python scripts/data_preparation/balance_dataset.py \
    --input-images datasets/dataset_v0_4_class/images \
    --input-labels datasets/dataset_v0_4_class/labels \
    --output-dir datasets/balanced/full_xray_4_class \
    --num-classes 4 \
    --aux-labels datasets/dataset_v0_4_class/labels_8_class \
                 datasets/dataset_v0_4_class/labels-knee

echo ""
echo "─────────────────────────────────────────────────────────"
echo "3.8 Balancing full X-rays (8-class)..."
echo "─────────────────────────────────────────────────────────"

python scripts/data_preparation/balance_dataset.py \
    --input-images datasets/dataset_v0_4_class/images \
    --input-labels datasets/dataset_v0_4_class/labels_8_class \
    --output-dir datasets/balanced/full_xray_8_class \
    --num-classes 8 \
    --aux-labels datasets/dataset_v0_4_class/labels \
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
show_balance_summary "Cropped knees (10-class)" "datasets/balanced/knees_cropped_10_class"
show_balance_summary "Cropped knees (4-class)" "datasets/balanced/knees_cropped_4_class"
show_balance_summary "Cropped knees (8-class)" "datasets/balanced/knees_cropped_8_class"
show_balance_summary "Full X-rays (5-class)" "datasets/balanced/full_xray"
show_balance_summary "Full X-rays (10-class)" "datasets/balanced/full_xray_10_class"
show_balance_summary "Full X-rays (4-class)" "datasets/balanced/full_xray_4_class"
show_balance_summary "Full X-rays (8-class)" "datasets/balanced/full_xray_8_class"

echo ""
echo "════════════════════════════════════════════════════════"
echo "✅ Step 3 Complete!"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Summary:"
echo "  ✅ 8 datasets balanced via flip augmentation"
echo "  ✅ All label variants copied and adjusted"
echo "  ✅ Balance reports generated"
echo ""
echo "Output structure:"
echo "  datasets/balanced/"
echo "    ├── knees_cropped/        # 5-class cropped knees"
echo "    ├── knees_cropped_10_class/ # 10-class cropped knees"
echo "    ├── knees_cropped_4_class/  # 4-class cropped knees"
echo "    ├── knees_cropped_8_class/  # 8-class cropped knees"
echo "    ├── full_xray/            # 5-class full X-rays"
echo "    ├── full_xray_10_class/     # 10-class full X-rays"
echo "    ├── full_xray_4_class/      # 4-class full X-rays"
echo "    └── full_xray_8class/      # 8-class full X-rays"
echo ""
echo "Next: Run step_4_preprocess_balanced.sh to preprocess balanced datasets"
