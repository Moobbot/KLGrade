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
echo "Step 4: Balance Dataset"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Input:  datasets/dataset_knees_cropped/"
echo "Output: datasets/balanced/knees_cropped/"
echo ""
echo "Strategy: Oversample minority classes to ~500 samples/class"
echo "Method:   Horizontal/vertical flips with label adjustment"
echo ""

echo "─────────────────────────────────────────────────────────"
echo "4.1 Balancing cropped knees dataset..."
echo "─────────────────────────────────────────────────────────"

python scripts/data_preparation/balance_dataset.py \
    --input-images datasets/dataset_knees_cropped/images \
    --input-labels datasets/dataset_knees_cropped/labels \
    --output-dir datasets/balanced/knees_cropped \
    --num-classes 5 \
    --aux-labels datasets/dataset_knees_cropped/labels_new \
                 datasets/dataset_knees_cropped/labels-knee

echo ""
echo "✅ Balancing complete"
echo ""

echo "─────────────────────────────────────────────────────────"
echo "4.2 Checking balanced dataset statistics..."
echo "─────────────────────────────────────────────────────────"

# Count images
if [ -d "datasets/balanced/knees_cropped/images" ]; then
    BALANCED_COUNT=$(find datasets/balanced/knees_cropped/images -type f -name "*.jpg" | wc -l)
    ORIGINAL_COUNT=$(find datasets/dataset_knees_cropped/images -type f -name "*.jpg" | wc -l)
    echo "  Original images: $ORIGINAL_COUNT"
    echo "  Balanced images: $BALANCED_COUNT"
    echo "  Augmented: $(($BALANCED_COUNT - $ORIGINAL_COUNT))"
fi

# Check balance report
if [ -f "datasets/balanced/knees_cropped/balance_report.txt" ]; then
    echo ""
    echo "  Balance report:"
    cat datasets/balanced/knees_cropped/balance_report.txt
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "✅ Step 4 Complete!"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Summary:"
echo "  ✅ Dataset balanced via oversampling"
echo "  ✅ All label variants copied and adjusted"
echo "  ✅ Balance report generated"
echo ""
echo "Output structure:"
echo "  datasets/balanced/knees_cropped/"
echo "    ├── images/              # Balanced images (~4600+)"
echo "    ├── labels/              # 5-class labels"
echo "    ├── labels_4_class/       # 4-class labels"
echo "    ├── labels_8_class/       # 8-class labels"
echo "    ├── labels-knee/         # Knee boxes"
echo "    ├── labels_new/          # Lesion labels"
echo "    ├── balance_report.txt   # Balancing statistics"
echo "    └── dataset_statistics.txt"
echo ""
echo "Next: Run step_5_create_splits.sh"
