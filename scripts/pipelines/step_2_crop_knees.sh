#!/bin/bash
#
# Step 2: Crop Knee Regions
#
# Input:  datasets/dataset/dataset_v0/images/
#         datasets/dataset/dataset_v0/labels/
#         datasets/dataset/dataset_v0/labels-knee/
# Output: datasets/dataset_knees_cropped/
#

set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate klgrade || echo "⚠️  Warning: Failed to activate klgrade env"

echo "════════════════════════════════════════════════════════"
echo "Step 2: Crop Knee Regions"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Input:  datasets/dataset/dataset_v0/"
echo "Output: datasets/dataset_knees_cropped/"
echo ""

echo "─────────────────────────────────────────────────────────"
echo "2.1 Cropping knee regions from full X-rays..."
echo "─────────────────────────────────────────────────────────"
python scripts/preprocessing/prepare_knee_crops.py \
    --input datasets/dataset/dataset_v0 \
    --output datasets/dataset_knees_cropped

echo ""
echo "✅ Knee cropping complete"
echo ""

echo "─────────────────────────────────────────────────────────"
echo "2.2 Checking dataset statistics..."
echo "─────────────────────────────────────────────────────────"

# Count images
if [ -d "datasets/dataset_knees_cropped/images" ]; then
    IMG_COUNT=$(find datasets/dataset_knees_cropped/images -type f -name "*.jpg" | wc -l)
    echo "  Total cropped images: $IMG_COUNT"
fi

# Check label directories
echo "  Label directories:"
for label_dir in datasets/dataset_knees_cropped/labels*; do
    if [ -d "$label_dir" ]; then
        label_count=$(find "$label_dir" -type f -name "*.txt" | wc -l)
        echo "    - $(basename $label_dir): $label_count labels"
    fi
done

echo ""
echo "════════════════════════════════════════════════════════"
echo "✅ Step 2 Complete!"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Summary:"
echo "  ✅ Cropped knee regions extracted"
echo "  ✅ All label variants copied"
echo "  ✅ Images without labels filtered to images-no-labels/"
echo ""
echo "Output structure:"
echo "  datasets/dataset_knees_cropped/"
echo "    ├── images/              # Cropped knee images"
echo "    ├── labels/              # 5-class labels"
echo "    ├── labels_4class/       # 4-class labels"
echo "    ├── labels_8class/       # 8-class labels"
echo "    ├── labels-knee/         # Knee boxes"
echo "    ├── labels_new/          # Lesion labels"
echo "    └── dataset_statistics.txt"
echo ""
echo "Next: Run step_3_preprocess.sh (optional)"
echo "   or step_4_balance.sh to balance dataset"
