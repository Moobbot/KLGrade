#!/bin/bash
#
# Step 1: Crop Knee Regions from Full X-rays
#
# Input:  datasets/dataset_v0/
# Output: datasets/dataset_knees_cropped/
#
# Note: Label variants (10-class, 4-class, 8-class) will be generated in Step 2
#

set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate klgrade || echo "⚠️  Warning: Failed to activate klgrade env"

echo "════════════════════════════════════════════════════════"
echo "Step 1: Crop Knee Regions"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Input:  datasets/dataset_v0/ (full X-rays, 5-class labels)"
echo "Output: datasets/dataset_knees_cropped/"
echo ""
echo "Note: This step only crops images. Label variants will be"
echo "      generated in Step 2 from the cropped dataset."
echo ""

echo "─────────────────────────────────────────────────────────"
echo "1.1 Cropping knee regions from full X-rays..."
echo "─────────────────────────────────────────────────────────"

# Set PYTHONPATH for module imports
python scripts/data_preparation/crop_knee_regions.py \
    --dataset_dir datasets/dataset_v0 \
    --output_dir datasets/dataset_knees_cropped \
    --margin 0.15 \
    --min_size 300

echo ""
echo "✅ Knee cropping complete"
echo ""

# Check statistics
echo "─────────────────────────────────────────────────────────"
echo "1.2 Dataset statistics..."
echo "─────────────────────────────────────────────────────────"

if [ -d "datasets/dataset_knees_cropped/images" ]; then
    IMG_COUNT=$(find datasets/dataset_knees_cropped/images -type f \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l)
    echo "  Total cropped images: $IMG_COUNT"
    
    LABEL_COUNT=$(find datasets/dataset_knees_cropped/labels -type f -name "*.txt" 2>/dev/null | wc -l)
    echo "  Total labels: $LABEL_COUNT"
fi

echo ""

echo "════════════════════════════════════════════════════════"
echo "✅ Step 1 Complete!"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Summary:"
echo "  ✅ Cropped knees from full X-rays"
echo "  ✅ Copied 5-class KL labels (transformed to crop space)"
echo "  ✅ Filtered out crops with no labels"
echo ""
echo "Output structure:"
echo "  datasets/dataset_knees_cropped/"
echo "    ├── images/              # Cropped knee images (only with labels)"
echo "    ├── labels/              # 5-class KL labels (KL0-4)"
echo "    ├── labels-knee/         # Knee boxes (full crop)"
echo "    └── crop_report.json     # Cropping statistics"
echo ""
echo "Note: Label variants (10-class, 4-class, 8-class) will be"
echo "      generated in Step 2 for BOTH full X-rays and cropped knees."
echo ""
echo "Next: Run step_2_generate_labels.sh"