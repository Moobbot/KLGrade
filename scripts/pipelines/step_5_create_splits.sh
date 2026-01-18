#!/bin/bash
#
# Step 5: Create Train/Val/Test Splits
#
# Input:  All datasets in datasets/
# Output: datasets/splits/[dataset_name]/
#

set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate klgrade || echo "⚠️  Warning: Failed to activate klgrade env"

echo "════════════════════════════════════════════════════════"
echo "Step 5: Create Train/Val/Test Splits"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Split ratio: 70% train / 15% val / 15% test"
echo "Method: Stratified sampling by class"
echo ""

echo "─────────────────────────────────────────────────────────"
echo "5.1 Creating splits for all datasets..."
echo "─────────────────────────────────────────────────────────"

bash scripts/pipelines/create_dataset_splits.sh all

echo ""
echo "════════════════════════════════════════════════════════"
echo "✅ Step 5 Complete!"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Summary:"
echo "  ✅ Stratified train/val/test splits created"
echo "  ✅ Splits for all dataset variants"
echo ""
echo "Output structure:"
echo "  datasets/splits/"
echo "    ├── dataset_knees_cropped/           # Base 5-class"
echo "    ├── dataset_knees_cropped_4_class/   # Base 4-class"
echo "    ├── dataset_knees_cropped_8_class/   # Base 8-class"
echo "    ├── dataset_knees_cropped_10_class/  # Base 10-class"
echo "    ├── balanced_knees_cropped/          # Balanced 5-class"
echo "    ├── balanced_knees_cropped_4_class/  # Balanced 4-class"
echo "    ├── balanced_knees_cropped_8_class/  # Balanced 8-class"
echo "    ├── balanced_knees_cropped_10_class/ # Balanced 10-class"
echo "    ├── knee_full_10_class/              # Base Full X-ray 10-class"
echo "    ├── knee_full_8_class/               # Base Full X-ray 8-class"
echo "    ├── knee_full_4_class/               # Base Full X-ray 4-class"
echo "    ├── balanced_full_xray_10_class/     # Balanced Full X-ray 10-class"
echo "    ├── balanced_full_xray_4_class/      # Balanced Full X-ray 4-class"
echo "    └── balanced_full_xray_8_class/      # Balanced Full X-ray 8-class"
echo ""
echo "Each split directory contains:"
echo "  - train.txt (70%)"
echo "  - val.txt   (15%)"
echo "  - test.txt  (15%)"
echo ""
echo "════════════════════════════════════════════════════════"
echo "🎉 Data Pipeline Complete!"
echo "════════════════════════════════════════════════════════"
echo ""
echo "All datasets ready for training!"
echo ""
echo "Next steps:"
echo "  1. Review DATASET_GUIDE.md for dataset details"
echo "  2. Use datasets in training - see TRAINING_WORKFLOW.md"
echo "  3. Run experiments with different configurations"
