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
echo "    ├── dataset_knees_cropped/        # Base unbalanced"
echo "    ├── dataset_knees_cropped_4class/"
echo "    ├── dataset_knees_cropped_8class/"
echo "    ├── balanced_knees_cropped/       # Balanced dataset"
echo "    ├── balanced_knees_cropped_4class/"
echo "    ├── balanced_knees_cropped_8class/"
echo "    └── knee_full_10_class/           # Full X-rays"
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
