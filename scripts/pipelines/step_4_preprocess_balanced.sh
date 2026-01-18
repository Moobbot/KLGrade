#!/bin/bash
#
# Step 4: Preprocess Balanced Datasets
#
# Input:  datasets/balanced/{knees_cropped, knees_cropped_4_class, full_xray, full_xray_4_class}
# Output: datasets/processed_balanced/
#
# Creates 4 preprocessing variants for each balanced dataset:
#   - resize_only
#   - blur_clahe2
#   - sharp_clahe4
#   - blur_clahe2_notebook
#

set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate klgrade || echo "⚠️  Warning: Failed to activate klgrade env"

echo "════════════════════════════════════════════════════════"
echo "Step 4: Preprocess Balanced Datasets"
echo "════════════════════════════════════════════════════════"
echo ""
echo "This step creates preprocessing variants for ALL balanced datasets:"
echo "  1. Cropped knees (5-class) - 4 variants"
echo "  2. Cropped knees (4-class) - 4 variants"
echo "  3. Full X-rays (5-class) - 4 variants"
echo "  4. Full X-rays (4-class) - 4 variants"
echo ""
echo "Preprocessing variants:"
echo "  - resize_only          (no enhancement)"
echo "  - blur_clahe2          (Gaussian blur + CLAHE clipLimit=2)"
echo "  - sharp_clahe4         (Sharpening + CLAHE clipLimit=4)"
echo "  - blur_clahe2_notebook (Notebook-based preprocessing)"
echo ""
echo "⚠️  Note: This creates 4 × 4 = 16 processed datasets"
echo "   This may take significant time and disk space."
echo ""

# Prompt user to continue
read -p "Continue with preprocessing? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Skipping preprocessing variants."
    echo ""
    echo "You can still run individual preprocessing scripts:"
    echo "  python scripts/preprocessing/preprocess_dataset.py --dataset <name>"
    echo ""
    echo "Or skip to: bash scripts/pipelines/step_5_create_splits.sh"
    exit 0
fi

echo ""
echo "─────────────────────────────────────────────────────────"
echo "4.1 Preprocessing balanced cropped knees (5-class)..."
echo "─────────────────────────────────────────────────────────"

python scripts/preprocessing/preprocess_balanced.py --dataset knees_cropped || {
    echo "❌ Failed to preprocess knees_cropped"
    exit 1
}

echo ""
echo "─────────────────────────────────────────────────────────"
echo "4.2 Preprocessing balanced cropped knees (4-class)..."
echo "─────────────────────────────────────────────────────────"

python scripts/preprocessing/preprocess_balanced.py --dataset knees_cropped_4_class || {
    echo "❌ Failed to preprocess knees_cropped_4_class"
    exit 1
}

echo ""
echo "─────────────────────────────────────────────────────────"
echo "4.3 Preprocessing balanced cropped knees (8-class)..."
echo "─────────────────────────────────────────────────────────"

python scripts/preprocessing/preprocess_balanced.py --dataset knees_cropped_8_class || {
    echo "❌ Failed to preprocess knees_cropped_8_class"
    exit 1
}

echo ""
echo "─────────────────────────────────────────────────────────"
echo "4.4 Preprocessing balanced cropped knees (10-class)..."
echo "─────────────────────────────────────────────────────────"

python scripts/preprocessing/preprocess_balanced.py --dataset knees_cropped_10_class || {
    echo "❌ Failed to preprocess knees_cropped_10_class"
    exit 1
}

echo ""
echo "─────────────────────────────────────────────────────────"
echo "4.5 Preprocessing balanced full X-rays (5-class)..."
echo "─────────────────────────────────────────────────────────"

python scripts/preprocessing/preprocess_balanced.py --dataset full_xray || {
    echo "❌ Failed to preprocess full_xray"
    exit 1
}

echo ""
echo "─────────────────────────────────────────────────────────"
echo "4.6 Preprocessing balanced full X-rays (4-class)..."
echo "─────────────────────────────────────────────────────────"

python scripts/preprocessing/preprocess_balanced.py --dataset full_xray_4_class || {
    echo "❌ Failed to preprocess full_xray_4_class"
    exit 1
}

echo ""
echo "─────────────────────────────────────────────────────────"
echo "4.7 Preprocessing balanced full X-rays (8-class)..."
echo "─────────────────────────────────────────────────────────"

python scripts/preprocessing/preprocess_balanced.py --dataset full_xray_8_class || {
    echo "❌ Failed to preprocess full_xray_8_class"
    exit 1
}

echo ""
echo "─────────────────────────────────────────────────────────"
echo "4.8 Preprocessing balanced full X-rays (10-class)..."
echo "─────────────────────────────────────────────────────────"

python scripts/preprocessing/preprocess_balanced.py --dataset full_xray_10_class || {
    echo "❌ Failed to preprocess full_xray_10_class"
    exit 1
}

echo ""
echo "════════════════════════════════════════════════════════"
echo "✅ Step 4 Complete!"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Summary:"
echo "  ✅ Preprocessed 8 balanced datasets × 4 variants = 32 processed datasets"
echo ""
echo "Output structure:"
echo "  datasets/processed_balanced/"
echo "    ├── knees_cropped/       # 5-class cropped knees"
echo "    │   ├── resize_only/"
echo "    │   ├── blur_clahe2/"
echo "    │   ├── sharp_clahe4/"
echo "    │   └── blur_clahe2_notebook/"
echo "    ├── knees_cropped_4_class/"
echo "    ├── knees_cropped_8_class/"
echo "    ├── knees_cropped_10_class/"
echo "    ├── full_xray/           # 5-class full x-rays"
echo "    ├── full_xray_4_class/"
echo "    ├── full_xray_8_class/"
echo "    └── full_xray_10_class/"
echo ""
echo "Next: Run step_5_create_splits.sh to create train/val/test splits"
