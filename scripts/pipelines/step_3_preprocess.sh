#!/bin/bash
#
# Step 3: Create Preprocessing Variants (Comprehensive)
#
# Preprocesses ALL 4 dataset variants with multiple preprocessing methods:
# 1. Full X-rays (5-class + 10-class)
# 2. Full X-rays (4-class + 8-class)
# 3. Cropped knees (5-class + 10-class)
# 4. Cropped knees (4-class + 8-class)
#

set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate klgrade || echo "⚠️  Warning: Failed to activate klgrade env"

echo "════════════════════════════════════════════════════════"
echo "Step 3: Create Preprocessing Variants (Comprehensive)"
echo "════════════════════════════════════════════════════════"
echo ""
echo "This step creates preprocessing variants for ALL datasets:"
echo ""
echo "  Datasets:"
echo "    1. Full X-rays (5-class + 10-class)"
echo "    2. Full X-rays (4-class + 8-class)"
echo "    3. Cropped knees (5-class + 10-class)"
echo "    4. Cropped knees (4-class + 8-class)"
echo ""
echo "  Preprocessing variants:"
echo "    - resize_only          (no enhancement)"
echo "    - blur_clahe2          (Gaussian blur + CLAHE clipLimit=2)"
echo "    - sharp_clahe4         (Sharpening + CLAHE clipLimit=4)"
echo "    - blur_clahe2_notebook (Notebook-based preprocessing)"
echo ""
echo "⚠️  Note: This creates 4 × 4 = 16 processed datasets"
echo "   This may take significant time and disk space."
echo ""

read -p "Continue with comprehensive preprocessing? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Skipping preprocessing variants."
    echo ""
    echo "You can still run individual preprocessing scripts:"
    echo "  python scripts/preprocessing/preprocess_dataset.py --dataset full_xrays"
    echo "  python scripts/preprocessing/preprocess_dataset.py --dataset full_xrays_4class"
    echo "  python scripts/preprocessing/preprocess_dataset.py --dataset knees_cropped"
    echo "  python scripts/preprocessing/preprocess_dataset.py --dataset knees_cropped_4class"
    echo ""
    echo "Or skip to: bash scripts/pipelines/step_4_balance.sh"
    exit 0
fi

echo ""
echo "─────────────────────────────────────────────────────────"
echo "3.1 Preprocessing full X-rays (5-class + 10-class)..."
echo "─────────────────────────────────────────────────────────"

python scripts/preprocessing/preprocess_dataset.py --dataset full_xrays || {
    echo "❌ Failed to preprocess full X-rays"
    exit 1
}

echo ""
echo "─────────────────────────────────────────────────────────"
echo "3.2 Preprocessing full X-rays 4-class (4-class + 8-class)..."
echo "─────────────────────────────────────────────────────────"

python scripts/preprocessing/preprocess_dataset.py --dataset full_xrays_4class || {
    echo "❌ Failed to preprocess full X-rays 4-class"
    exit 1
}

echo ""
echo "─────────────────────────────────────────────────────────"
echo "3.3 Preprocessing cropped knees (5-class + 10-class)..."
echo "─────────────────────────────────────────────────────────"

python scripts/preprocessing/preprocess_dataset.py --dataset knees_cropped || {
    echo "❌ Failed to preprocess cropped knees"
    exit 1
}

echo ""
echo "─────────────────────────────────────────────────────────"
echo "3.4 Preprocessing cropped knees 4-class (4-class + 8-class)..."
echo "─────────────────────────────────────────────────────────"

python scripts/preprocessing/preprocess_dataset.py --dataset knees_cropped_4class || {
    echo "❌ Failed to preprocess cropped knees 4-class"
    exit 1
}

echo ""
echo "════════════════════════════════════════════════════════"
echo "✅ Step 3 Complete!"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Summary:"
echo "  ✅ Preprocessed 4 datasets × 4 variants = 16 processed datasets"
echo ""
echo "Output structure:"
echo "  datasets/processed/"
echo "    ├── full_xray/           (5-class + 10-class)"
echo "    │   ├── resize_only/"
echo "    │   ├── blur_clahe2/"
echo "    │   ├── sharp_clahe4/"
echo "    │   └── blur_clahe2_notebook/"
echo "    ├── full_xray_4class/    (4-class + 8-class)"
echo "    │   ├── resize_only/"
echo "    │   ├── blur_clahe2/"
echo "    │   ├── sharp_clahe4/"
echo "    │   └── blur_clahe2_notebook/"
echo "    ├── knees_cropped/       (5-class + 10-class)"
echo "    │   ├── resize_only/"
echo "    │   ├── blur_clahe2/"
echo "    │   ├── sharp_clahe4/"
echo "    │   └── blur_clahe2_notebook/"
echo "    └── knees_cropped_4class/ (4-class + 8-class)"
echo "        ├── resize_only/"
echo "        ├── blur_clahe2/"
echo "        ├── sharp_clahe4/"
echo "        └── blur_clahe2_notebook/"
echo ""
echo "Next: Run step_4_balance.sh to balance datasets for training"
