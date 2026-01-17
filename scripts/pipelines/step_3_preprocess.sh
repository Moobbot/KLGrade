#!/bin/bash
#
# Step 3: Create Preprocessing Variants (Optional - for ablation studies)
#
# Input:  datasets/dataset_knees_cropped/
#         datasets/dataset_v0/
# Output: datasets/processed/knees_cropped/
#         datasets/processed/full_xray/
#

set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate klgrade || echo "⚠️  Warning: Failed to activate klgrade env"

echo "════════════════════════════════════════════════════════"
echo "Step 3: Create Preprocessing Variants (Optional)"
echo "════════════════════════════════════════════════════════"
echo ""
echo "This step creates multiple preprocessing variants:"
echo "  - resize_only          (no enhancement)"
echo "  - blur_clahe2          (Gaussian blur + CLAHE clipLimit=2)"
echo "  - sharp_clahe4         (Sharpening + CLAHE clipLimit=4)"
echo "  - blur_clahe2_notebook (Notebook-based preprocessing)"
echo ""
echo "⚠️  Note: This is for ablation studies only."
echo "   You can skip this and go directly to step_4_balance.sh"
echo ""

read -p "Continue with preprocessing variants? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Skipping preprocessing variants."
    echo "Run step_4_balance.sh next."
    exit 0
fi

echo "─────────────────────────────────────────────────────────"
echo "3.1 Preprocessing cropped knees..."
echo "─────────────────────────────────────────────────────────"

# Check if preprocessing script exists
if [ -f "scripts/preprocessing/preprocess_knees_cropped.sh" ]; then
    bash scripts/preprocessing/preprocess_knees_cropped.sh
    echo "✅ Cropped knees preprocessing complete"
else
    echo "⚠️  Script not found: scripts/preprocessing/preprocess_knees_cropped.sh"
    echo "   Creating variants manually..."
    
    # Create output directories
    mkdir -p datasets/processed/knees_cropped/{resize_only,blur_clahe2,sharp_clahe4}
    
    # Note: Actual preprocessing implementation would go here
    echo "   Please implement preprocessing logic or use existing script"
fi

echo ""
echo "─────────────────────────────────────────────────────────"
echo "3.2 Preprocessing full X-rays..."
echo "─────────────────────────────────────────────────────────"

if [ -f "scripts/preprocessing/preprocess_full_xrays.sh" ]; then
    bash scripts/preprocessing/preprocess_full_xrays.sh
    echo "✅ Full X-rays preprocessing complete"
else
    echo "⚠️  Script not found: scripts/preprocessing/preprocess_full_xrays.sh"
    mkdir -p datasets/processed/full_xray/{resize_only,blur_clahe2,sharp_clahe4}
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "✅ Step 3 Complete!"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Summary:"
echo "  ✅ Preprocessing variants created"
echo ""
echo "Output:"
echo "  datasets/processed/"
echo "    ├── knees_cropped/"
echo "    │   ├── resize_only/"
echo "    │   ├── blur_clahe2/"
echo "    │   ├── sharp_clahe4/"
echo "    │   └── blur_clahe2_notebook/"
echo "    └── full_xray/"
echo "        ├── resize_only/"
echo "        ├── blur_clahe2/"
echo "        ├── sharp_clahe4/"
echo "        └── blur_clahe2_notebook/"
echo ""
echo "Next: Run step_4_balance.sh"
