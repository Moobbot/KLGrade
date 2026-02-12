#!/bin/bash
# CDT-CAD Training - Recommended Fast Configuration
# Batch=12, img_size=384, hidden_dim=128
# Training time: ~50 minutes for 5-class, ~1.5 hours for 10-class

set -e  # Exit on error

echo "============================================================"
echo "CDT-CAD Training - Recommended Fast Configuration"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  - Batch size: 12"
echo "  - Image size: 384"
echo "  - Hidden dim: 128"
echo "  - Accumulation steps: 2 (effective batch = 24)"
echo "  - Early stopping: 15 epochs patience"
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate klgrade

# Load and export WandB API key if .wandb.env exists
if [ -f .wandb.env ]; then
    export $(grep -v '^#' .wandb.env | xargs)
    echo "✅ Loaded WandB configuration"
    echo "   Project: $WANDB_PROJECT"
    echo "   Entity: $WANDB_ENTITY"
fi

# Train 5-class model
echo "============================================================"
echo "Training 5-Class Model (KL0-KL4)"
echo "============================================================"
echo "Expected time: ~50 minutes"
echo ""

python api_kiocmil_cada/training/train_cdt_cad_lesion.py \
  --data datasets/balanced/knees_cropped/dataset_cdt_cad_5class.yaml \
  --epochs 100 \
  --batch 8 \
  --accumulation-steps 3 \
  --img-size 384 \
  --hidden-dim 128 \
  --num-encoder-layers 4 \
  --num-decoder-layers 4 \
  --num-classes 5 \
  --early-stopping-patience 15 \
  --device cuda \
  --save-dir runs/cdt_cad/balanced_5class_fast

echo ""
echo "✅ 5-Class training complete!"
echo ""

# Train 10-class model
echo "============================================================"
echo "Training 10-Class Model (KL0-a/b to KL4-a/b)"
echo "============================================================"
echo "Expected time: ~1.5 hours"
echo ""

python api_kiocmil_cada/training/train_cdt_cad_lesion.py \
  --data datasets/balanced/knees_cropped_10_class/dataset_cdt_cad_10class.yaml \
  --epochs 100 \
  --batch 12 \
  --accumulation-steps 2 \
  --img-size 384 \
  --hidden-dim 128 \
  --num-encoder-layers 4 \
  --num-decoder-layers 4 \
  --num-classes 10 \
  --early-stopping-patience 15 \
  --device cuda \
  --save-dir runs/cdt_cad/balanced_10class_fast

echo ""
echo "✅ 10-Class training complete!"
echo ""

echo "============================================================"
echo "All Training Complete!"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  - 5-class:  runs/cdt_cad/balanced_5class_fast/"
echo "  - 10-class: runs/cdt_cad/balanced_10class_fast/"
echo ""
echo "Best models:"
echo "  - runs/cdt_cad/balanced_5class_fast/best.pt"
echo "  - runs/cdt_cad/balanced_10class_fast/best.pt"
echo ""
