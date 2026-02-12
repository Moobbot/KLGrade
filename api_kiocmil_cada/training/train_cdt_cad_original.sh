#!/bin/bash
# CDT-CAD Training - Original Configuration
# Batch=1, img_size=512, hidden_dim=128
# Training time: ~10-13 hours for 5-class, ~17-20 hours for 10-class

set -e  # Exit on error

echo "============================================================"
echo "CDT-CAD Training - Original Configuration"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  - Batch size: 1"
echo "  - Image size: 512"
echo "  - Hidden dim: 128"
echo "  - Accumulation steps: 8 (effective batch = 8)"
echo "  - Early stopping: 15 epochs patience"
echo ""
echo "⚠️  WARNING: This is the SLOW configuration!"
echo "    Consider using train_cdt_cad_fast.sh instead (12x faster)"
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate klgrade

# Train 5-class model
echo "============================================================"
echo "Training 5-Class Model (KL0-KL4)"
echo "============================================================"
echo "Expected time: ~10-13 hours"
echo ""

python api_kiocmil_cada/training/train_cdt_cad_lesion.py \
  --data datasets/balanced/knees_cropped/dataset_cdt_cad_5class.yaml \
  --epochs 100 \
  --batch 1 \
  --accumulation-steps 8 \
  --img-size 512 \
  --hidden-dim 128 \
  --num-encoder-layers 4 \
  --num-decoder-layers 4 \
  --early-stopping-patience 15 \
  --device cuda \
  --save-dir runs/cdt_cad/balanced_5class_original

echo ""
echo "✅ 5-Class training complete!"
echo ""

# Train 10-class model
echo "============================================================"
echo "Training 10-Class Model (KL0-a/b to KL4-a/b)"
echo "============================================================"
echo "Expected time: ~17-20 hours"
echo ""

python api_kiocmil_cada/training/train_cdt_cad_lesion.py \
  --data datasets/balanced/knees_cropped_10_class/dataset_cdt_cad_10class.yaml \
  --epochs 100 \
  --batch 1 \
  --accumulation-steps 8 \
  --img-size 512 \
  --hidden-dim 128 \
  --num-encoder-layers 4 \
  --num-decoder-layers 4 \
  --num-classes 10 \
  --early-stopping-patience 15 \
  --device cuda \
  --save-dir runs/cdt_cad/balanced_10class_original

echo ""
echo "✅ 10-Class training complete!"
echo ""

echo "============================================================"
echo "All Training Complete!"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  - 5-class:  runs/cdt_cad/balanced_5class_original/"
echo "  - 10-class: runs/cdt_cad/balanced_10class_original/"
echo ""
echo "Best models:"
echo "  - runs/cdt_cad/balanced_5class_original/best.pt"
echo "  - runs/cdt_cad/balanced_10class_original/best.pt"
echo ""
