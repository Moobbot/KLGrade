#!/bin/bash
# Quick test of DETR with Class Imbalance Handling
# Tests RFS sampler + Focal Loss with CB weights

set -e

# Activate conda
eval "$(conda shell.bash hook)"
conda activate klgrade

# Load WandB credentials from .wandb.env
if [ -f ".wandb.env" ]; then
    set -a
    source .wandb.env
    set +a
    echo "✅ Loaded WandB credentials from .wandb.env"
else
    echo "❌ Error: .wandb.env file not found"
    exit 1
fi

# Verify credentials are loaded
if [ -z "$WANDB_API_KEY" ]; then
    echo "❌ Error: WANDB_API_KEY not set"
    exit 1
fi

wandb login $WANDB_API_KEY

echo "========================================================================
Testing DETR with Class Imbalance Solutions
========================================================================
"

# Test 1: RFS Sampler only (2 epochs)
echo "Test 1: RFS Sampler (Repeat Factor Sampling)"
python scripts/training/train_detr.py \
  --img_dir processed/knee/images \
  --label_dir processed/knee/labels \
  --split_dir processed/splits/knee_5_class \
  --num_classes 5 \
  --epochs 2 \
  --batch 4 \
  --device cuda \
  --output runs/detr/test_rfs

echo "✅ RFS test complete!"
echo ""

# Note: For full Focal Loss integration, we need to modify train_detr.py further
# This will be done in next iteration

echo "========================================================================
Test Complete!
========================================================================
"

# Next steps:
# 1. Check WandB dashboard for metrics
# 2. If mAP improves, run full experiments
# 3. Implement 2-stage training
