#!/bin/bash
# Test YOLO with WandB Logging
# This script will re-run the training with proper WandB integration

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

echo "========================================================================"
echo "Testing YOLO with WandB Logging"
echo "========================================================================"
echo ""

# Run training with WandB enabled
python scripts/training/train_yolo.py \
  --img_dir processed/knee/images \
  --label_dir processed/knee/labels \
  --model yolo11n.pt \
  --epochs 2 \
  --batch 4 \
  --device 0 \
  --project runs/detect \
  --name wandb_test_v2

echo ""
echo "✅ Test complete! Check WandB dashboard:"
echo "   https://wandb.ai/YOUR_USERNAME/KLGrade-Knee-OA"
echo ""
