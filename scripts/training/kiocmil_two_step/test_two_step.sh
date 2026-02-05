#!/bin/bash
# Quick test script for two-step pipeline
# Usage: ./test_two_step.sh [image_path]

echo "Two-Step Pipeline Test"
echo "======================"
echo ""

# Default paths
KNEE_DETECTOR="runs/my_knee_run_resplit/weights/best.pt"
KIOCMIL_CLASSIFIER="runs/kiocmil_cada/cada_10class_balanced/best_acc_model.pt"

# Get sample image
if [ -z "$1" ]; then
    # Use first image from dataset
    IMAGE=$(ls datasets/dataset_v0/images/*.jpg 2>/dev/null | head -1)
    if [ -z "$IMAGE" ]; then
        echo "❌ No images found in datasets/dataset_v0/images/"
        echo "Usage: $0 <image_path>"
        exit 1
    fi
    echo "Using sample image: $IMAGE"
else
    IMAGE="$1"
fi

# Check if files exist
if [ ! -f "$KNEE_DETECTOR" ]; then
    echo "❌ Knee detector not found: $KNEE_DETECTOR"
    exit 1
fi

if [ ! -f "$KIOCMIL_CLASSIFIER" ]; then
    echo "❌ KIOCMIL classifier not found: $KIOCMIL_CLASSIFIER"
    exit 1
fi

if [ ! -f "$IMAGE" ]; then
    echo "❌ Image not found: $IMAGE"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Knee Detector: $KNEE_DETECTOR"
echo "  KIOCMIL Classifier: $KIOCMIL_CLASSIFIER"
echo "  Test Image: $IMAGE"
echo ""

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate klgrade

# Change to project root
PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Run test
python scripts/training/kiocmil_two_step/test_two_step_pipeline.py \
  --knee-detector "$KNEE_DETECTOR" \
  --kiocmil-classifier "$KIOCMIL_CLASSIFIER" \
  --image "$IMAGE" \
  --num-classes 10 \
  --device cuda

echo ""
echo "Test completed!"
