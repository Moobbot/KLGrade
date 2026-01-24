#!/bin/bash
# Quick Start Script for KLGrade API

echo "=========================================="
echo "KLGrade API Quick Start"
echo "=========================================="

# Check if models exist
echo ""
echo "Checking model files..."

KIOCMIL_MODEL="runs/kiocmil_cada/cada_10class_balanced/best_acc_model.pt"
KNEE_MODEL="runs/detect/my_knee_run_resplit/weights/best.pt"

if [ ! -f "$KIOCMIL_MODEL" ]; then
    echo "❌ KIOCMIL model not found: $KIOCMIL_MODEL"
    exit 1
fi

if [ ! -f "$KNEE_MODEL" ]; then
    echo "❌ Knee detection model not found: $KNEE_MODEL"
    exit 1
fi

echo "✅ All models found"

# Start API server
echo ""
echo "Starting API server on port 8001..."
echo "API Documentation will be available at: http://localhost:8001/docs"
echo ""

python scripts/deployment/kiocmil_api_server.py \
  --kiocmil-model "$KIOCMIL_MODEL" \
  --knee-model "$KNEE_MODEL" \
  --lesion-model "$KNEE_MODEL" \
  --num-classes 10 \
  --host 0.0.0.0 \
  --port 8001
