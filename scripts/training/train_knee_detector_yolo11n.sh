#!/bin/bash
# Train YOLO11N knee detector with descriptive naming
# This avoids folder name conflicts in runs/detect/

set -e

echo "=========================================="
echo "YOLO11N Knee Detector Training"
echo "=========================================="

# Configuration
MODEL="yolo11n.pt"
DATA="datasets/splits/knee/dataset.yaml"
EPOCHS=100
BATCH=16
IMGSZ=640
DEVICE=0
PATIENCE=20  # Early stopping patience

# Generate unique run name with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="knee_yolo11n_${TIMESTAMP}"

echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Data: $DATA"
echo "  Epochs: $EPOCHS"
echo "  Batch: $BATCH"
echo "  Image Size: $IMGSZ"
echo "  Device: GPU $DEVICE"
echo "  Early Stopping: $PATIENCE epochs"
echo "  Run Name: $RUN_NAME"
echo ""

# Activate conda environment
echo "Activating conda environment: klgrade"
eval "$(conda shell.bash hook)"
conda activate klgrade

# Check if data file exists
if [ ! -f "$DATA" ]; then
    echo "ERROR: Data file not found: $DATA"
    exit 1
fi

# Start training
echo "Starting training..."
echo "=========================================="

python src/training/knee_detector/train.py \
    --model $MODEL \
    --data $DATA \
    --epochs $EPOCHS \
    --batch $BATCH \
    --imgsz $IMGSZ \
    --device $DEVICE \
    --project runs/detect \
    --name $RUN_NAME \
    --patience $PATIENCE \
    --save true \
    --save_period 10 \
    --cache false \
    --workers 8 \
    --pretrained true \
    --optimizer auto \
    --verbose true \
    --plots true \
    --val true

echo ""
echo "=========================================="
echo "Training completed!"
echo "Results saved to: runs/detect/$RUN_NAME"
echo "=========================================="

# Run evaluation on all splits
echo ""
echo "Running full evaluation..."
python scripts/evaluation/evaluate_knee_full.py \
    --model "runs/detect/$RUN_NAME/weights/best.pt" \
    --output "docs/knee_yolo11n_${TIMESTAMP}_evaluation.json"

echo ""
echo "✅ All done!"
echo "Model: runs/detect/$RUN_NAME/weights/best.pt"
