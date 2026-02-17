#!/bin/bash

# Activate conda environment if needed (adjust path as necessary)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate klgrade

echo "Starting Two-Step YOLO API Server on port 9090..."
echo "Models:"
echo "  - Knee: runs/detect/knee_detector/weights/best.pt"
echo "  - Lesion: runs/detect/lesion_8class_balanced/weights/best.pt"

# Run server
python -m api_two_step_yolo.inference.server
