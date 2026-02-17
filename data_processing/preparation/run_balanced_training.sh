#!/bin/bash

# Setup environments and paths
WORK_DIR="/home/ngoductam/KLGrade"
cd $WORK_DIR

# Activate environment (optional, assuming user runs this in their env but good to be safe if running as subprocess)
# source activate klgrade 

echo "=================================================="
echo "Step 1: Setting up datasets (Splits & Configs)"
echo "=================================================="
python setup_balanced_training.py

echo ""
echo "=================================================="
echo "Step 2: Starting Training"
echo "=================================================="

# Default parameters
EPOCHS=${1:-100}
BATCH=${2:-16}
IMG_SIZE=${3:-640}
DEVICE=${4:-"0"}

# 1. Train 5-class model
echo "Training 5-class Balanced Model..."
yolo detect train \
    data=datasets/processed_balanced/knees_cropped/resize_only/data.yaml \
    model=yolo11l.pt \
    epochs=$EPOCHS \
    batch=$BATCH \
    imgsz=$IMG_SIZE \
    device=$DEVICE \
    project=runs/detect \
    name=balanced_5class \
    exist_ok=True \
    val=True

# 2. Train 10-class model
echo ""
echo "Training 10-class Balanced Model..."
yolo detect train \
    data=datasets/processed_balanced/knees_cropped_10_class/resize_only/data.yaml \
    model=yolo11l.pt \
    epochs=$EPOCHS \
    batch=$BATCH \
    imgsz=$IMG_SIZE \
    device=$DEVICE \
    project=runs/detect \
    name=balanced_10class \
    exist_ok=True \
    val=True

echo ""
echo "Training Complete. Check runs/detect/balanced_5class and runs/detect/balanced_10class"
