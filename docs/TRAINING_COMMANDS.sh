#!/bin/bash
# YOLO Training Commands for Linux Server
# KLGrade - Knee OA Detection Project
# Run these commands sequentially for full baseline + augmentation experiments

# Make script exit on first error
set -e

# ============================================================================
# Virtual Environment Setup
# ============================================================================

echo "Activating virtual environment..."
source .venv/bin/activate

# Verify Python is from venv
which python
python --version

# ============================================================================
# PHASE 1: Baseline Training (No Augmentation)
# ============================================================================

echo "Starting Baseline Training Phase..."

# E001: 5-class Baseline
echo "Training E001: 5-class Baseline..."
yolo detect train \
  data=configs/yolo_5_class_baseline.yaml \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  device=0 \
  project=runs/detect \
  name=E001_5class_baseline \
  exist_ok=False

# E004: 10-class Baseline  
echo "Training E004: 10-class Baseline..."
yolo detect train \
  data=configs/yolo_10_class_baseline.yaml \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  device=0 \
  project=runs/detect \
  name=E004_10class_baseline \
  exist_ok=False

# E005: 4-class Baseline
echo "Training E005: 4-class Baseline..."
yolo detect train \
  data=configs/yolo_4_class_baseline.yaml \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  device=0 \
  project=runs/detect \
  name=E005_4class_baseline \
  exist_ok=False

# E006: 8-class Baseline
echo "Training E006: 8-class Baseline..."
yolo detect train \
  data=configs/yolo_8_class_baseline.yaml \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  device=0 \
  project=runs/detect \
  name=E006_8class_baseline \
  exist_ok=False

echo "Baseline training phase complete!"

# ============================================================================
# PHASE 2: Conservative Augmentation
# ============================================================================

echo "Starting Conservative Augmentation Phase..."

# E002: 5-class + Conservative Augmentation
echo "Training E002: 5-class + Conservative..."
yolo detect train \
  data=configs/yolo_5_class_conservative.yaml \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  device=0 \
  project=runs/detect \
  name=E002_5class_conservative \
  exist_ok=False

# E007: 10-class + Conservative Augmentation
echo "Training E007: 10-class + Conservative..."
yolo detect train \
  data=configs/yolo_10_class_conservative.yaml \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  device=0 \
  project=runs/detect \
  name=E007_10class_conservative \
  exist_ok=False

# E008: 4-class + Conservative Augmentation
echo "Training E008: 4-class + Conservative..."
yolo detect train \
  data=configs/yolo_4_class_conservative.yaml \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  device=0 \
  project=runs/detect \
  name=E008_4class_conservative \
  exist_ok=False

# E009: 8-class + Conservative Augmentation
echo "Training E009: 8-class + Conservative..."
yolo detect train \
  data=configs/yolo_8_class_conservative.yaml \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  device=0 \
  project=runs/detect \
  name=E009_8class_conservative \
  exist_ok=False

echo "Conservative augmentation phase complete!"

# ============================================================================
# OPTIONAL: Background Execution
# ============================================================================

# To run in background with logging:
# nohup bash docs/TRAINING_COMMANDS.sh > training.log 2>&1 &

# To monitor progress:
# tail -f training.log

# To check GPU usage:
# watch -n 1 nvidia-smi

echo "All training experiments complete!"
