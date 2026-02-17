#!/bin/bash
#
# YOLO Training Script
#
# Trains YOLO models on prepared datasets
#

set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate klgrade || echo "⚠️  Warning: Failed to activate klgrade env"

echo "════════════════════════════════════════════════════════"
echo "YOLO Training Pipeline"
echo "════════════════════════════════════════════════════════"
echo ""

# ============================================================
# Configuration
# ============================================================

DEFAULT_EPOCHS=100
DEFAULT_BATCH=16
DEFAULT_IMG_SIZE=640

echo "Training Configuration:"
echo "  Default Epochs: $DEFAULT_EPOCHS"
echo "  Default Batch:  $DEFAULT_BATCH"
echo "  Default ImgSz:  $DEFAULT_IMG_SIZE"
echo ""

# ============================================================
# Dataset Selection
# ============================================================

echo "Available datasets:"
echo "  1. dataset_knees_cropped (5-class, unbalanced)"
echo "  2. balanced_knees_cropped (5-class, balanced)"
echo "  3. dataset_knees_cropped_4_class (4-class, unbalanced)"
echo "  4. balanced_knees_cropped_4_class (4-class, balanced)"
echo "  5. dataset_knees_cropped_8_class (8-class, unbalanced)"
echo "  6. balanced_knees_cropped_8_class (8-class, balanced)"
echo "  7. knee_full_10_class (10-class, full X-rays)"
echo "  8. Custom config"
echo ""

read -p "Select dataset (1-8): " dataset_choice

case $dataset_choice in
    1)
        CONFIG="configs/yolo_5_class_baseline.yaml"
        DATASET_NAME="5-class Unbalanced"
        ;;
    2)
        CONFIG="configs/yolo_5_class_balanced.yaml"
        DATASET_NAME="5-class Balanced"
        ;;
    3)
        CONFIG="configs/yolo_4_class_baseline.yaml"
        DATASET_NAME="4-class Unbalanced"
        ;;
    4)
        CONFIG="configs/yolo_4_class_balanced.yaml"
        DATASET_NAME="4-class Balanced"
        ;;
    5)
        CONFIG="configs/yolo_8_class_baseline.yaml"
        DATASET_NAME="8-class Unbalanced"
        ;;
    6)
        CONFIG="configs/yolo_8_class_balanced.yaml"
        DATASET_NAME="8-class Balanced"
        ;;
    7)
        CONFIG="configs/yolo_10_class_full.yaml"
        DATASET_NAME="10-class Full X-rays"
        ;;
    8)
        read -p "Enter config path: " CONFIG
        DATASET_NAME="Custom"
        ;;
    *)
        echo "Invalid choice. Using default: 5-class unbalanced"
        CONFIG="configs/yolo_5_class_baseline.yaml"
        DATASET_NAME="5-class Unbalanced"
        ;;
esac

echo ""
echo "Selected: $DATASET_NAME"
echo "Config:   $CONFIG"
echo ""

# Check if config exists
if [ ! -f "$CONFIG" ]; then
    echo "❌ Error: Config file not found: $CONFIG"
    echo "   Please create it or select a different dataset."
    exit 1
fi

# ============================================================
# Training Parameters
# ============================================================

echo "Training parameters:"
read -p "Epochs [$DEFAULT_EPOCHS]: " epochs
epochs=${epochs:-$DEFAULT_EPOCHS}

read -p "Batch size [$DEFAULT_BATCH]: " batch
batch=${batch:-$DEFAULT_BATCH}

read -p "Image size [$DEFAULT_IMG_SIZE]: " imgsz
imgsz=${imgsz:-$DEFAULT_IMG_SIZE}

echo ""
echo "Final configuration:"
echo "  Dataset:   $DATASET_NAME"
echo "  Config:    $CONFIG"
echo "  Epochs:    $epochs"
echo "  Batch:     $batch"
echo "  Image size: $imgsz"
echo ""

read -p "Start training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

# ============================================================
# Training
# ============================================================

echo ""
echo "════════════════════════════════════════════════════════"
echo "Starting YOLO Training..."
echo "════════════════════════════════════════════════════════"
echo ""

yolo detect train \
    data="$CONFIG" \
    epochs=$epochs \
    batch=$batch \
    imgsz=$imgsz \
    patience=50 \
    save=True \
    device=0 \
    workers=8 \
    project=runs/detect \
    name="${DATASET_NAME// /_}_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "════════════════════════════════════════════════════════"
echo "✅ Training Complete!"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Results saved to: runs/detect/"
echo ""
echo "Next steps:"
echo "  1. Evaluate model: python scripts/evaluation/evaluate_yolo_standalone.py"
echo "  2. Visualize: python scripts/visualization/visualize_gradcam.py"
echo "  3. Test predictions: yolo detect predict model=runs/detect/train/weights/best.pt"
