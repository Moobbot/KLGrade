#!/bin/bash

# Visualize end-to-end model predictions on multiple images
# Uses CPU mode to avoid GPU OOM issues

CHECKPOINT="runs/end_to_end/test_5epochs/best.pt"
IMAGE_DIR="datasets/dataset_v0/images"
OUTPUT_DIR="runs/end_to_end/visualizations"
IMAGE_SIZE=384
NUM_SAMPLES=10

echo "Visualizing end-to-end model predictions (CPU mode)"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Image directory: $IMAGE_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Image size: $IMAGE_SIZE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get list of images
images=($(ls "$IMAGE_DIR"/*.jpg 2>/dev/null | head -n $NUM_SAMPLES))

if [ ${#images[@]} -eq 0 ]; then
    echo "❌ No images found in $IMAGE_DIR"
    exit 1
fi

echo "Found ${#images[@]} images to process"
echo ""

# Process each image separately using CPU (slower but no OOM)
success=0
for img in "${images[@]}"; do
    echo "Processing: $(basename $img)"
    
    # Run visualization for single image on CPU
    python scripts/training/visualize_end_to_end_single.py \
        --checkpoint "$CHECKPOINT" \
        --image "$img" \
        --output-dir "$OUTPUT_DIR" \
        --image-size $IMAGE_SIZE \
        --device cpu
    
    if [ $? -eq 0 ]; then
        ((success++))
    fi
    
    echo ""
done

echo "=========================================="
echo "Completed! $success/${#images[@]} images processed"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
