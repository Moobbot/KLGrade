#!/bin/bash

# Visualize end-to-end model predictions on multiple images
# Processes one image at a time to avoid OOM

CHECKPOINT="runs/end_to_end/test_5epochs/best.pt"
IMAGE_DIR="datasets/dataset_v0/images"
OUTPUT_DIR="runs/end_to_end/visualizations"
IMAGE_SIZE=384
NUM_SAMPLES=5

echo "Visualizing end-to-end model predictions"
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

# Process each image separately to avoid OOM
success=0
for img in "${images[@]}"; do
    echo "Processing: $(basename $img)"
    
    # Run visualization for single image (clears memory between runs)
    python scripts/training/visualize_end_to_end_single.py \
        --checkpoint "$CHECKPOINT" \
        --image "$img" \
        --output-dir "$OUTPUT_DIR" \
        --image-size $IMAGE_SIZE
    
    if [ $? -eq 0 ]; then
        ((success++))
    fi
    
    # Wait for GPU to fully release memory
    echo "  Waiting for GPU cleanup..."
    sleep 3
    
    echo ""
done

echo "=========================================="
echo "Completed! $success/${#images[@]} images processed"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
