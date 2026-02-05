#!/bin/bash

# Optimized visualization script with reduced image size to avoid OOM
# Uses 256x256 instead of 384x384 to reduce memory usage by ~50%

CHECKPOINT="runs/end_to_end/test_5epochs/best.pt"
IMAGE_DIR="datasets/dataset_v0/images"
OUTPUT_DIR="runs/end_to_end/visualizations_optimized"
IMAGE_SIZE=256  # Reduced from 384 to avoid OOM
NUM_SAMPLES=10

echo "Visualizing end-to-end model predictions (Optimized)"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Image directory: $IMAGE_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Image size: ${IMAGE_SIZE}x${IMAGE_SIZE} (optimized for memory)"
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
failed=0
for img in "${images[@]}"; do
    echo "Processing: $(basename $img)"
    
    # Run visualization for single image
    python scripts/training/visualize_end_to_end_single.py \
        --checkpoint "$CHECKPOINT" \
        --image "$img" \
        --output-dir "$OUTPUT_DIR" \
        --image-size $IMAGE_SIZE
    
    if [ $? -eq 0 ]; then
        ((success++))
    else
        ((failed++))
    fi
    
    # Wait for GPU to fully release memory
    echo "  Waiting for GPU cleanup..."
    sleep 5  # Increased from 3s to 5s
    
    echo ""
done

echo "=========================================="
echo "✅ Success: $success/${#images[@]} images"
if [ $failed -gt 0 ]; then
    echo "❌ Failed: $failed images"
fi
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
