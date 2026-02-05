# End-to-End Model Visualization Scripts

This directory contains scripts for visualizing end-to-end model predictions on X-ray images.

## Scripts Overview

### Core Scripts

**`visualize_end_to_end_single.py`** - Single image processor
- Processes one image at a time
- Memory-efficient with explicit GPU cleanup
- Configurable image size and device (CPU/CUDA)
- Usage: `python visualize_end_to_end_single.py --checkpoint <path> --image <path> --output-dir <path>`

### Batch Processing Scripts

**`visualize_end_to_end_optimized.sh`** ✅ **RECOMMENDED**
- **Image size**: 256×256 (memory-optimized)
- **Success rate**: 100% (10/10 images)
- **Delay**: 5s between images
- **Use case**: Production visualization, processing multiple images reliably

**`visualize_end_to_end_batch.sh`**
- **Image size**: 384×384 (higher resolution)
- **Success rate**: 60% (3/5 images, may OOM)
- **Delay**: 3s between images
- **Use case**: Higher quality visualization when GPU memory is sufficient

**`visualize_end_to_end_cpu.sh`**
- **Device**: CPU mode (no GPU)
- **Success rate**: 100% (no OOM)
- **Speed**: Slow (~10x slower than GPU)
- **Use case**: Fallback when GPU is unavailable or fully occupied

## Quick Start

### Recommended Usage (Optimized)
```bash
cd /path/to/KLGrade
./scripts/training/visualize_end_to_end_optimized.sh
```

### Custom Single Image
```bash
python scripts/training/visualize_end_to_end_single.py \
  --checkpoint runs/end_to_end/test_5epochs/best.pt \
  --image datasets/dataset_v0/images/sample.jpg \
  --output-dir runs/visualizations \
  --image-size 256
```

## Output Format

Visualizations include:
- 🟢 **Green boxes**: Detected knee regions
- 🔵 **Blue boxes**: Joint Space (JS) lesions  
- 🔴 **Red boxes**: Osteophyte (OST) lesions
- ⚫ **Top banner**: KL grade prediction with confidence percentage

## Memory Optimization

The optimized script (256×256) reduces memory usage by ~50% compared to 384×384:
- **384×384**: 3/5 success (60%), CUDA OOM errors
- **256×256**: 10/10 success (100%), no OOM ✅

## Configuration

Edit the shell scripts to customize:
- `CHECKPOINT`: Path to model checkpoint
- `IMAGE_DIR`: Directory containing input images
- `OUTPUT_DIR`: Where to save visualizations
- `IMAGE_SIZE`: Resolution (256 recommended, 384 for higher quality)
- `NUM_SAMPLES`: Number of images to process

## Troubleshooting

**CUDA OOM errors?**
1. Use `visualize_end_to_end_optimized.sh` (256×256)
2. Reduce `NUM_SAMPLES` in the script
3. Use CPU mode: `visualize_end_to_end_cpu.sh`

**Slow processing?**
- GPU mode: ~2s per image
- CPU mode: ~20s per image
- Consider reducing `NUM_SAMPLES` for faster completion
