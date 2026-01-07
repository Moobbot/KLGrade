# GPU Setup and Testing Guide

## Quick GPU Test

### 1. Basic GPU Test (PyTorch)

```bash
# Test if GPU is available
python scripts/test_gpu.py
```

This will check:

- ✅ CUDA availability
- ✅ GPU count and names
- ✅ Memory info
- ✅ Basic computation test

### 2. YOLO GPU Test (1 epoch)

```bash
# Test YOLO training on GPU
python scripts/test_yolo_gpu.py
```

This runs a quick 1-epoch training to verify:

- ✅ YOLO can use GPU
- ✅ No memory errors
- ✅ Training pipeline works

## Expected Output

### Successful GPU Test:

```
============================================================
GPU Test Report
============================================================

1. CUDA Available: True
2. CUDA Version: 12.1
3. GPU Count: 1

4. GPU Details:
   GPU 0: NVIDIA RTX 4090
      - Total Memory: 24.00 GB
      - Multi-Processors: 128
      - Compute Capability: 8.9

5. Current GPU Memory:
   GPU 0:
      - Allocated: 0.00 GB
      - Reserved: 0.00 GB

6. Running GPU computation test...
   ✅ GPU computation successful!

7. cuDNN:
   Enabled: True
   Version: 8902

============================================================
✅ All GPU tests passed!
============================================================
```

## Troubleshooting

### Error: "CUDA not available"

**Check NVIDIA driver:**

```bash
nvidia-smi
```

**Reinstall PyTorch with CUDA:**

```bash
# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Error: "Out of memory"

**Reduce batch size:**

```yaml
# In your config YAML
batch: 8 # Try 4 or 2 if OOM
```

**Clear GPU cache:**

```python
import torch
torch.cuda.empty_cache()
```

### Error: "RuntimeError: No CUDA GPUs are available"

**Check GPU visibility:**

```bash
# See available GPUs
echo $CUDA_VISIBLE_DEVICES

# Set specific GPU
export CUDA_VISIBLE_DEVICES=0
```

## GPU Monitoring

### Real-time monitoring:

```bash
# Update every 1 second
watch -n 1 nvidia-smi
```

### During training:

```bash
# In another terminal
watch -n 1 nvidia-smi
```

## Optimal Settings

### For RTX 4090 (24GB):

```yaml
batch: 16-32
imgsz: 640
workers: 8
```

### For RTX 3090 (24GB):

```yaml
batch: 16-24
imgsz: 640
workers: 8
```

### For RTX 3080 (10GB):

```yaml
batch: 8-12
imgsz: 640
workers: 4
```

### For RTX 3060 (12GB):

```yaml
batch: 8-16
imgsz: 640
workers: 4
```

## Multi-GPU Training

```bash
# Train on multiple GPUs
yolo detect train data=configs/yolo_5_class_baseline.yaml \
    epochs=100 \
    batch=32 \
    device=0,1,2,3  # Use GPUs 0,1,2,3
```

## Performance Tips

1. **Use AMP (Automatic Mixed Precision)**: Enabled by default in YOLO
2. **Increase workers**: `workers=8` for faster data loading
3. **Increase batch size**: Max out GPU memory for faster training
4. **Use cache**: `cache=True` if dataset fits in RAM
