# CDT-CAD Migration Guide: Moving to Larger GPU Server

## Overview

This guide explains how to migrate CDT-CAD lesion detector training from the current RTX 2080 Ti (10GB) to a server with larger GPU (e.g., A100 40GB, RTX 3090 24GB, or V100 32GB).

## Why Migrate?

**Current Limitation**: RTX 2080 Ti (10GB VRAM)
- CDT-CAD model requires ~12-16GB for training
- OOM errors even with batch_size=1 and img_size=384

**Target**: Server with ≥24GB VRAM
- Can train full CDT-CAD model
- Larger batch sizes → better performance
- Faster training with more VRAM

---

## Migration Steps

### 1. Prepare Code Package

Create a portable package with all necessary files:

```bash
# On current server (RTX 2080 Ti)
cd /home/ngoductam/KLGrade

# Create migration package
mkdir -p ~/cdt_cad_migration
cd ~/cdt_cad_migration

# Copy essential code
cp -r /home/ngoductam/KLGrade/src ./
cp -r /home/ngoductam/KLGrade/api_kiocmil_cada ./
cp /home/ngoductam/KLGrade/requirements.txt ./

# Copy training script and configs
cp /home/ngoductam/KLGrade/api_kiocmil_cada/training/train_cdt_cad_lesion.py ./
cp /home/ngoductam/KLGrade/datasets/dataset_knees_cropped/dataset_cdt_cad.yaml ./

# Create archive
tar -czf cdt_cad_code.tar.gz src/ api_kiocmil_cada/ requirements.txt train_cdt_cad_lesion.py dataset_cdt_cad.yaml

echo "✅ Code package created: cdt_cad_code.tar.gz"
```

### 2. Prepare Dataset

**Option A: Copy Full Dataset** (if network allows)

```bash
# Package dataset
cd /home/ngoductam/KLGrade/datasets
tar -czf dataset_knees_cropped.tar.gz dataset_knees_cropped/

# Package splits
cd /home/ngoductam/KLGrade/datasets/splits
tar -czf splits_knees_cropped.tar.gz dataset_knees_cropped_70_20_10/

echo "✅ Dataset packages created"
```

**Option B: Use Existing Dataset** (if already on target server)

Skip this step if dataset is already available on the target server.

### 3. Transfer Files to Target Server

```bash
# Using scp
scp ~/cdt_cad_migration/cdt_cad_code.tar.gz user@target-server:~/
scp ~/KLGrade/datasets/dataset_knees_cropped.tar.gz user@target-server:~/
scp ~/KLGrade/datasets/splits/splits_knees_cropped.tar.gz user@target-server:~/

# Or using rsync (faster for large files)
rsync -avz --progress ~/cdt_cad_migration/cdt_cad_code.tar.gz user@target-server:~/
rsync -avz --progress ~/KLGrade/datasets/ user@target-server:~/datasets/
```

### 4. Setup on Target Server

```bash
# SSH to target server
ssh user@target-server

# Extract code
mkdir -p ~/KLGrade
cd ~/KLGrade
tar -xzf ~/cdt_cad_code.tar.gz

# Extract dataset
mkdir -p datasets
cd datasets
tar -xzf ~/dataset_knees_cropped.tar.gz
tar -xzf ~/splits_knees_cropped.tar.gz

# Create conda environment
conda create -n klgrade python=3.10 -y
conda activate klgrade

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install PyWavelets  # For CDT-CAD

# Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')"
```

### 5. Update Paths in Training Script

```bash
# Edit train_cdt_cad_lesion.py
nano train_cdt_cad_lesion.py

# Update these paths if different:
# - PROJECT_ROOT (if needed)
# - Default paths for --data, --img-dir, --label-dir
```

### 6. Run CDT-CAD Training

```bash
# Activate environment
conda activate klgrade

# Test with small config first
python train_cdt_cad_lesion.py \
    --data dataset_cdt_cad.yaml \
    --img-dir datasets/dataset_knees_cropped/images \
    --label-dir datasets/dataset_knees_cropped/labels \
    --train-split datasets/splits/dataset_knees_cropped_70_20_10/train.txt \
    --val-split datasets/splits/dataset_knees_cropped_70_20_10/val.txt \
    --epochs 5 \
    --batch-size 4 \
    --img-size 640 \
    --device cuda:0

# If successful, run full training
nohup python train_cdt_cad_lesion.py \
    --data dataset_cdt_cad.yaml \
    --img-dir datasets/dataset_knees_cropped/images \
    --label-dir datasets/dataset_knees_cropped/labels \
    --train-split datasets/splits/dataset_knees_cropped_70_20_10/train.txt \
    --val-split datasets/splits/dataset_knees_cropped_70_20_10/val.txt \
    --epochs 100 \
    --batch-size 8 \
    --img-size 640 \
    --device cuda:0 \
    > cdt_cad_training.log 2>&1 &

# Monitor progress
tail -f cdt_cad_training.log
```

### 7. Retrieve Trained Model

```bash
# On target server, package trained model
cd ~/KLGrade
tar -czf cdt_cad_trained_model.tar.gz runs/cdt_cad_lesion/

# Transfer back to original server
scp cdt_cad_trained_model.tar.gz user@rtx-2080-server:~/

# On original server, extract
cd /home/ngoductam/KLGrade
tar -xzf ~/cdt_cad_trained_model.tar.gz
```

---

## Recommended Training Configurations by GPU

### RTX 3090 (24GB)
```bash
--batch-size 8
--img-size 640
--hidden-dim 256
--num-encoder-layers 6
--num-decoder-layers 6
```

### A100 (40GB)
```bash
--batch-size 16
--img-size 800
--hidden-dim 384
--num-encoder-layers 8
--num-decoder-layers 8
```

### V100 (32GB)
```bash
--batch-size 12
--img-size 640
--hidden-dim 256
--num-encoder-layers 6
--num-decoder-layers 6
```

---

## Troubleshooting

### Issue: Still OOM on Larger GPU

**Solution 1**: Enable Mixed Precision
```python
# Add to training script
scaler = torch.cuda.amp.GradScaler()

# In training loop
with torch.cuda.amp.autocast():
    outputs = model(images)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Solution 2**: Reduce Model Size
```bash
--hidden-dim 128
--num-encoder-layers 4
--num-decoder-layers 4
```

**Solution 3**: Gradient Checkpointing
```python
# In model initialization
model.gradient_checkpointing_enable()
```

### Issue: Slow Data Loading

**Solution**: Increase DataLoader workers
```python
# In train_cdt_cad_lesion.py
train_loader = DataLoader(
    ...,
    num_workers=8,  # Increase from 0
    pin_memory=True,
    persistent_workers=True,
)
```

### Issue: Different CUDA Version

```bash
# Check CUDA version on target server
nvidia-smi

# Install matching PyTorch
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Verification Checklist

Before full training, verify:

- [ ] GPU has ≥24GB VRAM
- [ ] CUDA and PyTorch versions match
- [ ] All dependencies installed (`PyWavelets`, `ultralytics`, etc.)
- [ ] Dataset paths are correct
- [ ] Test run (5 epochs) completes successfully
- [ ] WandB login (if using tracking)
- [ ] Sufficient disk space for checkpoints (~5GB per checkpoint)

---

## Expected Training Time

| GPU | Batch Size | Time per Epoch | Total (100 epochs) |
|-----|------------|----------------|-------------------|
| RTX 2080 Ti (10GB) | OOM | N/A | N/A |
| RTX 3090 (24GB) | 8 | ~15 min | ~25 hours |
| A100 (40GB) | 16 | ~8 min | ~13 hours |
| V100 (32GB) | 12 | ~10 min | ~17 hours |

---

## Post-Training: Bring Model Back

### Option 1: Use Model Directly on Target Server

Deploy inference API on the larger GPU server.

### Option 2: Transfer Model Back

```bash
# Package only the best model
tar -czf cdt_cad_best_model.tar.gz runs/cdt_cad_lesion/weights/best.pt

# Transfer
scp cdt_cad_best_model.tar.gz user@rtx-2080-server:~/

# Inference can run on RTX 2080 Ti (only training needs large GPU)
```

### Option 3: Quantize Model for Smaller GPU

```python
# Quantize to FP16 or INT8 for inference
import torch

model = torch.load('best.pt')
model_fp16 = model.half()  # FP16
torch.save(model_fp16, 'best_fp16.pt')
```

---

## Summary

1. **Package** code and dataset
2. **Transfer** to larger GPU server
3. **Setup** environment and dependencies
4. **Train** CDT-CAD with larger batch size
5. **Retrieve** trained model
6. **Integrate** into existing pipeline

**Estimated Total Time**: 2-3 days (including setup and training)

**Alternative**: Consider cloud GPU services (AWS, GCP, Lambda Labs) if no local server with large GPU is available.
