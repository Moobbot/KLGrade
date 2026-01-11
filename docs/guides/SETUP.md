# Environment Setup & Configuration

This guide covers setting up the KLGrade environment, including Conda virtual environment, dependencies, and GPU configuration.

## 1. Conda Environment

### Quick Setup

```bash
# Create environment with Python 3.10
conda create -n klgrade python=3.10 -y

# Activate environment
conda activate klgrade

# Install dependencies
pip install -r requirements.txt

# Install YOLO
pip install ultralytics

# Install WandB (optional)
pip install wandb
```

### Verification

```bash
# Check Python version
python --version  # Should be 3.10.x

# Check YOLO installation
yolo version
```

### Management Commands

- **Activate**: `conda activate klgrade`
- **Deactivate**: `conda deactivate`
- **List Envs**: `conda env list`
- **Remove**: `conda env remove -n klgrade`

### Export/Import

```bash
# Export
conda env export > environment.yml

# Import
conda env create -f environment.yml
```

---

## 2. GPU Setup & Testing

### Quick Test

1. **Basic GPU Test (PyTorch)**:
   ```bash
   python scripts/test_gpu.py
   ```
   Checks CUDA availability, GPU count, and memory.

2. **YOLO GPU Test**:
   ```bash
   python scripts/test_yolo_gpu.py
   ```
   Runs a 1-epoch training to verify YOLO can access the GPU.

### Expected Output

```
============================================================
GPU Test Report
============================================================
1. CUDA Available: True
2. CUDA Version: 12.1
3. GPU Count: 1
...
============================================================
✅ All GPU tests passed!
============================================================
```

### Troubleshooting GPU Issues

| Error | Solution |
|-------|----------|
| `"CUDA not available"` | Check NVIDIA driver (`nvidia-smi`). Reinstall CUDA-enabled PyTorch. |
| `"Out of memory"` | Reduce batch size in config (`batch: 4`). Clear cache (`torch.cuda.empty_cache()`). |
| `"No CUDA GPUs"` | Check visibility: `echo $CUDA_VISIBLE_DEVICES`. |

### Performance Tips

1. **AMP (Automatic Mixed Precision)**: Enabled by default in YOLO.
2. **Workers**: Set `workers=8` (or 4 on smaller CPUs) for faster data loading.
3. **Batch Size**: Maximize based on VRAM (e.g., 16-32 for 24GB VRAM, 8-12 for 10GB).
