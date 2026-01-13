# KIOCMIL Version Guide

## Version Comparison

| Version | Accuracy | Features | Status | Use Case |
|---------|----------|----------|--------|----------|
| **V1** | 42-48% | Baseline, no augmentation | ✅ **Stable** | **Production/Baseline** |
| **V2** | 19-20% | Geometric + Photometric augmentation | ⚠️ Experimental | Research only |

## Quick Start

### Default (V1 Baseline - Recommended)

```bash
conda activate klgrade
python src/training/train_kiocmil.py --epochs 50 --batch_size 8
```

### V1 Explicit

```bash
python src/training/train_kiocmil_v1.py --epochs 50
```

### V2 Experimental

```bash
python src/training/train_kiocmil_v2.py --use_v2_dataset --augmentation_level strong --epochs 50
```

## Version Details

### V1 - Baseline (42-48% Accuracy)

**Source**: Git commit `4578263`

**Features**:
- No data augmentation
- Simple transforms (resize, normalize, to tensor)
- Proven stable performance
- Fast training

**Files**:
- `src/training/train_kiocmil_v1.py`
- `src/datasets/kiocmil_dataset_v1.py`
- `src/datasets/kiocmil_transforms_v1.py`

**Performance (from evaluation_kiocmil.log)**:
```
Epoch 2: Val Acc = 42.47%
Epoch 4: Val Acc = 45.66%
Epoch 11: Val Acc = 48.86%
Epoch 15: Val Acc = 47.49%
```

### V2 - Experimental (19-20% Accuracy)

**Features**:
- Advanced augmentation pipeline
- Geometric transforms (rotate, flip, shift/scale/rotate)
- Photometric transforms (brightness, contrast, gamma, CLAHE, gaussian blur)
- Correct augmentation flow: geometric → crop → photometric → normalize → tensor

**Files**:
- `src/training/train_kiocmil_v2.py`
- `src/datasets/kiocmil_dataset_v2.py`
- `src/datasets/kiocmil_transforms_v2.py`

**Performance (from recent runs)**:
```
Epoch 1: Val Acc = 20.03%
Epoch 5: Val Acc = 19.08%
```

**⚠️ Known Issues**:
- Significantly lower accuracy than baseline
- May be overfitting or augmentation too aggressive
- Needs hyperparameter tuning

## File Organization

```
src/
├── training/
│   ├── train_kiocmil.py → train_kiocmil_v1.py (symlink)
│   ├── train_kiocmil_v1.py (baseline)  
│   └── train_kiocmil_v2.py (experimental)
└── datasets/
    ├── kiocmil_dataset.py → kiocmil_dataset_v1.py (symlink)
    ├── kiocmil_dataset_v1.py
    ├── kiocmil_dataset_v2.py
    ├── kiocmil_transforms.py → kiocmil_transforms_v1.py (symlink)
    ├── kiocmil_transforms_v1.py
    └── kiocmil_transforms_v2.py
```

## Log Directory Structure

Logs are automatically versioned:
```
log/
├── run_kiocmil_001/  # First run
├── run_kiocmil_002/  # Second run
└── run_kiocmil_003/  #Third run
```

Each directory contains:
- `config.json` - Training configuration
- Log files (if configured)

## Recommendations

### For Production/Baseline
Use V1 - It's proven stable and achieves good accuracy.

### For Research
Use V2 if you want to experiment with augmentation, but expect lower accuracy. Consider:
- Reducing augmentation strength (`--augmentation_level light`)
- Tuning hyperparameters
- Comparing with V1 baseline first

## Troubleshooting

### V1 Issues
- **Low accuracy**: Check data paths, ensure correct splits
- **Crashes**: Reduce batch size

### V2 Issues
- **Low accuracy**: This is expected, try reducing augmentation
- **Import errors**: Ensure `--use_v2_dataset` flag is used

## Rollback

To rollback to a specific version:

```bash
# Check symlinks
ls -la src/training/train_kiocmil.py
ls -la src/datasets/kiocmil_*.py

# Re-create symlinks if needed
cd src/training && ln -sf train_kiocmil_v1.py train_kiocmil.py
cd src/datasets && ln -sf kiocmil_dataset_v1.py kiocmil_dataset.py
cd src/datasets && ln -sf kiocmil_transforms_v1.py kiocmil_transforms.py
```
