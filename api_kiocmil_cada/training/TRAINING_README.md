# Automated Training Scripts

## Quick Reference

### Train All Datasets (15 variants)
```bash
conda activate klgrade
python api_kiocmil_cada/training/train_all_datasets.py
```

### Train by Category
```bash
# Cropped knee datasets only (8 variants)
python api_kiocmil_cada/training/train_all_datasets.py --category cropped

# Full X-ray datasets only (6 variants)
python api_kiocmil_cada/training/train_all_datasets.py --category full_xray

# Knee detection only (1 variant)
python api_kiocmil_cada/training/train_all_datasets.py --category detection
```

### Custom Training Parameters
```bash
# 50 epochs, batch size 8
python api_kiocmil_cada/training/train_all_datasets.py --epochs 50 --batch 8
```

## Dataset Variants

### Cropped Knee Datasets (8 variants)
**Base (unbalanced):**
- 5-class (KL0-KL4)
- 4-class (KL1-KL4)
- 8-class (KL1-a to KL4-b)
- 10-class (KL0-a to KL4-b)

**Balanced:**
- 5-class (KL0-KL4)
- 4-class (KL1-KL4)
- 8-class (KL1-a to KL4-b)
- 10-class (KL0-a to KL4-b)

### Full X-ray Datasets (6 variants)
**Base (unbalanced):**
- 4-class (KL1-KL4)
- 8-class (KL1-a to KL4-b)
- 10-class (KL0-a to KL4-b)

**Balanced:**
- 4-class (KL1-KL4)
- 8-class (KL1-a to KL4-b)
- 10-class (KL0-a to KL4-b)

### Detection (1 variant)
- Knee detection (single class)

## Training Configuration

- **Model**: YOLO11l (pretrained)
- **Epochs**: 100 (default, configurable)
- **Batch size**: 16 (default, configurable)
- **Early stopping**: patience=20
- **Augmentation**: Auto (mosaic, mixup, copy-paste)

## Output Structure

Results saved to:
```
runs/detect/
├── lesion_5class_base/
├── lesion_4class_base/
├── lesion_8class_base/
├── lesion_10class_base/
├── lesion_5class_balanced/
├── lesion_4class_balanced/
├── lesion_8class_balanced/
├── lesion_10class_balanced/
├── lesion_full_4class_base/
├── lesion_full_8class_base/
├── lesion_full_10class_base/
├── lesion_full_4class_balanced/
├── lesion_full_8class_balanced/
├── lesion_full_10class_balanced/
└── knee_detector/
```

Each directory contains:
- `weights/best.pt` - Best model
- `weights/last.pt` - Last epoch
- Training curves and metrics

## Estimated Time

- **Single dataset**: ~2-4 hours (100 epochs)
- **All 15 datasets**: ~30-60 hours total
- **Cropped only (8)**: ~16-32 hours
- **Full X-ray only (6)**: ~12-24 hours

## Notes

- Training runs sequentially (one at a time)
- Early stopping may reduce actual epochs
- Failed trainings are logged but don't stop the pipeline
- Progress is printed after each dataset
