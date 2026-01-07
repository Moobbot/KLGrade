# Experiment Tracking Log

## Overview

This document tracks all training experiments for knee OA detection project.

---

## Experiment Matrix

| Exp ID | Dataset       | Classes | Augmentation | IRFS | Loss | Epochs | Status     | mAP50 | mAP50-95 | Notes             |
| ------ | ------------- | ------- | ------------ | ---- | ---- | ------ | ---------- | ----- | -------- | ----------------- |
| E001   | knee_5_class  | 5       | None         | No   | CE   | 100    | ⏳ Pending | -     | -        | Baseline          |
| E002   | knee_5_class  | 5       | Conservative | No   | CE   | 100    | ⏳ Pending | -     | -        | + Augmentation    |
| E003   | knee_5_class  | 5       | Conservative | Yes  | CE   | 100    | ⏳ Pending | -     | -        | + IRFS            |
| E004   | knee_10_class | 10      | None         | No   | CE   | 100    | ⏳ Pending | -     | -        | Baseline 10-class |
| E005   | knee_4_class  | 4       | Conservative | No   | CE   | 100    | ⏳ Pending | -     | -        | No KL0            |
| E006   | knee_8_class  | 8       | None         | No   | CE   | 100    | ⏳ Pending | -     | -        | No KL0 10-class   |

**Legend:**

- CE = Cross Entropy
- IRFS = Instance-Aware Repeat Factor Sampling
- ⏳ = Pending, 🔄 = Running, ✅ = Complete, ❌ = Failed

---

## Experiment Details

### E001: 5-class Baseline

**Config:** `configs/yolo_5_class_baseline.yaml`

**Purpose:** Establish baseline performance without augmentation

**Hyperparameters:**

- Model: YOLOv11n
- Epochs: 100
- Batch: 16
- LR: 0.001 → 0.00001
- Augmentation: None

**Command:**

```powershell
.venv\Scripts\python.exe scripts\training\train_yolo.py `
    --config configs\yolo_5_class_baseline.yaml
```

**Results:**

- mAP50: TBD
- mAP50-95: TBD
- Per-class AP: TBD
- Training time: TBD

**Analysis:**

- TBD

---

### E002: 5-class + Conservative Augmentation

**Config:** `configs/yolo_5_class_conservative.yaml`

**Purpose:** Test conservative augmentation impact

**Changes from E001:**

- ✅ Rotation: ±5°
- ✅ Translation: ±10%
- ✅ Scale: ±10%
- ✅ H-Flip: 50%
- ✅ Brightness/Contrast: ±20% (via hsv_v)

**Command:**

```powershell
.venv\Scripts\python.exe scripts\training\train_yolo.py `
    --config configs\yolo_5_class_conservative.yaml
```

**Expected:**

- +5-10% mAP improvement
- Better generalization

**Results:**

- TBD

---

### E003: 5-class + Conservative Aug + IRFS

**Purpose:** Address class imbalance with IRFS

**Changes from E002:**

- ✅ IRFS with thresh=0.001
- Expected repeat factors:
  - KL0: ~6.6×
  - KL4: ~3.2×
  - Others: 1-2×

**Status:** ⏳ Pending (IRFS integration)

---

## Dataset Statistics

### 5-class (knee)

- Train: 1,181 images
- Val: 253 images
- Test: 254 images
- Total boxes: 3,141
- Class imbalance: 45:1 (KL2 vs KL0)

### 10-class (knee_10_class)

- Train: 1,181 images
- Val: 253 images
- Test: 254 images
- Total boxes: 3,141
- Severe imbalance in "b" classes

### 4-class (knee_4_class)

- Train: 1,122 images
- Val: 240 images
- Test: 241 images
- Total boxes: 3,042

### 8-class (knee_8_class)

- Train: 1,122 images
- Val: 240 images
- Test: 241 images
- Total boxes: 3,042

---

## Training Commands Cheatsheet

```powershell
# 5-class baseline
.venv\Scripts\python.exe scripts\training\train_yolo.py --config configs\yolo_5_class_baseline.yaml

# 5-class conservative
.venv\Scripts\python.exe scripts\training\train_yolo.py --config configs\yolo_5_class_conservative.yaml

# 10-class baseline
.venv\Scripts\python.exe scripts\training\train_yolo.py --config configs\yolo_10_class_baseline.yaml

# 4-class baseline
.venv\Scripts\python.exe scripts\training\train_yolo.py --config configs\yolo_4_class_baseline.yaml

# 8-class baseline
.venv\Scripts\python.exe scripts\training\train_yolo.py --config configs\yolo_8_class_baseline.yaml

# With conservative augmentation
.venv\Scripts\python.exe scripts\training\train_yolo.py --config configs\yolo_5_class_conservative.yaml
.venv\Scripts\python.exe scripts\training\train_yolo.py --config configs\yolo_10_class_conservative.yaml
.venv\Scripts\python.exe scripts\training\train_yolo.py --config configs\yolo_4_class_conservative.yaml
.venv\Scripts\python.exe scripts\training\train_yolo.py --config configs\yolo_8_class_conservative.yaml

# Resume from checkpoint
.venv\Scripts\python.exe scripts\training\train_yolo.py --config configs\yolo_5_class_baseline.yaml --resume runs/detect/knee_5_class_baseline/weights/last.pt
```

---

## Evaluation Commands

```powershell
# Validate model
yolo detect val model=runs/detect/knee_5_class_baseline/weights/best.pt data=configs/yolo_5_class_baseline.yaml

# Test on test set
yolo detect predict model=runs/detect/knee_5_class_baseline/weights/best.pt source=processed/knee/images conf=0.25
```

---

## Notes

- All experiments use seed=42 for reproducibility
- YOLOv11n chosen for fast iteration (can upgrade to yolo11m/l later)
- Early stopping patience=20 epochs
- Checkpoints saved every 10 epochs

---

**Last Updated:** 2026-01-07  
**Next:** Run E001 (5-class baseline)
