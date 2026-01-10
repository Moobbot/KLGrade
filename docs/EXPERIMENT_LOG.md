# Experiment Tracking Log

## Overview

This document tracks all training experiments for knee OA detection project.

---

## Experiment Matrix

| Exp ID | Model | Split | Enhancement | mAP50 | mAP50-95 | Date | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLO Runs (Enhanced Pipeline)** | | | | | | | |
| `004_knee10_full` | YOLOv11l | 10-Class | CLAHE+Bal | **0.259** | 0.092 | Jan 10 | ✅ Best 10-cls |
| `003_knee5_full` | YOLOv11l | 5-Class | CLAHE+Bal | **0.250** | 0.096 | Jan 10 | ✅ Good |
| `001_v0_full` | YOLOv11l | Dataset V0 | CLAHE+Bal | 0.182 | 0.065 | Jan 10 | - |
| `002_v0_clahe` | YOLOv11l | Dataset V0 | CLAHE Only | 0.182 | 0.065 | Jan 10 | - |
| **YOLO Runs (Standard)** | | | | | | | |
| `E005_4class` | YOLOv11n | 4-Class | Baseline | **0.289** | 0.104 | Jan 09 | ✅ Best Overall |
| `E002_5class` | YOLOv11n | 5-Class | Conservative | **0.269** | 0.105 | Jan 09 | ✅ Best 5-cls |
| `E006_8class` | YOLOv11n | 8-Class | Baseline | 0.258 | 0.104 | Jan 09 | - |
| `E004_10class` | YOLOv11n | 10-Class | Baseline | 0.206 | 0.086 | Jan 09 | - |
| **DETR Runs** | | | | | | | |
| `E012_4class` | DETR-R50 | 4-Class | Baseline | 0.0004 | 0.0001 | Jan 08 | ❌ Failed |
| `E010_5class` | DETR-R50 | 5-Class | Baseline | 0.0003 | 0.0001 | Jan 08 | ❌ Failed |
| `E011_10class` | DETR-R50 | 10-Class | Baseline | 0.0001 | 0.0000 | Jan 08 | ❌ Failed |

---

## Detailed Results (By Date)

### 📅 2026-01-10: Enhanced YOLO Training
**Focus**: `yolo11l` (large) model with CLAHE preprocessing and undersampling/balancing.

- **`yolo11l_E_ENHANCED_004_knee10_full`**
  - **Conf**: 10 Classes, CLAHE, Balanced
  - **mAP50**: 0.259 | **mAP50-95**: 0.092
  - **Note**: Performs surprisingly well for 10 classes, close to 5-class performance.

- **`yolo11l_E_ENHANCED_003_knee5_full`**
  - **Conf**: 5 Classes (Standard), CLAHE, Balanced
  - **mAP50**: 0.250 | **mAP50-95**: 0.096
  - **Note**: Solid performance, but slightly lower than the `yolo11n` conservative run from Jan 09 (0.269), possibly due to over-processing or model complexity differences?

- **`yolo11l_E_ENHANCED_001_v0_full`**
  - **Conf**: Dataset V0 (Original), CLAHE, Balanced
  - **mAP50**: 0.182 | **mAP50-95**: 0.065
  - **Note**: Significantly lower than `processed/knee` runs. Confirms V0 dataset quality/split issues resolved in `processed/knee`.

### 📅 2026-01-09: YOLO Baseline & Conservative
**Focus**: `yolo11n` (nano) model with standard augmentation.

- **`E005_4_class_baseline`** (4-Class)
  - **mAP50**: **0.289** (Highest recorded)
  - **Note**: Removing KL0 (healthy) and grouping into 4 classes yields the best detection results.

- **`E002_5class_conservative`** (5-Class)
  - **mAP50**: 0.269
  - **Note**: Best 5-class result. Suggests `yolo11n` + Conservative Aug is a very strong baseline, potentially outperforming complex enhancements on `yolo11l` for this dataset size.

- **`E006_8class_baseline`** (8-Class)
  - **mAP50**: 0.258
  - **Note**: removing KL0 allows for reasonable breakdown even with 8 classes.

### 📅 2026-01-08: DETR Baseline
**Focus**: DETR ResNet-50.

- **All Experiments (`E010`-`E013`)**
  - **mAP50**: ~0.000
  - **Status**: Failed to converge.
  - **Reason**: 50 epochs is insufficient for DETR (requires 300+). Dataset size likely too small for Transformer without massive pre-training/augmentation.

---

## Conclusions & Recommendations

1.  **Model Selection**: YOLOv11 outperforms DETR by a massive margin (25-30% mAP vs 0%). Stick with YOLO.
2.  **Dataset**: `processed/knee` (and its 4/8/10 splits) is superior to `dataset v0`.
3.  **Enhancements**:
    - **Class count**: 4-class split (No KL0) gives best raw metrics (0.289 mAP).
    - **Preprocessor**: `yolo11n` (Conservative Aug) achieved 0.269 vs `yolo11l` (CLAHE) 0.250.
    - **Recommendation**: Prioritize the **4-class** or **5-class** split. Consider revisiting `yolo11n` with tuned augmentation rather than just `yolo11l` with CLAHE, or combine them.
