# Augmentation Configuration Tracking

## Overview

This file documents all augmentation parameters used in training experiments for reproducibility and comparison.

---

## Conservative Augmentation (RECOMMENDED)

**File:** `src/datasets/augmentation.py::get_conservative_train_transform()`

**Status:** ✅ Active - Safe for medical X-ray images

### Geometric Transformations

| Transform       | Parameter     | Value | Probability | Notes                         |
| --------------- | ------------- | ----- | ----------- | ----------------------------- |
| Horizontal Flip | -             | -     | 50%         | L/R knee symmetry OK          |
| Rotation        | `limit`       | ±5°   | 50%         | Very conservative for anatomy |
| Shift           | `shift_limit` | ±10%  | 50%         | Patient positioning variation |
| Scale           | `scale_limit` | ±10%  | 50%         | Distance from X-ray source    |

### Intensity Adjustments

| Transform  | Parameter          | Value   | Probability | Notes                           |
| ---------- | ------------------ | ------- | ----------- | ------------------------------- |
| Brightness | `brightness_limit` | ±20%    | 50%         | X-ray exposure variation        |
| Contrast   | `contrast_limit`   | ±20%    | 50%         | Imaging settings variation      |
| Gamma      | `gamma_limit`      | 0.8-1.2 | 30%         | Light contrast curve adjustment |

### Image Quality

| Transform      | Parameter    | Value  | Probability | Notes                   |
| -------------- | ------------ | ------ | ----------- | ----------------------- |
| Gaussian Blur  | `blur_limit` | 3-5 px | 20%         | Motion/focus simulation |
| Gaussian Noise | `var_limit`  | 5-15   | 20%         | Sensor noise            |

### Preprocessing

- **Resize:** 640×640
- **Normalize:** mean=0.5, std=0.5
- **Format:** YOLO (center_x, center_y, width, height)

---

## Moderate Augmentation (EXPERIMENTAL)

**File:** `src/datasets/augmentation.py::get_moderate_train_transform()`

**Status:** ⚠️ Experimental - For comparison only

### Key Differences from Conservative:

- Rotation: ±20° (vs ±5°)
- Shift/Scale: ±15% (vs ±10%)
- Brightness/Contrast: ±30% (vs ±20%)
- Gamma: 0.7-1.3 (vs 0.8-1.2)
- Additional MotionBlur option

**Use case:** Test if heavier augmentation helps with small dataset

---

## Validation Transform

**File:** `src/datasets/augmentation.py::get_val_transform()`

**Transformations:**

- Resize to 640×640
- Normalize (mean=0.5, std=0.5)
- **NO augmentation** (deterministic)

---

## IRFS (Instance-Aware Repeat Factor Sampling)

**File:** `src/datasets/samplers.py::RepeatFactorSampler`

**Parameters:**

| Parameter       | Value | Description              |
| --------------- | ----- | ------------------------ |
| `repeat_thresh` | 0.001 | Target frequency (0.1%)  |
| `shuffle`       | True  | Shuffle repeated indices |

**Formula:** `r_i = max(1, sqrt(0.001 / f_i))`

**Expected Repeat Factors (5-class):**

- KL0 (3.2%): ~6.6×
- KL1 (25.3%): ~2.0×
- KL2 (43.3%): ~1.5×
- KL3 (18.5%): ~2.3×
- KL4 (9.5%): ~3.2×

**Effective Dataset Size:** ~1,181 → ~2,500 samples per epoch

---

## Avoided Augmentations (Medical Image Safety)

❌ **DO NOT USE:**

| Transform             | Reason                             |
| --------------------- | ---------------------------------- |
| Vertical Flip         | Anatomically incorrect orientation |
| Heavy Rotation (>20°) | Clinically unrealistic             |
| Color Jittering       | Grayscale X-rays                   |
| Cutout/Grid Mask      | May obscure lesions                |
| Elastic Deformation   | Distorts anatomy                   |

---

## Experiment Tracking

### Baseline (No Augmentation)

- Date: TBD
- mAP: -
- Per-class AP: -

### Conservative Augmentation

- Date: TBD
- mAP: -
- KL0 AP: -
- Notes: -

### Conservative + IRFS

- Date: TBD
- mAP: -
- KL0 AP: -
- Effective dataset size: ~2,500

### Moderate Augmentation

- Date: TBD
- mAP: -
- Notes: Comparison experiment

---

## Change Log

| Date       | Change              | Reason                                   |
| ---------- | ------------------- | ---------------------------------------- |
| 2026-01-07 | Initial config      | Conservative approach for medical images |
| 2026-01-07 | Rotation: 15° → 5°  | Too aggressive for anatomy               |
| 2026-01-07 | Gamma: Keep 0.8-1.2 | Light variation sufficient               |

---

## References

- Albumentations documentation
- Medical image augmentation best practices
- IRFS paper (Repeat Factor Sampling)

---

**Last Updated:** 2026-01-07  
**Maintained by:** Research Team
