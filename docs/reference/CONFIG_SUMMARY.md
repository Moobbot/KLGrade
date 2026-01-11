# Training Configuration Summary

## Complete Config Matrix

### Baseline Configs (No Augmentation)

| Config                        | Dataset       | Classes | Images | Command                |
| ----------------------------- | ------------- | ------- | ------ | ---------------------- |
| `yolo_5_class_baseline.yaml`  | knee          | 5       | 1,688  | [See below](#commands) |
| `yolo_10_class_baseline.yaml` | knee_10_class | 10      | 1,688  | [See below](#commands) |
| `yolo_4_class_baseline.yaml`  | knee_4_class  | 4       | 1,603  | [See below](#commands) |
| `yolo_8_class_baseline.yaml`  | knee_8_class  | 8       | 1,603  | [See below](#commands) |

### Conservative Augmentation Configs

| Config                            | Dataset       | Classes | Images | Command                |
| --------------------------------- | ------------- | ------- | ------ | ---------------------- |
| `yolo_5_class_conservative.yaml`  | knee          | 5       | 1,688  | [See below](#commands) |
| `yolo_10_class_conservative.yaml` | knee_10_class | 10      | 1,688  | [See below](#commands) |
| `yolo_4_class_conservative.yaml`  | knee_4_class  | 4       | 1,603  | [See below](#commands) |
| `yolo_8_class_conservative.yaml`  | knee_8_class  | 8       | 1,603  | [See below](#commands) |

**Total:** 8 configurations (4 baseline + 4 with augmentation)

---

## Augmentation Parameters

**Conservative augmentation:**

- Rotation: ±5°
- Translation: ±10%
- Scale: ±10%
- Horizontal flip: 50%
- HSV value: ±1.5% (brightness/contrast)
- No vertical flip, no mosaic, no mixup

---

## Commands

### Baseline Training

```powershell
# 5-class baseline
.venv\Scripts\python.exe scripts\training\train_yolo.py --config configs\yolo_5_class_baseline.yaml

# 10-class baseline
.venv\Scripts\python.exe scripts\training\train_yolo.py --config configs\yolo_10_class_baseline.yaml

# 4-class baseline
.venv\Scripts\python.exe scripts\training\train_yolo.py --config configs\yolo_4_class_baseline.yaml

# 8-class baseline
.venv\Scripts\python.exe scripts\training\train_yolo.py --config configs\yolo_8_class_baseline.yaml
```

### With Conservative Augmentation

```powershell
# 5-class + augmentation
.venv\Scripts\python.exe scripts\training\train_yolo.py --config configs\yolo_5_class_conservative.yaml

# 10-class + augmentation
.venv\Scripts\python.exe scripts\training\train_yolo.py --config configs\yolo_10_class_conservative.yaml

# 4-class + augmentation
.venv\Scripts\python.exe scripts\training\train_yolo.py --config configs\yolo_4_class_conservative.yaml

# 8-class + augmentation
.venv\Scripts\python.exe scripts\training\train_yolo.py --config configs\yolo_8_class_conservative.yaml
```

---

## Files Created

```
configs/
├── yolo_5_class_baseline.yaml        ✓
├── yolo_5_class_conservative.yaml    ✓
├── yolo_10_class_baseline.yaml       ✓
├── yolo_10_class_conservative.yaml   ✓
├── yolo_4_class_baseline.yaml        ✓
├── yolo_4_class_conservative.yaml    ✓
├── yolo_8_class_baseline.yaml        ✓
└── yolo_8_class_conservative.yaml    ✓
```

All configurations ready for training!
