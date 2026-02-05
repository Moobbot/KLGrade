# YOLO Detection Training

This folder contains scripts for training **YOLO models for knee and lesion detection**.

## 📁 Contents

### Training Scripts
- **`train_yolo.py`** - Standard YOLO training
- **`train_yolo_enhanced.py`** - Enhanced YOLO with augmentation
- **`train_knee_detection.py`** - Knee-specific detection training
- **`train_detr.py`** - DETR-based detection (experimental)

## 🚀 Quick Start

### Train Standard YOLO
```bash
python scripts/training/yolo_detection/train_yolo.py \
  --data datasets/dataset_v0/data.yaml \
  --epochs 100 \
  --batch 16 \
  --img 640
```

### Train Enhanced YOLO
```bash
python scripts/training/yolo_detection/train_yolo_enhanced.py \
  --data datasets/dataset_v0/data.yaml \
  --epochs 100 \
  --augment
```

## 📊 Detection Tasks

### Knee Detection
- Detect knee regions in full-leg X-rays
- Used as first step in two-step pipeline
- Typical mAP: 85-95%

### Lesion Detection
- Detect Joint Space (JS) narrowing
- Detect Osteophytes (OST)
- More challenging: smaller objects, subtle features

## 🎯 Use Cases

1. **Two-Step Pipeline**: Train YOLO → Use for KIOCMIL
2. **End-to-End Pipeline**: YOLO backbone shared with KIOCMIL
3. **Standalone Detection**: Pure detection task

## 📈 Typical Results

**Knee Detection:**
- mAP@0.5: **~90%**
- Fast inference: ~10ms per image

**Lesion Detection:**
- JS mAP@0.5: **~70-80%**
- OST mAP@0.5: **~65-75%**
- More challenging due to small size

## 📝 Notes

- For end-to-end training, see `../end_to_end/`
- For KIOCMIL classification, see `../kiocmil_two_step/`
- Detection results are used as input to KIOCMIL models
