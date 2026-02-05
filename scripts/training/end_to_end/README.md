# End-to-End Model Training

This folder contains all scripts for training and evaluating the **end-to-end KIOCMIL model** that combines detection and classification in a single pipeline.

## 📁 Contents

### Training Scripts
- **`train_end_to_end.py`** - Main training script
  - Trains YOLO detection heads + KIOCMIL classification jointly
  - Supports shared backbone architecture
  - Memory-optimized (256×256 images, yolo11s backbone)

- **`train_end_to_end_test.sh`** - Quick 5-epoch test run
  - Validates setup before full training
  - Uses optimized memory settings

### Inference & Visualization
- **`test_end_to_end_inference.py`** - Test inference on single image
- **`visualize_end_to_end_single.py`** - Process single image visualization
- **`visualize_end_to_end_optimized.sh`** ✅ **RECOMMENDED** - Batch visualization (256×256, 100% success)
- **`visualize_end_to_end_batch.sh`** - Higher res visualization (384×384, may OOM)
- **`visualize_end_to_end_cpu.sh`** - CPU fallback (slow but reliable)

### Documentation
- **`VISUALIZATION_README.md`** - Complete visualization guide
- **`MEMORY_OPTIMIZATION_SUMMARY.md`** - Memory optimization details

## 🚀 Quick Start

### Training (5 epochs test)
```bash
cd /path/to/KLGrade
./scripts/training/end_to_end/train_end_to_end_test.sh
```

### Visualization (Recommended)
```bash
cd /path/to/KLGrade
./scripts/training/end_to_end/visualize_end_to_end_optimized.sh
```

### Custom Training
```bash
python scripts/training/end_to_end/train_end_to_end.py \
  --backbone yolo11s \
  --num-classes 10 \
  --epochs 50 \
  --batch-size 1 \
  --lr 0.001 \
  --save-dir runs/end_to_end/my_experiment
```

## 📊 Model Architecture

```
Input Image (256×256)
    ↓
YOLO11s Backbone (shared)
    ↓
├─→ Knee Detection Head → Knee Boxes
├─→ Lesion Detection Head → JS/OST Boxes
└─→ KIOCMIL-CADA → KL Grade (10 classes)
```

## ⚙️ Memory Optimization

**Optimized Settings:**
- Image size: **256×256** (~50% memory vs 384×384)
- Backbone: **yolo11s** (~60% memory vs yolo11l)
- Batch size: **1** (stable training)
- **Total GPU memory**: ~5-8 GiB (vs ~22 GiB original)

## 📈 Training Results

**5-Epoch Test Run:**
- Best Val Accuracy: **35.85%**
- Best Val Loss: **1.6001**
- Training stable, no OOM errors ✅

## 🎯 Next Steps

1. Load real images (replace dummy data)
2. Use pretrained KIOCMIL weights
3. Implement gradient accumulation
4. Full 50-epoch training run
5. Upgrade to yolo11l if GPU allows

## 📝 Notes

- This is an **end-to-end** approach (single model)
- Alternative: Two-step approach (see `../kiocmil_two_step/`)
- For YOLO-only detection, see `../yolo_detection/`
