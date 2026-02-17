# Knee Detector Training Scripts

## Available Training Scripts

### 1. YOLO11N Training (Recommended)
**Script**: `scripts/training/train_knee_detector_yolo11n.sh`

**Model**: YOLO11N (Nano - 5.3MB)
- Lightweight and fast
- Best performance on knee detection task
- Recommended for production

**Configuration**:
```bash
Model: yolo11n.pt
Epochs: 100
Batch: 16
Image Size: 640
Early Stopping: 20 epochs patience
```

**Usage**:
```bash
bash scripts/training/train_knee_detector_yolo11n.sh
```

**Output**: `runs/detect/knee_yolo11n_YYYYMMDD_HHMMSS/`

**Expected Results**:
- mAP@50: ~99.4%
- mAP@50-95: ~80.4%
- Training time: ~30-45 minutes (RTX 2080 Ti)

---

### 2. YOLO11L Training
**Script**: `scripts/training/train_knee_detector_yolo11l.sh`

**Model**: YOLO11L (Large - 49MB)
- Larger model capacity
- Slower inference
- Not recommended (YOLO11N performs better)

**Configuration**:
```bash
Model: yolo11l.pt
Epochs: 100
Batch: 16
Image Size: 640
Early Stopping: 20 epochs patience
```

**Usage**:
```bash
bash scripts/training/train_knee_detector_yolo11l.sh
```

**Output**: `runs/detect/knee_yolo11l_YYYYMMDD_HHMMSS/`

**Expected Results**:
- mAP@50: ~99.3%
- mAP@50-95: ~73.1%
- Training time: ~60-90 minutes (RTX 2080 Ti)

---

## Features

### ✅ Timestamp-based Naming
All runs are saved with unique timestamps to prevent overwriting:
```
runs/detect/
├── knee_yolo11n_20260217_134500/
├── knee_yolo11n_20260217_150000/
└── knee_yolo11l_20260217_160000/
```

### ✅ Early Stopping
Both scripts include early stopping with **20 epochs patience**:
- Monitors validation mAP
- Stops if no improvement for 20 consecutive epochs
- Saves best model automatically
- Prevents overfitting

### ✅ Auto Evaluation
After training completes, automatically runs full evaluation:
- Evaluates on train, val, and test splits
- Generates comprehensive metrics
- Saves results to JSON

### ✅ Comprehensive Logging
- Training plots (loss, metrics curves)
- Confusion matrix
- PR curves
- F1 curves
- Validation predictions

---

## Comparison

| Feature | YOLO11N | YOLO11L |
|---------|---------|---------|
| **Model Size** | 5.3MB ✅ | 49MB |
| **mAP@50** | 99.4% ✅ | 99.3% |
| **mAP@50-95** | 80.4% ✅ | 73.1% |
| **Speed** | 1.8ms ✅ | 1.8ms |
| **Training Time** | 30-45min ✅ | 60-90min |
| **Recommended** | ✅ Yes | ❌ No |

**Conclusion**: YOLO11N is superior in all aspects for knee detection.

---

## Training Tips

### Monitor Training
```bash
# Watch training progress
tail -f runs/detect/knee_yolo11n_*/results.csv

# Check GPU usage
watch -n 1 nvidia-smi
```

### Resume Training
If training is interrupted, you can resume:
```bash
yolo detect train \
    model=runs/detect/knee_yolo11n_TIMESTAMP/weights/last.pt \
    resume=true
```

### Adjust Hyperparameters
Edit the script to modify:
- `EPOCHS`: Maximum training epochs
- `BATCH`: Batch size (reduce if OOM)
- `PATIENCE`: Early stopping patience
- `IMGSZ`: Input image size

---

## Output Structure

After training, each run contains:
```
runs/detect/knee_yolo11n_TIMESTAMP/
├── weights/
│   ├── best.pt          # Best model (use this)
│   ├── last.pt          # Last epoch
│   └── epoch*.pt        # Periodic checkpoints
├── results.csv          # Training metrics
├── results.png          # Metrics plots
├── confusion_matrix.png
├── PR_curve.png
├── F1_curve.png
└── args.yaml           # Training configuration
```

---

## Evaluation After Training

Results are automatically saved to:
- `docs/knee_yolo11n_TIMESTAMP_evaluation.json`
- `docs/knee_yolo11l_TIMESTAMP_evaluation.json`

To manually evaluate:
```bash
python scripts/evaluation/evaluate_knee_full.py \
    --model runs/detect/knee_yolo11n_TIMESTAMP/weights/best.pt \
    --output docs/my_evaluation.json
```
