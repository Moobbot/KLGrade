# YOLO Training Output Structure

Mỗi lần training, YOLO tự động lưu đầy đủ thông tin vào thư mục `runs/detect/<experiment_name>/`:

## 📁 Cấu trúc thư mục output:

```
runs/detect/E001_5class_baseline/
├── args.yaml                      # Toàn bộ config & hyperparameters
├── results.csv                    # Metrics từng epoch (mAP, loss, precision, recall...)
├── results.png                    # Đồ thị training curves
├── confusion_matrix.png           # Ma trận nhầm lẫn
├── confusion_matrix_normalized.png
├── F1_curve.png                   # F1 score curves
├── P_curve.png                    # Precision curves
├── R_curve.png                    # Recall curves
├── PR_curve.png                   # Precision-Recall curve
├── labels.jpg                     # Phân bố labels trong dataset
├── labels_correlogram.jpg         # Tương quan giữa các labels
├── train_batch*.jpg               # Sample training images với predictions
├── val_batch*_labels.jpg          # Validation images với ground truth
├── val_batch*_pred.jpg            # Validation images với predictions
└── weights/
    ├── best.pt                    # Best checkpoint (theo mAP)
    └── last.pt                    # Last epoch checkpoint
```

## 📊 File quan trọng nhất:

### 1. **args.yaml** - Toàn bộ config

- Dataset path
- Hyperparameters (lr, batch, epochs...)
- Augmentation settings
- Model architecture
- Device info

### 2. **results.csv** - Metrics chi tiết

Columns:

- epoch
- train/box_loss, train/cls_loss, train/dfl_loss
- metrics/precision(B), metrics/recall(B)
- metrics/mAP50(B), metrics/mAP50-95(B)
- val/box_loss, val/cls_loss, val/dfl_loss
- lr/pg0, lr/pg1, lr/pg2

### 3. **weights/best.pt** - Model tốt nhất

- Chứa toàn bộ model state
- Có thể load lại để inference hoặc fine-tune

## 🔍 Xem lại kết quả sau này:

```python
import pandas as pd
import yaml

# Đọc config
with open('runs/detect/E001_5class_baseline/args.yaml') as f:
    config = yaml.safe_load(f)
print(config)

# Đọc metrics
df = pd.read_csv('runs/detect/E001_5class_baseline/results.csv')
print(df[['epoch', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']])

# Load model
from ultralytics import YOLO
model = YOLO('runs/detect/E001_5class_baseline/weights/best.pt')
```

## ✅ Guaranteed Logging

Tất cả thông tin này được YOLO tự động lưu, không cần config thêm!
