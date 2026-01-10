# Training on Dataset V0 (Original Dataset)

## Tổng quan

Dataset V0 là dataset gốc chưa qua xử lý (cropping, resizing), giữ nguyên ảnh X-ray toàn bộ cơ thể.

**Đường dẫn:**
- Images: `dataset/dataset_v0/images/`
- Labels: `dataset/dataset_v0/labels/`

**Thống kê:**
- Tổng số ảnh: **1473**
- Số lớp: **5** (KL0, KL1, KL2, KL3, KL4)
- Splits: 70/15/15 (train/val/test)

**Phân bố classes:**
```
KL0:  93 images (6.3%)
KL1: 572 images (38.8%)
KL2: 771 images (52.3%) ← Đa số
KL3: 342 images (23.2%)
KL4: 207 images (14.0%)
```

## Configs đã tạo

### 1. Baseline (No Augmentation)
**File:** `configs/yolo_dataset_v0_baseline.yaml`

Không có augmentation để đánh giá performance baseline.

### 2. Conservative Augmentation
**File:** `configs/yolo_dataset_v0_conservative.yaml`

Augmentation thích hợp cho ảnh y khoa:
- Flip left-right: 50% (knees symmetric)
- Rotation: ±3°
- Scale: ±30%
- Translation: 5%
- HSV minimal adjustments
- NO mosaic, mixup, perspective

## Training Scripts

### Quick Test (2 epochs)
```bash
# Activate environment
conda activate klgrade

# Test baseline
yolo detect train \
  data=configs/yolo_dataset_v0_baseline.yaml \
  epochs=2 \
  batch=4 \
  device=0 \
  project=runs/detect \
  name=test_dataset_v0
```

### Full Training Pipeline
```bash
# Run both experiments
bash docs/TRAINING_COMMANDS_YOLO_DATASET_V0.sh

# Or run in background with logging
nohup bash docs/TRAINING_COMMANDS_YOLO_DATASET_V0.sh > training_dataset_v0.log 2>&1 &
```

### Monitor Progress
```bash
# Local logs
tail -f training_dataset_v0.log

# GPU usage
watch -n 1 nvidia-smi

# WandB dashboard
# https://wandb.ai/ngotam2k1-thuyloi-university/KLGrade-Knee-OA
```

## Experiments

### E_V0_001: Baseline
- Config: `yolo_dataset_v0_baseline.yaml`
- Augmentation: None
- Purpose: Establish baseline performance
- Output: `runs/detect/E_V0_001_baseline/`

### E_V0_002: Conservative
- Config: `yolo_dataset_v0_conservative.yaml`
- Augmentation: Conservative (medical-friendly)
- Purpose: Test if augmentation improves generalization
- Output: `runs/detect/E_V0_002_conservative/`

## So sánh với Processed Datasets

| Dataset | Images | Path | Preprocessing |
|---------|--------|------|---------------|
| **dataset_v0** | 1473 | `dataset/dataset_v0/` | None (raw X-rays) |
| knee_5_class | ~1688 | `processed/knee/` | Cropped knees, 640x640 |
| knee_10_class | ~1688 | `processed/knee_10_class/` | 10 classes (split by shape) |
| knee_4_class | ~1400 | `processed/knee_4_class/` | Filtered KL0 |
| knee_8_class | ~1400 | `processed/knee_8_class/` | 10-class without KL0 |

## Kết quả mong đợi

### Dataset V0 (Full X-ray)
**Ưu điểm:**
- Nhiều context (toàn bộ X-ray)
- Natural distribution

**Nhược điểm:**
- Bounding boxes nhỏ hơn trong ảnh lớn
- Có thể khó detect do nhiều noise

### Processed Datasets (Cropped Knees)
**Ưu điểm:**
- Focus vào vùng quan tâm
- Bounding boxes lớn hơn trong ảnh
- Có thể dễ detect hơn

**Nhược điểm:**
- Mất context xung quanh
- Tăng số lượng ảnh do duplicate (left/right knee)

## Next Steps

1. **Chạy training trên dataset_v0:**
   ```bash
   bash docs/TRAINING_COMMANDS_YOLO_DATASET_V0.sh
   ```

2. **So sánh kết quả:**
   - Compare mAP50, mAP50-95
   - Compare với processed datasets (knee_5_class)
   - Xem dataset nào cho kết quả tốt hơn

3. **Nếu dataset_v0 tốt hơn:**
   - Sử dụng dataset_v0 cho final model
   - Có thể thử thêm augmentation strategies

4. **Nếu processed datasets tốt hơn:**
   - Tiếp tục với pipeline hiện tại
   - Focus vào class imbalance và augmentation

## WandB Integration

WandB đã được enable trong script:
```bash
# Setup
export WANDB_API_KEY="..."
export WANDB_PROJECT="KLGrade-Knee-OA"
wandb login $WANDB_API_KEY
yolo settings wandb=True

# Metrics được log tự động:
# - box_loss, cls_loss, dfl_loss
# - mAP50, mAP50-95
# - Precision, Recall
# - Learning rate, GPU memory
```

## Troubleshooting

### Issue: "No labels found"
```bash
# Verify labels exist
ls dataset/dataset_v0/labels/ | head

# Check cache
rm dataset/dataset_v0/labels.cache
```

### Issue: Training slow
```bash
# Reduce batch size
yolo detect train data=configs/yolo_dataset_v0_baseline.yaml batch=8 ...

# Use smaller model
# Change model: yolo11n.pt → yolo11s.pt in config
```

### Issue: OOM (Out of Memory)
```bash
# Reduce image size
# Change imgsz: 640 → 512 in config

# Or reduce batch size
batch=4 # or even batch=2
```

## Summary

✅ **Dataset V0 setup complete:**
- Splits created: `splits/dataset_v0/`
- Configs created: `configs/yolo_dataset_v0_*.yaml`
- Training script ready: `docs/TRAINING_COMMANDS_YOLO_DATASET_V0.sh`
- WandB integration enabled

✅ **Ready to train!**
```bash
bash docs/TRAINING_COMMANDS_YOLO_DATASET_V0.sh
```
