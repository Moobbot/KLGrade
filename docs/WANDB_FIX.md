# WandB Integration Fix

## Vấn đề
Khi chạy test thì WandB hoạt động, nhưng khi chạy pipeline thì không thấy data lên WandB.

## Nguyên nhân
1. **YOLO CLI không tự động bật WandB**: Khi sử dụng `yolo detect train` (Ultralytics CLI), WandB integration bị tắt mặc định trong settings
2. **Thiếu bước enable trong script**: Các training script chưa có lệnh `yolo settings wandb=True`

## Giải pháp đã áp dụng

### 1. Kiểm tra YOLO Settings
```bash
# Kiểm tra settings hiện tại
yolo settings

# Output sẽ hiển thị:
{
  "wandb": false,  # ← VẤN ĐỀ Ở ĐÂY
  ...
}
```

### 2. Bật WandB trong YOLO Settings
```bash
# Bật WandB
yolo settings wandb=True

# Verify
yolo settings  # wandb should be true now
```

### 3. Cập nhật Training Scripts

Đã thêm dòng sau vào tất cả training scripts:
- `TRAINING_COMMANDS_YOLO_WANDB.sh`
- `TRAINING_COMMANDS_YOLO_WANDB_Data0.sh`

```bash
# Enable WandB in Ultralytics YOLO settings
echo "Enabling WandB integration in Ultralytics YOLO..."
yolo settings wandb=True
```

### 4. Thêm Parameters cho YOLO Commands

Đã thêm các parameters sau vào tất cả `yolo detect train` commands:
```bash
yolo detect train \
  data=configs/yolo_5_class_baseline.yaml \
  ... \
  patience=50 \        # ← MỚI: Early stopping patience
  save_period=10 \     # ← MỚI: Save checkpoint mỗi 10 epochs
  plots=True           # ← MỚI: Tạo training plots
```

## Cách sử dụng

### Nếu chưa bật WandB (chỉ cần làm 1 lần)
```bash
# Activate conda environment
conda activate klgrade

# Bật WandB trong YOLO
yolo settings wandb=True

# Verify
yolo settings
```

### Chạy Training với WandB
```bash
# Chạy toàn bộ pipeline
bash docs/TRAINING_COMMANDS_YOLO_WANDB.sh

# Hoặc chạy single experiment
yolo detect train \
  data=configs/yolo_5_class_baseline.yaml \
  epochs=100 \
  batch=16 \
  device=0 \
  project=runs/detect \
  name=E001_5class_baseline \
  patience=50 \
  save_period=10 \
  plots=True
```

### Kiểm tra WandB Dashboard
Sau khi training bắt đầu, kiểm tra:
- **WandB Dashboard**: https://wandb.ai/ngotam2k1-thuyloi-university/KLGrade-Knee-OA
- **Local logs**: Terminal output sẽ hiển thị WandB URL

## Verification

### Test nhỏ để verify
```bash
# Activate environment
conda activate klgrade

# Setup WandB
export WANDB_API_KEY="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"
export WANDB_PROJECT="KLGrade-Knee-OA"
wandb login $WANDB_API_KEY

# Enable WandB
yolo settings wandb=True

# Run quick test (2 epochs)
yolo detect train \
  data=configs/yolo_5_class_baseline.yaml \
  epochs=2 \
  batch=4 \
  device=0 \
  project=runs/detect \
  name=wandb_test \
  plots=True
```

## Sự khác biệt giữa Python Script vs CLI

### Python Script (`train_yolo.py`)
- ✅ Có `wandb.init()` và `wandb.finish()` trong code
- ✅ Tự động log metrics lên WandB
- Sử dụng: `python scripts/training/train_yolo.py --args`

### YOLO CLI (`yolo detect train`)
- ⚠️ Cần bật WandB trong settings: `yolo settings wandb=True`
- ⚠️ Cần có `WANDB_API_KEY` và `WANDB_PROJECT` environment variables
- ⚠️ Cần login: `wandb login`
- Sử dụng: `yolo detect train data=config.yaml ...`

## Troubleshooting

### Vấn đề: WandB vẫn không hoạt động
```bash
# 1. Kiểm tra wandb đã installed chưa
pip list | grep wandb

# 2. Kiểm tra đã login chưa
wandb login --verify

# 3. Kiểm tra YOLO settings
yolo settings

# 4. Reset settings nếu cần
yolo settings reset
yolo settings wandb=True
```

### Vấn đề: "wandb: ERROR Unable to fetch run"
```bash
# Verify API key
echo $WANDB_API_KEY

# Re-login
wandb login $WANDB_API_KEY --relogin
```

### Vấn đề: Training chạy nhưng không thấy trong WandB dashboard
- Kiểm tra project name: `echo $WANDB_PROJECT`
- Kiểm tra entity/username trên WandB dashboard
- Kiểm tra terminal output có dòng "View run at https://wandb.ai/..."

## Kết luận

✅ **Đã fix**: Tất cả training scripts đã được cập nhật để tự động bật WandB
✅ **Verified**: WandB settings đã được set to `True`
✅ **Ready**: Có thể chạy toàn bộ pipeline với WandB tracking

Giờ đây khi chạy `bash docs/TRAINING_COMMANDS_YOLO_WANDB.sh`, tất cả experiments sẽ tự động được log lên WandB dashboard.
