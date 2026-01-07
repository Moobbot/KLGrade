# Quick Start Guide - Linux Server

## 🚀 Chạy Pipeline Đầy Đủ (Từ Dataset Gốc)

### 1. Upload code lên server

```bash
# Trên máy local (Windows)
scp -r E:/CaoHoc/thesis/KLGrade your-user@server:/path/to/

# Hoặc dùng git
git clone your-repo-url
cd KLGrade
```

### 2. Setup conda environment

```bash
# Tạo environment
conda create -n klgrade python=3.10 -y
conda activate klgrade

# Install all dependencies (includes ultralytics, wandb, etc.)
pip install -r requirements.txt
```

### 3. Upload dataset

```bash
# Dataset phải có cấu trúc:
# dataset/dataset_v0/images/
# dataset/dataset_v0/labels/
# dataset/dataset_v0/labels-knee/
```

### 4. Chạy pipeline

```bash
# Làm file executable
chmod +x scripts/run_full_pipeline.sh

# Chạy!
./scripts/run_full_pipeline.sh
```

**Thời gian**: ~10-20 phút

**Output**:

- `processed/knee/` - Knee crops
- `processed/knee_10_class/` - 10-class dataset
- `processed/knee_4_class/` - 4-class dataset
- `processed/knee_8_class/` - 8-class dataset
- `splits/` - Train/val/test splits
- `configs/` - YOLO configs (already exist)

## 🎯 Sau khi pipeline xong

### Test GPU

```bash
python scripts/test_yolo_gpu.py
```

### Start training

```bash
# Làm file executable
chmod +x docs/TRAINING_COMMANDS_WANDB.sh

# Chạy training (8 experiments)
./docs/TRAINING_COMMANDS_WANDB.sh
```

Hoặc chạy từng experiment riêng:

```bash
yolo detect train \
  data=configs/yolo_5_class_baseline.yaml \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  device=0
```

## 📊 Monitor Training

```bash
# GPU usage
watch -n 1 nvidia-smi

# WandB (if enabled)
# Visit: https://wandb.ai
```

## ❓ Troubleshooting

### Lỗi: "Permission denied"

```bash
chmod +x scripts/run_full_pipeline.sh
chmod +x docs/TRAINING_COMMANDS_WANDB.sh
```

### Lỗi: "conda: command not found"

```bash
# Initialize conda
conda init bash
# Restart terminal
```

### Lỗi: "CUDA not available"

```bash
# Check GPU
nvidia-smi

# Reinstall PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 📁 File Structure After Pipeline

```
KLGrade/
├── dataset/
│   └── dataset_v0/          # Raw data (input)
├── processed/
│   ├── knee/                # 5-class crops
│   ├── knee_10_class/       # 10-class dataset
│   ├── knee_4_class/        # 4-class (no KL0)
│   └── knee_8_class/        # 8-class (10-class no KL0)
├── splits/                  # Train/val/test splits
│   ├── knee_5_class/
│   ├── knee_10_class/
│   ├── knee_4_class/
│   └── knee_8_class/
├── configs/                 # YOLO config files
└── runs/                    # Training results (created during training)
```

## 🎯 Summary

**One-time setup:**

1. `conda create -n klgrade python=3.10 -y`
2. `conda activate klgrade`
3. `pip install -r requirements.txt`

**Each time:**

1. `conda activate klgrade`
2. `./scripts/run_full_pipeline.sh` (if dataset changed)
3. `./docs/TRAINING_COMMANDS_WANDB.sh` (start training)

That's it! 🚀
