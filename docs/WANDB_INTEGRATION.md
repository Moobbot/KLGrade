# WandB Integration Guide

## 🚀 Quick Start

```bash
# 1. Verify credentials
cat .wandb.env

# 2. Run training
bash docs/TRAINING_COMMANDS_YOLO_WANDB.sh

# 3. View dashboard
# https://wandb.ai/ngotam2481-thuyloi-university/KLGrade-Knee-OA
```

## Available Scripts

```bash
# YOLO
bash docs/TRAINING_COMMANDS_YOLO_WANDB.sh
bash docs/TRAINING_ENHANCED.sh

# DETR
bash docs/TRAINING_COMMANDS_DETR_WANDB.sh
bash docs/EVALUATION_COMMANDS_DETR.sh
```

**Tự động:**
- ✅ Load `.wandb.env`
- ✅ Login WandB
- ✅ Log metrics, images, models

## Troubleshooting

### ".wandb.env not found"
```bash
ls -la .wandb.env  # Check exists
```

### "WANDB_API_KEY not set"
```bash
cat .wandb.env  # Verify WANDB_API_KEY=wandb_v1_xxx
```

### Dashboard empty
```bash
yolo settings wandb=True  # Enable WandB for YOLO
```

## Links

- [WANDB_FIX.md](WANDB_FIX.md) - Detailed info
- [WandB Docs](https://docs.wandb.ai/)
