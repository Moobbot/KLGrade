# WandB Tracking Guide

This guide explains how to use Weights & Biases (WandB) for experiment tracking in the KLGrade project.

## 🚀 Quick Start

1. **Verify Credentials**:
   Ensure `.wandb.env` exists with your API key:
   ```bash
   cat .wandb.env
   # Should show: export WANDB_API_KEY=...
   ```

2. **Run Training**:
   Run any training script (e.g., `bash docs/scripts/TRAINING_COMMANDS_YOLO_WANDB.sh`). WandB is enabled by default in our scripts.

3. **View Dashboard**:
   [KLGrade Knee OA Dashboard](https://wandb.ai/ngotam2k1-thuyloi-university/KLGrade-Knee-OA)

---

## ⚙️ Configuration Details

### How it Automation Works
Our scripts automatically handle WandB setup:
- **Load Env**: Source `.wandb.env` to set `WANDB_API_KEY` and `WANDB_PROJECT`.
- **Login**: `wandb login $WANDB_API_KEY`.
- **YOLO Setting**: `yolo settings wandb=True` (Critical for CLI usage).

### Manual Setup (One-time)
If running manually without our scripts:

```bash
# 1. Install
pip install wandb

# 2. Login
wandb login

# 3. Enable for YOLO
yolo settings wandb=True
```

---

## 🛠️ Troubleshooting

### Issue: Logging not appearing
- **Check YOLO Settings**: Run `yolo settings` and ensure `"wandb": true`.
- **Check Project Name**: Verify `WANDB_PROJECT` matches your dashboard project.
- **Offline Mode**: If internet is flaky, use `wandb offline` and sync later with `wandb sync runs/detect/exp_name`.

### Issue: "WANDB_API_KEY not set"
- Ensure `.wandb.env` file is present in the root directory.
- Run `source .wandb.env` manually before training.

### Issue: "Unable to fetch run"
- Verify your API key is correct.
- Re-login: `wandb login --relogin`.

---

## 📊 Features
- **Metrics**: mAP, Precision, Recall, Box Loss, CLS Loss.
- **System**: GPU usage, Memory, Temperature.
- **Artifacts**: Model checkpoints (`best.pt`), confusion matrices, validation batches.
