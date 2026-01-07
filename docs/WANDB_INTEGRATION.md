# WandB Integration Guide

## Setup (One-time)

### 1. Activate Virtual Environment

```bash
# IMPORTANT: Always activate .venv first!
source .venv/bin/activate
```

### 2. Install WandB

```bash
pip install wandb
```

### 3. Login

```bash
# Method 1: Using environment variable (recommended for servers)
export WANDB_API_KEY="wandb_v1_Hello"
wandb login $WANDB_API_KEY

# Method 2: Interactive login
wandb login
# Then paste your key when prompted
```

### 4. Set Project Name

```bash
export WANDB_PROJECT="KLGrade-Knee-OA"
```

## ✅ Automatic Integration

**Good news**: YOLO v8+ has built-in WandB support! Just login and it will automatically:

- Log all metrics (mAP, loss, precision, recall...)
- Upload training curves
- Save model checkpoints
- Track hyperparameters
- Log validation images with predictions

## Usage

### Option 1: Use the provided script

```bash
chmod +x docs/TRAINING_COMMANDS_WANDB.sh
./docs/TRAINING_COMMANDS_WANDB.sh
```

### Option 2: Manual commands

```bash
# Set API key first
export WANDB_API_KEY="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"
wandb login $WANDB_API_KEY

# Then run any YOLO training command
yolo detect train data=configs/yolo_5_class_baseline.yaml epochs=100 ...
```

## View Results

Access your experiments at:

```
https://wandb.ai/your-username/KLGrade-Knee-OA
```

## What Gets Logged

- **Metrics**: mAP50, mAP50-95, precision, recall, losses
- **Hyperparameters**: learning rate, batch size, augmentation settings
- **System info**: GPU usage, memory, training time
- **Media**: Training/validation images with bounding boxes
- **Model**: Checkpoints automatically saved

## Disable WandB (if needed)

```bash
export WANDB_MODE=disabled
```

## Tips

1. **Compare runs**: Click "Compare" in WandB to see multiple experiments side-by-side
2. **Custom tags**: Add tags in WandB UI to organize experiments
3. **Note taking**: Add notes directly in WandB dashboard
4. **Share results**: Generate shareable report links
