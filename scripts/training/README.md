# KIOCMIL CADA Training Scripts

**Location**: `scripts/training/`  
**Purpose**: Train KIOCMIL CADA models across all configurations

---

## 📂 Available Scripts

### Master Script
- **`run_all_cada_experiments.sh`** - Run ALL 20 experiments (complete training suite)

### Base Training (Unbalanced + Balanced, No Preprocessing)
- **`run_train_4_5_class.sh`** - Train 4/5-class models (4 experiments)
- **`run_train_8_10_class.sh`** - Train 8/10-class models (4 experiments)

### Preprocessed Training (Resize, Blur, Sharp)
- **`run_processed_4_8_class.sh`** - Train preprocessed 4/8-class models (6 experiments)
- **`run_processed_5_10_class.sh`** - Train preprocessed 5/10-class models (6 experiments)

### Evaluation
- **`evaluate_all_cada_experiments.sh`** - Evaluate all trained models

---

## 🎯 Quick Start

### Train Specific Configuration

```bash
cd /home/ngoductam/KLGrade

# Train 4/5-class base models
bash scripts/training/run_train_4_5_class.sh

# Train 8/10-class base models
bash scripts/training/run_train_8_10_class.sh

# Train preprocessed 4/8-class models
bash scripts/training/run_processed_4_8_class.sh

# Train preprocessed 5/10-class models
bash scripts/training/run_processed_5_10_class.sh
```

### Train Everything

```bash
# Run all 20 experiments
bash scripts/training/run_all_cada_experiments.sh
```

---

## 📊 Experiment Coverage

### `run_train_4_5_class.sh` (4 experiments)
- cada_4class_unbalanced
- cada_4class_balanced
- cada_5class_unbalanced
- cada_5class_balanced

### `run_train_8_10_class.sh` (4 experiments)
- cada_8class_unbalanced
- cada_8class_balanced  
- cada_10class_unbalanced
- cada_10class_balanced

### `run_processed_4_8_class.sh` (6 experiments)
- cada_4class_balanced_resize
- cada_4class_balanced_blur
- cada_4class_balanced_sharp
- cada_8class_balanced_resize
- cada_8class_balanced_blur
- cada_8class_balanced_sharp

### `run_processed_5_10_class.sh` (6 experiments)
- cada_5class_balanced_resize
- cada_5class_balanced_blur
- cada_5class_balanced_sharp
- cada_10class_balanced_resize
- cada_10class_balanced_blur
- cada_10class_balanced_sharp

### `run_all_cada_experiments.sh` (ALL 20 experiments)
Runs all of the above in sequence.

---

## 🔄 Complete Workflow

### Step-by-Step Training & Evaluation

```bash
cd /home/ngoductam/KLGrade

# Step 1: Train all models (choose one)
bash scripts/training/run_all_cada_experiments.sh          # All 20 experiments
# OR train selectively:
bash scripts/training/run_train_4_5_class.sh              # Just 4/5-class
bash scripts/training/run_train_8_10_class.sh             # Just 8/10-class  
bash scripts/training/run_processed_4_8_class.sh          # Just preprocessed 4/8
bash scripts/training/run_processed_5_10_class.sh         # Just preprocessed 5/10

# Step 2: Evaluate all trained models
bash scripts/training/evaluate_all_cada_experiments.sh

# Step 3: Aggregate results
python scripts/create_reports/aggregate_complete_results.py
```

---

## ⚙️ Training Configuration

All scripts use consistent settings:
- **Epochs**: 100
- **Batch Size**: 16
- **Early Stopping**: Patience 15
- **WandB Project**: klgrade-kiocmil-cada
- **Device**: Auto-detect (CUDA if available)

---

## 📁 Output Structure

```
runs/kiocmil_cada/
├── cada_4class_unbalanced/
│   ├── best_acc_model.pt
│   ├── last_model.pt
│   ├── train.log
│   └── config.json
├── cada_4class_balanced/
│   └── ...
└── [other experiments]/
    └── ...
```

---

## 🎨 Script Organization

### Why Split by Class Configuration?

1. **Faster Iteration**: Train only what you need
2. **Resource Management**: Spread training across multiple sessions
3. **Debugging**: Isolate issues to specific configurations
4. **Parallel Training**: Run different configs on different GPUs

### Naming Convention

- `run_train_*`: Base models (no preprocessing)
- `run_processed_*`: Preprocessed models (resize/blur/sharp)
- `run_all_*`: Master script (everything)

---

## 💡 Tips & Best Practices

### Before Training

1. **Check data splits exist**:
   ```bash
   ls datasets/splits/
   ```

2. **Verify W&B login**:
   ```bash
   wandb login
   ```

3. **Check disk space** (~50GB per full run):
   ```bash
   df -h
   ```

### During Training

- Monitor with W&B dashboard
- Check logs: `tail -f runs/kiocmil_cada/[exp_name]/train.log`
- Expect ~2-3 hours per experiment

### After Training

- Evaluate immediately for quality check
- Archive old runs if disk space limited
- Compare results with `aggregate_complete_results.py`

---

## 🐛 Troubleshooting

### Out of Memory
```bash
# Reduce batch size in script
BATCH_SIZE=8  # instead of 16
```

### Training Too Slow
```bash
# Reduce epochs for testing
EPOCHS=10  # instead of 100
```

### W&B Not Logging
```bash
# Disable W&B for testing
python src/training/train_kiocmil_cada.py --no_wandb ...
```

---

## 📜 Change Log

**2026-01-21**: Reorganized training scripts
- Created `run_train_4_5_class.sh`
- Created `run_train_8_10_class.sh`
- Created `run_processed_5_10_class.sh`
- Kept `run_processed_4_8_class.sh`
- Kept `run_all_cada_experiments.sh`
- Archived deprecated "retrain" scripts

**Previous**: All training via `run_all_cada_experiments.sh` only

---

**Last Updated**: 2026-01-21  
**Maintainer**: Project Team  
**Status**: Production Ready
