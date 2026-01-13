# KIOCMIL Logging Guide

## Overview

Both v1 and v2 now have versioned logging with automatic run numbering.

## Logging Utility Functions

Module tiện ích: `src/utils/logging_utils.py`

### 1. `get_next_log_dir(base_dir="log", module_name=None)`

Tự động tạo thư mục log có số thứ tự theo module, không ghi đè log cũ.

**Ví dụ:**
```python
from src.utils.logging_utils import get_next_log_dir

# Với module name
log_dir = get_next_log_dir("log", "kiocmil")
print(f"Logging to: {log_dir}")  # log/run_kiocmil_001

# Lần chạy tiếp theo
log_dir = get_next_log_dir("log", "kiocmil")  
# log/run_kiocmil_002

# Module khác
log_dir = get_next_log_dir("log", "detr")
# log/run_detr_001
```

**Format thư mục:**
- **Với module name:** `run_<tên_module>_<số>` 
  - Ví dụ: `run_kiocmil_001`, `run_kiocmil_002`, `run_detr_001`
- **Không có module name:** `run_<số>`
  - Ví dụ: `run_001`, `run_002`

### 2. `save_training_config(log_dir, args, additional_info=None)`

Lưu cấu hình training vào file JSON.

**Ví dụ:**
```python
from src.utils.logging_utils import save_training_config

config_path = save_training_config(
    log_dir, 
    args, 
    {"device": "cuda:0", "model": "resnet18"}
)
```

**File config.json:**
```json
{
  "timestamp": "2026-01-13 07:30:00",
  "args": {
    "epochs": 50,
    "batch_size": 8,
    ...
  },
  "device": "cuda:0",
  "model": "resnet18"
}
```

### 3. `setup_training_logging(base_dir, module_name, args, additional_info=None)`

Tạo log directory và lưu config trong một lần gọi.

**Ví dụ:**
```python
from src.utils.logging_utils import setup_training_logging

log_dir = setup_training_logging(
    "log", 
    module_name="kiocmil",
    args=args, 
    additional_info={"device": device, "model": "resnet18"}
)
```


## Log Directory Structure

```
log/
├── run_kiocmil_001/
│   ├── config.json          # Training configuration
│   └── training.log         # Training progress log
├── run_kiocmil_002/
│   ├── config.json
│   └── training.log
└── run_kiocmil_003/
    ├── config.json
    └── training.log
```

## Features

### 1. Versioned Directories

Each training run automatically gets a unique directory:
- **Format**: `log/run_kiocmil_XXX` where XXX is an auto-incrementing number
- **No overwrites**: Old logs are preserved
- **Easy comparison**: Compare logs from different runs

### 2. Configuration Tracking

Each run saves `config.json` with:
- All command-line arguments
- Device information (CPU/GPU)
- Timestamp
- Paths used

### 3. Training Log File

Each run creates `training.log` with:
- Timestamp for each event
- Epoch-by-epoch progress
- Validation metrics
- Model save events
- Early stopping triggers

## Example Log Files

### `config.json`
```json
{
  "timestamp": "2026-01-13 08:00:00",
  "args": {
    "epochs": 50,
    "batch_size": 2,
    "lr": 0.0001,
    "backbone": "resnet18",
    "use_oversampling": true
  },
  "device": "cuda:0",
  "save_dir": "runs/kiocmil_exp1",
  "log_dir": "log/run_kiocmil_004"
}
```

### `training.log`
```
2026-01-13 08:00:15 - INFO - Config saved to: log/run_kiocmil_004/config.json
2026-01-13 08:01:20 - INFO - Epoch 1: Train Loss=2.1925, Val Loss=1.8915, Val Acc=0.5297
2026-01-13 08:01:21 - INFO - Saved Best Model (Acc: 0.5297)
2026-01-13 08:02:30 - INFO - Epoch 2: Train Loss=1.7781, Val Loss=2.0338, Val Acc=0.4247
2026-01-13 08:03:40 - INFO - Epoch 3: Train Loss=1.6471, Val Loss=1.8762, Val Acc=0.5023
...
```

## Usage

### Training with Logging

```bash
# V1 (Baseline)
python src/training/train_kiocmil.py --epochs 50

# V2 (Experimental)
python src/training/train_kiocmil_v2.py --use_v2_dataset --epochs 50
```

Both automatically:
- Create versioned log directory
- Save config.json
- Write training.log
- Print to console AND file

### View Logs in Real-Time

```bash
# Watch training progress
tail -f log/run_kiocmil_004/training.log

# Watch only epoch summaries
tail -f log/run_kiocmil_004/training.log | grep "Epoch"
```

### Compare Runs

```bash
# Compare configs
diff log/run_kiocmil_001/config.json log/run_kiocmil_002/config.json

# Compare final accuracy
grep "Training completed" log/run_kiocmil_*/training.log
```

## Log Retention

- **All logs are kept** - No automatic deletion
- **Manual cleanup**: Delete old run directories if needed
```bash
# Remove specific run
rm -rf log/run_kiocmil_001

# Keep only latest 10 runs
ls -d log/run_kiocmil_* | head -n-10 | xargs rm -rf
```

## Log Analysis

### Find Best Run

```bash
# Extract all final accuracies
grep "Training completed" log/run_kiocmil_*/training.log

# Find runs that hit early stopping
grep "Early stopping" log/run_kiocmil_*/training.log
```

### Check Run Configuration

```bash
# See what hyperparameters were used
jq '.args' log/run_kiocmil_001/config.json

# Check which device was used
jq '.device' log/run_kiocmil_*/config.json
```

## Integration with WandB

Logs work alongside WandB:
- **File logs**: Permanent, local, always available
- **WandB**: Online dashboard, visualizations, comparisons

Use both together for best results!

## Troubleshooting

### Logs not appearing
- Check that `log/` directory exists and is writable
- Verify logging_utils.py is in `src/utils/`
- Ensure training script has proper imports

### Run numbers not incrementing
- Check `log/` directory for existing `run_kiocmil_XXX` folders
- Ensure script has write permissions

### Log file empty
- Training might have crashed early
- Check console output for errors
- Verify `logging.basicConfig()` is called before any logging
