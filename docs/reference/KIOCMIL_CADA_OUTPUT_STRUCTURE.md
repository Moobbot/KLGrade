# KIOCMIL CADA Training Output Structure

**Model**: KIOCMIL with Context-Aware Deformable Attention  
**Output Location**: `runs/kiocmil_cada/`

This document describes the output structure for KIOCMIL CADA training and evaluation.

---

## 📁 Training Output Structure

Each experiment creates a directory: `runs/kiocmil_cada/<experiment_name>/`

```
runs/kiocmil_cada/cada_5class_balanced/
├── config.json                    # Training configuration
├── train.log                      # Training logs
├── best_acc_model.pt             # Best model (by validation accuracy)
├── last_model.pt                 # Latest checkpoint
└── evaluation/                    # (Created after evaluation)
    ├── metrics.json              # Evaluation metrics
    ├── metrics.txt               # Human-readable report
    ├── cm_5_class.png           # Confusion matrix (main)
    ├── roc_5_class.png          # ROC curve (main)
    ├── cm_grade.png             # Grade confusion matrix (derived)
    └── cm_type.png              # Type confusion matrix (derived)
```

---

## 📄 Key Files

### 1. `config.json` - Training Configuration

Contains all hyperparameters and settings:

```json
{
  "num_classes": 5,
  "epochs": 100,
  "batch_size": 16,
  "learning_rate": 0.0001,
  "backbone": "yolo11l",
  "feature_dim": 256,
  "early_stopping_patience": 15,
  "train_img_dir": "datasets/balanced/full_xray/images",
  "val_img_dir": "datasets/balanced/full_xray/images",
  ...
}
```

### 2. `train.log` - Training Logs

Real-time training progress:
```
Epoch 1/100
Training:   3%|▎  | 2/65 [00:01<00:49, 1.28it/s, loss=1.46, acc=0.323]
...
Epoch 50/100 - Best acc: 0.7812 at epoch 47
```

### 3. Model Checkpoints

- **`best_acc_model.pt`**: Model with best validation accuracy
- **`last_model.pt`**: Most recent checkpoint

Both contain:
- Model state dict
- Optimizer state
- Training epoch
- Best metrics

---

## 📊 Evaluation Output Structure

Created by running `evaluate_kiocmil_cada.py`:

```
runs/kiocmil_cada/cada_5class_balanced/evaluation/
├── metrics.json                   # All metrics in JSON format
├── metrics.txt                    # Human-readable report
├── cm_5_class.png                # Main classification confusion matrix
├── roc_5_class.png               # Main classification ROC curve
├── cm_grade.png                  # Grade-only confusion matrix (10→5)
├── cm_type.png                   # Type-only confusion matrix (10→2)
└── roc_grade.png                 # Grade-only ROC curve (optional)
```

### `metrics.json` Structure

```json
{
  "accuracy": 0.7812,
  "kappa": 0.6973,
  "f1_macro": 0.7277,
  "precision_macro": 0.7421,
  "recall_macro": 0.7156,
  "auc_macro": 0.9425,
  "per_class_metrics": {
    "KL0": {"precision": 0.80, "recall": 0.75, "f1": 0.77},
    "KL1": {"precision": 0.85, "recall": 0.82, "f1": 0.83},
    ...
  },
  "derived_metrics": {
    "accuracy": 0.7850,
    "kappa": 0.7012,
    "f1_macro": 0.7310,
    "type_metrics": {
      "accuracy": 0.9945,
      "f1_macro": 0.9944
    }
  },
  "confusion_matrix": [[...], [...], ...]
}
```

### `metrics.txt` Example

```
=== KIOCMIL CADA Evaluation Results ===

Model: runs/kiocmil_cada/cada_5class_balanced/best_acc_model.pt
Classes: 5
Samples: 234

--- Main Metrics ---
Accuracy: 78.12%
Kappa: 0.6973
F1 (Macro): 0.7277
AUC (Macro): 0.9425

--- Per-Class Metrics ---
         Precision  Recall    F1     Support
KL0      0.8000    0.7500   0.7742      20
KL1      0.8500    0.8200   0.8347      50
KL2      0.7200    0.7800   0.7488      75
KL3      0.6800    0.7100   0.6947      55
KL4      0.7100    0.6500   0.6788      34

--- Confusion Matrix ---
[[15  3  2  0  0]
 [ 2 41  5  2  0]
 [ 0  4 59  9  3]
 [ 0  1  8 39  7]
 [ 0  0  5  7 22]]

--- Derived Metrics (10→5 class) ---
Grade Accuracy: 78.50%
Type Accuracy: 99.45%
```

---

## 🔍 Loading and Using Results

### Load Model Checkpoint

```python
import torch
from src.models.kiocmil_model_cada import KiocmilModelCADA

# Load checkpoint
checkpoint = torch.load(
    'runs/kiocmil_cada/cada_5class_balanced/best_acc_model.pt',
    map_location='cuda'
)

# Initialize model
model = KiocmilModelCADA(
    backbone_name='yolo11l',
    num_classes=5,
    feature_dim=256
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Read Evaluation Metrics

```python
import json

# Load metrics
with open('runs/kiocmil_cada/cada_5class_balanced/evaluation/metrics.json') as f:
    metrics = json.load(f)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"AUC: {metrics['auc_macro']:.4f}")
print(f"Per-class F1: {metrics['per_class_metrics']}")
```

### Read Training Config

```python
import json

# Load config
with open('runs/kiocmil_cada/cada_5class_balanced/config.json') as f:
    config = json.load(f)

print(f"Trained for {config['epochs']} epochs")
print(f"Batch size: {config['batch_size']}")
```

---

## 📦 Complete Results Directory

After running all experiments:

```
runs/kiocmil_cada/
├── cada_4class_unbalanced/
│   └── [as above]
├── cada_4class_balanced/
│   └── [as above]
├── cada_5class_unbalanced/
│   └── [as above]
├── cada_5class_balanced/
│   └── [as above]
├── ... (16 more experiments)
├── complete_results.csv          # Aggregated results
└── COMPLETE_RESULTS.md           # Formatted report
```

**Aggregated files** (created by `aggregate_complete_results.py`):
- `complete_results.csv`: All metrics from all experiments
- `COMPLETE_RESULTS.md`: Comprehensive comparison report

---

## 🎯 Key Differences from YOLO

| Aspect | YOLO | KIOCMIL CADA |
|--------|------|--------------|
| **Output dir** | `runs/detect/` | `runs/kiocmil_cada/` |
| **Best model** | `best.pt` (mAP) | `best_acc_model.pt` (accuracy) |
| **Metrics file** | `results.csv` | `metrics.json` |
| **Config** | `args.yaml` | `config.json` |
| **Eval output** | Inline during training | Separate evaluation step |

---

**Last Updated**: 2026-01-21  
**Model**: KIOCMIL CADA  
**Related**: [KIOCMIL CADA Training](../KIOCMIL_CADA_TRAINING.md)
