# KIOCMIL CADA Training

**Model**: KIOCMIL with Context-Aware Deformable Attention (CADA)  
**Task**: Knee Osteoarthritis Grading (Kellgren-Lawrence Scale)  
**Backbone**: YOLO11L

> **Note**: This documentation is specific to KIOCMIL CADA. Other models will have separate training documentation.

---

## Quick Start

### Train All KIOCMIL CADA Experiments

```bash
cd /home/ngoductam/KLGrade

# Train all 20 experiments
bash scripts/training/run_all_cada_experiments.sh

# Evaluate
bash scripts/training/evaluate_all_cada_experiments.sh

# Generate report
python scripts/create_reports/aggregate_complete_results.py
```

### Train Selective Configurations

```bash
# Base models (no preprocessing)
bash scripts/training/run_train_4_5_class.sh        # 4/5-class
bash scripts/training/run_train_8_10_class.sh       # 8/10-class

# Preprocessed models
bash scripts/training/run_processed_4_8_class.sh    # 4/8-class
bash scripts/training/run_processed_5_10_class.sh   # 5/10-class
```

---

## Model Configurations

### Class Configurations

| Configuration | Description | Classes | Use Case |
|---------------|-------------|---------|----------|  
| **4-Class** | KL grades 1-4 | KL1, KL2, KL3, KL4 | Standard grading (no healthy) |
| **5-Class** | KL grades 0-4 | KL0, KL1, KL2, KL3, KL4 | Full grading scale |
| **8-Class** | 4-class + lesion types | KL1-a/b, KL2-a/b, KL3-a/b, KL4-a/b | Lesion type analysis |
| **10-Class** | 5-class + lesion types | KL0-a/b, KL1-a/b, ..., KL4-a/b | Fine-grained analysis |

Where:
- **a** = Osteophytes (gai xương - bone spurs)
- **b** = Joint Space Narrowing (khe khớp - cartilage loss)

### Data Strategies

Each class configuration trained with 3 strategies:
1. **Unbalanced**: Natural class distribution
2. **Balanced**: Augmentation to balance classes
3. **Preprocessed** (Balanced + processing):
   - Resize only
   - Blur + CLAHE
   - Sharp + CLAHE

**Total**: 20 experiments (4 configs × 5 strategies each, minus duplicates)

---

## Training Scripts

See [Training Scripts README](../scripts/training/README.md) for detailed documentation.

### Available Scripts

| Script | Experiments | Description |
|--------|-------------|-------------|
| `run_all_cada_experiments.sh` | 20 | All KIOCMIL CADA experiments |
| `run_train_4_5_class.sh` | 4 | Base 4/5-class models |
| `run_train_8_10_class.sh` | 4 | Base 8/10-class models |
| `run_processed_4_8_class.sh` | 6 | Preprocessed 4/8-class |
| `run_processed_5_10_class.sh` | 6 | Preprocessed 5/10-class |
| `evaluate_all_cada_experiments.sh` | All | Evaluation script |

---

## Training Configuration

### Hyperparameters

```python
{
    "epochs": 100,
    "batch_size": 16,
    "learning_rate": 1e-4,
    "optimizer": "AdamW",
    "early_stopping_patience": 15,
    "backbone": "yolo11l",
    "feature_dim": 256,
    "num_deformable_points": 4,
    "num_context_scales": 3,
    "dropout": 0.1
}
```

### Data Augmentation

- Geometric: Random rotation (±15°), horizontal/vertical flip
- Photometric: CLAHE (optional), brightness/contrast adjustment
- None during validation

---

## Results Summary

### Best Models (Production Ready)

| Configuration | Model | Accuracy | Key Metric |
|---------------|-------|----------|------------|
| **4-Class** | cada_4class_balanced | **83.71%** | Kappa: 0.78 |
| **5-Class** | cada_5class_unbalanced | **81.11%** | F1: 0.75 |
| **8-Class** | cada_8class_balanced | **78.03%** | Type Acc: 99.45% |
| **10-Class** | cada_10class_balanced | **82.01%** | Type Acc: 99.79% |

### Key Findings

- **Type Classification**: Exceptional (99.79% for osteophytes vs joint space)
- **Data Balancing**: Critical for 8/10-class, optional for 4/5-class
- **Preprocessing**: Minimal works best; resize moderate improvement
- **AUC Metrics**: All models provide comprehensive ROC analysis

Full results: [CADA Complete Results](experiments/CADA_COMPLETE_RESULTS.md)

---

## Output Structure

```
runs/kiocmil_cada/
├── cada_4class_unbalanced/
│   ├── best_acc_model.pt           # Best model checkpoint
│   ├── last_model.pt               # Latest checkpoint
│   ├── train.log                   # Training logs
│   ├── config.json                 # Training configuration
│   └── evaluation/
│       ├── metrics.json            # Evaluation metrics
│       ├── metrics.txt             # Human-readable report
│       ├── cm_4_class.png          # Confusion matrix
│       └── roc_4_class.png         # ROC curve
└── [19 other experiments]/
    └── ...
```

See [KIOCMIL CADA Output Structure](reference/KIOCMIL_CADA_OUTPUT_STRUCTURE.md) for details.

---

## Workflow

### Complete Pipeline

```bash
# 1. Prepare data (if not done)
bash scripts/pipelines/run_all_steps.sh

# 2. Train models
bash scripts/training/run_all_cada_experiments.sh

# 3. Evaluate
bash scripts/training/evaluate_all_cada_experiments.sh

# 4. Generate reports
python scripts/create_reports/aggregate_complete_results.py
```

### Selective Training

Train only what you need for faster iteration:

```bash
# Just 5-class models
bash scripts/training/run_train_4_5_class.sh

# Evaluate 5-class only
python src/training/evaluate_kiocmil_cada.py \
    --model_path runs/kiocmil_cada/cada_5class_balanced/best_acc_model.pt \
    --num_classes 5 \
    ...
```

---

## Guides & References

### Detailed Guides
- [KIOCMIL CADA Training Guide](guides/KIOCMIL_CADA_TRAINING_GUIDE.md) - Step-by-step tutorial
- [Training Scripts README](../scripts/training/README.md) - Script documentation

### Technical References
- [KIOCMIL CADA Architecture](KIOCMIL_CADA_REFERENCE.md) - Model design details
- [Output Structure](reference/KIOCMIL_CADA_OUTPUT_STRUCTURE.md) - File organization

### Results & Analysis
- [Complete Results](experiments/CADA_COMPLETE_RESULTS.md) - All 30 experiments
- [Experiment Report](logs/EXPERIMENT_REPORT_FULL.md) - Comprehensive walkthrough

---

## Monitoring & Debugging

### Weights & Biases Integration

Training automatically logs to W&B:
- Real-time metrics (loss, accuracy, kappa)
- Confusion matrices per epoch
- Learning rate schedules
- System metrics (GPU, memory)

```bash
# View training progress
wandb login
# Check: https://wandb.ai/[your-entity]/klgrade-kiocmil-cada
```

### Training Logs

```bash
# Watch training progress
tail -f runs/kiocmil_cada/cada_5class_balanced/train.log

# Check for errors
grep -i "error\|warning" runs/kiocmil_cada/*/train.log
```

---

## Future Models

This documentation is specific to **KIOCMIL CADA**. When training other models:

- Create separate documentation (e.g., `YOLO_TRAINING.md`, `DETR_TRAINING.md`)
- Use different output directories (`runs/yolo/`, `runs/detr/`)
- Follow similar structure for consistency

---

**Model**: KIOCMIL CADA  
**Last Updated**: 2026-01-21  
**Status**: Production Ready  
**Total Experiments**: 20
