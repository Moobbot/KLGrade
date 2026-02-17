# Development Tools

This directory contains utility tools for development, debugging, and analysis.

## Structure

```
tools/
├── analysis/       # Training log analysis and metrics
├── debugging/      # Debugging utilities
├── evaluation/     # Model evaluation scripts
└── wandb/          # Weights & Biases utilities
```

## 📁 analysis/

- **`analyze_training_log.py`** - Parse and analyze training logs

```bash
python tools/analysis/analyze_training_log.py --log training.log
```

## 📁 debugging/

- **`debug_auc_issues.py`** - Debug AUC calculation issues
- **`debug_dataset.py`** - Dataset debugging utilities

```bash
python tools/debugging/debug_dataset.py --dataset datasets/balanced/
```

## 📁 evaluation/

- **`evaluate_two_step_v0.py`** - Evaluate two-step YOLO pipeline

```bash
python tools/evaluation/evaluate_two_step_v0.py \
    --knee-model runs/detect/knee_detector/weights/best.pt \
    --lesion-model runs/detect/lesion_detector/weights/best.pt
```

## 📁 wandb/

- **`check_wandb_entities.py`** - Check WandB configuration

```bash
python tools/wandb/check_wandb_entities.py
```

---

## Usage Notes

- Run all scripts from project root
- Activate `klgrade` conda environment first
- Use `--help` flag for detailed usage
