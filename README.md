# KLGrade - Auto-Grading Knee Osteoarthritis

An end-to-end pipeline for Knee Osteoarthritis grading from X-ray images using Deep Learning.

**Models**: KIOCMIL CADA (Context-Aware Deformable Attention) | YOLO Detection

---

## 📚 Documentation

Comprehensive documentation organized by topic:

### Core Documentation
1. **[Documentation Index](docs/README.md)** - Complete navigation guide
2. **[Data Pipeline](docs/guides/DATA_PIPELPE.md)** - From raw X-rays to processed datasets
3. **[Dataset Reference](docs/DATASETS.md)** - Dataset structure and statistics
4. **[Training Index](docs/TRAINING_INDEX.md)** - Central hub for all training docs

### Model-Specific Training
- **[KIOCMIL CADA Training](docs/KIOCMIL_CADA_TRAINING.md)** - Complete CADA training guide
- **[CADA Results](docs/experiments/CADA_COMPLETE_RESULTS.md)** - Evaluation results (20 experiments)

### Tools & Analysis
- **[Dataset Analysis Tools](tools/dataset_analysis/README.md)** - Dataset validation & visualization
- **[Visualization Organization](docs/reference/VISUALIZATION_ORGANIZATION.md)** - Viz code structure

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
conda create -n klgrade python=3.10 -y
conda activate klgrade
pip install -r requirements.txt
```

### 2. Prepare Data
From raw X-rays in `datasets/dataset_v0`, run the pipeline:
```bash
# Complete pipeline (recommended)
bash scripts/pipelines/run_full_pipeline.sh

# Or step-by-step:
bash scripts/pipelines/step_1_crop_knees.sh
bash scripts/pipelines/step_2_generate_labels.sh
bash scripts/pipelines/step_3_balance.sh
bash scripts/pipelines/step_4_preprocess_balanced.sh
bash scripts/pipelines/step_5_create_splits.sh
```

### 3. Train KIOCMIL CADA Model
Train the best-performing model (5-class balanced):
```bash
python src/training/train_kiocmil_cada.py \
    --num_classes 5 \
    --balanced \
    --epochs 100 \
    --batch_size 16 \
    --experiment_name cada_5class_balanced
```

Or use training scripts:
```bash
bash scripts/training/run_train_4_5_class.sh
```

### 4. Evaluate
```bash
bash scripts/training/evaluate_all_cada_experiments.sh
```

---

## 🎯 Key Results (KIOCMIL CADA)

| Configuration | Best Model | Accuracy | Highlights |
|---------------|-----------|----------|------------|
| **4-Class** | Unbalanced | **83.71%** | Best overall accuracy |
| **5-Class** | Unbalanced | **80.65%** | Full KL grading |
| **8-Class** | Balanced + Resize | **78.03%** | Lesion type analysis |
| **10-Class** | Balanced | **82.66%** | Fine-grained classification |

**Highlights:**
- **Type Classification**: 99.79% accuracy (osteophytes vs joint space)
- **Complete Evaluation**: All 20 models with comprehensive metrics
- **AUC Calculation**: Multi-class ROC analysis for all configurations

See [CADA Complete Results](docs/experiments/CADA_COMPLETE_RESULTS.md) for details.

---

## 📂 Project Structure
```
KLGrade/
├── datasets/               # Data (original, balanced, processed)
├── docs/                   # Comprehensive documentation
│   ├── guides/            # How-to guides
│   ├── reference/         # Technical references
│   ├── experiments/       # Experiment results
│   └── logs/              # Experiment logs
├── configs/                # Training configurations
├── scripts/                # Executable scripts
│   ├── pipelines/         # Data preparation pipeline
│   ├── training/          # Model training scripts
│   ├── analyzes/          # Dataset analysis
│   └── evaluation/        # Model evaluation
├── src/                    # Source code
│   ├── models/            # Model implementations
│   ├── datasets/          # Dataset loaders
│   ├── training/          # Training & evaluation
│   ├── utils/             # Shared utilities
│   └── visualization/     # Model interpretation
├── tools/                  # Analysis & validation tools
│   └── dataset_analysis/  # Dataset tools
└── runs/                   # Experiment outputs
    └── kiocmil_cada/      # CADA model results
```

---

## 📖 For More Details

- **Getting Started**: See [docs/README.md](docs/README.md)
- **Training Models**: See [docs/TRAINING_INDEX.md](docs/TRAINING_INDEX.md)
- **Data Pipeline**: See [docs/guides/DATA_PIPELINE.md](docs/guides/DATA_PIPELINE.md)
- **Experiment Results**: See [docs/experiments/](docs/experiments/)

---

**Status**: Production Ready ✅  
**Last Updated**: 2026-01-21
