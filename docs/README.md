# KLGrade Documentation

**Project**: Knee Osteoarthritis Grading using Deep Learning  
**Last Updated**: 2026-01-21

---

## 📚 Quick Links

### Getting Started
- [Setup Guide](guides/SETUP.md) - Installation and environment setup
- [Data Pipeline](guides/DATA_PIPELINE.md) - Dataset preparation workflow

### Model Training
- [**Training Index**](TRAINING_INDEX.md) - Overview of all models
- [KIOCMIL CADA Training](KIOCMIL_CADA_TRAINING.md) - KIOCMIL CADA model training
- [Training Guide](guides/KIOCMIL_CADA_TRAINING_GUIDE.md) - Step-by-step tutorial

### Experimental Results
- [**CADA Complete Results**](experiments/CADA_COMPLETE_RESULTS.md) - Full KIOCMIL CADA evaluation (20 experiments)
- [Experiment Report](logs/EXPERIMENT_REPORT_FULL.md) - Comprehensive training & evaluation walkthrough
- [Results CSV](experiments/cada_complete_results.csv) - Raw experimental data

### Technical References
- [KIOCMIL CADA Reference](KIOCMIL_CADA_REFERENCE.md) - Model architecture & implementation
- [Code Structure](reference/CODE_STRUCTURE.md) - Project organization
- [Dependencies](reference/DEPENDENCIES.md) - Required packages

### Data & Preprocessing
- [Datasets Overview](DATASETS.md) - Available datasets and configurations
- [Preprocessing Labels](PREPROCESSING_LABELS.md) - Label generation and balancing
- [Output Structure](reference/KIOCMIL_CADA_OUTPUT_STRUCTURE.md) - Understanding results

---

## 🎯 Project Overview

This project implements state-of-the-art deep learning models for automated knee osteoarthritis grading from X-ray images using the Kellgren-Lawrence (KL) scale.

### Key Features
- **Multiple Classifications**: 4, 5, 8, and 10-class configurations
- **Context-Aware Attention**: KIOCMIL with Deformable Attention (CADA)
- **Data Balancing**: Advanced augmentation strategies
- **Comprehensive Evaluation**: Including derived metrics (grade, type)

### Best Results
| Configuration | Model | Accuracy |
|---------------|-------|----------|
| 4-Class (KL1-4) | cada_4class_balanced | 83.71% |
| 5-Class (KL0-4) | cada_5class_unbalanced_corrected | 81.11% |
| 8-Class (compartments) | cada_8class_balanced | 78.03% |
| 10-Class (fine-grained) | cada_10class_balanced | 82.01% |

---

## 📂 Documentation Structure

```
docs/
├── README.md                          # This file
├── DATASETS.md                        # Dataset information
├── PREPROCESSING_LABELS.md            # Label preprocessing
├── TRAINING.md                        # Training overview
├── KIOCMIL_CADA_REFERENCE.md         # CADA model reference
├── experiments/                       # Experimental results
│   ├── CADA_COMPLETE_RESULTS.md      # All 30 experiments
│   └── cada_complete_results.csv     # Raw data
├── guides/                            # Step-by-step guides
│   ├── SETUP.md                      # Environment setup
│   ├── TRAINING.md                   # Training guide
│   ├── DATA_PIPELINE.md              # Data preparation
│   └── WANDB.md                      # W&B integration
├── logs/                              # Experiment logs
│   ├── EXPERIMENT_REPORT_FULL.md     # Final report
│   ├── EXPERIMENT_LOG.md             # Training history
│   └── archive_processing_log.md     # Historical logs
└── reference/                         # Technical references
    ├── CODE_STRUCTURE.md             # Project structure
    ├── CONFIG_SUMMARY.md             # Configuration options
    ├── DEPENDENCIES.md               # Package requirements
    └── TRAINING_OUTPUT_STRUCTURE.md  # Results organization
```

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Create conda environment
conda env create -f environment.yml
conda activate klgrade

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data
```bash
# Run data pipeline
bash scripts/pipelines/run_all_steps.sh
```

### 3. Train Model
```bash
# Train KIOCMIL CADA (10-class)
bash scripts/training/run_all_cada_experiments.sh
```

### 4. Evaluate
```bash
# Evaluate all models
bash scripts/training/evaluate_all_cada_experiments.sh
```

---

## 🔬 Research Highlights

### Critical Bug Fix (2026-01-21)
Discovered and fixed training script bug where `num_classes` was hardcoded to 10:
- **Impact**: 4/5-class models trained with wrong architecture
- **Fix**: Updated to use `args.num_classes`
- **Result**: Re-trained 10 models with +0.8% avg improvement

[Full details](experiments/CADA_COMPLETE_RESULTS.md#critical-bug-discovery--fix)

### Key Findings
1. **Data balancing crucial** for fine-grained classification (8/10-class)
2. **No preprocessing best** for most configurations
3. **Type classification exceptional**: 99.79% accuracy (osteophytes vs joint space)
4. **Corrected models enable proper AUC calculation**

---

## 📊 Available Models

### KIOCMIL CADA
- **Architecture**: YOLO11L backbone + Deformable Attention
- **Configurations**: 4, 5, 8, 10 classes
- **Training Strategies**: Balanced/Unbalanced with various preprocessing
- **Total Models**: 30 (20 original + 10 corrected)

[Model Details](KIOCMIL_CADA_REFERENCE.md)

---

## 📝 Contributing

When adding new documentation:
1. Place in appropriate subdirectory (`guides/`, `reference/`, `experiments/`, `logs/`)
2. Update this README with links
3. Follow markdown formatting conventions
4. Include date and version information

---

## 📞 Contact & Support

For questions or issues:
- Check existing documentation first
- Review [Experiment Log](logs/EXPERIMENT_LOG.md) for historical context
- Consult [Training Output Structure](reference/TRAINING_OUTPUT_STRUCTURE.md) for file locations

---

**Last Updated**: 2026-01-21  
**Version**: 1.0  
**Status**: Production Ready
