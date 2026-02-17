# Scripts Directory

This directory contains executable scripts for various tasks in the KLGrade project.

## Structure

```
scripts/
├── training/          # Training scripts
│   ├── kiocmil/       # KIOCMIL training
│   ├── yolo/          # YOLO training
│   ├── end_to_end/    # End-to-end training
│   └── experiments/   # Experimental training runs
├── evaluation/        # Model evaluation scripts
├── analysis/          # Analysis and reporting
├── visualization/     # Visualization scripts
├── deployment/        # Deployment scripts
├── inference/         # Inference scripts
├── pipelines/         # Training/deployment pipelines
├── create_reports/    # Report generation
└── legacy/            # Deprecated scripts
```

## Usage

All scripts should be run from the project root:

```bash
cd /home/ngoductam/KLGrade

# Activate environment
conda activate klgrade

# Run script
python scripts/training/kiocmil/train_script.py
bash scripts/training/experiments/run_experiment.sh
```

## Categories

### Training
Scripts for training models (KIOCMIL, YOLO, end-to-end)

### Evaluation
Scripts for evaluating trained models

### Analysis
Scripts for analyzing results, generating reports

### Visualization
Scripts for creating visualizations and plots

### Deployment
Scripts for deploying models and APIs

### Inference
Scripts for running inference on datasets

### Pipelines
Automated workflows combining multiple steps

---

## Related Directories

- [`data_processing/`](../data_processing/README.md) - Data preparation and preprocessing
- [`tools/`](../tools/README.md) - Development tools
- [`tests/`](../tests/README.md) - Test scripts
