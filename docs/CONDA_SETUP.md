# Conda Environment Setup for KLGrade

## Create Conda Environment

```bash
# Create environment with Python 3.10
conda create -n klgrade python=3.10 -y

# Activate environment
conda activate klgrade

# Install dependencies
pip install -r requirements.txt

# Install YOLO
pip install ultralytics

# Install WandB (optional)
pip install wandb
```

## Verify Installation

```bash
# Check Python version
python --version  # Should be 3.10.x

# Check YOLO installation
yolo version

# Check WandB (optional)
wandb --version
```

## Environment Management

### Activate Environment

```bash
conda activate klgrade
```

### Deactivate Environment

```bash
conda deactivate
```

### List Environments

```bash
conda env list
```

### Remove Environment (if needed)

```bash
conda env remove -n klgrade
```

## Export/Import Environment

### Export (for reproducibility)

```bash
# Export to YAML
conda env export > environment.yml

# Or export pip requirements
pip freeze > requirements_conda.txt
```

### Import (on another machine)

```bash
# From YAML
conda env create -f environment.yml

# Or from requirements
conda create -n klgrade python=3.10
conda activate klgrade
pip install -r requirements.txt
```

## Running Scripts

All training scripts now automatically activate conda:

```bash
# Training with WandB
./docs/TRAINING_COMMANDS_WANDB.sh

# Full pipeline
./scripts/run_full_pipeline.sh
```

## Troubleshooting

### "conda: command not found"

```bash
# Initialize conda for bash
conda init bash
# Then restart terminal
```

### Environment activation fails

```bash
# Check if environment exists
conda env list

# If not, create it
conda create -n klgrade python=3.10
```

### Permission denied

```bash
# Make scripts executable
chmod +x docs/TRAINING_COMMANDS_WANDB.sh
chmod +x scripts/run_full_pipeline.sh
```
