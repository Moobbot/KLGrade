#!/bin/bash
#
# Experiment: Generate splits with different ratios
#

set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate klgrade || echo "⚠️  Warning: Failed to activate klgrade env"

echo "=============================================="
echo "Running Data Split Experiments"
echo "=============================================="

# Experiment 1: 80% Train, 10% Val, 10% Test
echo ""
echo "Experiment 1: 80/10/10"
echo "----------------------------------------------"
export TRAIN_RATIO=0.8
export VAL_RATIO=0.1
export TEST_RATIO=0.1
export SPLIT_SUFFIX="_80_10_10"

bash scripts/pipelines/create_dataset_splits.sh all

# Experiment 2: 60% Train, 20% Val, 20% Test
echo ""
echo "Experiment 2: 60/20/20"
echo "----------------------------------------------"
export TRAIN_RATIO=0.6
export VAL_RATIO=0.2
export TEST_RATIO=0.2
export SPLIT_SUFFIX="_60_20_20"

bash scripts/pipelines/create_dataset_splits.sh all

# Experiment 3: 70% Train, 20% Val, 10% Test
echo ""
echo "Experiment 3: 70/20/10"
echo "----------------------------------------------"
export TRAIN_RATIO=0.7
export VAL_RATIO=0.2
export TEST_RATIO=0.1
export SPLIT_SUFFIX="_70_20_10"

bash scripts/pipelines/create_dataset_splits.sh all

echo ""
echo "=============================================="
echo "✅ All Experiments Complete!"
echo "=============================================="
echo "Check output in datasets/splits/"
