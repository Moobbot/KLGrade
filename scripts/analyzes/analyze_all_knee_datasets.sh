#!/bin/bash
# Analyze all knee datasets - from original to all processed versions

ANALYZE_SCRIPT="scripts/analyzes/analyze_knee_dataset.py"

echo "============================================================"
echo "ANALYZING ALL KNEE DATASETS"
echo "============================================================"
echo ""

# Array of datasets to analyze
datasets=(
    "datasets/dataset_v0"
    "datasets/dataset_knees_cropped"
    "datasets/dataset_knees_cropped_balanced"
    "datasets/data_processed/resize_only"
    "datasets/data_processed/blur_clahe2"
    "datasets/data_processed/sharp_clahe4"
    "datasets/data_processed/blur_clahe2_notebook"
    "datasets/data_processed_balanced/resize_only"
    "datasets/data_processed_balanced/blur_clahe2"
    "datasets/data_processed_balanced/sharp_clahe4"
    "datasets/data_processed_balanced/blur_clahe2_notebook"
)

# Counter for progress
total=${#datasets[@]}
current=0

# Analyze each dataset
for dataset in "${datasets[@]}"; do
    current=$((current + 1))
    echo "[$current/$total] Analyzing: $dataset"
    echo "---"
    
    cd /home/ngoductam/KLGrade
    PYTHONPATH=/home/ngoductam/KLGrade /home/ngoductam/miniconda3/envs/klgrade/bin/python $ANALYZE_SCRIPT --dataset "$dataset"
    
    if [ $? -eq 0 ]; then
        echo "✅ Complete"
    else
        echo "❌ Failed"
    fi
    echo ""
done

echo "============================================================"
echo "ALL ANALYSES COMPLETE"
echo "============================================================"
echo ""
echo "Reports saved to:"
for dataset in "${datasets[@]}"; do
    echo "  - $dataset/dataset_statistics.txt"
done
