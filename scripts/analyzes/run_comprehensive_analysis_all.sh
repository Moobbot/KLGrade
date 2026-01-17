#!/bin/bash
# Run comprehensive analysis for all knee datasets

ANALYZE_SCRIPT="tools/check_dataset/comprehensive_analysis.py"

echo "============================================================"
echo "COMPREHENSIVE ANALYSIS FOR ALL DATASETS"
echo "============================================================"
echo ""

# Array of datasets and their output directories
declare -A datasets
datasets=(
    ["datasets/dataset/dataset_v0"]="analysis/dataset_v0"
    ["datasets/dataset_knees_cropped"]="analysis/knees_cropped"
    ["datasets/dataset_knees_cropped_balanced"]="analysis/knees_cropped_balanced"
    ["datasets/data_processed/resize_only"]="analysis/processed/resize_only"
    ["datasets/data_processed/blur_clahe2"]="analysis/processed/blur_clahe2"
    ["datasets/data_processed/sharp_clahe4"]="analysis/processed/sharp_clahe4"
    ["datasets/data_processed/blur_clahe2_notebook"]="analysis/processed/blur_clahe2_notebook"
    ["datasets/data_processed_balanced/resize_only"]="analysis/processed_balanced/resize_only"
    ["datasets/data_processed_balanced/blur_clahe2"]="analysis/processed_balanced/blur_clahe2"
    ["datasets/data_processed_balanced/sharp_clahe4"]="analysis/processed_balanced/sharp_clahe4"
    ["datasets/data_processed_balanced/blur_clahe2_notebook"]="analysis/processed_balanced/blur_clahe2_notebook"
)

# Counter
total=${#datasets[@]}
current=0

# Analyze each dataset
for dataset_dir in "${!datasets[@]}"; do
    current=$((current + 1))
    output_dir="${datasets[$dataset_dir]}"
    
    echo "[$current/$total] Analyzing: $dataset_dir"
    echo "Output: $output_dir"
    echo "---"
    
    cd /home/ngoductam/KLGrade
    PYTHONPATH=/home/ngoductam/KLGrade /home/ngoductam/miniconda3/envs/klgrade/bin/python $ANALYZE_SCRIPT \
        --dataset_dir "$dataset_dir" \
        --output "$output_dir"
    
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
echo "Reports saved to analysis/ directory:"
for output_dir in "${datasets[@]}"; do
    echo "  - $output_dir/"
done
