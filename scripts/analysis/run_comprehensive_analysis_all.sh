#!/bin/bash
# Run comprehensive analysis for all knee datasets

ANALYZE_SCRIPT="tools/dataset_analysis/analyze.py"

echo "============================================================"
echo "COMPREHENSIVE ANALYSIS FOR ALL DATASETS"
echo "============================================================"
echo ""

# Array of datasets and their output directories
declare -A datasets
datasets=(
    # Original datasets
    ["datasets/dataset_v0"]="dataset_analysis/dataset_v0"
    ["datasets/dataset_v0_4_class"]="dataset_analysis/dataset_v0_4_class"
    ["datasets/dataset_knees_cropped"]="dataset_analysis/knees_cropped"
    ["datasets/dataset_knees_cropped_4_class"]="dataset_analysis/knees_cropped_4_class"
    
    # Balanced datasets (current structure)
    ["datasets/balanced/full_xray"]="dataset_analysis/balanced/full_xray_5class"
    ["datasets/balanced/full_xray_4_class"]="dataset_analysis/balanced/full_xray_4class"
    ["datasets/balanced/full_xray_8_class"]="dataset_analysis/balanced/full_xray_8class"
    ["datasets/balanced/full_xray_10_class"]="dataset_analysis/balanced/full_xray_10class"
    ["datasets/balanced/knees_cropped"]="dataset_analysis/balanced/knees_cropped_5class"
    ["datasets/balanced/knees_cropped_4_class"]="dataset_analysis/balanced/knees_cropped_4class"
    ["datasets/balanced/knees_cropped_8_class"]="dataset_analysis/balanced/knees_cropped_8class"
    ["datasets/balanced/knees_cropped_10_class"]="dataset_analysis/balanced/knees_cropped_10class"
    
    # Processed datasets (current structure)
    ["datasets/processed_balanced/full_xray/blur_clahe2"]="dataset_analysis/processed_balanced/full_xray/blur_clahe2"
    ["datasets/processed_balanced/full_xray/resize_only"]="dataset_analysis/processed_balanced/full_xray/resize_only"
    ["datasets/processed_balanced/full_xray/sharp_clahe4"]="dataset_analysis/processed_balanced/full_xray/sharp_clahe4"
    ["datasets/processed_balanced/full_xray_4_class/blur_clahe2"]="dataset_analysis/processed_balanced/full_xray_4_class/blur_clahe2"
    ["datasets/processed_balanced/full_xray_4_class/resize_only"]="dataset_analysis/processed_balanced/full_xray_4_class/resize_only"
    ["datasets/processed_balanced/full_xray_4_class/sharp_clahe4"]="dataset_analysis/processed_balanced/full_xray/sharp_clahe4"
    ["datasets/processed_balanced/full_xray_8_class/blur_clahe2"]="dataset_analysis/processed_balanced/full_xray_8_class/blur_clahe2"
    ["datasets/processed_balanced/full_xray_8_class/resize_only"]="dataset_analysis/processed_balanced/full_xray_8_class/resize_only"
    ["datasets/processed_balanced/full_xray_8_class/sharp_clahe4"]="dataset_analysis/processed_balanced/full_xray_8_class/sharp_clahe4"
    ["datasets/processed_balanced/full_xray_10_class/blur_clahe2"]="dataset_analysis/processed_balanced/full_xray_10_class/blur_clahe2"
    ["datasets/processed_balanced/full_xray_10_class/resize_only"]="dataset_analysis/processed_balanced/full_xray_10_class/resize_only"
    ["datasets/processed_balanced/full_xray_10_class/sharp_clahe4"]="dataset_analysis/processed_balanced/full_xray_10_class/sharp_clahe4"
    ["datasets/processed_balanced/knees_cropped/blur_clahe2"]="dataset_analysis/processed_balanced/knees_cropped_5class/blur_clahe2"
    ["datasets/processed_balanced/knees_cropped/resize_only"]="dataset_analysis/processed_balanced/knees_cropped_5class/resize_only"
    ["datasets/processed_balanced/knees_cropped/sharp_clahe4"]="dataset_analysis/processed_balanced/knees_cropped_5class/sharp_clahe4"
    ["datasets/processed_balanced/knees_cropped_4_class/blur_clahe2"]="dataset_analysis/processed_balanced/knees_cropped_4class/blur_clahe2"
    ["datasets/processed_balanced/knees_cropped_4_class/resize_only"]="dataset_analysis/processed_balanced/knees_cropped_4class/resize_only"
    ["datasets/processed_balanced/knees_cropped_4_class/sharp_clahe4"]="dataset_analysis/processed_balanced/knees_cropped_4class/sharp_clahe4"
    ["datasets/processed_balanced/knees_cropped_8_class/blur_clahe2"]="dataset_analysis/processed_balanced/knees_cropped_8class/blur_clahe2"
    ["datasets/processed_balanced/knees_cropped_8_class/resize_only"]="dataset_analysis/processed_balanced/knees_cropped_8class/resize_only"
    ["datasets/processed_balanced/knees_cropped_8_class/sharp_clahe4"]="dataset_analysis/processed_balanced/knees_cropped_8class/sharp_clahe4"
    ["datasets/processed_balanced/knees_cropped_10_class/blur_clahe2"]="dataset_analysis/processed_balanced/knees_cropped_10class/blur_clahe2"
    ["datasets/processed_balanced/knees_cropped_10_class/resize_only"]="dataset_analysis/processed_balanced/knees_cropped_10class/resize_only"
    ["datasets/processed_balanced/knees_cropped_10_class/sharp_clahe4"]="dataset_analysis/processed_balanced/knees_cropped_10class/sharp_clahe4"
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
echo "Reports saved to dataset_analysis/ directory:"
for output_dir in "${datasets[@]}"; do
    echo "  - $output_dir/"
done
