#!/bin/bash
# Analyze all knee datasets - from original to all processed versions

ANALYZE_SCRIPT="scripts/analyzes/analyze_knee_dataset.py"

echo "============================================================"
echo "ANALYZING ALL KNEE DATASETS"
echo "============================================================"
echo ""

# Array of datasets to analyze
datasets=()

# 1. Base Datasets
base_candidates=(
    "datasets/dataset_v0"
    "datasets/dataset_v0_4_class"
    "datasets/dataset_knees_cropped"
    "datasets/dataset_knees_cropped_4_class"
)

for d in "${base_candidates[@]}"; do
    if [ -d "$d" ]; then
        datasets+=("$d")
    fi
done

# 2. Balanced Datasets (all variants in datasets/balanced/)
if [ -d "datasets/balanced" ]; then
    for d in datasets/balanced/*; do
        if [ -d "$d" ]; then
            datasets+=("$d")
        fi
    done
fi

# 3. Processed Balanced Datasets (nested: dataset/variant)
# Structure: datasets/processed_balanced/{dataset_name}/{variant_name}
if [ -d "datasets/processed_balanced" ]; then
    for dataset_dir in datasets/processed_balanced/*; do
        if [ -d "$dataset_dir" ]; then
            # Check for variant subdirectories (look for 'images' inside to confirm it's a dataset)
            for variant_dir in "$dataset_dir"/*; do
                if [ -d "$variant_dir" ] && [ -d "$variant_dir/images" ]; then
                    datasets+=("$variant_dir")
                fi
            done
        fi
    done
fi

# Counter for progress
total=${#datasets[@]}
current=0

# Analyze each dataset
for dataset in "${datasets[@]}"; do
    current=$((current + 1))
    echo "[$current/$total] Analyzing: $dataset"
    echo "---"
    
    python $ANALYZE_SCRIPT --dataset "$dataset"
    
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
