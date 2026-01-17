#!/bin/bash
# Filter empty labels from all processed datasets

echo "============================================================"
echo "FILTERING EMPTY LABELS FROM PROCESSED DATASETS"
echo "============================================================"
echo ""

datasets=(
    "datasets/data_processed/resize_only"
    "datasets/data_processed/blur_clahe2"
    "datasets/data_processed/sharp_clahe4"
    "datasets/data_processed/blur_clahe2_notebook"
    "datasets/data_processed_balanced/resize_only"
    "datasets/data_processed_balanced/blur_clahe2"
    "datasets/data_processed_balanced/sharp_clahe4"
    "datasets/data_processed_balanced/blur_clahe2_notebook"
)

total=${#datasets[@]}
current=0

for dataset in "${datasets[@]}"; do
    current=$((current + 1))
    echo "[$current/$total] Filtering: $dataset"
    
    cd /home/ngoductam/KLGrade
    PYTHONPATH=/home/ngoductam/KLGrade /home/ngoductam/miniconda3/envs/klgrade/bin/python \
        scripts/data_preparation/filter_no_labels.py --input "$dataset"
    
    echo ""
done

echo "============================================================"
echo "COMPLETE"
echo "============================================================"
