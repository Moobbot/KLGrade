#!/bin/bash
#
# Regenerate all YAML configs and dataset splits
# This script creates splits for all class configurations and updates YAML configs
#

set -e  # Exit on error

echo "========================================"
echo "Regenerating Dataset Splits & Configs"
echo "========================================"
echo ""

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate klgrade || echo "Warning: Failed to activate klgrade env"

# Base dataset
DATASET_DIR="datasets/dataset_knees_cropped"
IMG_DIR="$DATASET_DIR/images"

# Create splits for each class configuration
echo "Creating dataset splits..."
echo ""

# 5-class (default labels)
echo "=== 5-class split ==="
python scripts/data_preparation/split_dataset.py \
    --img_dir "$IMG_DIR" \
    --label_dir "datasets/dataset_knees_cropped/labels" \
    --out_dir "datasets/dataset_knees_cropped" \
    --train 0.7 --val 0.15 --test 0.15 \
    --seed 42

# 4-class  
echo ""
echo "=== 4-class split ==="
python scripts/data_preparation/split_dataset.py \
    --img_dir "$IMG_DIR" \
    --label_dir "datasets/dataset_knees_cropped/labels_4_class" \
    --out_dir "datasets/dataset_knees_cropped/labels_4_class" \
    --train 0.7 --val 0.15 --test 0.15 \
    --seed 42

# 8-class
echo ""
echo "=== 8-class split ==="
python scripts/data_preparation/split_dataset.py \
    --img_dir "$IMG_DIR" \
    --label_dir "datasets/dataset_knees_cropped_4_class/labels_8_class" \
    --out_dir "datasets/dataset_knees_cropped_8_class" \
    --train 0.7 --val 0.15 --test 0.15 \
    --seed 42

# 10-class (from dataset_v0)
if [ -d "datasets/dataset_v0/labels_10_class" ]; then
    echo ""
    echo "=== 10-class split ==="
    python scripts/data_preparation/split_dataset.py \
        --img_dir "datasets/dataset_v0/images" \
        --label_dir "datasets/dataset_v0/labels_10_class" \
        --out_dir "datasets/dataset_v0_10_class" \
        --train 0.7 --val 0.15 --test 0.15 \
        --seed 42
fi

echo ""
echo "========================================"
echo "✅ All splits created!"
echo "========================================"
echo ""
echo "Now updating YAML configs..."

# Update configs to use new split locations
python - << 'EOF'
import yaml
from pathlib import Path

configs = {
    '5_class': {
        'dataset_dir': 'datasets/dataset_knees_cropped',
        'nc': 5,
        'names': ['KL0', 'KL1', 'KL2', 'KL3', 'KL4']
    },
    '4_class': {
        'dataset_dir': 'datasets/dataset_knees_cropped_4_class',
        'nc': 4,
        'names': ['KL1', 'KL2', 'KL3', 'KL4']
    },
    '8_class': {
        'dataset_dir': 'datasets/dataset_knees_cropped_8_class',
        'nc': 8,
        'names': ['KL0', 'KL1-a', 'KL1-b', 'KL2-a', 'KL2-b', 'KL3-a', 'KL3-b', 'KL4']
    },
    '10_class': {
        'dataset_dir': 'datasets/dataset_v0_10_class',
        'nc': 10,
        'names': ['KL0', 'KL1-a', 'KL1-b', 'KL1-c', 'KL2-a', 'KL2-b', 'KL3-a', 'KL3-b', 'KL3-c', 'KL4']
    }
}

for class_type, info in configs.items():
    dataset_dir = Path(info['dataset_dir'])
    
    for variant in ['baseline', 'conservative']:
        config_file = Path(f'configs/yolo_{class_type}_{variant}.yaml')
        
        if config_file.exists():
            # Load existing config
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Update paths
            config['path'] = '.'
            config['train'] = str(dataset_dir / 'train.txt')
            config['val'] = str(dataset_dir / 'val.txt')
            config['test'] = str(dataset_dir / 'test.txt')
            config['nc'] = info['nc']
            config['names'] = info['names']
            
            # Save updated config
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            print(f"✅ Updated {config_file}")

print("\n✅ All configs updated!")
EOF

echo ""
echo "========================================"
echo "✅ Complete!"
echo "========================================"
echo ""
echo "Verify splits exist:"
ls -lh datasets/dataset_knees_cropped/*.txt datasets/dataset_knees_cropped_*/*.txt 2>/dev/null | awk '{print "  " $9, "(" $5 ")"}'
