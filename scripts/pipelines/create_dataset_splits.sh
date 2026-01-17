#!/bin/bash
#
# Create train/val/test splits for each dataset independently
# Usage: 
#   ./scripts/pipelines/create_dataset_splits.sh all        # Create all splits
#   ./scripts/pipelines/create_dataset_splits.sh 5_class     # Create 5-class only
#   ./scripts/pipelines/create_dataset_splits.sh balanced   # Create balanced only
#

set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate klgrade || echo "⚠️  Warning: Failed to activate klgrade env"

# Configuration
SEED=42
TRAIN_RATIO=0.7
VAL_RATIO=0.15
TEST_RATIO=0.15

echo "========================================"
echo "Dataset Splits Creation"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Train: ${TRAIN_RATIO} (70%)"
echo "  Val:   ${VAL_RATIO} (15%)"
echo "  Test:  ${TEST_RATIO} (15%)"
echo "  Seed:  ${SEED}"
echo ""

# Function to create splits for a dataset
create_splits() {
    local dataset_name=$1
    local img_dir=$2
    local label_dir=$3
    local out_dir=$4
    
    echo "========================================" 
    echo "Creating splits for: $dataset_name"
    echo "========================================"
    echo "  Images:  $img_dir"
    echo "  Labels:  $label_dir"
    echo "  Output:  $out_dir"
    echo ""
    
    if [ ! -d "$img_dir" ]; then
        echo "❌ Error: Image directory not found: $img_dir"
        return 1
    fi
    
    if [ ! -d "$label_dir" ]; then
        echo "❌ Error: Label directory not found: $label_dir"
        return 1
    fi
    
    python scripts/data_preparation/split_dataset.py \
        --img_dir "$img_dir" \
        --label_dir "$label_dir" \
        --out_dir "$out_dir" \
        --train $TRAIN_RATIO \
        --val $VAL_RATIO \
        --test $TEST_RATIO \
        --seed $SEED
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Splits created successfully!"
        echo "   Files:"
        ls -lh "$out_dir"/*.txt 2>/dev/null | awk '{print "     " $9, "(" $5 ")"}'
        echo ""
    else
        echo "❌ Failed to create splits for $dataset_name"
        return 1
    fi
}

# Parse command line argument
MODE=${1:-all}

case $MODE in
    5_class|5-class|cropped-5)
        echo "Mode: Creating 5-class cropped splits only"
        echo ""
        create_splits \
            "Cropped Knees - 5 Class" \
            "datasets/dataset_knees_cropped/images" \
            "datasets/dataset_knees_cropped/labels" \
            "datasets/splits/dataset_knees_cropped"
        ;;
        
    4_class|4-class|cropped-4)
        echo "Mode: Creating 4-class cropped splits only"
        echo ""
        create_splits \
            "Cropped Knees - 4 Class" \
            "datasets/dataset_knees_cropped/images" \
            "datasets/dataset_knees_cropped/labels_4_class" \
            "datasets/splits/dataset_knees_cropped_4_class"
        ;;
        
    8_class|8-class|cropped-8)
        echo "Mode: Creating 8-class cropped splits only"
        echo ""
        create_splits \
            "Cropped Knees - 8 Class" \
            "datasets/dataset_knees_cropped/images" \
            "datasets/dataset_knees_cropped/labels_8_class" \
            "datasets/splits/dataset_knees_cropped_8_class"
        ;;
        
    balanced|balanced-5)
        echo "Mode: Creating balanced 5-class splits only"
        echo ""
        if [ ! -d "datasets/balanced/knees_cropped" ]; then
            echo "❌ Error: Balanced dataset not found!"
            echo "   Run balance_dataset.py first:"
            echo "   python scripts/data_preparation/balance_dataset.py \\"
            echo "       --input datasets/dataset_knees_cropped \\"
            echo "       --output datasets/balanced/knees_cropped"
            exit 1
        fi
        
        create_splits \
            "Balanced Knees - 5 Class" \
            "datasets/balanced/knees_cropped/images" \
            "datasets/balanced/knees_cropped/labels" \
            "datasets/splits/balanced_knees_cropped"
        ;;
        
    balanced-4)
        echo "Mode: Creating balanced 4-class splits only"
        echo ""
        create_splits \
            "Balanced Knees - 4 Class" \
            "datasets/balanced/knees_cropped/images" \
            "datasets/balanced/knees_cropped/labels_4_class" \
            "datasets/splits/balanced_knees_cropped_4_class"
        ;;
        
    balanced-8)
        echo "Mode: Creating balanced 8-class splits only"
        echo ""
        create_splits \
            "Balanced Knees - 8 Class" \
            "datasets/balanced/knees_cropped/images" \
            "datasets/balanced/knees_cropped/labels_8_class" \
            "datasets/splits/balanced_knees_cropped_8_class"
        ;;
        
    10_class|10-class|full)
        echo "Mode: Creating 10-class full X-ray splits only"
        echo ""
        create_splits \
            "Full X-rays - 10 Class" \
            "datasets/dataset/dataset_v0/images" \
            "datasets/dataset/dataset_v0/labels_10_class" \
            "datasets/splits/knee_full_10_class"
        ;;
        
    all)
        echo "Mode: Creating ALL splits"
        echo ""
        
        # 1. Cropped 5-class
        create_splits \
            "Cropped Knees - 5 Class" \
            "datasets/dataset_knees_cropped/images" \
            "datasets/dataset_knees_cropped/labels" \
            "datasets/splits/dataset_knees_cropped"
        
        # 2. Cropped 4-class (separate output dir)
        create_splits \
            "Cropped Knees - 4 Class" \
            "datasets/dataset_knees_cropped/images" \
            "datasets/dataset_knees_cropped/labels_4_class" \
            "datasets/splits/dataset_knees_cropped_4_class"
        
        # 3. Cropped 8-class (separate output dir)
        create_splits \
            "Cropped Knees - 8 Class" \
            "datasets/dataset_knees_cropped/images" \
            "datasets/dataset_knees_cropped/labels_8_class" \
            "datasets/splits/dataset_knees_cropped_8_class"
        
        # 4. Balanced 5-class (if exists)
        if [ -d "datasets/balanced/knees_cropped" ]; then
            create_splits \
                "Balanced Knees - 5 Class" \
                "datasets/balanced/knees_cropped/images" \
                "datasets/balanced/knees_cropped/labels" \
                "datasets/splits/balanced_knees_cropped"
            
            # 4a. Balanced 4-class
            create_splits \
                "Balanced Knees - 4 Class" \
                "datasets/balanced/knees_cropped/images" \
                "datasets/balanced/knees_cropped/labels_4_class" \
                "datasets/splits/balanced_knees_cropped_4_class"
            
            # 4b. Balanced 8-class
            create_splits \
                "Balanced Knees - 8 Class" \
                "datasets/balanced/knees_cropped/images" \
                "datasets/balanced/knees_cropped/labels_8_class" \
                "datasets/splits/balanced_knees_cropped_8_class"
        else
            echo "⚠️  Skipping balanced dataset (not found)"
            echo ""
        fi
        
        # 5. Full X-rays 10-class
        if [ -d "datasets/dataset/dataset_v0" ]; then
            create_splits \
                "Full X-rays - 10 Class" \
                "datasets/dataset/dataset_v0/images" \
                "datasets/dataset/dataset_v0/labels_10_class" \
                "datasets/splits/knee_full_10_class"
        else
            echo "⚠️  Skipping full X-rays dataset (not found)"
            echo ""
        fi
        ;;
        
    help|--help|-h)
        echo "Usage: $0 [MODE]"
        echo ""
        echo "Modes:"
        echo "  all          - Create all splits (default)"
        echo "  5_class       - Cropped 5-class only"
        echo "  4_class       - Cropped 4-class only"
        echo "  8_class       - Cropped 8-class only"
        echo "  balanced     - Balanced 5-class only"
        echo "  balanced-4   - Balanced 4-class only"
        echo "  balanced-8   - Balanced 8-class only"
        echo "  10_class      - Full X-rays 10-class only"
        echo ""
        echo "Examples:"
        echo "  $0 all              # Create all splits"
        echo "  $0 5_class           # Only 5-class cropped"
        echo "  $0 balanced         # Only balanced dataset"
        exit 0
        ;;
        
    *)
        echo "❌ Error: Unknown mode '$MODE'"
        echo "   Run '$0 help' for usage information"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "✅ Dataset Splits Creation Complete!"
echo "========================================"
echo ""
echo "Summary of created split files:"
find datasets/splits -name "train.txt" -o -name "val.txt" -o -name "test.txt" 2>/dev/null | while read file; do
    lines=$(wc -l < "$file" 2>/dev/null || echo "0")
    printf "  %-70s %5s images\n" "$file" "$lines"
done
