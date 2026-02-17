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
            "datasets/dataset_v0/images" \
            "datasets/dataset_v0/labels_10_class" \
            "datasets/splits/knee_full_10_class"
        ;;

    balanced-10)
        echo "Mode: Creating balanced 10-class cropped splits only"
        echo ""
        create_splits \
            "Balanced Knees - 10 Class" \
            "datasets/balanced/knees_cropped_10_class/images" \
            "datasets/balanced/knees_cropped_10_class/labels" \
            "datasets/splits/balanced_knees_cropped_10_class"
        ;;

    balanced-full-10)
        echo "Mode: Creating balanced 10-class full splits only"
        echo ""
        create_splits \
            "Balanced Full X-rays - 10 Class" \
            "datasets/balanced/full_xray_10_class/images" \
            "datasets/balanced/full_xray_10_class/labels" \
            "datasets/splits/balanced_full_xray_10_class"
        ;;

    balanced-full-4)
        echo "Mode: Creating balanced 4-class full splits only"
        echo ""
        create_splits \
            "Balanced Full X-rays - 4 Class" \
            "datasets/balanced/full_xray_4_class/images" \
            "datasets/balanced/full_xray_4_class/labels" \
            "datasets/splits/balanced_full_xray_4_class"
        ;;

    balanced-full-8)
        echo "Mode: Creating balanced 8-class full splits only"
        echo ""
        create_splits \
            "Balanced Full X-rays - 8 Class" \
            "datasets/balanced/full_xray_8_class/images" \
            "datasets/balanced/full_xray_8_class/labels" \
            "datasets/splits/balanced_full_xray_8_class"
        ;;
        
    all)
        echo "Mode: Creating ALL splits"
        echo ""
        
        # 1. Cropped 5-class
        if [ -d "datasets/dataset_knees_cropped/labels" ]; then
            create_splits \
                "Cropped Knees - 5 Class" \
                "datasets/dataset_knees_cropped/images" \
                "datasets/dataset_knees_cropped/labels" \
                "datasets/splits/dataset_knees_cropped"
        fi
        
        # 2. Cropped 4-class
        if [ -d "datasets/dataset_knees_cropped_4_class/labels" ]; then
             create_splits \
                "Cropped Knees - 4 Class" \
                "datasets/dataset_knees_cropped_4_class/images" \
                "datasets/dataset_knees_cropped_4_class/labels" \
                "datasets/splits/dataset_knees_cropped_4_class"
        elif [ -d "datasets/dataset_knees_cropped/labels_4_class" ]; then
             create_splits \
                "Cropped Knees - 4 Class" \
                "datasets/dataset_knees_cropped/images" \
                "datasets/dataset_knees_cropped/labels_4_class" \
                "datasets/splits/dataset_knees_cropped_4_class"
        fi
        
        # 3. Cropped 8-class
        if [ -d "datasets/dataset_knees_cropped_4_class/labels_8_class" ]; then
            create_splits \
                "Cropped Knees - 8 Class" \
                "datasets/dataset_knees_cropped_4_class/images" \
                "datasets/dataset_knees_cropped_4_class/labels_8_class" \
                "datasets/splits/dataset_knees_cropped_8_class"
        fi

        # 3. Cropped 10-class
        if [ -d "datasets/dataset_knees_cropped/labels_10_class" ]; then
            create_splits \
                "Cropped Knees - 10 Class" \
                "datasets/dataset_knees_cropped/images" \
                "datasets/dataset_knees_cropped/labels_10_class" \
                "datasets/splits/dataset_knees_cropped_10_class"
        fi
        
        # 4. Balanced Variants (if they exist)
        
        # Balanced 5-class
        if [ -d "datasets/balanced/knees_cropped" ]; then
            create_splits \
                "Balanced Knees - 5 Class" \
                "datasets/balanced/knees_cropped/images" \
                "datasets/balanced/knees_cropped/labels" \
                "datasets/splits/balanced_knees_cropped"
        fi

        # Balanced 4-class
        if [ -d "datasets/balanced/knees_cropped_4_class" ]; then
            create_splits \
                "Balanced Knees - 4 Class" \
                "datasets/balanced/knees_cropped_4_class/images" \
                "datasets/balanced/knees_cropped_4_class/labels" \
                "datasets/splits/balanced_knees_cropped_4_class"
        fi

        # Balanced 8-class
        if [ -d "datasets/balanced/knees_cropped_8_class" ]; then
            create_splits \
                "Balanced Knees - 8 Class" \
                "datasets/balanced/knees_cropped_8_class/images" \
                "datasets/balanced/knees_cropped_8_class/labels" \
                "datasets/splits/balanced_knees_cropped_8_class"
        fi
        
        # Balanced 10-class
        if [ -d "datasets/balanced/knees_cropped_10_class" ]; then
            create_splits \
                "Balanced Knees - 10 Class" \
                "datasets/balanced/knees_cropped_10_class/images" \
                "datasets/balanced/knees_cropped_10_class/labels" \
                "datasets/splits/balanced_knees_cropped_10_class"
        fi

        # 5. Full X-rays
        
        # Full 10-class (Base)
        if [ -d "datasets/dataset_v0/labels_10_class" ]; then
            create_splits \
                "Full X-rays - 10 Class" \
                "datasets/dataset_v0/images" \
                "datasets/dataset_v0/labels_10_class" \
                "datasets/splits/knee_full_10_class"
        fi

        # Full 4-class (Base)
        if [ -d "datasets/dataset_v0_4_class" ]; then
            create_splits \
                "Full X-rays - 4 Class" \
                "datasets/dataset_v0_4_class/images" \
                "datasets/dataset_v0_4_class/labels" \
                "datasets/splits/knee_full_4_class"
        fi


        # Full 8-class (Base)
        if [ -d "datasets/dataset_v0_4_class/labels_8_class" ]; then
            create_splits \
                "Full X-rays - 8 Class" \
                "datasets/dataset_v0_4_class/images" \
                "datasets/dataset_v0_4_class/labels_8_class" \
                "datasets/splits/knee_full_8_class"
        fi

        # Balanced Full 10-class
        if [ -d "datasets/balanced/full_xray_10_class" ]; then
            create_splits \
                "Balanced Full X-rays - 10 Class" \
                "datasets/balanced/full_xray_10_class/images" \
                "datasets/balanced/full_xray_10_class/labels" \
                "datasets/splits/balanced_full_xray_10_class"
        fi

        # Balanced Full 4-class
        if [ -d "datasets/balanced/full_xray_4_class" ]; then
             create_splits \
                "Balanced Full X-rays - 4 Class" \
                "datasets/balanced/full_xray_4_class/images" \
                "datasets/balanced/full_xray_4_class/labels" \
                "datasets/splits/balanced_full_xray_4_class"
        fi

        # Balanced Full 8-class
        if [ -d "datasets/balanced/full_xray_8_class" ]; then
             create_splits \
                "Balanced Full X-rays - 8 Class" \
                "datasets/balanced/full_xray_8_class/images" \
                "datasets/balanced/full_xray_8_class/labels" \
                "datasets/splits/balanced_full_xray_8_class"
        fi

        # 6. Processed Balanced Datasets
        if [ -d "datasets/processed_balanced" ]; then
            echo ""
            echo "----------------------------------------"
            echo "Processing Processed Balanced Datasets..."
            echo "----------------------------------------"
            
            for dataset_dir in datasets/processed_balanced/*; do
                if [ -d "$dataset_dir" ]; then
                    dataset_name=$(basename "$dataset_dir")
                    
                    for variant_dir in "$dataset_dir"/*; do
                        if [ -d "$variant_dir" ]; then
                            variant_name=$(basename "$variant_dir")
                            
                            # Check if valid dataset structure
                            if [ -d "$variant_dir/images" ] && [ -d "$variant_dir/labels" ]; then
                                create_splits \
                                    "Processed Balanced - $dataset_name - $variant_name" \
                                    "$variant_dir/images" \
                                    "$variant_dir/labels" \
                                    "datasets/splits/processed_balanced/$dataset_name/$variant_name"
                            fi
                        fi
                    done
                fi
            done
        fi
        ;;

    help|--help|-h)
        echo "Usage: $0 [MODE]"
        echo ""
        echo "Modes:"
        echo "  all                 - Create all splits (default)"
        echo "  5_class              - Cropped 5-class only"
        echo "  4_class              - Cropped 4-class only"
        echo "  8_class              - Cropped 8-class only"
        echo "  10_class             - Full X-rays 10-class only"
        echo "  balanced            - Balanced 5-class only"
        echo "  balanced-4          - Balanced 4-class only"
        echo "  balanced-8          - Balanced 8-class only"
        echo "  balanced-10         - Balanced 10-class only"
        echo "  balanced-full-10    - Balanced Full X-ray 10-class"
        echo "  balanced-full-4     - Balanced Full X-ray 4-class"
        echo "  balanced-full-8     - Balanced Full X-ray 8-class"
        echo ""
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
