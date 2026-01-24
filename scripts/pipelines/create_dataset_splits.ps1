# Create train/val/test splits for each dataset independently
# Usage: 
#   .\scripts\pipelines\create_dataset_splits.ps1 -Mode all        # Create all splits
#   .\scripts\pipelines\create_dataset_splits.ps1 -Mode 5_class     # Create 5-class only
#   .\scripts\pipelines\create_dataset_splits.ps1 -Mode balanced   # Create balanced only

param (
    [string]$Mode = "all"
)

$ErrorActionPreference = "Stop"

# Activate conda environment
Write-Host "Activating conda environment..." -ForegroundColor Cyan
try {
    conda activate klgrade_api
} catch {
    Write-Host "⚠️  Warning: Failed to activate klgrade_api env" -ForegroundColor Yellow
}

# Configuration
$SEED = 42
$TRAIN_RATIO = 0.7
$VAL_RATIO = 0.15
$TEST_RATIO = 0.15

Write-Host "========================================"
Write-Host "Dataset Splits Creation"
Write-Host "========================================"
Write-Host ""
Write-Host "Configuration:"
Write-Host "  Train: $TRAIN_RATIO (70%)"
Write-Host "  Val:   $VAL_RATIO (15%)"
Write-Host "  Test:  $TEST_RATIO (15%)"
Write-Host "  Seed:  $SEED"
Write-Host ""

# Function to create splits for a dataset
function Create-Splits {
    param (
        [string]$DatasetName,
        [string]$ImgDir,
        [string]$LabelDir,
        [string]$OutDir
    )
    
    Write-Host "========================================" 
    Write-Host "Creating splits for: $DatasetName"
    Write-Host "========================================"
    Write-Host "  Images:  $ImgDir"
    Write-Host "  Labels:  $LabelDir"
    Write-Host "  Output:  $OutDir"
    Write-Host ""
    
    if (-not (Test-Path $ImgDir)) {
        Write-Host "Error: Image directory not found: $ImgDir" -ForegroundColor Red
        return $false
    }
    
    if (-not (Test-Path $LabelDir)) {
        Write-Host "Error: Label directory not found: $LabelDir" -ForegroundColor Red
        return $false
    }
    
    python scripts/data_preparation/split_dataset.py `
        --img_dir "$ImgDir" `
        --label_dir "$LabelDir" `
        --out_dir "$OutDir" `
        --train $TRAIN_RATIO `
        --val $VAL_RATIO `
        --test $TEST_RATIO `
        --seed $SEED
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "Split created successfully!"
        Write-Host "   Files:"
        Get-ChildItem -Path "$OutDir" -Filter "*.txt" | ForEach-Object {
            $count = (Get-Content $_.FullName | Measure-Object -Line).Lines
            Write-Host "     $($_.Name) ($count)"
        }
        Write-Host ""
        return $true
    } else {
        Write-Host "Failed to create splits for $DatasetName" -ForegroundColor Red
        return $false
    }
}

switch -Regex ($Mode) {
    "^(5_class|5-class|cropped-5)$" {
        Write-Host "Mode: Creating 5-class cropped splits only"
        Write-Host ""
        Create-Splits `
            -DatasetName "Cropped Knees - 5 Class" `
            -ImgDir "datasets/dataset_knees_cropped/images" `
            -LabelDir "datasets/dataset_knees_cropped/labels" `
            -OutDir "datasets/splits/dataset_knees_cropped"
    }

    "^(4_class|4-class|cropped-4)$" {
        Write-Host "Mode: Creating 4-class cropped splits only"
        Write-Host ""
        Create-Splits `
            -DatasetName "Cropped Knees - 4 Class" `
            -ImgDir "datasets/dataset_knees_cropped/images" `
            -LabelDir "datasets/dataset_knees_cropped/labels_4_class" `
            -OutDir "datasets/splits/dataset_knees_cropped_4_class"
    }

    "^(8_class|8-class|cropped-8)$" {
        Write-Host "Mode: Creating 8-class cropped splits only"
        Write-Host ""
        Create-Splits `
            -DatasetName "Cropped Knees - 8 Class" `
            -ImgDir "datasets/dataset_knees_cropped/images" `
            -LabelDir "datasets/dataset_knees_cropped/labels_8_class" `
            -OutDir "datasets/splits/dataset_knees_cropped_8_class"
    }

    "^(balanced|balanced-5)$" {
        Write-Host "Mode: Creating balanced 5-class splits only"
        Write-Host ""
        if (-not (Test-Path "datasets/balanced/knees_cropped")) {
            Write-Host "❌ Error: Balanced dataset not found!" -ForegroundColor Red
            Write-Host "   Run balance_dataset.py first:"
            Write-Host "   python scripts/data_preparation/balance_dataset.py ..."
            exit 1
        }
        
        Create-Splits `
            -DatasetName "Balanced Knees - 5 Class" `
            -ImgDir "datasets/balanced/knees_cropped/images" `
            -LabelDir "datasets/balanced/knees_cropped/labels" `
            -OutDir "datasets/splits/balanced_knees_cropped"
    }

    "^balanced-4$" {
        Write-Host "Mode: Creating balanced 4-class splits only"
        Write-Host ""
        Create-Splits `
            -DatasetName "Balanced Knees - 4 Class" `
            -ImgDir "datasets/balanced/knees_cropped/images" `
            -LabelDir "datasets/balanced/knees_cropped/labels_4_class" `
            -OutDir "datasets/splits/balanced_knees_cropped_4_class"
    }

    "^balanced-8$" {
        Write-Host "Mode: Creating balanced 8-class splits only"
        Write-Host ""
        Create-Splits `
            -DatasetName "Balanced Knees - 8 Class" `
            -ImgDir "datasets/balanced/knees_cropped/images" `
            -LabelDir "datasets/balanced/knees_cropped/labels_8_class" `
            -OutDir "datasets/splits/balanced_knees_cropped_8_class"
    }

    "^(10_class|10-class|full)$" {
        Write-Host "Mode: Creating 10-class full X-ray splits only"
        Write-Host ""
        Create-Splits `
            -DatasetName "Full X-rays - 10 Class" `
            -ImgDir "datasets/dataset_v0/images" `
            -LabelDir "datasets/dataset_v0/labels_10_class" `
            -OutDir "datasets/splits/knee_full_10_class"
    }

    "^balanced-10$" {
        Write-Host "Mode: Creating balanced 10-class cropped splits only"
        Write-Host ""
        Create-Splits `
            -DatasetName "Balanced Knees - 10 Class" `
            -ImgDir "datasets/balanced/knees_cropped_10_class/images" `
            -LabelDir "datasets/balanced/knees_cropped_10_class/labels" `
            -OutDir "datasets/splits/balanced_knees_cropped_10_class"
    }

    "^balanced-full-10$" {
        Write-Host "Mode: Creating balanced 10-class full splits only"
        Write-Host ""
        Create-Splits `
            -DatasetName "Balanced Full X-rays - 10 Class" `
            -ImgDir "datasets/balanced/full_xray_10_class/images" `
            -LabelDir "datasets/balanced/full_xray_10_class/labels" `
            -OutDir "datasets/splits/balanced_full_xray_10_class"
    }

    "^balanced-full-4$" {
        Write-Host "Mode: Creating balanced 4-class full splits only"
        Write-Host ""
        Create-Splits `
            -DatasetName "Balanced Full X-rays - 4 Class" `
            -ImgDir "datasets/balanced/full_xray_4_class/images" `
            -LabelDir "datasets/balanced/full_xray_4_class/labels" `
            -OutDir "datasets/splits/balanced_full_xray_4_class"
    }

    "^balanced-full-8$" {
        Write-Host "Mode: Creating balanced 8-class full splits only"
        Write-Host ""
        Create-Splits `
            -DatasetName "Balanced Full X-rays - 8 Class" `
            -ImgDir "datasets/balanced/full_xray_8_class/images" `
            -LabelDir "datasets/balanced/full_xray_8_class/labels" `
            -OutDir "datasets/splits/balanced_full_xray_8_class"
    }

    "^all$" {
        Write-Host "Mode: Creating ALL splits"
        Write-Host ""
        
        # 1. Cropped 5-class
        if (Test-Path "datasets/dataset_knees_cropped/labels") {
            Create-Splits `
                -DatasetName "Cropped Knees - 5 Class" `
                -ImgDir "datasets/dataset_knees_cropped/images" `
                -LabelDir "datasets/dataset_knees_cropped/labels" `
                -OutDir "datasets/splits/dataset_knees_cropped" | Out-Null
        }
        
        # 2. Cropped 4-class
        if (Test-Path "datasets/dataset_knees_cropped_4_class/labels") {
             Create-Splits `
                -DatasetName "Cropped Knees - 4 Class" `
                -ImgDir "datasets/dataset_knees_cropped_4_class/images" `
                -LabelDir "datasets/dataset_knees_cropped_4_class/labels" `
                -OutDir "datasets/splits/dataset_knees_cropped_4_class" | Out-Null
        } elseif (Test-Path "datasets/dataset_knees_cropped/labels_4_class") {
             Create-Splits `
                -DatasetName "Cropped Knees - 4 Class" `
                -ImgDir "datasets/dataset_knees_cropped/images" `
                -LabelDir "datasets/dataset_knees_cropped/labels_4_class" `
                -OutDir "datasets/splits/dataset_knees_cropped_4_class" | Out-Null
        }
        
        # 3. Cropped 8-class
        if (Test-Path "datasets/dataset_knees_cropped_4_class/labels_8_class") {
            Create-Splits `
                -DatasetName "Cropped Knees - 8 Class" `
                -ImgDir "datasets/dataset_knees_cropped_4_class/images" `
                -LabelDir "datasets/dataset_knees_cropped_4_class/labels_8_class" `
                -OutDir "datasets/splits/dataset_knees_cropped_8_class" | Out-Null
        }

        # 3. Cropped 10-class
        if (Test-Path "datasets/dataset_knees_cropped/labels_10_class") {
            Create-Splits `
                -DatasetName "Cropped Knees - 10 Class" `
                -ImgDir "datasets/dataset_knees_cropped/images" `
                -LabelDir "datasets/dataset_knees_cropped/labels_10_class" `
                -OutDir "datasets/splits/dataset_knees_cropped_10_class" | Out-Null
        }
        
        # 4. Balanced Variants (if they exist)
        
        # Balanced 5-class
        if (Test-Path "datasets/balanced/knees_cropped") {
            Create-Splits `
                -DatasetName "Balanced Knees - 5 Class" `
                -ImgDir "datasets/balanced/knees_cropped/images" `
                -LabelDir "datasets/balanced/knees_cropped/labels" `
                -OutDir "datasets/splits/balanced_knees_cropped" | Out-Null
        }

        # Balanced 4-class
        if (Test-Path "datasets/balanced/knees_cropped_4_class") {
            Create-Splits `
                -DatasetName "Balanced Knees - 4 Class" `
                -ImgDir "datasets/balanced/knees_cropped_4_class/images" `
                -LabelDir "datasets/balanced/knees_cropped_4_class/labels" `
                -OutDir "datasets/splits/balanced_knees_cropped_4_class" | Out-Null
        }

        # Balanced 8-class
        if (Test-Path "datasets/balanced/knees_cropped_8_class") {
            Create-Splits `
                -DatasetName "Balanced Knees - 8 Class" `
                -ImgDir "datasets/balanced/knees_cropped_8_class/images" `
                -LabelDir "datasets/balanced/knees_cropped_8_class/labels" `
                -OutDir "datasets/splits/balanced_knees_cropped_8_class" | Out-Null
        }
        
        # Balanced 10-class
        if (Test-Path "datasets/balanced/knees_cropped_10_class") {
            Create-Splits `
                -DatasetName "Balanced Knees - 10 Class" `
                -ImgDir "datasets/balanced/knees_cropped_10_class/images" `
                -LabelDir "datasets/balanced/knees_cropped_10_class/labels" `
                -OutDir "datasets/splits/balanced_knees_cropped_10_class" | Out-Null
        }

        # 5. Full X-rays
        
        # Full 10-class (Base)
        if (Test-Path "datasets/dataset_v0/labels_10_class") {
            Create-Splits `
                -DatasetName "Full X-rays - 10 Class" `
                -ImgDir "datasets/dataset_v0/images" `
                -LabelDir "datasets/dataset_v0/labels_10_class" `
                -OutDir "datasets/splits/knee_full_10_class" | Out-Null
        }

        # Full 4-class (Base)
        if (Test-Path "datasets/dataset_v0_4_class") {
            Create-Splits `
                -DatasetName "Full X-rays - 4 Class" `
                -ImgDir "datasets/dataset_v0_4_class/images" `
                -LabelDir "datasets/dataset_v0_4_class/labels" `
                -OutDir "datasets/splits/knee_full_4_class" | Out-Null
        }


        # Full 8-class (Base)
        if (Test-Path "datasets/dataset_v0_4_class/labels_8_class") {
            Create-Splits `
                -DatasetName "Full X-rays - 8 Class" `
                -ImgDir "datasets/dataset_v0_4_class/images" `
                -LabelDir "datasets/dataset_v0_4_class/labels_8_class" `
                -OutDir "datasets/splits/knee_full_8_class" | Out-Null
        }

        # Balanced Full 10-class
        if (Test-Path "datasets/balanced/full_xray_10_class") {
            Create-Splits `
                -DatasetName "Balanced Full X-rays - 10 Class" `
                -ImgDir "datasets/balanced/full_xray_10_class/images" `
                -LabelDir "datasets/balanced/full_xray_10_class/labels" `
                -OutDir "datasets/splits/balanced_full_xray_10_class" | Out-Null
        }

        # Balanced Full 4-class
        if (Test-Path "datasets/balanced/full_xray_4_class") {
             Create-Splits `
                -DatasetName "Balanced Full X-rays - 4 Class" `
                -ImgDir "datasets/balanced/full_xray_4_class/images" `
                -LabelDir "datasets/balanced/full_xray_4_class/labels" `
                -OutDir "datasets/splits/balanced_full_xray_4_class" | Out-Null
        }

        # Balanced Full 8-class
        if (Test-Path "datasets/balanced/full_xray_8_class") {
             Create-Splits `
                -DatasetName "Balanced Full X-rays - 8 Class" `
                -ImgDir "datasets/balanced/full_xray_8_class/images" `
                -LabelDir "datasets/balanced/full_xray_8_class/labels" `
                -OutDir "datasets/splits/balanced_full_xray_8_class" | Out-Null
        }

        # 6. Processed Balanced Datasets
        if (Test-Path "datasets/processed_balanced") {
            Write-Host ""
            Write-Host "----------------------------------------"
            Write-Host "Processing Processed Balanced Datasets..."
            Write-Host "----------------------------------------"
            
            Get-ChildItem -Path "datasets/processed_balanced" -Directory | ForEach-Object {
                $datasetName = $_.Name
                $datasetPath = $_.FullName
                
                Get-ChildItem -Path $datasetPath -Directory | ForEach-Object {
                    $variantName = $_.Name
                    $variantDir = $_.FullName
                    
                    if ((Test-Path "$variantDir/images") -and (Test-Path "$variantDir/labels")) {
                        Create-Splits `
                            -DatasetName "Processed Balanced - $datasetName - $variantName" `
                            -ImgDir "$variantDir/images" `
                            -LabelDir "$variantDir/labels" `
                            -OutDir "datasets/splits/processed_balanced/$datasetName/$variantName" | Out-Null
                    }
                }
            }
        }
    }

    "^(help|--help|-h)$" {
        Write-Host "Usage: .\scripts\pipelines\create_dataset_splits.ps1 [-Mode] <MODE>"
        Write-Host ""
        Write-Host "Modes:"
        Write-Host "  all                 - Create all splits (default)"
        Write-Host "  5_class              - Cropped 5-class only"
        Write-Host "  4_class              - Cropped 4-class only"
        Write-Host "  8_class              - Cropped 8-class only"
        Write-Host "  10_class             - Full X-rays 10-class only"
        Write-Host "  balanced            - Balanced 5-class only"
        Write-Host "  balanced-4          - Balanced 4-class only"
        Write-Host "  balanced-8          - Balanced 8-class only"
        Write-Host "  balanced-10         - Balanced 10-class only"
        Write-Host "  balanced-full-10    - Balanced Full X-ray 10-class"
        Write-Host "  balanced-full-4     - Balanced Full X-ray 4-class"
        Write-Host "  balanced-full-8     - Balanced Full X-ray 8-class"
        Write-Host ""
        exit 0
    }

    Default {
        Write-Host "Error: Unknown mode '$Mode'" -ForegroundColor Red
        Write-Host "   Run '.\scripts\pipelines\create_dataset_splits.ps1 -Mode help' for usage information"
        exit 1
    }
}

Write-Host ""
Write-Host "========================================"
Write-Host "Dataset Splits Creation Complete!"
Write-Host "========================================"
Write-Host ""
Write-Host "Summary of created split files:"
Get-ChildItem -Path "datasets/splits" -Recurse -Include "train.txt","val.txt","test.txt" | ForEach-Object {
    $lines = 0
    try {
        $lines = (Get-Content $_.FullName | Measure-Object -Line).Lines
    } catch {
        $lines = 0
    }
    Write-Host ("  {0,-70} {1,5} images" -f $_.FullName, $lines)
}
