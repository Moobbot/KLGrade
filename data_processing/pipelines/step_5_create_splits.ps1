# Step 5: Create Train/Val/Test Splits
#
# Input:  All datasets in datasets/
# Output: datasets/splits/[dataset_name]/

$ErrorActionPreference = "Stop"

# Activate conda environment
Write-Host "Activating conda environment..." -ForegroundColor Cyan
try {
    conda activate klgrade_api
} catch {
    Write-Host "⚠️  Warning: Failed to activate klgrade_api env" -ForegroundColor Yellow
}

Write-Host "========================================================"
Write-Host "Step 5: Create Train/Val/Test Splits"
Write-Host "========================================================"
Write-Host ""
Write-Host "Split ratio: 70% train / 15% val / 15% test"
Write-Host "Method: Stratified sampling by class"
Write-Host ""

Write-Host "--------------------------------------------------------"
Write-Host "5.1 Creating splits for all datasets..."
Write-Host "--------------------------------------------------------"

# Call the PowerShell version of create_dataset_splits
.\scripts\pipelines\create_dataset_splits.ps1 -Mode all

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to create splits" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================================"
Write-Host "Step 5 Complete!"
Write-Host "========================================================"
Write-Host ""
Write-Host "Summary:"
Write-Host "  [OK] Stratified train/val/test splits created"
Write-Host "  [OK] Splits for all dataset variants"
Write-Host ""
Write-Host "Output structure:"
Write-Host "  datasets/splits/"
Write-Host "    |-- dataset_knees_cropped/           # Base 5-class"
Write-Host "    |-- dataset_knees_cropped_4_class/   # Base 4-class"
Write-Host "    |-- dataset_knees_cropped_8_class/   # Base 8-class"
Write-Host "    |-- dataset_knees_cropped_10_class/  # Base 10-class"
Write-Host "    |-- balanced_knees_cropped/          # Balanced 5-class"
Write-Host "    |-- balanced_knees_cropped_4_class/  # Balanced 4-class"
Write-Host "    |-- balanced_knees_cropped_8_class/  # Balanced 8-class"
Write-Host "    |-- balanced_knees_cropped_10_class/ # Balanced 10-class"
Write-Host "    |-- knee_full_10_class/              # Base Full X-ray 10-class"
Write-Host "    |-- knee_full_8_class/               # Base Full X-ray 8-class"
Write-Host "    |-- knee_full_4_class/               # Base Full X-ray 4-class"
Write-Host "    |-- balanced_full_xray_10_class/     # Balanced Full X-ray 10-class"
Write-Host "    |-- balanced_full_xray_4_class/      # Balanced Full X-ray 4-class"
Write-Host "    |-- balanced_full_xray_8_class/      # Balanced Full X-ray 8-class"
Write-Host "    +-- processed_balanced/              # Processed Balanced Datasets (all variants)"
Write-Host ""
Write-Host "Each split directory contains:"
Write-Host "  - train.txt (70%)"
Write-Host "  - val.txt   (15%)"
Write-Host "  - test.txt  (15%)"
Write-Host ""
Write-Host "════════════════════════════════════════════════════════"
Write-Host "🎉 Data Pipeline Complete!"
Write-Host "════════════════════════════════════════════════════════"
Write-Host ""
Write-Host "All datasets ready for training!"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Review DATASET_GUIDE.md for dataset details"
Write-Host "  2. Use datasets in training - see TRAINING_WORKFLOW.md"
Write-Host "  3. Run experiments with different configurations"
