# Step 3: Balance Datasets via Oversampling
#
# Creates balanced versions of all dataset variants by oversampling minority classes.
# Balanced datasets enable training models with better class representation.

$ErrorActionPreference = "Stop"

# Activate conda environment
Write-Host "Activating conda environment..." -ForegroundColor Cyan
try {
    conda activate klgrade_api
} catch {
    Write-Host "⚠️  Warning: Failed to activate klgrade_api env" -ForegroundColor Yellow
}

Write-Host "========================================================"
Write-Host "Step 3: Balance Datasets (Comprehensive)"
Write-Host "========================================================"
Write-Host ""
Write-Host "This step balances ALL dataset variants:"
Write-Host "  1. Cropped knees (5-class)"
Write-Host "  2. Cropped knees (10-class)"
Write-Host "  3. Cropped knees (4-class)"
Write-Host "  4. Cropped knees (8-class)"
Write-Host "  5. Full X-rays (5-class)"
Write-Host "  6. Full X-rays (10-class)"
Write-Host "  7. Full X-rays (4-class)"
Write-Host "  8. Full X-rays (8-class)"
Write-Host ""
Write-Host "Strategy: Oversample minority classes to max class count"
Write-Host "Method:   Horizontal/vertical flips with label adjustment"
Write-Host ""
Write-Host "Warning: This will create 8 balanced datasets"
Write-Host ""

# Prompt user to continue
$Continueary = Read-Host "Continue with dataset balancing? (y/n)"
if ($Continueary -notmatch "^[Yy]$") {
    Write-Host "Skipping dataset balancing."
    Write-Host ""
    Write-Host "Next: .\scripts\pipelines\step_4_preprocess_balanced.ps1"
    exit 0
}

Write-Host ""
Write-Host "--------------------------------------------------------"
Write-Host "3.1 Balancing cropped knees (5-class)..."
Write-Host "--------------------------------------------------------"

python scripts/data_preparation/balance_dataset.py `
    --input-images datasets/dataset_knees_cropped/images `
    --input-labels datasets/dataset_knees_cropped/labels `
    --output-dir datasets/balanced/knees_cropped `
    --num-classes 5 `
    --aux-labels datasets/dataset_knees_cropped/labels_10_class `
                 datasets/dataset_knees_cropped/labels-knee

Write-Host ""
Write-Host "--------------------------------------------------------"
Write-Host "3.2 Balancing cropped knees (10-class)..."
Write-Host "--------------------------------------------------------"

python scripts/data_preparation/balance_dataset.py `
    --input-images datasets/dataset_knees_cropped/images `
    --input-labels datasets/dataset_knees_cropped/labels_10_class `
    --output-dir datasets/balanced/knees_cropped_10_class `
    --num-classes 10 `
    --aux-labels datasets/dataset_knees_cropped/labels `
                 datasets/dataset_knees_cropped/labels-knee

Write-Host ""
Write-Host "--------------------------------------------------------"
Write-Host "3.3 Balancing cropped knees (4-class)..."
Write-Host "--------------------------------------------------------"

python scripts/data_preparation/balance_dataset.py `
    --input-images datasets/dataset_knees_cropped_4_class/images `
    --input-labels datasets/dataset_knees_cropped_4_class/labels `
    --output-dir datasets/balanced/knees_cropped_4_class `
    --num-classes 4 `
    --aux-labels datasets/dataset_knees_cropped_4_class/labels_8_class `
                 datasets/dataset_knees_cropped/labels-knee

Write-Host ""
Write-Host "--------------------------------------------------------"
Write-Host "3.4 Balancing cropped knees (8-class)..."
Write-Host "--------------------------------------------------------"

python scripts/data_preparation/balance_dataset.py `
    --input-images datasets/dataset_knees_cropped_4_class/images `
    --input-labels datasets/dataset_knees_cropped_4_class/labels_8_class `
    --output-dir datasets/balanced/knees_cropped_8_class `
    --num-classes 8 `
    --aux-labels datasets/dataset_knees_cropped_4_class/labels `
                 datasets/dataset_knees_cropped/labels-knee

Write-Host ""
Write-Host "--------------------------------------------------------"
Write-Host "3.5 Balancing full X-rays (5-class)..."
Write-Host "--------------------------------------------------------"

python scripts/data_preparation/balance_dataset.py `
    --input-images datasets/dataset_v0/images `
    --input-labels datasets/dataset_v0/labels `
    --output-dir datasets/balanced/full_xray `
    --num-classes 5 `
    --aux-labels datasets/dataset_v0/labels_10_class `
                 datasets/dataset_v0/labels-knee

Write-Host ""
Write-Host "--------------------------------------------------------"
Write-Host "3.6 Balancing full X-rays (10-class)..."
Write-Host "--------------------------------------------------------"

python scripts/data_preparation/balance_dataset.py `
    --input-images datasets/dataset_v0/images `
    --input-labels datasets/dataset_v0/labels_10_class `
    --output-dir datasets/balanced/full_xray_10_class `
    --num-classes 10 `
    --aux-labels datasets/dataset_v0/labels `
                 datasets/dataset_v0/labels-knee

Write-Host ""
Write-Host "--------------------------------------------------------"
Write-Host "3.7 Balancing full X-rays (4-class)..."
Write-Host "--------------------------------------------------------"

python scripts/data_preparation/balance_dataset.py `
    --input-images datasets/dataset_v0_4_class/images `
    --input-labels datasets/dataset_v0_4_class/labels `
    --output-dir datasets/balanced/full_xray_4_class `
    --num-classes 4 `
    --aux-labels datasets/dataset_v0_4_class/labels_8_class `
                 datasets/dataset_v0/labels-knee

Write-Host ""
Write-Host "--------------------------------------------------------"
Write-Host "3.8 Balancing full X-rays (8-class)..."
Write-Host "--------------------------------------------------------"

python scripts/data_preparation/balance_dataset.py `
    --input-images datasets/dataset_v0_4_class/images `
    --input-labels datasets/dataset_v0_4_class/labels_8_class `
    --output-dir datasets/balanced/full_xray_8_class `
    --num-classes 8 `
    --aux-labels datasets/dataset_v0_4_class/labels `
                 datasets/dataset_v0/labels-knee

Write-Host ""
Write-Host "OK Balancing complete for all datasets"
Write-Host ""

Write-Host "--------------------------------------------------------"
Write-Host "Checking balanced dataset statistics..."
Write-Host "--------------------------------------------------------"

# Summary function
function Show-BalanceSummary {
    param([string]$Name, [string]$Dir)
    
    if (Test-Path "$Dir/images") {
        $count = (Get-ChildItem -Path "$Dir/images" -Recurse -Include *.jpg,*.png).Count
        Write-Host ""
        Write-Host "  $Name`: $count images"
        
        $reportPath = "$Dir/balance_report.txt"
        if (Test-Path $reportPath) {
            $lastLine = Get-Content $reportPath | Select-Object -Last 1
            if ($lastLine -match "augmented\)") {
                Write-Host "    $lastLine"
            }
        }
    }
}

Write-Host ""
Show-BalanceSummary -Name "Cropped knees (5-class)" -Dir "datasets/balanced/knees_cropped"
Show-BalanceSummary -Name "Cropped knees (10-class)" -Dir "datasets/balanced/knees_cropped_10_class"
Show-BalanceSummary -Name "Cropped knees (4-class)" -Dir "datasets/balanced/knees_cropped_4_class"
Show-BalanceSummary -Name "Cropped knees (8-class)" -Dir "datasets/balanced/knees_cropped_8_class"
Show-BalanceSummary -Name "Full X-rays (5-class)" -Dir "datasets/balanced/full_xray"
Show-BalanceSummary -Name "Full X-rays (10-class)" -Dir "datasets/balanced/full_xray_10_class"
Show-BalanceSummary -Name "Full X-rays (4-class)" -Dir "datasets/balanced/full_xray_4_class"
Show-BalanceSummary -Name "Full X-rays (8-class)" -Dir "datasets/balanced/full_xray_8_class"

Write-Host ""
Write-Host "========================================================"
Write-Host "Step 3 Complete!"
Write-Host "========================================================"
Write-Host ""
Write-Host "Summary:"
Write-Host "  [OK] 8 datasets balanced via flip augmentation"
Write-Host "  [OK] All label variants copied and adjusted"
Write-Host "  [OK] Balance reports generated"
Write-Host ""
Write-Host "Output structure:"
Write-Host "  datasets/balanced/"
Write-Host "    |-- knees_cropped/        # 5-class cropped knees"
Write-Host "    |-- knees_cropped_10_class/ # 10-class cropped knees"
Write-Host "    |-- knees_cropped_4_class/  # 4-class cropped knees"
Write-Host "    |-- knees_cropped_8_class/  # 8-class cropped knees"
Write-Host "    |-- full_xray/            # 5-class full X-rays"
Write-Host "    |-- full_xray_10_class/     # 10-class full X-rays"
Write-Host "    |-- full_xray_4_class/      # 4-class full X-rays"
Write-Host "    +-- full_xray_8class/      # 8-class full X-rays"
Write-Host ""
Write-Host "Next: Run .\scripts\pipelines\step_4_preprocess_balanced.ps1 to preprocess balanced datasets"
