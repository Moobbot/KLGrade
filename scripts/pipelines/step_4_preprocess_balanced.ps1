# Step 4: Preprocess Balanced Datasets
#
# Input:  datasets/balanced/{knees_cropped, knees_cropped_4_class, full_xray, full_xray_4_class}
# Output: datasets/processed_balanced/
#
# Creates 4 preprocessing variants for each balanced dataset:
#   - resize_only
#   - blur_clahe2
#   - sharp_clahe4
# #   - blur_clahe2_notebook

$ErrorActionPreference = "Stop"

# Activate conda environment
Write-Host "Activating conda environment..." -ForegroundColor Cyan
try {
    conda activate klgrade_api
} catch {
    Write-Host "⚠️  Warning: Failed to activate klgrade_api env" -ForegroundColor Yellow
}

Write-Host "========================================================"
Write-Host "Step 4: Preprocess Balanced Datasets"
Write-Host "========================================================"
Write-Host ""
Write-Host "This step creates preprocessing variants for ALL balanced datasets:"
Write-Host "  1. Cropped knees (5-class) - 4 variants"
Write-Host "  2. Cropped knees (4-class) - 4 variants"
Write-Host "  3. Full X-rays (5-class) - 4 variants"
Write-Host "  4. Full X-rays (4-class) - 4 variants"
Write-Host ""
Write-Host "Preprocessing variants:"
Write-Host "  - resize_only          (no enhancement)"
Write-Host "  - blur_clahe2          (Gaussian blur + CLAHE clipLimit=2)"
Write-Host "  - sharp_clahe4         (Sharpening + CLAHE clipLimit=4)"
# Write-Host "  - blur_clahe2_notebook (Notebook-based preprocessing)"
Write-Host ""
Write-Host "Warning: This will create 4 times 4 = 16 processed datasets"
Write-Host "   This may take significant time and disk space."
Write-Host ""

# Prompt user to continue
$Continueary = Read-Host "Continue with preprocessing? (y/n)"
if ($Continueary -notmatch "^[Yy]$") {
    Write-Host "Skipping preprocessing variants."
    Write-Host ""
    Write-Host "You can still run individual preprocessing scripts:"
    Write-Host "  python scripts/preprocessing/preprocess_dataset.py --dataset <name>"
    Write-Host ""
    Write-Host "Or skip to: .\scripts\pipelines\step_5_create_splits.ps1"
    exit 0
}

Write-Host ""
Write-Host "--------------------------------------------------------"
Write-Host "4.1 Preprocessing balanced cropped knees (5-class)..."
Write-Host "--------------------------------------------------------"

python scripts/preprocessing/preprocess_balanced.py --dataset knees_cropped
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to preprocess knees_cropped" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "--------------------------------------------------------"
Write-Host "4.2 Preprocessing balanced cropped knees (4-class)..."
Write-Host "--------------------------------------------------------"

python scripts/preprocessing/preprocess_balanced.py --dataset knees_cropped_4_class
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to preprocess knees_cropped_4_class" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "--------------------------------------------------------"
Write-Host "4.3 Preprocessing balanced cropped knees (8-class)..."
Write-Host "--------------------------------------------------------"

python scripts/preprocessing/preprocess_balanced.py --dataset knees_cropped_8_class
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to preprocess knees_cropped_8_class" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "--------------------------------------------------------"
Write-Host "4.4 Preprocessing balanced cropped knees (10-class)..."
Write-Host "--------------------------------------------------------"

python scripts/preprocessing/preprocess_balanced.py --dataset knees_cropped_10_class
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to preprocess knees_cropped_10_class" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "--------------------------------------------------------"
Write-Host "4.5 Preprocessing balanced full X-rays (5-class)..."
Write-Host "--------------------------------------------------------"

python scripts/preprocessing/preprocess_balanced.py --dataset full_xray
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to preprocess full_xray" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "--------------------------------------------------------"
Write-Host "4.6 Preprocessing balanced full X-rays (4-class)..."
Write-Host "--------------------------------------------------------"

python scripts/preprocessing/preprocess_balanced.py --dataset full_xray_4_class
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to preprocess full_xray_4_class" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "--------------------------------------------------------"
Write-Host "4.7 Preprocessing balanced full X-rays (8-class)..."
Write-Host "--------------------------------------------------------"

python scripts/preprocessing/preprocess_balanced.py --dataset full_xray_8_class
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to preprocess full_xray_8_class" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "--------------------------------------------------------"
Write-Host "4.8 Preprocessing balanced full X-rays (10-class)..."
Write-Host "--------------------------------------------------------"

python scripts/preprocessing/preprocess_balanced.py --dataset full_xray_10_class
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to preprocess full_xray_10_class" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================================"
Write-Host "Step 4 Complete!"
Write-Host "========================================================"
Write-Host ""
Write-Host "Summary:"
Write-Host "  [OK] Preprocessed 8 balanced datasets x 4 variants = 32 processed datasets"
Write-Host ""
Write-Host "Output structure:"
Write-Host "  datasets/processed_balanced/"
Write-Host "    |-- knees_cropped/       # 5-class cropped knees"
Write-Host "    |   |-- resize_only/"
Write-Host "    |   |-- blur_clahe2/"
Write-Host "    |   |-- sharp_clahe4/"
# Write-Host "    |   +-- blur_clahe2_notebook/"
Write-Host "    |-- knees_cropped_4_class/"
Write-Host "    |-- knees_cropped_8_class/"
Write-Host "    |-- knees_cropped_10_class/"
Write-Host "    |-- full_xray/           # 5-class full x-rays"
Write-Host "    |-- full_xray_4_class/"
Write-Host "    |-- full_xray_8_class/"
Write-Host "    +-- full_xray_10_class/"
Write-Host ""
Write-Host "Next: Run .\scripts\pipelines\step_5_create_splits.ps1 to create train/val/test splits"
