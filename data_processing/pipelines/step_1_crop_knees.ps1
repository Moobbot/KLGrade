# Step 1: Crop Knee Regions from Full X-rays
#
# Input:  datasets/dataset_v0/
# Output: datasets/dataset_knees_cropped/
#
# Note: Label variants (10-class, 4-class, 8-class) will be generated in Step 2

$ErrorActionPreference = "Stop"

# Activate conda environment
Write-Host "Activating conda environment..." -ForegroundColor Cyan
try {
    conda activate klgrade_api
} catch {
    Write-Host "⚠️  Warning: Failed to activate klgrade_api env" -ForegroundColor Yellow
}

Write-Host "========================================================"
Write-Host "Step 1: Crop Knee Regions"
Write-Host "========================================================"
Write-Host ""
Write-Host "Input:  datasets/dataset_v0/ (full X-rays, 5-class labels)"
Write-Host "Output: datasets/dataset_knees_cropped/"
Write-Host ""
Write-Host "Note: This step only crops images. Label variants will be"
Write-Host "      generated in Step 2 from the cropped dataset."
Write-Host ""

Write-Host "--------------------------------------------------------"
Write-Host "1.1 Cropping knee regions from full X-rays..."
Write-Host "--------------------------------------------------------"

# Set PYTHONPATH for module imports if needed, but python usually handles current dir
python scripts/data_preparation/crop_knee_regions.py `
    --dataset_dir datasets/dataset_v0 `
    --output_dir datasets/dataset_knees_cropped `
    --margin 0.15 `
    --min_size 300

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to crop knee regions" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✅ Knee cropping complete"
Write-Host ""

# Check statistics
Write-Host "--------------------------------------------------------"
Write-Host "1.2 Dataset statistics..."
Write-Host "--------------------------------------------------------"

if (Test-Path "datasets/dataset_knees_cropped/images") {
    $imgCount = (Get-ChildItem -Path "datasets/dataset_knees_cropped/images" -Recurse -Include *.jpg,*.png).Count
    Write-Host "  Total cropped images: $imgCount"
    
    $labelCount = (Get-ChildItem -Path "datasets/dataset_knees_cropped/labels" -Recurse -Filter *.txt).Count
    Write-Host "  Total labels: $labelCount"
}

Write-Host ""

Write-Host "========================================================"
Write-Host "Step 1 Complete!"
Write-Host "========================================================"
Write-Host ""
Write-Host "Summary:"
Write-Host "  ✅ Cropped knees from full X-rays"
Write-Host "  ✅ Copied 5-class KL labels (transformed to crop space)"
Write-Host "  ✅ Filtered out crops with no labels"
Write-Host ""
Write-Host "Output structure:"
Write-Host "  datasets/dataset_knees_cropped/"
Write-Host "    ├── images/              # Cropped knee images (only with labels)"
Write-Host "    ├── labels/              # 5-class KL labels (KL0-4)"
Write-Host "    ├── labels-knee/         # Knee boxes (full crop)"
Write-Host "    └── crop_report.json     # Cropping statistics"
Write-Host ""
Write-Host "Note: Label variants (10-class, 4-class, 8-class) will be"
Write-Host "      generated in Step 2 for BOTH full X-rays and cropped knees."
Write-Host ""
Write-Host "Next: Run .\scripts\pipelines\step_2_generate_labels.ps1"
