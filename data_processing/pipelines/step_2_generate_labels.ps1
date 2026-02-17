# Step 2: Generate Label Variants for Both Datasets
#
# Input:  datasets/dataset_v0/labels/              (full X-rays, 5-class)
#         datasets/dataset_knees_cropped/labels/   (cropped knees, 5-class)
# Output: 10-class, 4-class, 8-class labels for BOTH datasets
#
# Logic:
#   For each dataset:
#     1. Generate 10-class labels from 5-class
#     2. Filter KL0 to create 4-class labels
#     3. Generate 8-class labels from 4-class labels

$ErrorActionPreference = "Stop"

# Activate conda environment
Write-Host "Activating conda environment..." -ForegroundColor Cyan
try {
    conda activate klgrade_api
} catch {
    Write-Host "⚠️  Warning: Failed to activate klgrade_api env" -ForegroundColor Yellow
}

Write-Host "========================================================"
Write-Host "Step 2: Generate Label Variants"
Write-Host "========================================================"
Write-Host ""
Write-Host "This step generates label variants for 2 datasets:"
Write-Host "  1. dataset_v0            (full X-rays)"
Write-Host "  2. dataset_knees_cropped (cropped knees)"
Write-Host ""

# ============================================================
# 2.1: Generate labels for FULL X-RAYS (dataset_v0)
# ============================================================
Write-Host "--------------------------------------------------------"
Write-Host "2.1 Generating labels for FULL X-RAYS (dataset_v0)..."
Write-Host "--------------------------------------------------------"
Write-Host ""

# 2.1.1 Generate 10-class
Write-Host "  2.1.1 Generating 10-class labels..."
python tools/dataset_analysis/class_split_report.py `
    --labels-dir datasets/dataset_v0/labels `
    --knee-labels-dir datasets/dataset_v0/labels-knee `
    --save-dir datasets/dataset_v0/labels_10_class `
    --limit 0

Write-Host "  ✅ 10-class: datasets/dataset_v0/labels_10_class/"

# 2.1.2 Generate 4-class
Write-Host ""
Write-Host "  2.1.2 Generating 4-class labels (filter KL0)..."
python scripts/data_preparation/filter_kl0.py `
    --input datasets/dataset_v0 `
    --output datasets/dataset_v0_4_class `
    --num_classes 5 `
    --classes 0

Write-Host "  ✅ 4-class: datasets/dataset_v0_4_class/labels/"

# 2.1.3 Generate 8-class
Write-Host ""
Write-Host "  2.1.3 Generating 8-class labels (from 4-class)..."
python tools/dataset_analysis/class_split_report.py `
    --labels-dir datasets/dataset_v0_4_class/labels `
    --knee-labels-dir datasets/dataset_v0/labels-knee `
    --save-dir datasets/dataset_v0_4_class/labels_8_class `
    --limit 0

Write-Host "  ✅ 8-class: datasets/dataset_v0_4_class/labels_8_class/"

Write-Host ""
Write-Host "✅ Full X-rays labels complete"
Write-Host ""

# ============================================================
# 2.2: Generate labels for CROPPED KNEES
# ============================================================
Write-Host "--------------------------------------------------------"
Write-Host "2.2 Generating labels for CROPPED KNEES..."
Write-Host "--------------------------------------------------------"
Write-Host ""

if (Test-Path "datasets/dataset_knees_cropped/labels") {
    # 2.2.1 Generate 10-class
    Write-Host "  2.2.1 Generating 10-class labels..."
    python tools/dataset_analysis/class_split_report.py `
        --labels-dir datasets/dataset_knees_cropped/labels `
        --save-dir datasets/dataset_knees_cropped/labels_10_class `
        --limit 0
    
    Write-Host "  ✅ 10-class: datasets/dataset_knees_cropped/labels_10_class/"
    
    # 2.2.2 Generate 4-class
    Write-Host ""
    Write-Host "  2.2.2 Generating 4-class labels (filter KL0)..."
    python scripts/data_preparation/filter_kl0.py `
        --input datasets/dataset_knees_cropped `
        --output datasets/dataset_knees_cropped_4_class `
        --num_classes 5 `
        --classes 0
    
    Write-Host "  ✅ 4-class: datasets/dataset_knees_cropped_4_class/labels/"
    
    # 2.2.3 Generate 8-class
    Write-Host ""
    Write-Host "  2.2.3 Generating 8-class labels (from 4-class)..."
    python tools/dataset_analysis/class_split_report.py `
        --labels-dir datasets/dataset_knees_cropped_4_class/labels `
        --save-dir datasets/dataset_knees_cropped_4_class/labels_8_class `
        --limit 0
    
    Write-Host "  ✅ 8-class: datasets/dataset_knees_cropped_4_class/labels_8_class/"
    
    Write-Host ""
    Write-Host "✅ Cropped knees labels complete"
} else {
    Write-Host "⚠️  Skipping: datasets/dataset_knees_cropped not found" -ForegroundColor Yellow
    Write-Host "   Run .\scripts\pipelines\step_1_crop_knees.ps1 first"
}

Write-Host ""

# ============================================================
# Summary
# ============================================================
Write-Host "========================================================"
Write-Host "Step 2 Complete!"
Write-Host "========================================================"
Write-Host ""
Write-Host "Summary:"
Write-Host ""
Write-Host "  Full X-rays:"
Write-Host "    [OK] datasets/dataset_v0/labels_10_class/          (10 classes)"
Write-Host "    [OK] datasets/dataset_v0_4_class/labels/           (4 classes)"
Write-Host "    [OK] datasets/dataset_v0_4_class/labels_8_class/   (8 classes)"
Write-Host ""
Write-Host "  Cropped Knees:"
Write-Host "    [OK] datasets/dataset_knees_cropped/labels_10_class/         (10 classes)"
Write-Host "    [OK] datasets/dataset_knees_cropped_4_class/labels/          (4 classes)"
Write-Host "    [OK] datasets/dataset_knees_cropped_4_class/labels_8_class/  (8 classes)"
Write-Host ""
Write-Host "Benefits:"
Write-Host "  - Consistent label variants across both datasets"
Write-Host "  - Can train on full X-rays OR cropped knees"
Write-Host "  - All class configurations available (5, 10, 4, 8)"
Write-Host ""
Write-Host "Next: Run .\scripts\pipelines\step_3_balance.ps1"
Write-Host "      (step_4 is optional for ablation studies)"
