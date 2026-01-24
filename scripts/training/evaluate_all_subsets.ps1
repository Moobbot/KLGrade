# Script to evaluate model on Train, Val, and Full Dataset
# Usage: .\scripts\training\evaluate_all_subsets.ps1

$ErrorActionPreference = "Stop"

# Activate conda environment
Write-Host "Activating conda environment..." -ForegroundColor Cyan
try {
    conda activate klgrade_api
} catch {
    Write-Host "⚠️  Warning: Failed to activate klgrade_api env" -ForegroundColor Yellow
}

Write-Host "========================================================"
Write-Host "COMPREHENSIVE EVALUATION (Train / Val / Full)"
Write-Host "========================================================"

# --- CONFIGURATION ---
# Experiment settings (Modify these for different experiments)
$ExpName = "cada_10class_balanced"
$NumClasses = 10
$ImgDir = "datasets/balanced/full_xray_10_class/images"
$SplitDir = "datasets/splits/balanced_full_xray_10_class"
$KneeLabelDir = "datasets/balanced/full_xray_10_class/labels-knee"
$LesionLabelDir = "datasets/balanced/full_xray_10_class/labels"

$ModelPath = "runs/kiocmil_cada/$ExpName/best_acc_model.pt"

# Check model existence
if (-not (Test-Path $ModelPath)) {
    $ModelPath = "runs/kiocmil_cada/$ExpName/best_model.pt"
}
if (-not (Test-Path $ModelPath)) {
    Write-Host "❌ Model not found for $ExpName : $ModelPath" -ForegroundColor Red
    exit 1
}

Write-Host "Experiment: $ExpName"
Write-Host "Model: $ModelPath"
Write-Host ""

# Function to run evaluation
function Run-Eval {
    param(
        [string]$Subset,
        [string]$SplitFile,
        [string]$SaveSuffix
    )

    $SaveDir = "runs/kiocmil_cada/$ExpName/evaluation_$SaveSuffix"
    
    Write-Host "--------------------------------------------------------"
    Write-Host "Evaluating on: $Subset"
    Write-Host "Split File: $SplitFile"
    Write-Host "Output: $SaveDir"
    Write-Host "--------------------------------------------------------"

    python src/training/evaluate_kiocmil_cada.py `
        --model_path "$ModelPath" `
        --num_classes $NumClasses `
        --img_dir "$ImgDir" `
        --split_file "$SplitFile" `
        --knee_label_dir "$KneeLabelDir" `
        --lesion_label_dir "$LesionLabelDir" `
        --save_dir "$SaveDir" `
        --batch_size 16

    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Evaluation failed for $Subset" -ForegroundColor Red
    } else {
        Write-Host "✅ Evaluation complete for $Subset" -ForegroundColor Green
    }
    Write-Host ""
}

# 1. Evaluate on Validation Set (Standard)
Run-Eval -Subset "Validation Set" -SplitFile "$SplitDir/val.txt" -SaveSuffix "val"

# 2. Evaluate on Training Set
Run-Eval -Subset "Training Set" -SplitFile "$SplitDir/train.txt" -SaveSuffix "train"

# 3. Evaluate on Full Dataset (Balanced)
# Note: Passing a non-existent file path causes the dataset loader to fallback to loading ALL images in img_dir
Run-Eval -Subset "Full Dataset (Balanced)" -SplitFile "USE_ALL_IMAGES_IN_DIR" -SaveSuffix "full_balanced"

# 4. Evaluate on Original Unbalanced Dataset (dataset_v0)
Write-Host "--------------------------------------------------------"
Write-Host "Evaluating on: Original Unbalanced Dataset (v0)"
Write-Host "Output: runs/kiocmil_cada/$ExpName/evaluation_v0_unbalanced"
Write-Host "--------------------------------------------------------"

# Define v0 paths
$V0_ImgDir = "datasets/dataset_v0/images"
$V0_KneeLabelDir = "datasets/dataset_v0/labels-knee"
$V0_LesionLabelDir = "datasets/dataset_v0/labels_10_class" # Assuming 10-class model
if ($NumClasses -eq 5) { $V0_LesionLabelDir = "datasets/dataset_v0/labels" }

python src/training/evaluate_kiocmil_cada.py `
    --model_path "$ModelPath" `
    --num_classes $NumClasses `
    --img_dir "$V0_ImgDir" `
    --split_file "USE_ALL_IMAGES_IN_DIR" `
    --knee_label_dir "$V0_KneeLabelDir" `
    --lesion_label_dir "$V0_LesionLabelDir" `
    --save_dir "runs/kiocmil_cada/$ExpName/evaluation_v0_unbalanced" `
    --batch_size 16

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Evaluation failed for dataset_v0" -ForegroundColor Red
} else {
    Write-Host "✅ Evaluation complete for dataset_v0" -ForegroundColor Green
}
Write-Host ""

Write-Host "========================================================"
Write-Host "ALL EVALUATIONS COMPLETED"
Write-Host "========================================================"
