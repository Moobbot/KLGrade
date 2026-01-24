# Script to evaluate all KIOCMIL CADA experiments
# Usage: .\scripts\training\evaluate_all_cada_experiments.ps1

$ErrorActionPreference = "Stop"

# Activate conda environment
Write-Host "Activating conda environment..." -ForegroundColor Cyan
conda activate klgrade_api
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Warning: Failed to activate klgrade_api env" -ForegroundColor Yellow
}

Write-Host "========================================================"
Write-Host "STARTING EVALUATION OF ALL KIOCMIL CADA EXPERIMENTS"
Write-Host "========================================================"

# Function to run evaluation
function Evaluate-Experiment {
    param(
        [string]$ExpName,
        [int]$NumClasses,
        [string]$ImgDir,
        [string]$SplitDir,
        [string]$KneeLabelDir,
        [string]$LesionLabelDir
    )
    
    $ModelPath = "runs/kiocmil_cada/$ExpName/best_acc_model.pt"
    
    # Check if best_acc_model exists, fallback to best_model if not
    if (-not (Test-Path $ModelPath)) {
        $ModelPath = "runs/kiocmil_cada/$ExpName/best_model.pt"
    }

    if (-not (Test-Path $ModelPath)) {
        Write-Host "❌ Model not found for $ExpName : $ModelPath" -ForegroundColor Red
        return
    }
    
    Write-Host ""
    Write-Host "--------------------------------------------------------"
    Write-Host "Evaluating Experiment: $ExpName"
    Write-Host "Model: $ModelPath"
    Write-Host "Classes: $NumClasses"
    Write-Host "--------------------------------------------------------"

    python src/training/evaluate_kiocmil_cada.py `
        --model_path "$ModelPath" `
        --num_classes $NumClasses `
        --img_dir "$ImgDir" `
        --split_file "$SplitDir/val.txt" `
        --knee_label_dir "$KneeLabelDir" `
        --lesion_label_dir "$LesionLabelDir" `
        --save_dir "runs/kiocmil_cada/$ExpName/evaluation" `
        --batch_size 16
}

# ============================================================
# 1. 10-Class Experiments
# ============================================================
Evaluate-Experiment `
    -ExpName "cada_10class_unbalanced" `
    -NumClasses 10 `
    -ImgDir "datasets/dataset_v0/images" `
    -SplitDir "datasets/splits/knee_full_10_class" `
    -KneeLabelDir "datasets/dataset_v0/labels-knee" `
    -LesionLabelDir "datasets/dataset_v0/labels_10_class"

Evaluate-Experiment `
    -ExpName "cada_10class_balanced" `
    -NumClasses 10 `
    -ImgDir "datasets/balanced/full_xray_10_class/images" `
    -SplitDir "datasets/splits/balanced_full_xray_10_class" `
    -KneeLabelDir "datasets/balanced/full_xray_10_class/labels-knee" `
    -LesionLabelDir "datasets/balanced/full_xray_10_class/labels"

# Balanced + Resize Only
Evaluate-Experiment `
    -ExpName "cada_10class_balanced_resize" `
    -NumClasses 10 `
    -ImgDir "datasets/processed_balanced/full_xray/resize_only/images" `
    -SplitDir "datasets/splits/balanced_full_xray_10_class" `
    -KneeLabelDir "datasets/processed_balanced/full_xray/resize_only/labels-knee" `
    -LesionLabelDir "datasets/processed_balanced/full_xray/resize_only/labels_10_class"

# Balanced + Blur CLAHE
Evaluate-Experiment `
    -ExpName "cada_10class_balanced_blur" `
    -NumClasses 10 `
    -ImgDir "datasets/processed_balanced/full_xray/blur_clahe2/images" `
    -SplitDir "datasets/splits/balanced_full_xray_10_class" `
    -KneeLabelDir "datasets/processed_balanced/full_xray/blur_clahe2/labels-knee" `
    -LesionLabelDir "datasets/processed_balanced/full_xray/blur_clahe2/labels_10_class"

# Balanced + Sharp CLAHE
Evaluate-Experiment `
    -ExpName "cada_10class_balanced_sharp" `
    -NumClasses 10 `
    -ImgDir "datasets/processed_balanced/full_xray/sharp_clahe4/images" `
    -SplitDir "datasets/splits/balanced_full_xray_10_class" `
    -KneeLabelDir "datasets/processed_balanced/full_xray/sharp_clahe4/labels-knee" `
    -LesionLabelDir "datasets/processed_balanced/full_xray/sharp_clahe4/labels_10_class"

# ============================================================
# 2. 5-Class Experiments
# ============================================================
Evaluate-Experiment `
    -ExpName "cada_5class_unbalanced" `
    -NumClasses 5 `
    -ImgDir "datasets/dataset_v0/images" `
    -SplitDir "datasets/splits/knee_full_10_class" `
    -KneeLabelDir "datasets/dataset_v0/labels-knee" `
    -LesionLabelDir "datasets/dataset_v0/labels"

Evaluate-Experiment `
    -ExpName "cada_5class_balanced" `
    -NumClasses 5 `
    -ImgDir "datasets/balanced/full_xray/images" `
    -SplitDir "datasets/splits/balanced_full_xray_10_class" `
    -KneeLabelDir "datasets/balanced/full_xray/labels-knee" `
    -LesionLabelDir "datasets/balanced/full_xray/labels"

# Balanced + Resize Only
Evaluate-Experiment `
    -ExpName "cada_5class_balanced_resize" `
    -NumClasses 5 `
    -ImgDir "datasets/processed_balanced/full_xray/resize_only/images" `
    -SplitDir "datasets/splits/balanced_full_xray_10_class" `
    -KneeLabelDir "datasets/processed_balanced/full_xray/resize_only/labels-knee" `
    -LesionLabelDir "datasets/processed_balanced/full_xray/resize_only/labels"

# Balanced + Blur CLAHE
Evaluate-Experiment `
    -ExpName "cada_5class_balanced_blur" `
    -NumClasses 5 `
    -ImgDir "datasets/processed_balanced/full_xray/blur_clahe2/images" `
    -SplitDir "datasets/splits/balanced_full_xray_10_class" `
    -KneeLabelDir "datasets/processed_balanced/full_xray/blur_clahe2/labels-knee" `
    -LesionLabelDir "datasets/processed_balanced/full_xray/blur_clahe2/labels"

# Balanced + Sharp CLAHE
Evaluate-Experiment `
    -ExpName "cada_5class_balanced_sharp" `
    -NumClasses 5 `
    -ImgDir "datasets/processed_balanced/full_xray/sharp_clahe4/images" `
    -SplitDir "datasets/splits/balanced_full_xray_10_class" `
    -KneeLabelDir "datasets/processed_balanced/full_xray/sharp_clahe4/labels-knee" `
    -LesionLabelDir "datasets/processed_balanced/full_xray/sharp_clahe4/labels"

# ============================================================
# 3. 4-Class Experiments
# ============================================================
Evaluate-Experiment `
    -ExpName "cada_4class_unbalanced" `
    -NumClasses 4 `
    -ImgDir "datasets/dataset_v0_4_class/images" `
    -SplitDir "datasets/splits/knee_full_4_class" `
    -KneeLabelDir "datasets/dataset_v0/labels-knee" `
    -LesionLabelDir "datasets/dataset_v0_4_class/labels"

Evaluate-Experiment `
    -ExpName "cada_4class_balanced" `
    -NumClasses 4 `
    -ImgDir "datasets/balanced/full_xray_4_class/images" `
    -SplitDir "datasets/splits/balanced_full_xray_4_class" `
    -KneeLabelDir "datasets/balanced/full_xray_4_class/labels-knee" `
    -LesionLabelDir "datasets/balanced/full_xray_4_class/labels"

# Balanced + Resize Only
Evaluate-Experiment `
    -ExpName "cada_4class_balanced_resize" `
    -NumClasses 4 `
    -ImgDir "datasets/processed_balanced/full_xray_4_class/resize_only/images" `
    -SplitDir "datasets/splits/balanced_full_xray_4_class" `
    -KneeLabelDir "datasets/processed_balanced/full_xray_4_class/resize_only/labels-knee" `
    -LesionLabelDir "datasets/processed_balanced/full_xray_4_class/resize_only/labels"

# Balanced + Blur CLAHE
Evaluate-Experiment `
    -ExpName "cada_4class_balanced_blur" `
    -NumClasses 4 `
    -ImgDir "datasets/processed_balanced/full_xray_4_class/blur_clahe2/images" `
    -SplitDir "datasets/splits/balanced_full_xray_4_class" `
    -KneeLabelDir "datasets/processed_balanced/full_xray_4_class/blur_clahe2/labels-knee" `
    -LesionLabelDir "datasets/processed_balanced/full_xray_4_class/blur_clahe2/labels"

# Balanced + Sharp CLAHE
Evaluate-Experiment `
    -ExpName "cada_4class_balanced_sharp" `
    -NumClasses 4 `
    -ImgDir "datasets/processed_balanced/full_xray_4_class/sharp_clahe4/images" `
    -SplitDir "datasets/splits/balanced_full_xray_4_class" `
    -KneeLabelDir "datasets/processed_balanced/full_xray_4_class/sharp_clahe4/labels-knee" `
    -LesionLabelDir "datasets/processed_balanced/full_xray_4_class/sharp_clahe4/labels"

# ============================================================
# 4. 8-Class Experiments
# ============================================================
Evaluate-Experiment `
    -ExpName "cada_8class_unbalanced" `
    -NumClasses 8 `
    -ImgDir "datasets/dataset_v0_4_class/images" `
    -SplitDir "datasets/splits/knee_full_8_class" `
    -KneeLabelDir "datasets/dataset_v0/labels-knee" `
    -LesionLabelDir "datasets/dataset_v0_4_class/labels_8_class"

Evaluate-Experiment `
    -ExpName "cada_8class_balanced" `
    -NumClasses 8 `
    -ImgDir "datasets/balanced/full_xray_8_class/images" `
    -SplitDir "datasets/splits/balanced_full_xray_8_class" `
    -KneeLabelDir "datasets/balanced/full_xray_8_class/labels-knee" `
    -LesionLabelDir "datasets/balanced/full_xray_8_class/labels"

# Balanced + Resize Only
Evaluate-Experiment `
    -ExpName "cada_8class_balanced_resize" `
    -NumClasses 8 `
    -ImgDir "datasets/processed_balanced/full_xray_8_class/resize_only/images" `
    -SplitDir "datasets/splits/balanced_full_xray_8_class" `
    -KneeLabelDir "datasets/processed_balanced/full_xray_8_class/resize_only/labels-knee" `
    -LesionLabelDir "datasets/processed_balanced/full_xray_8_class/resize_only/labels"

# Balanced + Blur CLAHE
Evaluate-Experiment `
    -ExpName "cada_8class_balanced_blur" `
    -NumClasses 8 `
    -ImgDir "datasets/processed_balanced/full_xray_8_class/blur_clahe2/images" `
    -SplitDir "datasets/splits/balanced_full_xray_8_class" `
    -KneeLabelDir "datasets/processed_balanced/full_xray_8_class/blur_clahe2/labels-knee" `
    -LesionLabelDir "datasets/processed_balanced/full_xray_8_class/blur_clahe2/labels"

# Balanced + Sharp CLAHE
Evaluate-Experiment `
    -ExpName "cada_8class_balanced_sharp" `
    -NumClasses 8 `
    -ImgDir "datasets/processed_balanced/full_xray_8_class/sharp_clahe4/images" `
    -SplitDir "datasets/splits/balanced_full_xray_8_class" `
    -KneeLabelDir "datasets/processed_balanced/full_xray_8_class/sharp_clahe4/labels-knee" `
    -LesionLabelDir "datasets/processed_balanced/full_xray_8_class/sharp_clahe4/labels"

Write-Host "========================================================"
Write-Host "ALL EVALUATIONS COMPLETED"
Write-Host "========================================================"
