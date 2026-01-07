# Quick Training Scripts for KLGrade Object Detection
# Run these commands in PowerShell with .venv activated

# ============================================
# EXPERIMENT 1: Baseline (5 Classes)
# ============================================

# Quick test (5 epochs)
.venv\Scripts\python.exe examples\train_yolo11.py `
    --img_dir dataset\dataset_v1\images `
    --label_dir dataset\dataset_v1\labels `
    --model yolo11n.pt `
    --epochs 5 `
    --batch 4 `
    --device 0 `
    --name exp1_baseline_test

# Full training (100 epochs)
.venv\Scripts\python.exe examples\train_yolo11.py `
    --img_dir dataset\dataset_v1\images `
    --label_dir dataset\dataset_v1\labels `
    --model yolo11n.pt `
    --epochs 100 `
    --batch 16 `
    --device 0 `
    --name exp1_baseline_5classes

# ============================================
# EXPERIMENT 2: Fine-grained (10 Classes)
# ============================================

# Quick test (5 epochs)
.venv\Scripts\python.exe examples\train_yolo11.py `
    --img_dir dataset\dataset_v1\images `
    --label_dir dataset\dataset_v1\labels `
    --use_labels_new `
    --model yolo11n.pt `
    --epochs 5 `
    --batch 4 `
    --device 0 `
    --name exp2_finegrained_test

# Full training (100 epochs)
.venv\Scripts\python.exe examples\train_yolo11.py `
    --img_dir dataset\dataset_v1\images `
    --label_dir dataset\dataset_v1\labels `
    --use_labels_new `
    --model yolo11n.pt `
    --epochs 100 `
    --batch 16 `
    --device 0 `
    --name exp2_finegrained_10classes

# ============================================
# EXPERIMENT 3: Filtered Optimal (7 Classes)
# ============================================

# Quick test (5 epochs)
.venv\Scripts\python.exe examples\train_yolo11.py `
    --img_dir dataset\dataset_filtered\images `
    --label_dir dataset\dataset_filtered\labels `
    --use_filtered `
    --model yolo11n.pt `
    --epochs 5 `
    --batch 4 `
    --device 0 `
    --name exp3_filtered_test

# Full training (100 epochs)
.venv\Scripts\python.exe examples\train_yolo11.py `
    --img_dir dataset\dataset_filtered\images `
    --label_dir dataset\dataset_filtered\labels `
    --use_filtered `
    --model yolo11n.pt `
    --epochs 100 `
    --batch 16 `
    --device 0 `
    --name exp3_filtered_7classes

# ============================================
# NOTES:
# ============================================
# - Model variants: yolo11n.pt (nano), yolo11s.pt (small), yolo11m.pt (medium)
# - Adjust --batch based on GPU memory
# - Results saved to: runs/detect/<name>/
# - Best weights: runs/detect/<name>/weights/best.pt
