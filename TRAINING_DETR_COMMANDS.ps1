# DETR Training Commands for KLGrade Object Detection
# Run these commands in PowerShell with .venv activated

# ============================================
# EXPERIMENT 1: DETR Baseline (5 Classes)
# ============================================

# Quick test (5 epochs) - ResNet-50
.venv\Scripts\python.exe examples\train_detr.py `
    --img_dir dataset/dataset_v1/images `
    --label_dir dataset/dataset_v1/labels `
    --model facebook/detr-resnet-50 `
    --epochs 5 `
    --batch 4 `
    --lr 1e-4 `
    --device cuda `
    --output runs/detr/exp1_test_5classes

# Full training (50 epochs) - ResNet-50
.venv\Scripts\python.exe examples\train_detr.py `
    --img_dir dataset/dataset_v1/images `
    --label_dir dataset/dataset_v1/labels `
    --model facebook/detr-resnet-50 `
    --epochs 50 `
    --batch 4 `
    --lr 1e-4 `
    --device cuda `
    --output runs/detr/exp1_detr_5classes

# Extended training (100 epochs) - ResNet-50
.venv\Scripts\python.exe examples\train_detr.py `
    --img_dir dataset/dataset_v1/images `
    --label_dir dataset/dataset_v1/labels `
    --model facebook/detr-resnet-50 `
    --epochs 100 `
    --batch 4 `
    --lr 1e-4 `
    --device cuda `
    --output runs/detr/exp1_detr_5classes_100ep

# ============================================
# EXPERIMENT 2: DETR Fine-grained (10 Classes)
# ============================================

# Quick test (5 epochs)
.venv\Scripts\python.exe examples\train_detr.py `
    --img_dir dataset/dataset_v1/images `
    --label_dir dataset/dataset_v1/labels `
    --use_labels_new `
    --model facebook/detr-resnet-50 `
    --epochs 5 `
    --batch 4 `
    --lr 1e-4 `
    --device cuda `
    --output runs/detr/exp2_test_10classes

# Full training (50 epochs)
.venv\Scripts\python.exe examples\train_detr.py `
    --img_dir dataset/dataset_v1/images `
    --label_dir dataset/dataset_v1/labels `
    --use_labels_new `
    --model facebook/detr-resnet-50 `
    --epochs 50 `
    --batch 4 `
    --lr 1e-4 `
    --device cuda `
    --output runs/detr/exp2_detr_10classes

# Extended training (100 epochs)
.venv\Scripts\python.exe examples\train_detr.py `
    --img_dir dataset/dataset_v1/images `
    --label_dir dataset/dataset_v1/labels `
    --use_labels_new `
    --model facebook/detr-resnet-50 `
    --epochs 100 `
    --batch 4 `
    --lr 1e-4 `
    --device cuda `
    --output runs/detr/exp2_detr_10classes_100ep

# ============================================
# EXPERIMENT 3: DETR with ResNet-101 (More powerful)
# ============================================

# Full training (50 epochs) - 5 Classes
.venv\Scripts\python.exe examples\train_detr.py `
    --img_dir dataset/dataset_v1/images `
    --label_dir dataset/dataset_v1/labels `
    --model facebook/detr-resnet-101 `
    --epochs 50 `
    --batch 2 `
    --lr 1e-4 `
    --device cuda `
    --output runs/detr/exp3_resnet101_5classes

# Full training (50 epochs) - 10 Classes
.venv\Scripts\python.exe examples\train_detr.py `
    --img_dir dataset/dataset_v1/images `
    --label_dir dataset/dataset_v1/labels `
    --use_labels_new `
    --model facebook/detr-resnet-101 `
    --epochs 50 `
    --batch 2 `
    --lr 1e-4 `
    --device cuda `
    --output runs/detr/exp3_resnet101_10classes

# ============================================
# EXPERIMENT 4: Lower Learning Rate (Fine-tuning)
# ============================================

# Fine-tuning with lower LR (5 classes)
.venv\Scripts\python.exe examples\train_detr.py `
    --img_dir dataset/dataset_v1/images `
    --label_dir dataset/dataset_v1/labels `
    --model facebook/detr-resnet-50 `
    --epochs 50 `
    --batch 4 `
    --lr 5e-5 `
    --device cuda `
    --output runs/detr/exp4_lower_lr_5classes

# Fine-tuning with lower LR (10 classes)
.venv\Scripts\python.exe examples\train_detr.py `
    --img_dir dataset/dataset_v1/images `
    --label_dir dataset/dataset_v1/labels `
    --use_labels_new `
    --model facebook/detr-resnet-50 `
    --epochs 50 `
    --batch 4 `
    --lr 5e-5 `
    --device cuda `
    --output runs/detr/exp4_lower_lr_10classes

# ============================================
# NOTES & RECOMMENDATIONS:
# ============================================
# 
# MODEL VARIANTS:
# - facebook/detr-resnet-50:  ~41M params, cân bằng tốc độ/độ chính xác
# - facebook/detr-resnet-101: ~60M params, chậm hơn nhưng mạnh hơn
#
# BATCH SIZE vs GPU MEMORY:
# - Batch 2:  ~4-6GB VRAM
# - Batch 4:  ~8-10GB VRAM
# - Batch 8:  ~16GB VRAM
#
# EPOCHS RECOMMENDATION:
# - Quick test:    5-10 epochs (kiểm tra setup)
# - Development:   50 epochs (training thông thường)
# - Production:    100+ epochs (maximize performance)
#
# LEARNING RATE TUNING:
# - Default:       1e-4 (0.0001) - good starting point
# - Lower:         5e-5 (0.00005) - if loss oscillates
# - Higher:        2e-4 (0.0002) - if loss decreases too slowly
#
# OUTPUT STRUCTURE:
# - Best model:       runs/detr/<name>/best_model.pt
# - Checkpoints:      runs/detr/<name>/checkpoint_epoch_*.pt
# - COCO annotations: processed/coco/annotations_train*.json
#
# AUTOMATIC FEATURES:
# ✅ Auto convert YOLO → COCO format
# ✅ Auto save best model based on validation loss
# ✅ Auto save checkpoint every 10 epochs
# ✅ Progress bar with loss tracking
# ✅ Learning rate scheduling (StepLR)
#

# ============================================
# EVALUATION COMMANDS
# ============================================
# Evaluate trained DETR models to get COCO metrics (mAP, precision, recall)
# Similar to YOLO evaluation

# Evaluate Experiment 1 (5 classes)
.venv\Scripts\python.exe examples\evaluate_detr.py `
    --model_path runs/detr/exp1_test_5classes/best_model.pt `
    --img_dir dataset/dataset_v1/images `
    --label_dir dataset/dataset_v1/labels `
    --conf_threshold 0.5 `
    --device cuda `
    --output runs/detr/exp1_test_5classes/evaluation

# Evaluate Experiment 2 (10 classes)
.venv\Scripts\python.exe examples\evaluate_detr.py `
    --model_path runs/detr/exp2_detr_10classes/best_model.pt `
    --img_dir dataset/dataset_v1/images `
    --label_dir dataset/dataset_v1/labels `
    --use_labels_new `
    --conf_threshold 0.5 `
    --device cuda `
    --output runs/detr/exp2_detr_10classes/evaluation

# Evaluate with different confidence threshold
.venv\Scripts\python.exe examples\evaluate_detr.py `
    --model_path runs/detr/exp1_detr_5classes/best_model.pt `
    --img_dir dataset/dataset_v1/images `
    --label_dir dataset/dataset_v1/labels `
    --conf_threshold 0.3 `
    --device cuda `
    --output runs/detr/exp1_detr_5classes/evaluation_conf03

# ============================================
# EVALUATION OUTPUT:
# ============================================
# - metrics.json:  Complete COCO metrics in JSON format
#   - mAP50-95 (main metric, similar to YOLO mAP50-95)
#   - mAP50 (similar to YOLO mAP50)
#   - Per-class metrics
#   - Average Recall (AR) metrics
#
# - results.txt:   Human-readable summary
#   - Overall metrics table
#   - Per-class breakdown
#

# ============================================
# ERROR ANALYSIS COMMANDS
# ============================================
# Detailed error analysis with confusion matrix, false positives/negatives
# Works with both YOLO and DETR predictions

# Analyze DETR predictions
.venv\Scripts\python.exe examples\error_analysis.py `
    --predictions runs/detr/exp1_test_5classes/evaluation/predictions.json `
    --ground_truth processed/coco/annotations_val.json `
    --iou_threshold 0.5 `
    --conf_threshold 0.25 `
    --output runs/detr/exp1_test_5classes/error_analysis

# Analyze with different IoU threshold
.venv\Scripts\python.exe examples\error_analysis.py `
    --predictions runs/detr/exp1_test_5classes/evaluation/predictions.json `
    --ground_truth processed/coco/annotations_val.json `
    --iou_threshold 0.75 `
    --conf_threshold 0.25 `
    --output runs/detr/exp1_test_5classes/error_analysis_iou75

# ============================================
# ERROR ANALYSIS OUTPUT:
# ============================================
# - error_analysis.csv:  Detailed per-image and per-class statistics
#   - Image-level: GT count, Pred count, TP, FP, FN, Precision, Recall
#   - Class-level: Per-class metrics breakdown
#
# - error_report.txt:    Human-readable error analysis
#   - Overall statistics (TP, FP, FN rates)
#   - Error breakdown (correct, false positives, false negatives, classification errors)  
#   - Per-class statistics table
#   - Confusion matrix (which classes are confused)
#
# - statistics.json:     Raw statistics for programmatic analysis
#   - Confusion matrix data
#   - Error examples
#   - Per-class breakdowns
#


