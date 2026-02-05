# KIOCMIL Two-Step Training

This folder contains scripts for the **two-step KIOCMIL training approach**:
1. **Step 1**: Train YOLO for knee/lesion detection
2. **Step 2**: Train KIOCMIL for KL grade classification using detected regions

## 📁 Contents

### Training Scripts
- **`run_all_cada_experiments.sh`** - Run all CADA experiments
- **`evaluate_all_cada_experiments.sh`** - Evaluate all trained models
- **`train_kiocmil.sh`** - Train KIOCMIL model
- **`train_kiocmil_v3_wandb.sh`** - Train with WandB logging
- **`train_cdt_cad.py`** - Train CDT-CAD variant

### Dataset-Specific Scripts
- **`run_processed_4_8_class.sh`** - 4-class and 8-class experiments
- **`run_processed_5_10_class.sh`** - 5-class and 10-class experiments
- **`run_train_4_5_class.sh`** - Training for 4/5 class configs
- **`run_train_8_10_class.sh`** - Training for 8/10 class configs

## 🚀 Quick Start

### Run All CADA Experiments
```bash
cd /path/to/KLGrade
./scripts/training/kiocmil_two_step/run_all_cada_experiments.sh
```

### Evaluate All Models
```bash
./scripts/training/kiocmil_two_step/evaluate_all_cada_experiments.sh
```

### Train Single Model
```bash
./scripts/training/kiocmil_two_step/train_kiocmil.sh
```

## 📊 Two-Step Pipeline

```
Step 1: YOLO Detection
    Input Image → YOLO → Knee Boxes + Lesion Boxes

Step 2: KIOCMIL Classification
    Cropped Knees + Lesions → KIOCMIL-CADA → KL Grade
```

## 🎯 Advantages vs End-to-End

**Two-Step Approach:**
- ✅ Modular: Train detection and classification separately
- ✅ Easier debugging: Isolate issues to specific components
- ✅ Flexible: Can swap detection/classification models
- ✅ Proven: KIOCMIL-CADA achieves good results

**End-to-End Approach:**
- ✅ Joint optimization: Detection and classification learn together
- ✅ Simpler deployment: Single model
- ❌ Harder to debug
- ❌ Requires more memory

## 📈 Best Results

**KIOCMIL-CADA (5-class):**
- Validation Accuracy: **42.47%**
- Best configuration: Processed dataset with augmentation

**KIOCMIL-CADA (10-class):**
- Validation Accuracy: **~35-40%**
- More challenging due to finer-grained classification

## 📝 Datasets Supported

- **5-class**: KL0, KL1, KL2, KL3, KL4
- **10-class**: KL0-JS, KL1-JS, ..., KL4-OST
- **4-class**: Balanced 4-class variant
- **8-class**: Balanced 8-class variant

## 🔗 Related

- Detection training: `../yolo_detection/`
- End-to-end training: `../end_to_end/`
