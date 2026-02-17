# Model Evaluation Results

> **Note**: This file will be populated with actual evaluation results after running comprehensive experiments. Currently contains placeholder structure.

## Overview
This document tracks evaluation results for all KIOCMIL architectures across different configurations and datasets.

## Evaluation Datasets

### Test Sets
- **10-class**: `splits/knee_10_class/test.txt`
- **8-class**: `splits/knee_8_class/test.txt`
- **5-class**: `splits/knee_5_class/test.txt`
- **4-class**: `splits/knee_4_class/test.txt`

### Balanced Variants
- `processed_balanced_10_class/`
- `processed_balanced_8_class/`
- `processed_balanced_5_class/`
- `processed_balanced_4_class/`

---

## CADA Architecture

### 10-Class Results
| Metric | Value | Notes |
|--------|-------|-------|
| Accuracy | TBD | - |
| Macro F1 | TBD | - |
| Weighted F1 | TBD | - |
| Training Time | TBD | per epoch |
| Inference Speed | TBD | images/sec |

### 5-Class Results
| Metric | Value | Notes |
|--------|-------|-------|
| Accuracy | TBD | - |
| Macro F1 | TBD | - |
| Weighted F1 | TBD | - |

### Configuration Used
```python
backbone = "yolo11l"
num_classes = 10
feature_dim = 256
batch_size = 16
learning_rate = 1e-4
epochs = 100
```

---

## YOLO Architecture

### 10-Class Results
| Metric | Value | Notes |
|--------|-------|-------|
| Accuracy | TBD | - |
| Macro F1 | TBD | - |
| Weighted F1 | TBD | - |
| Training Time | TBD | per epoch |
| Inference Speed | TBD | images/sec |

### Configuration Used
```python
backbone = "yolo11l"
num_classes = 10
feature_dim = 256
batch_size = 4
learning_rate = 1e-4
epochs = 50
```

---

## ResNet Architecture

### ResNet18 - 10-Class Results
| Metric | Value | Notes |
|--------|-------|-------|
| Accuracy | TBD | - |
| Macro F1 | TBD | - |
| Weighted F1 | TBD | - |
| Training Time | TBD | per epoch |
| Inference Speed | TBD | images/sec |

### ResNet50 - 10-Class Results
| Metric | Value | Notes |
|--------|-------|-------|
| Accuracy | TBD | - |
| Macro F1 | TBD | - |
| Weighted F1 | TBD | - |

### Configuration Used
```python
backbone = "resnet18"  # or "resnet50"
num_classes = 10
feature_dim = 256
batch_size = 8  # ResNet18: 8, ResNet50: 4
learning_rate = 1e-4
epochs = 50
```

---

## Architecture Comparison

### Performance Summary (10-Class)
| Architecture | Accuracy | Macro F1 | Training Time | Inference Speed | GPU Memory |
|--------------|----------|----------|---------------|-----------------|------------|
| CADA | TBD | TBD | TBD | TBD | ~16GB |
| YOLO | TBD | TBD | TBD | TBD | ~12GB |
| ResNet18 | TBD | TBD | TBD | TBD | ~6GB |
| ResNet50 | TBD | TBD | TBD | TBD | ~8GB |

### Speed vs Accuracy Trade-off
```
                Accuracy
                   ↑
                   |
        CADA       |
                   |
           YOLO    |
                   |
      ResNet50     |
                   |
    ResNet18       |
                   |
                   └─────────────────→ Speed
```

---

## Detailed Results by Class

### Confusion Matrices
> To be added after evaluation

### Per-Class Metrics
> To be added after evaluation

---

## Ablation Studies

### CADA Components
| Configuration | Accuracy | Notes |
|---------------|----------|-------|
| Full CADA | TBD | All components |
| w/o Deformable Attention | TBD | - |
| w/o Cross-Attention | TBD | - |
| w/o Multi-scale Context | TBD | - |

### Data Augmentation Impact
| Augmentation Level | Accuracy | Notes |
|-------------------|----------|-------|
| None | TBD | - |
| Light | TBD | - |
| Strong | TBD | - |

### Loss Function Comparison
| Loss Type | Accuracy | Notes |
|-----------|----------|-------|
| Hierarchical (0.5/0.3/0.2) | TBD | Current best |
| Simple CE | TBD | - |
| Focal Loss | TBD | - |

---

## Training Curves
> To be added: Loss curves, accuracy curves, learning rate schedules

---

## Hardware Requirements

### Minimum Requirements
- **GPU**: RTX 2080 (8GB VRAM)
- **RAM**: 16GB
- **Storage**: 50GB

### Recommended
- **GPU**: RTX 3090 / A100 (24GB VRAM)
- **RAM**: 32GB
- **Storage**: 100GB SSD

---

## Evaluation Commands

### CADA
```bash
python src/training/cada/evaluate.py \
    --checkpoint runs/kiocmil_cada/best_model.pt \
    --img_dir processed/knee_10_class/images \
    --knee_label_dir processed/knee_10_class/labels_knee \
    --lesion_label_dir processed/knee_10_class/labels_lesion \
    --split_file splits/knee_10_class/test.txt \
    --output_dir evaluation_results/cada_10class
```

### YOLO
```bash
python src/training/yolo/evaluate.py \
    --checkpoint runs/kiocmil_yolo/best_model.pt \
    --img_dir dataset/dataset_v0/images \
    --knee_labels dataset/dataset_v0/labels-knee \
    --lesion_labels dataset/dataset_v0/labels_10_class \
    --test_split splits/knee_full_10_class/test.txt \
    --output_dir evaluation_results/yolo_10class
```

### ResNet
```bash
python src/training/evaluate_kiocmil.py \
    --checkpoint runs/kiocmil_resnet/best_model.pth \
    --img_dir dataset/dataset_v0/images \
    --knee_labels dataset/dataset_v0/labels-knee \
    --lesion_labels dataset/dataset_v0/labels_10_class \
    --test_split splits/knee_full_10_class/test.txt \
    --output_dir evaluation_results/resnet_10class
```

---

## Notes
- All results to be updated after running comprehensive evaluation
- Use consistent test sets across all architectures
- Report both balanced and imbalanced dataset results
- Include confidence intervals where applicable
