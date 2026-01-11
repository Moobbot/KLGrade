# Training Guide: YOLO & DETR

This guide covers training procedures for the KLGrade Knee OA Detection project, ranging from baseline experiments to enhanced training with augmentation and class balancing.

---

## 🏗️ 1. Basic Training (Baseline)

Baseline training uses standard datasets without advanced preprocessing to establish a performance benchmark.

### Experiments
- **E001**: 5-class Baseline (KL0-4)
- **E004**: 10-class Baseline (Split A/B)
- **E010**: DETR Baseline (Comparison)

### Running Baseline Training

```bash
# YOLO
bash docs/scripts/TRAINING_COMMANDS_YOLO_DATASET_V0.sh

# DETR
bash docs/scripts/TRAINING_COMMANDS_DETR_WANDB.sh
```

### Manual Command Example
```bash
python scripts/training/train_yolo.py \
    --config configs/yolo_5_class_baseline.yaml \
    --epochs 100 \
    --device 0
```

---

## 🚀 2. Enhanced Training

Enhanced training applies medical-specific preprocessing and class balancing to improved performance, especially on minority classes (KL0, KL4).

### Key Techniques
1.  **CLAHE**: Contrast Limited Adaptive Histogram Equalization to enhance X-ray details.
2.  **Gaussian Blur**: Reduces sensor noise.
3.  **Flip Balancing**: Horizontally flips minority class images to balance the dataset.
4.  **Label Scaling**: Ensures bboxes remain accurate after resizing.

### Running Enhanced Training

```bash
# Run full enhanced suite
bash docs/TRAINING_ENHANCED.sh
```

### Script Usage (`train_yolo_enhanced.py`)
```bash
python scripts/training/train_yolo_enhanced.py \
  --img_dir dataset/dataset_v0/images \
  --label_dir dataset/dataset_v0/labels \
  --split_dir splits/dataset_v0 \
  --num_classes 5 \
  --epochs 100 \
  --name experiment_enhanced
```

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--no-preprocessing` | Disable CLAHE/Blur | `False` |
| `--no-balancing` | Disable flip balancing | `False` |

---

## 🧪 3. Augmentation Configuration

We use **Albumentations** for robust, medically-safe data augmentation.

### Conservative Augmentation (Recommended)
*Defined in `src/datasets/augmentation.py`*

| Transform | Probability | Details |
| :--- | :--- | :--- |
| **Horizontal Flip** | 50% | Anatomically valid for knees |
| **Rotation** | 50% | ±5° (Safe limit) |
| **Shift/Scale** | 50% | ±10% |
| **Brightness/Contrast** | 50% | ±20% |
| **Gaussian Noise** | 20% | Visibly realistic sensor noise |

### ❌ Prohibited Augmentations
- **Vertical Flip**: Anatomically incorrect.
- **Heavy Rotation (>20°)**: Unrealistic.
- **Mixup/Mosaic**: Can obscure crucial osteophyte features.

---

## 📈 4. Advanced: Class Imbalance Strategy

### Instance-Aware Repeat Factor Sampling (IRFS)
Implemented in `src/datasets/samplers.py`. Increases sampling frequency for rare classes based on the formula:
`r_i = max(1, sqrt(t / f_i))`

### Implementation Steps
1.  **Sampler**: Uses `RecallFactorSampler` in DataLoader.
2.  **Loss**: Optional Focal Loss integration (enabled via `--focal-loss` in DETR training).

---

## 📊 5. Evaluation & Monitoring

### Metrics
- **mAP50**: Primary metric.
- **mAP50-95**: Strict metric.
- **Per-class AP**: Critical for analyzing minority class performance.

### WandB Integration
All scripts support WandB logging automatically.
- Dashboard: [Link to WandB Project](https://wandb.ai/ngotam2k1-thuyloi-university/KLGrade-Knee-OA)
- See `docs/guides/WANDB.md` for setup.
