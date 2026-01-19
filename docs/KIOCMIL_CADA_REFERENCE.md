# KIOCMIL CADA - Complete Reference Guide

**Context-Aware Deformable Attention (CADA) for Knee Osteoarthritis Grading**

This document consolidates all documentation for the KIOCMIL CADA project, replacing the previous `INDEX`, `GUIDE`, `REPORT`, `PIPELINE`, and `VERSIONS` files. It serves as the single source of truth for architecture, implementation, training, and evaluation.

---

## 📚 Table of Contents
1.  [Overview & Problem Statement](#1-overview--problem-statement)
2.  [Quick Start](#2-quick-start)
3.  [Architecture (CADA)](#3-architecture-cada)
4.  [Pipeline & Data Preparation](#4-pipeline--data-preparation)
5.  [Implementation Details](#5-implementation-details)
6.  [Training & Usage](#6-training--usage)
7.  [Evaluation Results](#7-evaluation-results)
8.  [Legacy Versions (V1/V2)](#8-legacy-versions-v1v2)
9.  [Troubleshooting](#9-troubleshooting)

---

## 1. Overview & Problem Statement

**Problem**: The previous KIOCMIL V3 model plateaued at **~50% accuracy** due to:
-   **Loss of spatial information**: Bounding boxes (bboxes) were discarded after cropping.
-   **Max pooling limitations**: "Hard" pooling discarded 99% of lesion information.
-   **Lack of context**: Lesions were treated independently without considering the global knee context.
-   **Simple fusion**: Linear concatenation was insufficient for complex feature relationships.

**Solution**: **KIOCMIL CADA (Context-Aware Deformable Attention)**
An enhanced Multi-Instance Learning (MIL) architecture that:
-   ✅ **Preserves Spatial Info**: Encodes bbox coordinates using Positional Encoding.
-   ✅ **Adaptive Sampling**: Uses Deformable Attention to focus on extensive lesion boundaries.
-   ✅ **Context Interaction**: Cross-attention between global context and local lesions.
-   ✅ **Learned Aggregation**: Soft, weighted pooling instead of max pooling.
-   ✅ **Hierarchical Fusion**: Transformer-based fusion of context and lesion features.

**Expected Result**: **+36% accuracy improvement** (50% → ~68-85%).

---

## 2. Quick Start

### Prerequisites
-   **GPU**: 5-7GB memory (min), RTX 2080 (rec).
-   **Environment**: PyTorch, Ultralytics, WandB (optional).

### Step 1: Verify Data
Ensure your dataset follows the structure with bbox info:
```bash
processed/knee_10_class/
├── images/           # Full X-rays
├── labels_knee/      # Knee bboxes (YOLO format: class cx cy w h)
└── labels_lesion/    # Lesion bboxes (class cx cy w h)
```

### Step 2: Install Dependencies
```bash
pip install torch torchvision ultralytics wandb
```

### Step 3: Train
```bash
python src/training/train_kiocmil_cada.py \
    --train_img_dir processed/knee_10_class/images \
    --train_knee_label_dir processed/knee_10_class/labels_knee \
    --train_lesion_label_dir processed/knee_10_class/labels_lesion \
    --train_split_file splits/knee_10_class/train.txt \
    --val_img_dir processed/knee_10_class/images \
    --val_knee_label_dir processed/knee_10_class/labels_knee \
    --val_lesion_label_dir processed/knee_10_class/labels_lesion \
    --val_split_file splits/knee_10_class/val.txt \
    --epochs 100 \
    --batch_size 16
```

---

## 3. Architecture (CADA)

**Model File**: `src/models/kiocmil_model_cada.py`

The architecture consists of 6 key modules:

1.  **Positional Encoding**: Encodes normalized bbox [cx, cy, w, h] into sinusoidal embeddings.
2.  **Context Encoder**: Extracts multi-scale features from the global knee image.
3.  **Backbone (YOLO11l)**: Extracts local features from lesion patches.
4.  **Deformable Cross-Attention**:
    -   *Query*: Lesion features.
    -   *Key/Value*: Context features.
    -   *Mechanism*: Adaptive sampling offsets to align lesions with relevant context.
5.  **Lesion Instance Aggregation**: Learned attention pooling to aggregate multiple variable-count lesions into a single representation per type (Osteophytes/JS).
6.  **Fusion Transformer**: Fuses Context, Aggregated Osteophyte, and Aggregated JS features.

### Comparison: V3 vs CADA
| Feature | V3 (Old) | CADA (New) | Benefit |
| :--- | :--- | :--- | :--- |
| **Spatial Info** | Discarded | Preserved (PosEnc) | Model knows *where* the lesion is |
| **Pooling** | Max Pooling | Learned Attention | Interpretable, no info loss |
| **Context** | Unused | Cross-Attention | Lesions inform context & vice versa |
| **Fusion** | Linear MLP | Transformer | Better feature integration |

---

## 4. Pipeline & Data Preparation

### 4.1 Data Pipeline Steps (Full X-ray)
1.  **Generate 10-Class Labels**: Convert raw YOLO labels to specific 10-class labels (KL0-a to KL4-b).
    ```bash
    python tools/check_dataset/class_split_report.py --labels-dir ... --save-dir ...
    ```
2.  **Dataset Splits**: Generate stratified Train/Val/Test splits.
    ```bash
    python scripts/data_preparation/split_dataset.py ...
    ```
3.  **Preprocessing**: Resize, CLAHE, Blur (optional but recommended for robustness).
    -   See `scripts/data_preparation/preprocess_knee_dataset.py`.

### 4.2 Handling Balanced Data
-   Oversample minority classes (flip augmentation).
-   Ensure separate `labels-knee` and `labels_lesion` directories exist for balanced datasets.

---

## 5. Implementation Details

### File Structure
```
src/
├── models/
│   ├── attention_modules.py          # Core CADA mechanisms (PosEnc, DeformableAttn)
│   ├── kiocmil_model_cada.py         # Main CADA model assembly
│   ├── kiocmil_model_v3.py           # Legacy V3 model
│   └── ...
├── datasets/
│   ├── kiocmil_dataset_v3.py         # CADA-compatible dataset (loads bboxes)
│   └── ...
└── training/
    ├── train_kiocmil_cada.py         # CADA training script
    └── ...
```

### Key Classes
-   **`KiocmilDatasetV3`**: Returns dictionary with `knees` list, containing `ctx` (context image), `ctx_bbox`, `js` (patches), `js_bboxes`, `ost`, `ost_bboxes`.
-   **`KiocmilModelCADA`**: The `nn.Module` implementing the full forward pass.

---

## 6. Training & Usage

### Best Practices
-   **Loss Function**: Weighted Multi-Task Loss works best.
    -   `Loss = 0.5 * CE(10-Class) + 0.3 * CE(Grade) + 0.2 * BCE(Type)`
-   **Learning Rate**: Use Cosine Annealing (`lr=1e-4` to `1e-6`).
-   **Augmentation**: Geometric (Flip/Shift) + Photometric (CLAHE/Blur).

### Monitoring
-   **WandB**: Highly recommended to track loss convergence and accuracy.
-   **Attention Weights**: CADA provides interpretable attention maps—visualize these to debug "where" the model is looking.

---

## 7. Evaluation Results

**Current Status (Jan 2026)**: Evaluation completed on all 8 experimental configurations.

| Experiment | Classes | Accuracy | Grade Acc (Derived) | Samples | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **cada_5class_unbalanced** | 5 | **80.65%** | N/A | 217 | ✅ Strong baseline |
| **cada_5class_balanced** | 5 | 78.21% | N/A | 234 | ✅ Good performance |
| **cada_4class_unbalanced** | 4 | **83.09%** | N/A | 207 | ✅ Best performance |
| **cada_4class_balanced** | 4 | 25.34% | N/A | 442 | ⚠️ Low accuracy (Data mix) |
| **cada_10class_unbalanced** | 10 | 56.68% | 58.53% | 217 | ⚠️ Hard task |
| **cada_10class_balanced** | 10 | 100.00% | 100.00% | 1401 | ❓ To be investigated (overfit?) |
| **cada_8class_unbalanced** | 8 | 52.17% | 53.14% | 207 | ⚠️ Hard task |
| **cada_8class_balanced** | 8 | 0.00% | 0.00% | 1097 | ❌ Data/Label Mismatch |

### Key Findings
1.  **4/5-Class Models Perform Best**: ~80-83% accuracy suggests the model handles the core grading task well when simplified.
2.  **Unbalanced > Balanced**: Surprisingly, unbalanced datasets performed better in key metrics, possibly due to artifacts in the balanced (augmented) data or label generation issues.
3.  **10-Class Complexity**: The 10-class task handling specific lesion combinations remains challenging (~57%).
4.  **Data Quality Issues**: The "Balanced" datasets for 4/8 classes show signs of label corruption or mismatch (0-25% accuracy), requiring regeneration of those specific label sets.

---

## 8. Legacy Versions (V1/V2)

For historical reference or baseline comparison.

| Version | Accuracy | Features | Status |
| :--- | :--- | :--- | :--- |
| **V1** | 42-48% | Baseline, no augmentation | ✅ Stable |
| **V2** | 19-20% | Experimental Augmentation | ⚠️ Deprecated |

**V1 Files**: `train_kiocmil_v1.py`, `kiocmil_dataset_v1.py`.
**V2 Files**: `train_kiocmil_v2.py`, `kiocmil_dataset_v2.py`.

Use V1 if you need a lightweight, simple baseline.

---

## 9. Troubleshooting

### Common Issues
1.  **No Samples Evaluated / "Empty Dataset"**:
    -   **Cause**: Missing `labels-knee` or `labels_lesion` for the specific dataset split (especially augmented/balanced ones).
    -   **Fix**: Ensure `labels-knee` exists and contains files for *all* images in the split, including flipped ones.

2.  **CUDA Out of Memory**:
    -   **Cause**: Deformable attention requires more memory.
    -   **Fix**: Reduce `batch_size` (e.g., to 8) or decrease `feature_dim` (e.g., to 128).

3.  **NaN Loss**:
    -   **Cause**: Unstable gradients in attention or bbox coordinates outside [0, 1].
    -   **Fix**: Clip gradients (`clip_grad_norm_`), check bbox normalization, reduce learning rate.

4.  **Low Accuracy (Plateau)**:
    -   **Fix**: Verify that bboxes are loaded correctly (visualize them). Ensure loss weights are balanced.

---
**Report Generated By**: Antigravity Agent
