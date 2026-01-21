# Visualization Code Organization

## Current Structure

### ✅ Shared Utilities (Reusable)

#### `src/utils/visualization.py` 
**Purpose**: Model evaluation plots  
**Used by**: All model evaluation scripts  
**Functions**:
- `plot_confusion_matrix()` - Standard CM
- `plot_confusion_matrix_normalized()` - Normalized CM  
- `plot_roc_curve()` - ROC curves
- `plot_precision_recall_curve()` - PR curves
- `plot_metric_curves()` - F1, Precision, Recall bars
- `plot_label_distribution()` - Label distribution
- `plot_training_curves()` - Training history

#### `src/utils/dataset_viz.py` ✨ NEW
**Purpose**: Dataset visualization utilities  
**Used by**: Dataset analysis tools  
**Functions**:
- `draw_yolo_boxes()` - Draw bounding boxes on images
- `plot_class_distribution()` - Class distribution charts
- `create_image_grid()` - Grid of images
- `plot_box_size_distribution()` - Box size histograms
- `compare_distributions()` - Compare multiple datasets

---

### 🎯 Specific Implementations

#### `src/visualization/` (Model Interpretation)
- `gradcam.py` - GradCAM heatmaps for model interpretation
- Future: `predictions.py`, `attention.py`, etc.

#### `tools/dataset_analysis/visualize/` (Dataset Analysis)
- `augmentations.py` - Augmentation visualization (uses `dataset_viz` utils)
- `samples.py` - Sample visualization (uses `dataset_viz` utils)
- `class_samples.py` - Class-wise samples (uses `dataset_viz` utils)

#### `scripts/evaluation/` (Evaluation Workflows)
- `visualize_predictions.py` - Prediction visualization workflows

---

## Design Pattern

```
┌─────────────────────────────────────────┐
│     SHARED UTILITIES (Reusable)         │
│  src/utils/visualization.py             │ ← Model eval plots
│  src/utils/dataset_viz.py               │ ← Dataset viz utils
└─────────────────────────────────────────┘
                    ↑
                    │ Use
                    │
┌─────────────────────────────────────────┐
│   SPECIFIC IMPLEMENTATIONS              │
│  tools/dataset_analysis/visualize/      │ ← Dataset analysis
│  src/visualization/                     │ ← Model interpretation
│  scripts/evaluation/                    │ ← Workflows
└─────────────────────────────────────────┘
```

**Principle**: 
- **Shared utils** = Low-level, reusable plotting functions
- **Specific implementations** = High-level, domain-specific visualizations that USE shared utils

---

## Usage Examples

### Using Shared Utils in Dataset Analysis

```python
from src.utils.dataset_viz import draw_yolo_boxes, plot_class_distribution

# In tools/dataset_analysis/visualize/samples.py
img_with_boxes = draw_yolo_boxes(image, boxes, class_names)
```

### Using Shared Utils in Model Evaluation

```python
from src.utils.visualization import plot_confusion_matrix, plot_roc_curve

# In src/training/evaluate_kiocmil_cada.py  
plot_confusion_matrix(cm, classes, save_path)
plot_roc_curve(targets, probs, n_classes, class_names, save_path)
```

---

**Status**: Organized ✅  
**Last Updated**: 2026-01-21
