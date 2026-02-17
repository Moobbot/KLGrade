# CADA (Context-Aware Deformable Attention) Architecture

## Overview
CADA is the current state-of-the-art architecture for KL grading in the KLGrade project. It uses deformable attention mechanisms to better capture spatial relationships between context and lesion features.

## Architecture Components

### 1. KiocmilModelCADA
Main model implementing the CADA architecture with:
- **YOLO11L Backbone**: Feature extraction from images
- **Context Encoder**: Multi-scale context feature extraction
- **Deformable Attention**: Spatial sampling with learned offsets
- **Cross-Attention**: Context-lesion feature interaction
- **Lesion Instance Aggregation**: MIL-based lesion pooling
- **Fusion Transformer**: Multi-modal feature fusion

### 2. Attention Modules
Specialized attention components:
- `DeformableAttention`: Learns spatial sampling offsets
- `CrossAttentionWithDeformable`: Cross-attention between context and lesions
- `ContextEncoder`: Multi-scale context feature extraction
- `LesionInstanceAggregation`: Aggregates multiple lesion instances
- `FusionTransformer`: Fuses context and lesion features
- `PositionalEncoding`: Adds spatial position information

## Usage

### Training
```bash
python src/training/cada/train.py \
    --train_img_dir processed/knee_10_class/images \
    --train_knee_label_dir processed/knee_10_class/labels_knee \
    --train_lesion_label_dir processed/knee_10_class/labels_lesion \
    --train_split_file splits/knee_10_class/train.txt \
    --val_img_dir processed/knee_10_class/images \
    --val_knee_label_dir processed/knee_10_class/labels_knee \
    --val_lesion_label_dir processed/knee_10_class/labels_lesion \
    --val_split_file splits/knee_10_class/val.txt \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-4
```

### Evaluation
```bash
python src/training/cada/evaluate.py \
    --checkpoint runs/kiocmil_cada/best_model.pt \
    --img_dir processed/knee_10_class/images \
    --knee_label_dir processed/knee_10_class/labels_knee \
    --lesion_label_dir processed/knee_10_class/labels_lesion \
    --split_file splits/knee_10_class/test.txt
```

### Inference
```python
from src.models.cada import KiocmilModelCADA

model = KiocmilModelCADA(
    backbone_name="yolo11l",
    num_classes=10,
    feature_dim=256,
    num_deformable_points=4,
)

# Load checkpoint
checkpoint = torch.load("runs/kiocmil_cada/best_model.pt")
model.load_state_dict(checkpoint)

# Inference
output = model(batch_data)
predictions = output["logits_10"].argmax(dim=1)
```

## Key Features

### Deformable Attention
- Learns spatial offsets for each attention point
- Adapts to irregular lesion shapes and positions
- More flexible than fixed grid sampling

### Multi-Scale Context
- Extracts features at multiple scales (3 levels)
- Captures both local details and global context
- Improves robustness to scale variations

### Hierarchical Classification
- Main task: 10-class KL grading (0a, 0b, 1a, 1b, ..., 4a, 4b)
- Auxiliary tasks: 5-class grade (0-4) and binary type (a/b)
- Multi-task learning improves overall performance

## Model Checkpoints
Checkpoints are saved in `runs/kiocmil_cada/`:
- `best_acc_model.pt`: Best validation accuracy
- `best_model.pt`: Best validation loss (early stopping)
- `checkpoint_epoch_N.pt`: Periodic checkpoints every N epochs

## Configuration

### Recommended Hyperparameters
```python
# Model
backbone_name = "yolo11l"
num_classes = 10  # or 4, 5, 8 depending on task
feature_dim = 256
num_deformable_points = 4
num_context_scales = 3

# Training
epochs = 100
batch_size = 16
learning_rate = 1e-4
weight_decay = 0.01

# Data Augmentation
augmentation_level = "strong"  # none, light, strong
use_clahe = True
```

### Loss Configuration
```python
# Hierarchical loss for 10-class
loss = 0.5 * loss_main + 0.3 * loss_grade + 0.2 * loss_type

# Simple loss for 4/5-class
loss = cross_entropy(logits, targets)
```

## Notes
- CADA is the most complex architecture with best potential accuracy
- Requires significant GPU memory (recommend 16GB+)
- Training time: ~2-3 hours per epoch on RTX 2080
- Use mixed precision training to reduce memory usage

## References
- Original KIOCMIL paper (if applicable)
- Deformable DETR: Deformable Transformers for End-to-End Object Detection
- YOLO11: Latest YOLO architecture for feature extraction
