# YOLO-Based KIOCMIL Models

## Overview
YOLO-based models use YOLO backbones (YOLO11L) for feature extraction instead of traditional ResNet architectures. This provides better feature representations and improved performance.

## Models

### 1. KiocmilModel (V3)
Main YOLO-based KIOCMIL model with:
- **YOLO11L Backbone**: Extracts features from C5 layer (1024D)
- **Attention Pooling**: Multi-head attention for MIL aggregation
- **Fusion MLP**: Combines context, joint space, and osteophyte features
- **Multi-task Heads**: 10-class, 5-grade, and binary type classification

**File**: `kiocmil_yolo.py`

### 2. YOLOWithClassification
Simpler approach combining YOLO detection with classification head:
- **YOLO Detection**: Detects knees and lesions
- **Classification Head**: MLP for KL grade prediction
- **End-to-End**: Single model for detection + classification

**File**: `yolo_with_classification.py`

## Usage

### Training V3 (YOLO11L Backbone)
```bash
python src/training/yolo/train_v3.py \
    --img_dir dataset/dataset_v0/images \
    --knee_labels dataset/dataset_v0/labels-knee \
    --lesion_labels dataset/dataset_v0/labels_10_class \
    --train_split splits/knee_full_10_class/train.txt \
    --val_split splits/knee_full_10_class/val.txt \
    --backbone yolo11l \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-4
```

### Training V2 (ResNet + YOLO Dataset)
```bash
python src/training/yolo/train_v2.py \
    --img_dir dataset/dataset_v0/images \
    --knee_labels dataset/dataset_v0/labels-knee \
    --lesion_labels dataset/dataset_v0/labels_10_class \
    --train_split splits/knee_full_10_class/train.txt \
    --val_split splits/knee_full_10_class/val.txt \
    --backbone resnet18 \
    --epochs 50
```

### Inference
```python
from src.models.yolo import KiocmilModel

model = KiocmilModel(
    backbone_name="yolo11l",
    num_classes=10,
    feature_dim=256,
    use_attention=True,
)

# Load checkpoint
checkpoint = torch.load("runs/kiocmil_yolo/best_model.pt")
model.load_state_dict(checkpoint)

# Inference
output = model(batch_data)
predictions = output["logits_10"].argmax(dim=1)
```

## Architecture Details

### YOLO11L Backbone
- **Input**: (B, 3, H, W) images
- **Output**: (B, 1024, H/32, W/32) features
- **Layers**: First 10 layers (up to C5)
- **Pretrained**: Uses YOLO11L pretrained weights

### Feature Extraction Pipeline
1. **Backbone**: Extract features from context and lesion patches
2. **Projection**: Project 1024D → 256D features
3. **Pooling**: Max pooling for lesion instances (MIL)
4. **Fusion**: Concatenate [context, JS, OST] → MLP
5. **Aggregation**: Attention pooling over knees
6. **Classification**: Multi-task heads (10-class, grade, type)

### Attention Pooling
- **Multi-head Attention**: 4 heads for diverse patterns
- **Learnable Query**: Single query attends to all knees
- **Feature Refinement**: MLP for enhanced representations

## Model Checkpoints
Checkpoints saved in `runs/kiocmil_yolo/`:
- `best_model.pt`: Best validation accuracy
- `last_model.pth`: Latest checkpoint
- `early_stop_best.pth`: Early stopping checkpoint

## Configuration

### Recommended Hyperparameters
```python
# Model
backbone_name = "yolo11l"
num_classes = 10  # or 4, 5, 8
feature_dim = 256
use_attention = True

# Training
epochs = 50
batch_size = 4  # YOLO11L is large
learning_rate = 1e-4
weight_decay = 1e-4

# Data Augmentation
augmentation_level = "strong"
use_clahe = True
```

### Loss Configuration
```python
# Hierarchical loss for 10-class
loss = 0.5 * loss_10class + 0.3 * loss_grade + 0.2 * loss_type
```

## Notes
- YOLO backbone provides better features than ResNet
- Larger model size (~50M params) requires more memory
- Use `batch_size=4` for YOLO11L to fit in GPU memory
- Enable mixed precision training for memory efficiency

## Comparison with Other Architectures

| Feature | ResNet | YOLO | CADA |
|---------|--------|------|------|
| Speed | Fast | Medium | Slow |
| Complexity | Simple | Medium | Complex |
| Memory | Low | Medium | High |

## Tips
- Use `batch_size=4` for YOLO11L (large model)
- Enable `use_attention=True` for better MIL aggregation
- Try `feature_dim=256` for good balance of performance/speed
- Use mixed precision training to reduce memory usage
