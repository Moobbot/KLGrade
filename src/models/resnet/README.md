# ResNet-Based KIOCMIL Models

## Overview
ResNet-based models are the baseline KIOCMIL architecture using traditional ResNet backbones (ResNet18/50) for feature extraction. These models are lightweight, fast, and serve as a strong baseline.

## Models

### KiocmilModel (Base)
Baseline KIOCMIL model with:
- **ResNet Backbone**: ResNet18 or ResNet50
- **Attention Pooling**: Multi-head attention for MIL
- **Fusion MLP**: Combines context, JS, and OST features
- **Multi-task Heads**: 10-class, 5-grade, and binary type

**File**: `kiocmil_resnet.py`

## Usage

### Training V1 (Original)
```bash
python src/training/resnet/train_v1.py \
    --img_dir dataset/dataset_v0/images \
    --knee_labels dataset/dataset_v0/labels-knee \
    --lesion_labels dataset/dataset_v0/labels_10_class \
    --train_split splits/knee_full_10_class/train.txt \
    --val_split splits/knee_full_10_class/val.txt \
    --backbone resnet18 \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-4
```

### Inference
```python
from src.models.resnet import KiocmilModel

model = KiocmilModel(
    backbone_name="resnet18",  # or "resnet50"
    num_classes=10,
    feature_dim=256,
    use_attention=True,
)

# Load checkpoint
checkpoint = torch.load("runs/kiocmil_exp1/best_model.pth")
model.load_state_dict(checkpoint)

# Inference
output = model(batch_data)
predictions = output["logits_10"].argmax(dim=1)
```

## Architecture Details

### ResNet Backbones

#### ResNet18
- **Parameters**: ~11M
- **Feature Dim**: 512D
- **Speed**: Fast (~200 images/sec)
- **Use Case**: Quick experiments, baseline

#### ResNet50
- **Parameters**: ~23M
- **Feature Dim**: 2048D
- **Speed**: Medium (~100 images/sec)
- **Use Case**: Better accuracy, production

### Feature Extraction Pipeline
1. **Backbone**: Extract features from patches
   - Context: (3, 384, 384) → (512,) or (2048,)
   - Lesions: (N, 3, 224, 224) → (N, 512) or (N, 2048)
2. **Projection**: Project to 256D
3. **Pooling**: Max pooling for lesion instances
4. **Fusion**: Concat [ctx, js, ost] → MLP → 256D
5. **Aggregation**: Attention over knees → 256D
6. **Classification**: 3 heads (10-class, grade, type)

### Attention Pooling
```python
class AttentionPool(nn.Module):
    - Multi-head attention (4 heads)
    - Learnable query vector
    - Feature refinement MLP
    - Returns: pooled features + attention weights
```

## Model Checkpoints
Saved in `runs/kiocmil_resnet/`:
- `best_model.pth`: Best validation accuracy
- `last_model.pth`: Latest checkpoint
- `early_stop_best.pth`: Early stopping checkpoint

## Configuration

### Recommended Hyperparameters
```python
# Model
backbone_name = "resnet18"  # or "resnet50"
num_classes = 10  # or 4, 5, 8
feature_dim = 256
use_attention = True

# Training
epochs = 50
batch_size = 8  # ResNet18: 8, ResNet50: 4
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
- ResNet models are lightweight and fast
- Good baseline for quick experiments
- ResNet18 for speed, ResNet50 for better accuracy
- Lower memory requirements than YOLO/CADA

## Comparison with Other Architectures

| Feature | ResNet | YOLO | CADA |
|---------|--------|------|------|
| Speed | Fast | Medium | Slow |
| Complexity | Simple | Medium | Complex |
| Memory | Low | Medium | High |

## When to Use
- **Quick experiments**: ResNet18 for fast iteration
- **Baseline**: Establish performance floor
- **Resource-constrained**: Limited GPU memory
- **Production (simple)**: When speed > accuracy
