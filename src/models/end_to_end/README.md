# End-to-End Detection + Classification Models

## Overview
End-to-end models combine knee/lesion detection and KL grade classification in a single model, eliminating the need for a separate 2-stage pipeline.

## Approaches

### 1. KiocmilEndToEnd
KIOCMIL with integrated detection heads.

**Architecture**:
1. Shared YOLO11L backbone
2. Knee detection head → knee boxes
3. Lesion detection head → lesion boxes (per knee)
4. KIOCMIL-CADA classification → KL grade

**Advantages**:
- ✅ Single model for detection + classification
- ✅ End-to-end training possible
- ✅ Shared backbone reduces parameters

**Disadvantages**:
- ❌ Complex training (multi-task loss)
- ❌ Harder to debug
- ❌ Requires careful loss balancing

**File**: `kiocmil_end_to_end.py`

### 2. KiocmilWithDetection
KIOCMIL-CADA with detection capabilities.

**Architecture**:
- CADA model with added detection heads
- Can leverage pretrained CADA weights
- Simpler than full end-to-end approach

**File**: `kiocmil_with_detection.py`

### 3. YOLOWithClassification
YOLO model with classification head.

**Architecture**:
1. YOLO for detection (knees, JS, OST)
2. Classification head on knee features
3. Simpler than KIOCMIL attention

**Advantages**:
- ✅ Leverages proven YOLO detection
- ✅ Simpler architecture
- ✅ Easier to train

**Disadvantages**:
- ❌ Less sophisticated than CADA
- ❌ May lose accuracy on classification

**File**: `yolo_with_classification.py` (in yolo/)

## Usage

### Training End-to-End Model
```python
from src.models.end_to_end import KiocmilEndToEnd

model = KiocmilEndToEnd(
    backbone_name="yolo11l",
    num_classes=10,
    pretrained_kiocmil="runs/kiocmil_cada/best_model.pt",
    pretrained_knee_detector="runs/yolo_knee/best.pt",
)

# Training loop
for images, targets in dataloader:
    output = model(images, mode="train")
    
    # Multi-task loss
    loss_detection = detection_loss(
        output["knee_boxes"], 
        output["knee_confs"],
        targets["knee_boxes"]
    )
    
    loss_classification = classification_loss(
        output["logits_10"],
        targets["labels"]
    )
    
    total_loss = loss_detection + loss_classification
    total_loss.backward()
```

### Inference
```python
model.eval()
with torch.no_grad():
    output = model(images, mode="inference")
    
    # Detection results
    knee_boxes = output["knee_boxes"]
    knee_confs = output["knee_confs"]
    
    # Classification results
    predictions = output["logits_10"].argmax(dim=1)
```

## Implementation Status

### ✅ Implemented
- Detection head architecture
- Model structure
- Forward pass logic

### ⚠️ Partial
- Training script (needs multi-task loss)
- Pretrained weight loading
- Evaluation metrics

### ❌ Not Implemented
- Full training pipeline
- Loss balancing strategy
- Comprehensive evaluation

## Multi-Task Loss Design

### Proposed Loss Function
```python
total_loss = (
    α * loss_knee_detection +      # Knee box regression + confidence
    β * loss_lesion_detection +    # Lesion box regression + confidence
    γ * loss_classification        # KL grade classification
)

# Suggested weights
α = 1.0  # Knee detection
β = 0.5  # Lesion detection (harder task)
γ = 2.0  # Classification (main task)
```

### Loss Components
1. **Knee Detection**: IoU loss + BCE for confidence
2. **Lesion Detection**: IoU loss + BCE for confidence + class loss
3. **Classification**: Cross-entropy for KL grade

## Training Strategy

### Stage 1: Pretrain Components
```bash
# 1. Train YOLO knee detector
python train_yolo_knee.py

# 2. Train KIOCMIL classifier (with GT boxes)
python src/training/cada/train.py

# 3. Train YOLO lesion detector
python train_yolo_lesion.py
```

### Stage 2: End-to-End Fine-tuning
```bash
# Fine-tune entire model end-to-end
python train_end_to_end.py \
    --pretrained_kiocmil runs/kiocmil_cada/best_model.pt \
    --pretrained_knee_detector runs/yolo_knee/best.pt \
    --freeze_backbone  # Optional: freeze backbone initially
```

## Configuration

### Recommended Hyperparameters
```python
# Model
backbone_name = "yolo11l"
num_classes = 10
pretrained_kiocmil = "runs/kiocmil_cada/best_model.pt"
pretrained_knee_detector = "runs/yolo_knee/best.pt"

# Training
epochs = 50
batch_size = 4
learning_rate = 1e-4

# Loss weights
alpha = 1.0  # Knee detection
beta = 0.5   # Lesion detection
gamma = 2.0  # Classification (main task)
```

## Notes
- End-to-end models are experimental
- Require careful loss balancing
- Training is more complex than 2-stage pipeline
- Consider 2-stage approach for production use

## Future Work

### Improvements
1. **Attention-based Detection**: Use deformable attention for detection
2. **Multi-scale Features**: Leverage FPN for better detection
3. **Cascade Refinement**: Iterative box refinement
4. **Uncertainty Estimation**: Confidence calibration

### Research Directions
1. Compare end-to-end vs 2-stage pipeline
2. Ablation study on loss weights
3. Impact of pretrained weights
4. Generalization to other datasets

## References
- DETR: End-to-End Object Detection with Transformers
- Deformable DETR: Deformable Transformers for End-to-End Object Detection
- YOLO series: Real-time object detection
