# KIOCMIL CADA Implementation Guide

## 📌 Overview

This guide explains how to implement and train the **Context-Aware Deformable Attention (CADA)** enhanced KIOCMIL model to overcome the **50% accuracy plateau** of the current V3 model.

---

## 🎯 Problem Statement

### Current KIOCMIL V3 Issues

- **Accuracy plateau at ~50%**: Basic attention mechanism insufficient
- **Loss of spatial information**: Bboxes discarded after cropping
- **Fixed max pooling**: Cannot adapt to variable numbers of lesions
- **Linear fusion**: Simple concatenation + MLP insufficient for complex relationships
- **No context-lesion interaction**: Lesions treated independently

### CADA Solution

Implement a hierarchical attention mechanism that:

1. ✅ Preserves spatial information (bbox coordinates)
2. ✅ Uses deformable attention for adaptive spatial sampling
3. ✅ Applies cross-attention between context and lesions
4. ✅ Learns instance-level aggregation weights
5. ✅ Fuses information hierarchically

---

## 📂 File Structure

```
src/
├── models/
│   ├── attention_modules.py          ✨ NEW: Attention mechanisms
│   ├── kiocmil_model_cada.py        ✨ NEW: CADA model
│   ├── kiocmil_model_v3.py          (existing - reference)
│   └── kiocmil_model.py             (existing - reference)
├── datasets/
│   ├── kiocmil_dataset_v3.py        ✨ NEW: V3 dataset with bbox info
│   ├── kiocmil_dataset_v2.py        (existing)
│   └── kiocmil_transforms_v2.py     (existing)
└── training/
    ├── train_kiocmil_cada.py        ✨ NEW: CADA training script
    ├── train_kiocmil_v3.py          (existing - reference)
    └── focal_loss.py                 (existing - reuse)

docs/
└── KIOCMIL_ENHANCEMENT_REPORT.md    ✨ NEW: Detailed analysis
```

---

## 🚀 Quick Start

### Step 1: Prepare Data with Bbox Information

Your dataset should have:

```
processed/knee_10_class/
├── images/                      # Full X-ray images
├── labels_knee/                 # Knee bboxes (YOLO format)
└── labels_lesion/              # Lesion bboxes (YOLO format)

splits/knee_10_class/
├── train.txt                   # Image stems for training
├── val.txt                     # Image stems for validation
└── test.txt                    # Image stems for testing
```

**YOLO Format** (normalized coordinates):

```
class_id center_x center_y width height
```

Example:

```
0 0.45 0.35 0.30 0.40    # Knee at center (0.45, 0.35) with size (0.30, 0.40)
4 0.50 0.40 0.10 0.15    # JS lesion (class 4) at...
1 0.40 0.30 0.08 0.12    # Ost lesion (class 1) at...
```

### Step 2: Install Dependencies

```bash
# Core dependencies
pip install torch torchvision
pip install ultralytics

# Optional: WandB logging
pip install wandb
wandb login
```

### Step 3: Train CADA Model

```bash
# Basic training
python src/training/train_kiocmil_cada.py \
    --train_img_dir processed/knee_10_class/images \
    --train_knee_label_dir processed/knee_10_class/labels_knee \
    --train_lesion_label_dir processed/knee_10_class/labels_lesion \
    --train_split_file splits/knee_10_class/train.txt \
    --epochs 100 \
    --batch_size 16 \
    --augmentation_level strong \
    --use_clahe True

# With WandB logging
python src/training/train_kiocmil_cada.py \
    ... \
    --wandb_name "kiocmil_cada_experiment_001" \
    --wandb_project "klgrade-kiocmil"
```

### Step 4: Monitor Training

```bash
# View logs
tail -f log/kiocmil_cada_*/training.log

# Or in WandB dashboard:
# https://wandb.ai/your-entity/klgrade-kiocmil
```

---

## 🔧 Architecture Details

### Module 1: Positional Encoding

**File**: `src/models/attention_modules.py::PositionalEncoding`

Encodes spatial bbox information into embeddings using sinusoidal positional encoding.

```python
from src.models.attention_modules import PositionalEncoding

pos_encoder = PositionalEncoding(feature_dim=256, max_seq_len=1000)

# Encode bbox [cx, cy, w, h] (normalized 0-1)
bbox = torch.tensor([[0.5, 0.4, 0.3, 0.2]])
pos_emb = pos_encoder(bbox)  # (1, 256)
```

**Key Features**:

- Sinusoidal encoding (similar to Transformer)
- Handles variable bbox ranges
- Gradient-friendly

### Module 2: Deformable Attention

**File**: `src/models/attention_modules.py::DeformableAttention`

Learns adaptive spatial sampling offsets to focus on relevant context regions.

```python
from src.models.attention_modules import DeformableAttention

deform_attn = DeformableAttention(
    feature_dim=256,
    num_points=4,      # 4 sample points per lesion
    num_heads=4,
    dropout=0.1,
)

# Query: lesion embedding, Feature map: context spatial features, Bbox: lesion location
query = torch.randn(B, 256)
feature_map = torch.randn(B, 256, 96, 96)  # From context encoder
bbox_center = torch.tensor([[0.5, 0.4]])
output, weights = deform_attn(query, feature_map, bbox_center)
# output: (B, 256), weights: (B, 4)
```

**How it works**:

1. Generate learnable offset grids: Δp ∼ MLP(query)
2. Compute sampling locations: p_sample = bbox_center + Δp
3. Bilinear sampling from feature map
4. Weight aggregation: Σ(w_i \* f_sampled_i)

**Expected behavior**:

- Offsets converge to relevant regions (e.g., lesion boundaries)
- Weights concentrate on informative sample points
- 20-30% increase in computation (acceptable trade-off)

### Module 3: Cross-Attention with Deformable Sampling

**File**: `src/models/attention_modules.py::CrossAttentionWithDeformable`

Combines deformable attention with cross-attention for context interaction.

```python
cross_attn = CrossAttentionWithDeformable(
    feature_dim=256,
    num_points=4,
    num_heads=4,
)

lesion_query = torch.randn(B, 256)
context_feature_map = torch.randn(B, 256, 96, 96)
lesion_bbox = torch.tensor([[0.5, 0.4]])

contextualized_lesion, weights = cross_attn(
    lesion_query,
    context_feature_map,
    lesion_bbox,
)
# contextualized_lesion: (B, 256) with context awareness
```

**Key improvements over simple pooling**:

- Context features sampled adaptively
- Cross-attention learns feature relationships
- Residual connections for stability
- Layer normalization for training stability

### Module 4: Lesion Instance Aggregation

**File**: `src/models/attention_modules.py::LesionInstanceAggregation`

Aggregates multiple lesion instances using learned attention weights.

```python
lesion_agg = LesionInstanceAggregation(
    feature_dim=256,
    hidden_dim=128,
    num_heads=4,
)

# Multiple lesion features: (n_lesions, 256)
lesion_features = torch.randn(5, 256)
aggregated, weights = lesion_agg(lesion_features)
# aggregated: (256,) pooled representation
# weights: (5,) attention weights per lesion
```

**Advantages over max pooling**:

- Learns which lesions matter
- Soft aggregation preserves information
- Interpretable attention weights

### Module 5: Fusion Transformer

**File**: `src/models/attention_modules.py::FusionTransformer`

Combines context and lesion embeddings using transformer layers.

```python
fusion = FusionTransformer(
    feature_dim=256,
    num_heads=4,
    num_layers=2,
)

context_emb = torch.randn(B, 256)
js_agg = torch.randn(B, 256)
ost_agg = torch.randn(B, 256)

fused = fusion(context_emb, js_agg, ost_agg)
# fused: (B, 256) combined representation
```

**What it learns**:

- How to weight context vs lesions
- Cross-type lesion interactions
- Feature importance

### Module 6: Full CADA Model

**File**: `src/models/kiocmil_model_cada.py::KiocmilModelCADA`

Complete end-to-end model with all components.

```python
from src.models.kiocmil_model_cada import KiocmilModelCADA

model = KiocmilModelCADA(
    backbone_name="yolo11l",
    num_classes=10,
    feature_dim=256,
    num_deformable_points=4,
    num_context_scales=3,
    use_positional_encoding=True,
)

# Forward pass with batch data
output = model(batch_data)
# output['logits_10']: (B, 10) - 10-class predictions
# output['logits_grade']: (B, 5) - KL grade
# output['logits_type']: (B, 1) - lesion type
# output['embedding']: (B, 256) - image embedding
```

---

## 📊 Dataset Format

### V3 Dataset Structure

The `KiocmilDatasetV3` returns items with bbox information:

```python
{
    'knees': [
        {
            'ctx': torch.Tensor,              # (3, 384, 384) context patch
            'ctx_bbox': torch.Tensor,        # (4,) [cx, cy, w, h] normalized
            'js': torch.Tensor,              # (n, 3, 224, 224) JS lesion patches
            'js_bboxes': torch.Tensor,       # (n, 4) JS bboxes
            'ost': torch.Tensor,             # (m, 3, 224, 224) Ost lesion patches
            'ost_bboxes': torch.Tensor,      # (m, 4) Ost bboxes
        },
        # ... more knees
    ],
    'label': int,  # Image-level label
}
```

### Creating Custom Dataset

If you want to use different data format:

```python
from src.datasets.kiocmil_dataset_v3 import KiocmilDatasetV3

class MyCustomDataset(KiocmilDatasetV3):
    def __init__(self, ...):
        # Your initialization
        pass

    def __getitem__(self, idx):
        # Load your data
        # Return in CADA format:
        return {
            'knees': [
                {
                    'ctx': ctx_tensor,
                    'ctx_bbox': ctx_bbox,
                    'js': js_tensors,
                    'js_bboxes': js_bboxes,
                    'ost': ost_tensors,
                    'ost_bboxes': ost_bboxes,
                },
            ],
            'label': label,
        }
```

---

## 🎓 Training Best Practices

### 1. Learning Rate Schedule

```python
# Cosine annealing works well
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,           # Total epochs
    eta_min=1e-6,
)
```

### 2. Loss Function

Use weighted combination:

```
loss = 0.5 * loss_10_class + 0.3 * loss_grade + 0.2 * loss_type
```

Why this weighting:

- **loss_10_class** (50%): Primary task, directly optimizes for correct lesion type/severity
- **loss_grade** (30%): Grade (0-4) helps distinguish severity levels
- **loss_type** (20%): Type (Ost vs JS) provides additional constraint

### 3. Data Augmentation

Recommended for medical imaging:

```python
# Geometric (before cropping)
- Random horizontal flip
- Small shifts/scales/rotations
- NOT vertical flip (anatomy matters)

# Photometric (after cropping)
- CLAHE for contrast enhancement
- Random brightness/contrast adjustments
- Gaussian blur for noise robustness
```

### 4. Batch Size

- **GPU memory**: 16 (typical), 8 (limited memory), 32 (large GPU)
- **Gradient accumulation**: If batch < 16, consider accumulating
- **Deformable attention**: Slightly more memory than standard attention

### 5. Early Stopping

```python
early_stopping = EarlyStopping(
    patience=15,         # Stop after 15 epochs without improvement
    verbose=True,
)
```

---

## 🔍 Debugging & Visualization

### Check Deformable Attention Offsets

```python
# Add this to training loop to visualize offset learning
def visualize_deformable_offsets(model, batch_data, epoch):
    """Visualize where deformable attention samples from."""
    # Get deformable attention modules
    # Log offset statistics to WandB
    # Plot heatmaps of sampling locations
    pass
```

### Monitor Attention Weights

```python
# In training:
output = model(batch_data)
# output includes attention weights

# Log to WandB:
wandb.log({
    'js_attention_weights': wandb.Histogram(js_weights.cpu().numpy()),
    'ost_attention_weights': wandb.Histogram(ost_weights.cpu().numpy()),
})
```

### Check Gradient Flow

```python
# Add gradient clipping (already in script)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Monitor in WandB:
for name, param in model.named_parameters():
    if param.grad is not None:
        wandb.log({f'grad/{name}': wandb.Histogram(param.grad.cpu().numpy())})
```

---

## 📈 Expected Results

### Performance Improvements

| Metric         | V3 Baseline | CADA Expected | Improvement |
| -------------- | ----------- | ------------- | ----------- |
| Accuracy       | ~50%        | ~65-70%       | +15-20%     |
| mAP (10-class) | ~0.25-0.29  | ~0.40-0.50    | +50-70%     |
| Grade MAE      | ~1.2        | ~0.8          | -33%        |
| Type F1        | ~0.65       | ~0.80         | +23%        |

### Convergence Pattern

Expected training curves:

```
Epoch 1-10:    Loss drops 50%, accuracy jumps
Epoch 10-30:   Slower improvement, deformable offsets learning
Epoch 30-60:   Fine-tuning, attention weights stabilizing
Epoch 60-100:  Plateau, minor improvements
```

### Typical Training Time

- **1 GPU (RTX 3090)**: ~3-5 hours for 100 epochs, batch size 16
- **2 GPUs**: ~2-3 hours (with DataParallel)
- **CPU only**: ~24-48 hours (not recommended)

---

## 🚨 Common Issues & Solutions

### Issue 1: CUDA Out of Memory

**Solution**:

```python
# Reduce batch size
--batch_size 8

# Reduce feature dimension
feature_dim=128  # instead of 256

# Gradient accumulation
gradient_accumulation_steps=2
```

### Issue 2: NaN Losses

**Cause**: Deformable attention sampling going out of bounds

**Solution**:

```python
# In attention_modules.py, ensure clamping:
sampling_locations = torch.clamp(sampling_locations, -1, 1)

# Reduce learning rate
--lr 5e-5  # instead of 1e-4
```

### Issue 3: No Improvement After 20 Epochs

**Possible causes**:

1. Data format mismatch (bboxes not in 0-1 range)
2. Frozen backbone (check requires_grad)
3. Wrong loss weights

**Debug**:

```python
# Check bbox ranges
assert bbox.min() >= 0 and bbox.max() <= 1

# Check gradients flow to all modules
for name, param in model.named_parameters():
    if param.grad is None:
        print(f"⚠️ No gradient: {name}")
```

### Issue 4: Memory Leak

**Solution**:

```python
# Clear cache periodically
if (batch_idx + 1) % 100 == 0:
    torch.cuda.empty_cache()

# Use context managers
with torch.no_grad():
    ...
```

---

## 🔗 Integration with Existing Code

### Use with Existing YOLO Detection

```python
# Your existing YOLO detection code
yolo = YOLO('yolo11l.pt')
results = yolo.predict('image.jpg')

# Extract bboxes in YOLO format
detections = results[0]
boxes = detections.boxes.xywhn  # Normalized [x, y, w, h]

# Create dataset item:
dataset_item = {
    'knees': [{
        'ctx': ctx_patch,
        'ctx_bbox': torch.tensor(boxes[0]),  # First detection
        ...
    }],
}

# Feed to CADA model
output = model([dataset_item])
```

### Replace V3 with CADA in Existing Pipeline

```python
# Old code:
# from src.models.kiocmil_model_v3 import KiocmilModelV3
# model = KiocmilModelV3()

# New code:
from src.models.kiocmil_model_cada import KiocmilModelCADA
model = KiocmilModelCADA()

# Rest of code remains the same
```

---

## 📚 References

- **CDT-CAD Paper**: Context-Aware Deformable Transformers for End-to-End Chest Abnormality Detection
- **Deformable DETR**: Deformable Transformers for End-to-End Object Detection
- **KIOCMIL Enhancement Report**: `docs/KIOCMIL_ENHANCEMENT_REPORT.md`

---

## ✅ Checklist for Implementation

- [ ] Data prepared with bbox information
- [ ] Install dependencies
- [ ] Run training script with small dataset first (5-10 images)
- [ ] Check loss curve (should decrease)
- [ ] Monitor attention weights
- [ ] Scale to full dataset
- [ ] Compare with V3 baseline
- [ ] Log results to EXPERIMENT_LOG.md
- [ ] Save best checkpoint
- [ ] Evaluate on test set

---

## 📞 Support

For issues or questions:

1. Check this guide's troubleshooting section
2. Review `docs/KIOCMIL_ENHANCEMENT_REPORT.md` for architecture details
3. Check logs: `tail -f log/kiocmil_cada_*/training.log`
4. Review WandB dashboard for training metrics
