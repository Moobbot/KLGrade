# Dependencies for KLGrade Dataset Loaders

## Core Requirements

```bash
pip install torch torchvision
pip install ultralytics  # For YOLO11
pip install transformers timm  # For DETR
pip install albumentations opencv-python pillow
pip install tqdm matplotlib pyyaml scipy
```

## Detailed Package List

### Deep Learning Frameworks
- `torch>=2.0.0` - PyTorch for neural networks
- `torchvision>=0.15.0` - Vision utilities and transforms

### Object Detection Models
- `ultralytics>=8.0.0` - YOLO11 implementation
- `transformers>=4.30.0` - HuggingFace transformers for DETR
- `timm>=0.9.0` - Vision transformer backbones

### Data Processing
- `numpy>=1.24.0` - Numerical operations
- `opencv-python>=4.8.0` - Image processing
- `Pillow>=10.0.0` - Image loading
- `albumentations>=1.3.0` - Advanced Image augmentations with bbox support
- `tqdm>=4.65.0` - Progress bars

### Utilities
- `PyYAML>=6.0` - YAML configuration files
- `scipy>=1.10.0` - Scientific computing
- `matplotlib>=3.7.0` - Visualization

### Optional
- `pycocotools>=2.0.0` - COCO evaluation metrics (if needed)

## Installation

### Quick Install (All at once)
```powershell
pip install torch torchvision ultralytics transformers timm albumentations opencv-python pillow tqdm matplotlib pyyaml scipy
```

### GPU Support (CUDA)
For GPU training, install PyTorch with CUDA:

```powershell
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Then install other packages
pip install ultralytics transformers timm albumentations opencv-python pillow tqdm matplotlib pyyaml scipy
```

### Verify Installation

```python
import torch
import ultralytics
import transformers
import albumentations

print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Ultralytics: {ultralytics.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"Albumentations: {albumentations.__version__}")
```
