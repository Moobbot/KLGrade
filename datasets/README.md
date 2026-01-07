# Dataset Loaders for KLGrade Object Detection

This directory contains dataset loaders and utilities for training object detection models on the KLGrade dataset.

## Directory Structure

```
datasets/
├── __init__.py           # Package initialization
├── coco_dataset.py       # COCO format dataset for DETR
├── converters.py         # YOLO → COCO format conversion
├── detr_transforms.py    # DETR-specific transforms and collate functions
└── README.md            # This file

../dataset.py            # YOLO dataset loader (main)
../examples/
├── test_yolo_dataset.py # Test YOLO dataset
├── test_coco_dataset.py # Test COCO dataset  
├── train_yolo11.py      # YOLO11 training example
└── train_detr.py        # DETR training example
```

## Available Datasets

### 1. YoloDataset (for YOLO11)

Located in `../dataset.py`. Loads YOLO format labels and images.

**Features:**
- YOLO format labels (normalized bounding boxes)
- Supports train/val/test splits
- Albumentations augmentation
- Memory caching
- Pascal VOC or YOLO output format

**Usage:**
```python
from dataset import YoloDataset, get_default_train_transform
from config import CLASSES

# Create dataset
transform = get_default_train_transform(img_size=(640, 640))
dataset = YoloDataset(
    img_dir="processed/knee/images",
    label_dir="processed/knee/labels",
    transform=transform,
    split_file="splits/train.txt",
    bbox_format='pascal_voc',
    return_dict=True
)

# Get a sample
sample = dataset[0]
print(sample['image'].shape)  # torch.Size([3, 640, 640])
print(sample['boxes'].shape)  # torch.Size([N, 4])
print(sample['labels'].shape) # torch.Size([N])
```

### 2. CocoDataset (for DETR)

Located in `datasets/coco_dataset.py`. Loads COCO JSON annotations.

**Features:**
- COCO JSON format
- HuggingFace DetrImageProcessor support
- Custom collate function for batching
- Automatic padding and pixel masks

**Usage:**
```python
from datasets import CocoDataset, get_detr_processor, detr_collate_fn
from torch.utils.data import DataLoader

# Get DETR processor
processor = get_detr_processor("facebook/detr-resnet-50")

# Create dataset
dataset = CocoDataset(
    coco_json_path="processed/coco/annotations_train.json",
    img_dir="processed/knee/images",
    processor=processor
)

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=4,
    collate_fn=detr_collate_fn
)

# Get a batch
batch = next(iter(dataloader))
print(batch['pixel_values'].shape)  # [B, 3, H, W]
print(batch['pixel_mask'].shape)    # [B, H, W]
```

## Utilities

### Format Conversion

Convert YOLO labels to COCO JSON:

```python
from datasets import create_coco_json
from config import CLASSES

create_coco_json(
    yolo_label_dir="processed/knee/labels",
    img_dir="processed/knee/images",
    output_path="processed/coco/annotations_train.json",
    class_names=CLASSES,
    split_file="splits/train.txt"
)
```

### DETR Transforms

```python
from datasets import get_detr_processor

# Get processor with custom size
processor = get_detr_processor(
    model_name="facebook/detr-resnet-50",
    size={'height': 800, 'width': 800}
)
```

## Testing

### Test YOLO Dataset
```powershell
python examples/test_yolo_dataset.py
```

Output:
- Loads train and val datasets
- Validates bbox coordinates
- Saves visualization samples to `check_vis/test_yolo/`
- Tests DataLoader

### Test COCO Dataset
```powershell
python examples/test_coco_dataset.py
```

Output:
- Converts YOLO labels to COCO JSON
- Loads COCO dataset (raw and with DETR processor)
- Tests collate function
- Saves visualizations to `check_vis/test_coco/`

## Training Examples

### Train YOLO11
```powershell
python examples/train_yolo11.py --epochs 100 --batch 16 --img_size 640
```

### Train DETR
```powershell
python examples/train_detr.py --epochs 50 --batch 4 --lr 1e-4
```

## Class Configuration

The dataset supports two label formats:

### Original Labels (5 classes)
```python
from config import CLASSES
# KL0, KL1, KL2, KL3, KL4
```

### Extended Labels (10 classes)
```python  
from config import CLASSES_LABEL_NEW
# KL0-a, KL0-b, KL1-a, KL1-b, ..., KL4-a, KL4-b
```

To use extended labels:
- YOLO: Set `use_labels_new=True` in YoloDataset
- COCO: Use `labels_new` directory when calling `create_coco_json`

## Common Issues

### Issue: DETR processor not found
**Solution:** Install transformers: `pip install transformers`

### Issue: Labels not found
**Solution:** Make sure you have run `class_split_report.py` to generate `labels_new/` if using extended classes

### Issue: CUDA out of memory
**Solution:** Reduce batch size or image size

### Issue: Slow training
**Solution:** Enable image caching: `cache_images=True` (requires sufficient RAM)

## References

- [YOLO11 Documentation](https://docs.ultralytics.com/)
- [DETR Paper](https://arxiv.org/abs/2005.12872)
- [HuggingFace DETR](https://huggingface.co/docs/transformers/model_doc/detr)
- [Albumentations](https://albumentations.ai/)
