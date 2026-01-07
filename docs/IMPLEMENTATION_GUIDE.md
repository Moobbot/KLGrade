# Hướng dẫn Thực hiện: Data Augmentation & Long-tail Handling

## Tổng quan

Tài liệu này hướng dẫn chi tiết từng bước để implement data augmentation và các phương pháp xử lý long-tail distribution cho bài toán knee OA detection.

---

## Bước 1: Chuẩn bị - Augmentation Transforms

### 1.1 Tạo file augmentation transforms

**File:** `src/datasets/augmentation.py`

**Mục đích:** Centralize all augmentation logic

**Code:**

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

def get_conservative_train_transform(img_size=(640, 640)):
    """
    Conservative augmentation for medical X-ray images.

    Safe transformations that preserve anatomical correctness:
    - Horizontal flip (L/R knee symmetry)
    - Slight rotation (±15°)
    - Brightness/contrast adjustment
    - Minimal blur/noise
    """
    return A.Compose([
        # Geometric (anatomically valid)
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5, border_mode=0),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=0,  # Rotation handled above
            p=0.5,
            border_mode=0
        ),

        # Intensity (X-ray exposure variation)
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),

        # Image quality (simulate capture conditions)
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(var_limit=(5, 15), p=0.2),

        # Resize to target size
        A.Resize(img_size[0], img_size[1]),

        # Normalize
        A.Normalize(mean=0.5, std=0.5),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels']
    ))

def get_val_transform(img_size=(640, 640)):
    """Validation transform - no augmentation"""
    return A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.Normalize(mean=0.5, std=0.5),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels']
    ))
```

**Kiểm tra:**

```python
# Test augmentation
transform = get_conservative_train_transform()
sample = transform(
    image=img,
    bboxes=[[0.5, 0.5, 0.2, 0.3]],
    class_labels=[2]
)
print("Augmented image shape:", sample['image'].shape)
print("Augmented bbox:", sample['bboxes'])
```

---

## Bước 2: Tạo Tool Kiểm tra Augmentation

### 2.1 Visualization tool

**File:** `tools/check_dataset/visualize_augmentations.py`

**Mục đích:** Visual verification của augmentations

**Code:**

```python
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.datasets.augmentation import get_conservative_train_transform
from src.datasets import YoloDataset

def visualize_augmentations(
    img_dir,
    label_dir,
    num_samples=10,
    save_dir='analysis/augmentation_checks'
):
    """
    Visualize original vs augmented samples.
    """
    # Load dataset
    dataset = YoloDataset(
        img_dir=img_dir,
        label_dir=label_dir,
        transform=get_conservative_train_transform()
    )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(num_samples):
        sample = dataset[idx]

        # Create comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Original (reload without transform)
        orig_dataset = YoloDataset(img_dir, label_dir, transform=None)
        orig = orig_dataset[idx]

        axes[0].imshow(orig['image'], cmap='gray')
        axes[0].set_title('Original')

        # Augmented
        axes[1].imshow(sample['image'].permute(1,2,0), cmap='gray')
        axes[1].set_title('Augmented')

        plt.savefig(save_dir / f'aug_{idx:03d}.png', dpi=100)
        plt.close()

    print(f"✅ Saved {num_samples} augmentation samples to {save_dir}")

if __name__ == '__main__':
    visualize_augmentations(
        img_dir='processed/knee/images',
        label_dir='processed/knee/labels',
        num_samples=20
    )
```

**Chạy:**

```powershell
.venv\Scripts\python.exe tools\check_dataset\visualize_augmentations.py
```

**Kiểm tra:** Xem folder `analysis/augmentation_checks/` để verify augmentations hợp lý

---

## Bước 3: Baseline Training Setup

### 3.1 Tạo YOLO training config

**File:** `configs/yolo_5_class_baseline.yaml`

```yaml
# Dataset
data:
  train: splits/knee_5_class/train.txt
  val: splits/knee_5_class/val.txt
  nc: 5 # Number of classes
  names: ["KL0", "KL1", "KL2", "KL3", "KL4"]

# Model
model: yolo11n.pt # Start with nano for faster iteration

# Hyperparameters
imgsz: 640
epochs: 100
batch: 16
optimizer: AdamW
lr0: 0.001
lrf: 0.01

# Augmentation (YOLO built-in, set to minimal for baseline)
hsv_h: 0.0
hsv_s: 0.0
hsv_v: 0.0
degrees: 0.0
translate: 0.0
scale: 0.0
flipud: 0.0
fliplr: 0.0 # Will use custom augmentation instead
mosaic: 0.0
mixup: 0.0

# Training
patience: 20
save_period: 10
workers: 4
device: 0 # GPU 0
project: runs/detect
name: knee_5_class_baseline
```

**Chạy baseline:**

```powershell
.venv\Scripts\python.exe scripts\training\train_yolo.py `
    --config configs\yolo_5_class_baseline.yaml
```

---

## Bước 4: Instance-Aware Repeat Factor Sampling (IRFS)

### 4.1 Implement IRFS trong dataset loader

**File:** `src/datasets/samplers.py`

```python
import math
from collections import defaultdict
from torch.utils.data import Sampler
import torch

class RepeatFactorSampler(Sampler):
    """
    Instance-Aware Repeat Factor Sampling.

    Rare classes get higher repeat factors based on:
    r_i = max(1, sqrt(t / f_i))

    where t = target frequency (default 0.001 = 0.1%)
    """
    def __init__(self, dataset, repeat_thresh=0.001, shuffle=True):
        self.dataset = dataset
        self.repeat_thresh = repeat_thresh
        self.shuffle = shuffle

        # Compute class frequencies
        self.class_freq = self._compute_class_frequencies()

        # Compute repeat factors per image
        self.repeat_factors = self._compute_repeat_factors()

        # Generate repeated indices
        self.indices = self._generate_indices()

    def _compute_class_frequencies(self):
        """Count instances per class"""
        freq = defaultdict(int)
        total = 0

        for sample in self.dataset:
            for box_class in sample['class_labels']:
                freq[box_class] += 1
                total += 1

        # Normalize
        for k in freq:
            freq[k] = freq[k] / total

        return freq

    def _compute_repeat_factors(self):
        """Compute repeat factor for each image"""
        factors = []

        for idx, sample in enumerate(self.dataset):
            # Get max repeat factor among all classes in image
            max_factor = 1.0
            for box_class in sample['class_labels']:
                class_frac = self.class_freq[box_class]
                factor = math.sqrt(self.repeat_thresh / class_frac)
                max_factor = max(max_factor, factor)

            factors.append(max_factor)

        return factors

    def _generate_indices(self):
        """Generate repeated indices based on factors"""
        indices = []
        for idx, factor in enumerate(self.repeat_factors):
            # Repeat image ceiling(factor) times
            repeat_count = math.ceil(factor)
            indices.extend([idx] * repeat_count)

        return indices

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(len(self.indices)).tolist()
            return iter([self.indices[i] for i in indices])
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
```

**Usage:**

```python
from src.datasets.samplers import RepeatFactorSampler

# Create sampler
sampler = RepeatFactorSampler(train_dataset, repeat_thresh=0.001)

# Use in dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    sampler=sampler,  # Instead of shuffle=True
    num_workers=4
)
```

---

## Bước 5: Validation & Metrics

### 5.1 Track per-class performance

**File:** `scripts/evaluation/evaluate_per_class.py`

```python
def compute_per_class_ap(results, class_names):
    """
    Compute AP for each class separately.
    """
    per_class_metrics = {}

    for class_id, class_name in class_names.items():
        # Filter predictions and GT for this class
        class_preds = [p for p in results if p['class'] == class_id]

        # Compute AP
        ap = compute_ap(class_preds)

        per_class_metrics[class_name] = {
            'AP': ap,
            'num_predictions': len(class_preds),
            'precision': ...,
            'recall': ...
        }

    return per_class_metrics
```

---

## Checklist Implementation

- [ ] **Bước 1:** Tạo `src/datasets/augmentation.py`
- [ ] **Bước 2:** Tạo visualization tool
- [ ] **Bước 3:** Baseline training config
- [ ] **Bước 4:** IRFS sampler
- [ ] **Bước 5:** Per-class evaluation
- [ ] **Bước 6:** Train baseline
- [ ] **Bước 7:** Train với IRFS
- [ ] **Bước 8:** So sánh results

---

## Expected Timeline

- **Week 1:** Steps 1-3 (Augmentation + Baseline)
- **Week 2:** Steps 4-5 (IRFS + Evaluation)
- **Week 3:** EQL v2 implementation
- **Week 4:** Decoupled training
- **Week 5:** Final evaluation

---

## Troubleshooting

### Issue: Augmentation làm mất bbox

**Solution:** Verify `bbox_params` trong Albumentations

### Issue: IRFS làm training chậm

**Solution:** Reduce `repeat_thresh` hoặc cache augmented samples

### Issue: Rare class vẫn AP thấp

**Solution:** Combine IRFS + EQL v2 + copy-paste
