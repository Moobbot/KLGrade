# KLGrade - Restructuring Plan

## 📁 Target Folder Structure

```
KLGrade/
│
├── 📂 data/                        # Data directories (gitignored)
│   ├── dataset/                    # Raw datasets
│   │   ├── dataset_v0/             # Original full X-ray images
│   │   │   ├── images/             # 1,473 full X-rays
│   │   │   ├── labels/             # KL 0-4 (5 classes) - 3,157 boxes
│   │   │   ├── labels-knee/        # Knee detection boxes
│   │   │   └── labels-knee-2box/   # Left/Right knee boxes
│   │   └── dataset_v0/
│   │       ├── images/
│   │       ├── labels/             # 5 classes
│   │       └── labels_new/         # 10 classes
│   │
│   ├── processed/                  # Processed datasets
│   │   ├── knee/                   # ✨ Cropped knee regions
│   │   │   ├── images/             # 1,460 square knee crops
│   │   │   ├── labels/             # Transformed KL labels (2,637 boxes, 84.2% retention)
│   │   │   └── crop_stats.json    # Crop statistics
│   │   ├── coco/                   # COCO format annotations
│   │   └── yolo11_labels.yaml      # YOLO config
│   │
│   └── splits/                     # Train/val/test split files
│       ├── train.txt
│       ├── val.txt
│       └── test.txt
│
├── 📂 src/                         # Source code modules
│   ├── __init__.py
│   ├── datasets/                   # Dataset modules
│   │   ├── __init__.py
│   │   ├── yolo_dataset.py         # YOLO format dataset
│   │   ├── coco_dataset.py         # COCO format dataset
│   │   ├── converters.py           # Format converters
│   │   └── transforms.py           # Data transforms
│   │
│   ├── models/                     # Model wrappers (future)
│   │   ├── __init__.py
│   │   ├── yolo.py
│   │   └── detr.py
│   │
│   ├── utils/                      # Utilities
│   │   ├── __init__.py
│   │   ├── bbox.py                 # Bounding box utilities
│   │   ├── metrics.py              # Metrics computation
│   │   └── visualization.py        # Visualization helpers
│   │
│   └── config.py                   # Configuration
│
├── 📂 scripts/                     # Executable scripts
│   ├── preprocessing/              # ✨ Preprocessing scripts
│   │   └── crop_knee_regions.py   # Crop knee from full X-rays
│   │
│   ├── data_preparation/           # Data prep scripts
│   │   ├── split_dataset.py
│   │   ├── filter_dataset.py
│   │   ├── remap_labels.py
│   │   └── analyze_dataset.py
│   │
│   ├── training/                   # Training scripts
│   │   ├── train_yolo.py
│   │   └── train_detr.py
│   │
│   ├── evaluation/                 # Evaluation scripts
│   │   ├── evaluate_yolo.py
│   │   ├── evaluate_detr.py
│   │   ├── error_analysis.py
│   │   └── visualize_predictions.py
│   │
│   └── testing/                    # Test scripts
│       ├── test_yolo_dataset.py
│       └── test_coco_dataset.py
│
├── 📂 tools/                       # Development tools
│   ├── check_dataset/              # ✨ Consolidated tools (6 tools)
│   │   ├── comprehensive_analysis.py  # Main dataset analysis
│   │   ├── validate_dataset.py        # Data validation
│   │   ├── visualize_samples.py       # Sample visualization
│   │   ├── class_split_report.py      # Post-split analysis
│   │   ├── check_augment.py           # Augmentation testing
│   │   ├── resize_images.py           # Image preprocessing
│   │   └── PREPROCESSING_LABELS.md    # Documentation
│   └── check_vis/                  # Visualization outputs
│
├── 📂 runs/                        # Training outputs
│   ├── detect/                     # YOLO runs
│   └── detr/                       # DETR runs
│
├── 📂 docs/                        # Documentation
│   ├── README.md                   # Main documentation
│   ├── DATA_PREPARATION.md
│   ├── TRAINING_GUIDE.md
│   ├── EVALUATION_GUIDE.md
│   └── API_REFERENCE.md
│
├── 📂 notebooks/                   # Jupyter notebooks (future)
│   └── exploratory_analysis.ipynb
│
├── 📂 tests/                       # Unit tests (future)
│   ├── __init__.py
│   ├── test_datasets.py
│   ├── test_converters.py
│   └── test_utils.py
│
├── .gitignore
├── requirements.txt
├── setup.py                        # Package setup (future)
├── README.md                       # Project README
└── RESTRUCTURING_PLAN.md          # This file
```

---

## 🔄 Migration Strategy - Step by Step

### Phase 1: Setup New Structure (No Breaking Changes)

#### Step 1.1: Create New Directories

```powershell
# Create src/ structure
mkdir src
mkdir src/datasets
mkdir src/models
mkdir src/utils

# Create scripts/ structure
mkdir scripts
mkdir scripts/data_preparation
mkdir scripts/training
mkdir scripts/evaluation
mkdir scripts/testing

# Create tools/
mkdir tools
mv check_dataset tools/

# Create docs/
mkdir docs
```

**Test**: Verify all directories created

---

#### Step 1.2: Move Documentation

```powershell
# Move docs to docs/
mv DATA_PREPARATION_GUIDE.md docs/DATA_PREPARATION.md
mv DEPENDENCIES.md docs/DEPENDENCIES.md
# Keep README.md at root

# Move command files to docs/
mv TRAINING_COMMANDS.ps1 docs/
mv TRAINING_DETR_COMMANDS.ps1 docs/
```

**Test**: Verify docs accessible

---

### Phase 2: Migrate Source Code

#### Step 2.1: Create src/datasets Module

```powershell
# Copy dataset modules to new location
cp datasets/coco_dataset.py src/datasets/
cp datasets/converters.py src/datasets/
cp datasets/detr_transforms.py src/datasets/transforms.py

# Copy old dataset.py as yolo_dataset.py
cp dataset.py src/datasets/yolo_dataset.py

# Create __init__.py
# (manually create with proper imports)
```

**Update `src/datasets/__init__.py`**:

```python
from .yolo_dataset import YoloDataset
from .coco_dataset import CocoDataset
from .converters import *
from .transforms import *

__all__ = [
    'YoloDataset',
    'CocoDataset',
    # ... export all
]
```

**Test**: `python -c "from src.datasets import YoloDataset; print('OK')"`

---

#### Step 2.2: Create src/utils Module

```python
# src/utils/bbox.py
def yolo_to_xyxy_norm(cx, cy, w, h):
    """From utils.py"""
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return x1, y1, x2, y2

# Add more bbox utilities
```

**Update `src/utils/__init__.py`**:

```python
from .bbox import *

__all__ = ['yolo_to_xyxy_norm', ...]
```

**Test**: `python -c "from src.utils import yolo_to_xyxy_norm; print('OK')"`

---

#### Step 2.3: Move config.py

```powershell
cp config.py src/config.py
```

**Test**: `python -c "from src.config import CLASSES; print('OK')"`

---

### Phase 3: Migrate Scripts (One by One)

#### Step 3.1: Move Data Preparation Scripts

```powershell
# Move and update imports
mv split_dataset.py scripts/data_preparation/
mv filter_dataset_by_class.py scripts/data_preparation/filter_dataset.py
mv remap_filtered_labels.py scripts/data_preparation/remap_labels.py
```

**Update imports in each file**:

```python
# FROM:
from config import CLASSES

# TO:
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import CLASSES
```

**Test each script**:

```powershell
python scripts/data_preparation/split_dataset.py --help
python scripts/data_preparation/filter_dataset.py --help
python scripts/data_preparation/remap_labels.py --help
```

---

#### Step 3.2: Move Training Scripts

```powershell
mv examples/train_yolo11.py scripts/training/train_yolo.py
mv examples/train_detr.py scripts/training/train_detr.py
```

**Update imports**:

```python
# FROM:
from dataset import YoloDataset
from datasets import create_coco_json

# TO:
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.datasets import YoloDataset, create_coco_json
```

**Test**:

```powershell
python scripts/training/train_yolo.py --help
python scripts/training/train_detr.py --help
```

---

#### Step 3.3: Move Evaluation Scripts

```powershell
mv examples/evaluate_detr.py scripts/evaluation/
mv examples/validate_yolo.py scripts/evaluation/evaluate_yolo.py
mv examples/error_analysis.py scripts/evaluation/
mv examples/visualize_predictions.py scripts/evaluation/
```

**Update imports** in each file

**Test**:

```powershell
python scripts/evaluation/evaluate_detr.py --help
python scripts/evaluation/evaluate_yolo.py --help
python scripts/evaluation/error_analysis.py --help
```

---

#### Step 3.4: Move Test Scripts

```powershell
mv examples/test_yolo_dataset.py scripts/testing/
mv examples/test_coco_dataset.py scripts/testing/
```

**Update imports**

**Test**: Run each test script

---

### Phase 4: Update Tools

#### Step 4.1: Update check_dataset scripts

```powershell
# Already moved to tools/check_dataset/
# Update imports in:
# - check_augment.py
# - check_labels.py
# etc.
```

**Update imports**:

```python
# FROM:
from dataset import YoloDataset
from utils import yolo_to_xyxy_norm

# TO:
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.datasets import YoloDataset
from src.utils import yolo_to_xyxy_norm
```

**Test each tool**

---

### Phase 5: Cleanup Old Files

#### Step 5.1: Safety Check

Before deleting, verify all tests pass:

```powershell
# Verify imports work
python -c "from src.datasets import YoloDataset, CocoDataset; print('Datasets OK')"
python -c "from src.utils import yolo_to_xyxy_norm; print('Utils OK')"
python -c "from src.config import CLASSES; print('Config OK')"

# Verify all scripts have --help
python scripts/training/train_yolo.py --help
python scripts/evaluation/evaluate_detr.py --help
# ... test all
```

#### Step 5.2: Remove Old Files (Only if all tests pass!)

```powershell
# Remove old files from root
rm dataset.py
rm utils.py
rm config.py  # (if moved to src/)

# Remove old directories
rm -r datasets  # (if fully migrated to src/)
rm -r examples  # (if empty)
```

---

### Phase 6: Update Documentation

#### Step 6.1: Update README.md

Update all command examples with new paths:

```powershell
# OLD:
python examples/train_yolo11.py --data ...

# NEW:
python scripts/training/train_yolo.py --data ...
```

#### Step 6.2: Create MIGRATION_GUIDE.md

Document the changes for users

---

## 📋 Migration Checklist

### Preparation

- [ ] Backup entire project
- [ ] Create new branch: `git checkout -b restructure`
- [ ] Create directory structure

### Source Code Migration

- [ ] Step 1: Setup directories
- [ ] Step 2: Migrate src/datasets
- [ ] Step 3: Migrate src/utils
- [ ] Step 4: Migrate src/config
- [ ] Test: All imports work

### Scripts Migration

- [ ] Step 5: Migrate data_preparation scripts
- [ ] Step 6: Migrate training scripts
- [ ] Step 7: Migrate evaluation scripts
- [ ] Step 8: Migrate testing scripts
- [ ] Test: All scripts --help work

### Tools Migration

- [ ] Step 9: Update check_dataset tools
- [ ] Test: All tools run

### Cleanup

- [ ] Step 10: Safety verification
- [ ] Step 11: Remove old files
- [ ] Step 12: Update documentation
- [ ] Test: Full workflow end-to-end

### Finalization

- [ ] Run complete test suite
- [ ] Update .gitignore
- [ ] Commit changes
- [ ] Create PR for review

---

## ⚠️ Safety Rules

1. **Never delete before migration complete**
2. **Test after each step**
3. **Keep old files until 100% verified**
4. **Use git branches**
5. **Document every change**

---

## 🧪 Testing Strategy

After each phase:

```powershell
# Test imports
python -c "from src.datasets import YoloDataset; print('OK')"

# Test scripts
python scripts/training/train_yolo.py --help

# Test full workflow (if possible)
python scripts/data_preparation/split_dataset.py --image_dir ... --dry-run
```

---

## 📝 Notes

- Keep this file updated during migration
- Mark steps as completed: `- [x]`
- Document any issues encountered
- Add rollback procedures if needed
