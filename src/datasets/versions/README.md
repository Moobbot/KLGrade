# Dataset Versions

This directory contains older versions of KIOCMIL dataset and transform implementations.

## Current Active Versions (in parent directory)

- **`kiocmil_dataset.py`** - Main dataset (copied from v3)
- **`kiocmil_dataset_v3.py`** - CADA dataset with bounding boxes
- **`kiocmil_transforms.py`** - Main transforms (copied from v2)
- **`kiocmil_transforms_v2.py`** - Photometric transforms with geometric support

## Legacy Versions (in this directory)

### Datasets
- **`kiocmil_dataset_base.py`** - Original base dataset
  - Used by: `train_kiocmil_v1.py`, `train_kiocmil_v2.py`, `train_kiocmil_v3.py`
  - Status: Legacy, kept for backward compatibility

- **`kiocmil_dataset_v1.py`** - First iteration
  - Status: Deprecated

- **`kiocmil_dataset_v2.py`** - Second iteration with improvements
  - Used by: `train_kiocmil_v2.py`, `train_kiocmil_v3.py`
  - Status: Legacy

### Transforms
- **`kiocmil_transforms.py`** (base) - Original transforms
  - Used by: `train_kiocmil_v1.py`, `train_kiocmil_v2.py`, `train_kiocmil_v3.py`
  - Status: Legacy, kept for backward compatibility

- **`kiocmil_transforms_v1.py`** - First iteration
  - Status: Deprecated

## Migration Notes

If you need to use legacy versions, import from `versions/`:

```python
# Old way (still works for backward compatibility)
from src.datasets.kiocmil_dataset import KiocmilDataset

# New way (recommended)
from src.datasets.kiocmil_dataset_v3 import KiocmilDatasetV3

# Legacy versions
from src.datasets.versions.kiocmil_dataset_base import KiocmilDataset
from src.datasets.versions.kiocmil_dataset_v1 import KiocmilDatasetV1
from src.datasets.versions.kiocmil_dataset_v2 import KiocmilDatasetV2
```

## Cleanup Plan

Once all training scripts are updated to use v3:
1. Update imports in `train_kiocmil_v*.py` files
2. Test all training pipelines
3. Remove legacy versions from this directory
