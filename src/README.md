# Source Code (src/)

This directory contains the core library code for the KLGrade project.

## Structure

```
src/
├── __init__.py
├── config.py              # Global configuration
├── models/                # Model definitions
│   ├── kiocmil_*.py       # KIOCMIL models
│   ├── attention_modules.py
│   ├── cdt_cad/           # CDT-CAD models
│   └── yolo_*.py          # YOLO wrappers
├── datasets/              # Dataset classes
│   ├── kiocmil_dataset*.py
│   ├── cdt_cad_dataset.py
│   └── coco_dataset.py
├── data/                  # Data utilities
│   ├── preprocessing/     # Preprocessing functions
│   ├── balancing/         # Class balancing
│   ├── filters/           # Data filters
│   └── utils/             # Data utilities
├── training/              # Training utilities
│   ├── train_*.py         # Training loops
│   ├── evaluate_*.py      # Evaluation functions
│   └── early_stopping.py
├── losses/                # Loss functions
│   ├── focal_loss.py
│   ├── giou_loss.py
│   └── cdt_cad_loss.py
├── utils/                 # General utilities
│   ├── bbox.py            # Bounding box utilities
│   ├── detection_metrics.py
│   └── logging_utils.py
├── api/                   # API utilities
│   ├── inference.py
│   ├── kiocmil_inference.py
│   └── end_to_end_inference.py
├── evaluation/            # Evaluation utilities
└── visualization/         # Visualization utilities
```

## Usage

This is library code meant to be imported by scripts and APIs:

```python
# Import models
from src.models.kiocmil_model_cada import KIOCMILModelCADA

# Import datasets
from src.datasets.kiocmil_dataset_v3 import KiocmilDatasetV3

# Import utilities
from src.utils.bbox import calculate_iou
from src.data.preprocessing import apply_clahe
```

## Design Principles

- **Reusable**: Code here should be importable and reusable
- **Well-tested**: Core functionality should have tests
- **Documented**: Functions should have docstrings
- **Modular**: Clear separation of concerns

## Key Modules

### models/
Model architectures and definitions

### datasets/
PyTorch Dataset classes for loading data

### data/
Data processing utilities (preprocessing, augmentation, filtering)

### training/
Training loops, evaluation, and training utilities

### losses/
Custom loss functions

### utils/
General-purpose utilities

### api/
API-specific code for inference

---

## Related Directories

- [`scripts/`](../scripts/README.md) - Executable scripts that use this library
- [`api_*/`](../) - API packages that import from src/
