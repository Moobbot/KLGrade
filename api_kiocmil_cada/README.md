# KIOCMIL-CADA Inference API

Standalone inference package for the 2-step KIOCMIL-CADA KL grading pipeline.

## Features

- **2-Step Pipeline**: 
  1. **Detection**: Specific YOLO models for Knee localization and Lesion (Osteophyte/Joint Space) detection.
  2. **Classification**: KIOCMIL-CADA model using Deformable Attention on detected regions.
- **Multiple Configurations**: Support for 4-class, 5-class, 8-class, and 10-class KL grading schemes.
- **Flexible Output**: JSON, CSV, and Visualized Images.
- **Standardized API**: Easy to integrate into other applications.

## Directory Structure

```
api_kiocmil_cada/
├── checkpoints/       # Store your .pt model files here
├── configs/           # JSON configurations for different models
├── detectors/         # Detection model wrappers
├── examples/          # Usage scripts
├── models/            # Core KIOCMIL-CADA model definitions
├── preprocessing/     # Image transformations
├── utils/             # Helper functions
├── inference.py       # Main Python API class
└── run_inference.py   # CLI entry point
```

## Setup

1. **Install Dependencies**:
   ```bash
   pip install torch torchvision opencv-python ultralytics numpy tqdm
   ```

2. **Prepare Checkpoints**:
   Place your trained model weights in the `checkpoints/` directory (or update paths in configs).
   You need 3 models:
   - `kiocmil_cada_*.pt`: The classification model.
   - `knee_detector.pt`: YOLO model for detecting knees in full X-rays.
   - `lesion_detector.pt`: YOLO model for detecting lesions (JS/OST) in knee crops.

   > **Note**: If you don't have a trained `lesion_detector.pt`, you will need to train a YOLOv8/11 model on the lesion dataset (classes: 0-3 for Osteophytes, 4-5 for Joint Space).

## Usage

### Command Line Interface (CLI)

Run inference on a single image:
```bash
python api_kiocmil_cada/run_inference.py \
  --config api_kiocmil_cada/configs/config_10class.json \
  --image path/to/xray.jpg \
  --output-dir results/ \
  --visualize
```

Run on a directory of images:
```bash
python api_kiocmil_cada/run_inference.py \
  --config api_kiocmil_cada/configs/config_5class.json \
  --image-dir data/test_images/ \
  --output-dir results/ \
  --csv --json
```

### Python API

```python
import cv2
from api_kiocmil_cada.inference import KiocmilInference

# Initialize
engine = KiocmilInference(
    kiocmil_model_path="checkpoints/kiocmil_cada_10class.pt",
    knee_model_path="checkpoints/knee_detector.pt",
    lesion_model_path="checkpoints/lesion_detector.pt",
    num_classes=10
)

# Load image
image = cv2.imread("xray.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Predict
results = engine.predict(image)

# Access results
for knee in results:
    print(f"Class: {knee['predicted_class']}")
    print(f"Conf: {knee['confidence']}")
```

## Output Format

**JSON Output**:
```json
[
  {
    "image_path": "...",
    "knees": [
      {
        "knee_bbox": [x1, y1, x2, y2],
        "predicted_class": "KL2-a",
        "confidence": 0.85,
        "num_js_lesions": 1,
        "num_ost_lesions": 2,
        "js_lesion_bboxes": [...],
        "ost_lesion_bboxes": [...]
      }
    ]
  }
]
```

## Model Configurations

- **10-Class**: Full granularity (KL0-KL4, split by type a/b).
- **8-Class**: KL1-KL4, split by type a/b.
- **5-Class**: Standard KL grading (KL0-KL4).
- **4-Class**: KL1-KL4 only.

## Troubleshooting

- **Missing Lesion Detector**: The pipeline strictly requires a lesion detector. Ensure `lesion_detector.pt` is trained and available.
- **CUDA Out of Memory**: Reduce `batch_size` (if applicable) or run on CPU using `--device cpu`.
