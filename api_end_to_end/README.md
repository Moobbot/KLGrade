# End-to-End KIOCMIL Inference API

Standalone package for KL grade classification using End-to-End KIOCMIL models.

## Features

- ✅ Support for 4-class, 5-class, 8-class, and 10-class KL grade classification
- ✅ JSON-based configuration for easy model switching
- ✅ Single image and batch inference
- ✅ Multiple output formats (JSON, CSV, visualization)
- ✅ GPU and CPU support
- ✅ Progress tracking for batch processing

## Quick Start

### 1. Single Image Inference

```bash
python api_end_to_end/run_inference.py \
    --config api_end_to_end/configs/config_10class.json \
    --image path/to/xray.jpg \
    --output results.json
```

### 2. Batch Inference

```bash
python api_end_to_end/run_inference.py \
    --config api_end_to_end/configs/config_5class.json \
    --image-dir datasets/test/ \
    --output-dir outputs/predictions/ \
    --csv
```

### 3. With Visualization

```bash
python api_end_to_end/run_inference.py \
    --config api_end_to_end/configs/config_8class.json \
    --image path/to/xray.jpg \
    --visualize \
    --output-dir outputs/viz/
```

## Package Structure

```
api_end_to_end/
├── __init__.py              # Package initialization
├── inference.py             # Core inference class
├── output_formatters.py     # Output formatting utilities
├── run_inference.py         # CLI script
├── configs/                 # Model configurations
│   ├── config_10class.json  # 10-class model (85.92% acc)
│   ├── config_8class.json   # 8-class model (83.41% acc)
│   ├── config_5class.json   # 5-class model (84.82% acc)
│   └── config_4class.json   # 4-class model (72.13% acc)
├── utils/                   # Utility modules
│   ├── __init__.py
│   └── config.py           # Class name mappings
├── examples/               # Example scripts
└── README.md              # This file
```

## Available Models

| Model | Classes | Accuracy | Description |
|-------|---------|----------|-------------|
| **10-class** | KL0-a → KL4-b | **85.92%** | Detailed structure classification |
| **8-class** | KL1-a → KL4-b | **83.41%** | Pathological cases with detail |
| **5-class** | KL0 → KL4 | **84.82%** | Traditional KL grading |
| **4-class** | KL1 → KL4 | **72.13%** | Pathological cases only |

## Python API Usage

```python
from api_end_to_end import EndToEndInference

# Initialize
inference = EndToEndInference(
    checkpoint_path="runs/end_to_end/e2e_10class_balanced/best.pt",
    num_classes=10,
    device="cuda",
)

# Single image
result = inference.predict_single("path/to/xray.jpg")
print(f"Predicted: {result['predicted_class']} ({result['confidence']:.2%})")

# Batch processing
results = inference.predict_batch(["img1.jpg", "img2.jpg"])
```

## CLI Arguments

### Required
- `--config PATH` - Path to JSON configuration file
- `--image PATH` OR `--image-dir PATH` - Input image(s)

### Optional
- `--output PATH` - Output file (single image)
- `--output-dir PATH` - Output directory (batch)
- `--csv` - Save as CSV
- `--visualize` - Create annotated images
- `--max-images N` - Limit number of images
- `--device {cuda,cpu}` - Override device

## Configuration Format

```json
{
  "model_type": "10-class",
  "checkpoint_path": "runs/end_to_end/e2e_10class_balanced/best.pt",
  "num_classes": 10,
  "device": "cuda",
  "image_size": 640,
  "confidence_threshold": 0.0,
  "output_dir": "outputs/inference/10class"
}
```

## Output Formats

### JSON
```json
{
  "image_path": "xray.jpg",
  "predicted_class": "KL2-a",
  "confidence": 0.8532,
  "class_probabilities": {...}
}
```

### CSV
| image_path | predicted_class | confidence |
|-----------|----------------|------------|
| xray.jpg | KL2-a | 0.8532 |

### Visualization
Annotated images with predictions overlaid

## Requirements

- Python 3.8+
- PyTorch 1.10+
- torchvision
- OpenCV (cv2)
- NumPy
- PIL
- tqdm (optional, for progress bars)

## Model Checkpoints

Ensure model checkpoints are available at:
- `runs/end_to_end/e2e_10class_balanced/best.pt`
- `runs/end_to_end/e2e_8class_balanced/best.pt`
- `runs/end_to_end/e2e_5class_balanced/best.pt`
- `runs/end_to_end/e2e_4class_balanced/best.pt`

## Troubleshooting

**CUDA Out of Memory**
```bash
python api_end_to_end/run_inference.py --config ... --device cpu
```

**Import Errors**
```bash
# Run from project root
cd /path/to/KLGrade
python api_end_to_end/run_inference.py ...
```

## License

Part of the KLGrade project.
