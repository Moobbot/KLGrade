# Two-Step YOLO Pipeline API

## Overview

Two-Step YOLO Pipeline là phương pháp phát hiện và phân loại KL grade sử dụng 2 YOLO models tuần tự:

1. **Step 1**: YOLO Knee Detector - Phát hiện vị trí khớp gối
2. **Step 2**: YOLO Lesion Detector - Phát hiện và phân loại tổn thương (KL grade)

Pipeline này đơn giản hơn KIOCMIL-CADA nhưng vẫn đạt hiệu quả cao với tốc độ xử lý nhanh.

## Architecture

```
Input Image
    ↓
┌─────────────────────┐
│  YOLO Knee Detector │  → Detect knee regions
└─────────────────────┘
    ↓
  Crop knees
    ↓
┌─────────────────────┐
│ YOLO Lesion Detector│  → Detect lesions & classify KL grade
└─────────────────────┘
    ↓
  Aggregate results
    ↓
Output: KL grades + bounding boxes
```

## Model Details

### Knee Detector (Step 1)

- **Architecture**: **YOLOv11n** (Nano - Selected for production)
- **Classes**: 1 (knee)
- **Input**: Full X-ray image (automatically resized to 640x640 by model)
- **Output**: Knee bounding boxes with confidence scores
- **Performance**:
  - mAP@50: **99.5%**
  - mAP@50-95: **82.6%**
  - Inference Time: **~1.8ms** (RTX 2080 Ti)

### Lesion Detector (Step 2)

- **Architecture**: YOLOv11n/YOLOv11l
- **Classes**: 8 or 10 (KL grades)
  - 8-class: KL1a, KL1b, KL2a, KL2b, KL3a, KL3b, KL4a, KL4b
  - 10-class: KL0a, KL0b, KL1a, KL1b, ..., KL4a, KL4b
- **Input**: Cropped knee regions
- **Output**: Lesion bounding boxes with KL grade classification
- **Performance**: mAP@50-95 = 0.667 (8-class)

## API Endpoints

### 1. Health Check

```http
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "models_loaded": true
}
```

### 2. Model Info

```http
GET /model_info
```

**Response:**

```json
{
  "knee_detector": {
    "model_path": "runs/detect/knee_yolo11n_20260217_134003/weights/best.pt",
    "classes": ["knee"],
    "input_size": [640, 640]
  },
  "lesion_detector": {
    "model_path": "runs/detect/lesion_8class/weights/best.pt",
    "classes": ["KL1a", "KL1b", "KL2a", "KL2b", ...],
    "input_size": [640, 640]
  }
}
```

### 3. Predict (Two-Step Pipeline)

```http
POST /predict
```

**Parameters:**

- `file`: Image file (multipart/form-data)
- `visualize`: Return annotated image in base64 (default: false)

**Example Request (cURL):**

```bash
curl -X POST "http://localhost:8002/predict/" \
  -F "file=@xray_image.jpg" \
  -F "visualize=true"
```

**Response:**

```json
{
  "filename": "xray_image.jpg",
  "kl_grade": 2,
  "image_size": [1920, 1080],
  "knees_count": 2,
  "lesions_count": 2,
  "knees": [
    {
      "bbox": [245, 156, 512, 678],
      "confidence": 0.92,
      "knee_id": 0
    }
  ],
  "lesions": [
    {
      "bbox": [280, 320, 350, 420],
      "class_name": "KL2a",
      "confidence": 0.87,
      "knee_id": 0
    }
  ],
  "visualization": "..." // Base64 string if visualize=true
}
```

### 4. Predict (DICOM Support)

```http
POST /predict/dicom/
```

**Parameters:**

- `file`: DICOM file (`.dcm`) (multipart/form-data)
- `visualize`: Return annotated image in base64 (default: false)

**Example Request (cURL):**

```bash
# Standard Prediction
curl -X POST "http://localhost:8002/predict/dicom/" \
  -F "file=@knee_xray.dcm"

# With Visualization
curl -X POST "http://localhost:8002/predict/dicom/?visualize=true" \
  -F "file=@knee_xray.dcm"
```

**Response:**

```json
{
  "filename": "knee_xray.dcm",
  "kl_grade": 2,
  "image_size": [1024, 1024],
  "knees_count": 1,
  "lesions_count": 2,
  "knees": [
    {
      "bbox": [200, 300, 600, 800],
      "confidence": 0.95,
      "knee_id": 0
    }
  ],
  "lesions": [
    {
      "bbox": [250, 400, 300, 450],
      "class_name": "KL2a",
      "confidence": 0.88,
      "knee_id": 0
    }
  ],
  "visualization": "iVBORw0KGgoAAAANSUhEUgAA..." // Base64 string if visualize=true
}
```

## Setup & Installation

### 1. Environment Setup

```bash
# Activate environment
conda activate klgrade

# Install dependencies (if not already installed)
pip install ultralytics opencv-python fastapi uvicorn
```

### 2. Download Models

Models should be placed in:

```
runs/detect/
├── knee_yolo11n_*/
│   └── weights/
│       └── best.pt
└── lesion_8class/  # or lesion_10class
    └── weights/
        └── best.pt
```

### 3. Start API Server

```bash
python scripts/api/start_two_step_api.py \
  --knee-model runs/detect/knee_yolo11n_20260217_134003/weights/best.pt \
  --lesion-model runs/detect/lesion_8class/weights/best.pt \
  --port 8002 \
  --host 0.0.0.0
```

**Parameters:**

- `--knee-model`: Path to knee detector weights
- `--lesion-model`: Path to lesion detector weights
- `--port`: API port (default: 8002)
- `--host`: Host address (default: 0.0.0.0)
- `--device`: Device to use (default: "cuda" if available, else "cpu")

## Usage Examples

### Python Client

```python
import requests

# Prepare image
with open("xray_image.jpg", "rb") as f:
    files = {"file": f}
    data = {
        "knee_conf": 0.5,
        "lesion_conf": 0.5,
        "include_visualization": True
    }

    # Send request
    response = requests.post(
        "http://localhost:8002/predict",
        files=files,
        data=data
    )

    # Parse response
    result = response.json()
    print(f"Detected {result['num_knees_detected']} knees")

    for pred in result["predictions"]:
        print(f"Knee {pred['knee_id']}: Grade {pred['predicted_grade']}")
        print(f"  Confidence: {pred['grade_confidence']:.2f}")
        print(f"  Lesions: {len(pred['lesions'])}")
```

### Batch Processing

```python
import os
import requests
from pathlib import Path

def process_directory(image_dir, output_dir):
    """Process all images in a directory"""
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    for img_path in image_dir.glob("*.jpg"):
        print(f"Processing {img_path.name}...")

        with open(img_path, "rb") as f:
            files = {"file": f}
            data = {"knee_conf": 0.5, "lesion_conf": 0.5}

            response = requests.post(
                "http://localhost:8002/predict",
                files=files,
                data=data
            )

            # Save results
            result = response.json()
            output_file = output_dir / f"{img_path.stem}_result.json"

            with open(output_file, "w") as out:
                json.dump(result, out, indent=2)

# Usage
process_directory("test_images/", "results/")
```

## Performance

### Speed

| Component                   | Time (GPU) | Time (CPU) |
| --------------------------- | ---------- | ---------- |
| Knee Detection              | **~2ms**   | ~50ms      |
| Lesion Detection (per knee) | ~80ms      | ~300ms     |
| Total (2 knees)             | ~165ms     | ~650ms     |

### Accuracy

| Metric    | 8-Class | 10-Class |
| --------- | ------- | -------- |
| mAP@50    | 0.763   | 0.633    |
| mAP@50-95 | 0.667   | 0.571    |
| Precision | 0.78    | 0.71     |
| Recall    | 0.72    | 0.65     |

## Comparison with KIOCMIL-CADA

| Feature        | Two-Step YOLO        | KIOCMIL-CADA         |
| -------------- | -------------------- | -------------------- |
| **Speed**      | ⭐⭐⭐ Fast (~200ms) | ⭐⭐ Medium (~500ms) |
| **Accuracy**   | ⭐⭐ Good            | ⭐⭐⭐ Better        |
| **Complexity** | ⭐ Simple            | ⭐⭐⭐ Complex       |
| **Memory**     | ⭐⭐⭐ Low (~2GB)    | ⭐⭐ Medium (~6GB)   |
| **Deployment** | ⭐⭐⭐ Easy          | ⭐⭐ Moderate        |

**Use Two-Step YOLO when:**

- Speed is critical
- Limited GPU memory
- Simple deployment needed
- Good accuracy is sufficient

**Use KIOCMIL-CADA when:**

- Maximum accuracy required
- GPU resources available
- Complex attention mechanisms beneficial

## Troubleshooting

### 1. No Knees Detected

**Issue:** API returns 0 knees

**Solutions:**

- Lower `knee_conf` threshold (try 0.3)
- Check image quality and resolution
- Verify knee detector model is loaded correctly

### 2. Poor Lesion Detection

**Issue:** Low confidence or wrong classifications

**Solutions:**

- Lower `lesion_conf` threshold
- Ensure cropped knee regions are clear
- Check if using correct model (8-class vs 10-class)

### 3. Slow Inference

**Issue:** Processing takes too long

**Solutions:**

- Enable GPU: Check `torch.cuda.is_available()`
- Reduce image resolution before sending
- Use batch processing for multiple images
- **Note**: The API automatically handles resizing to 640x640. Manual resizing is only recommended if source images are massive (>4K) to save network bandwidth.

### 4. Out of Memory

**Issue:** CUDA out of memory error

**Solutions:**

- Use smaller YOLO model (n instead of l)
- Process on CPU: `--device cpu`
- Reduce image resolution

## Advanced Configuration

### Custom Confidence Thresholds

```python
# Different thresholds for different use cases
configs = {
    "high_precision": {"knee_conf": 0.7, "lesion_conf": 0.7},
    "balanced": {"knee_conf": 0.5, "lesion_conf": 0.5},
    "high_recall": {"knee_conf": 0.3, "lesion_conf": 0.3}
}
```

### Model Ensemble

```python
# Use multiple lesion detectors and vote
lesion_models = [
    "runs/detect/lesion_8class_v1/weights/best.pt",
    "runs/detect/lesion_8class_v2/weights/best.pt",
    "runs/detect/lesion_8class_v3/weights/best.pt"
]

# Majority voting for final prediction
```

## API Server Configuration

### Environment Variables

```bash
# Set in .env file or export
export KNEE_MODEL_PATH="runs/detect/knee_detector/weights/best.pt"
export LESION_MODEL_PATH="runs/detect/lesion_8class/weights/best.pt"
export API_PORT=8002
export DEVICE="cuda"  # or "cpu"
export LOG_LEVEL="INFO"
```

### Production Deployment

```bash
# Use gunicorn for production
gunicorn scripts.api.start_two_step_api:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8002 \
  --timeout 120
```

## Logging

Logs are saved to `logs/two_step_api_YYYY-MM-DD.log`

**Log Levels:**

- `DEBUG`: Detailed inference information
- `INFO`: Request/response summaries
- `WARNING`: Low confidence predictions
- `ERROR`: Processing failures

## References

- **YOLO Documentation**: https://docs.ultralytics.com/
- **Model Training**: `docs/TRAINING.md`
- **Evaluation Results**: `docs/YOLO_EVALUATION_RESULTS.md`
- **Main Setup**: `docs/SETUP.md`
