# Two-Step YOLO API Documentation

## Overview

The Two-Step YOLO API provides end-to-end KL grading using two sequential YOLO11l models:

1. **Step 1**: Knee Detection - Detect knee regions in full X-ray images
2. **Step 2**: Lesion Detection - Detect lesions (osteophytes, joint space narrowing) in cropped knees

This approach is **memory-efficient** and runs on RTX 2080 Ti (10GB VRAM).

---

## Quick Start

### Installation

```bash
conda activate klgrade
pip install ultralytics opencv-python numpy
```

### Basic Usage

```python
from api_two_step_yolo.inference.two_step_yolo_api import TwoStepYOLOInference

# Initialize pipeline
pipeline = TwoStepYOLOInference(
    knee_model_path="runs/detect/knee_detector/weights/best.pt",
    lesion_model_path="runs/detect/lesion_8class_balanced/weights/best.pt",
    device="cuda:0",
)

# Run inference
result = pipeline.predict("path/to/xray.jpg")

# Print results
print(f"Knees detected: {len(result['knees'])}")
print(f"Lesions detected: {len(result['lesions'])}")
print(f"KL Grade: {result['kl_grade']}")

# Visualize
pipeline.visualize("path/to/xray.jpg", "output.jpg")
```

### Command Line

```bash
python api_two_step_yolo/inference/two_step_yolo_api.py \
    --image path/to/xray.jpg \
    --lesion-model runs/detect/lesion_8class_balanced/weights/best.pt \
    --knee-conf 0.75 --lesion-conf 0.25 \
    --output result.jpg
```

### REST API (Web Server)

Start the API server:

```bash
python -m api_two_step_yolo.inference.server
# Runs on http://0.0.0.0:9090
```

Make a prediction:

```bash
# Get JSON result
curl -X POST "http://localhost:9090/predict/" \
     -F "file=@path/to/xray.jpg"

# Get Annotated Image
curl -X POST "http://localhost:9090/predict/?visualize=true" \
     -F "file=@path/to/xray.jpg" --output result.jpg
```

---

## Available Models

### Knee Detection
- **Model**: `runs/detect/knee_detector/weights/best.pt`
- **Classes**: 1 (Knee)
- **mAP@50**: 0.995
- **Use**: Always use this for Step 1

### Lesion Detection (Choose One)

#### Cropped Knee Images (Recommended)

| Model | Classes | Dataset | mAP@50 | mAP@50-95 | Use Case |
|-------|---------|---------|--------|-----------|----------|
| `lesion_8class_balanced` | 8 (OST/JS split) | Balanced | **0.763** | **0.667** | **Best Performance (Recommended)** |
| `lesion_5class_balanced` | 5 (KL0-4) | Balanced | 0.734 | 0.571 | Good alternative |
| `lesion_10class_balanced` | 10 (KL0-a to KL4-b) | Balanced | 0.633 | 0.571 | Fine-grained grading |
| `lesion_4class_balanced` | 4 (KL1-4) | Balanced | 0.584 | 0.348 | Not recommended |

#### Full X-ray Images

| Model | Classes | Dataset | mAP@50 | mAP@50-95 | Note |
|-------|---------|---------|--------|-----------|------|
| `lesion_full_10class_balanced` | 10 | Balanced | 0.681 | 0.527 | Best Full X-ray model |
| `lesion_full_8class_balanced` | 8 | Balanced | 0.657 | 0.514 | |

**Recommendation**: Always use **`lesion_8class_balanced`** for the best accuracy.

---

## API Reference

### `TwoStepYOLOInference`

#### Constructor

```python
TwoStepYOLOInference(
    knee_model_path: str,
    lesion_model_path: str,
    device: str = "cuda:0",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
)
```

**Parameters:**
| Parameter | Description | Default/Example |
|---|---|---|
| `knee_model_path` | Path to knee detection model (`.pt` file) | `runs/detect/knee_detector/weights/best.pt` |
| `lesion_model_path` | Path to lesion detection model (`.pt` file) | `runs/detect/lesion_8class_balanced/weights/best.pt` |
| `device` | Device to run inference on | `cuda:0`, `cpu` |
| `knee_conf_threshold` | Confidence threshold for knee detections (0.0-1.0) | `0.75` |
| `lesion_conf_threshold` | Confidence threshold for lesion detections (0.0-1.0) | `0.25` |
| `iou_threshold` | IoU threshold for NMS (0.0-1.0) | `0.45` |

#### Methods

##### `predict(image_path, return_crops=False)`

Run full two-step inference pipeline.

**Parameters:**
- `image_path` (str): Path to input X-ray image
- `return_crops` (bool): Whether to return cropped knee images

**Returns:**
```python
{
    "image_path": str,
    "image_size": [height, width],
    "knees": [
        {
            "bbox": [x1, y1, x2, y2],
            "confidence": float,
            "knee_id": int,
        },
        ...
    ],
    "lesions": [
        {
            "bbox": [x1, y1, x2, y2],  # Local coordinates
            "bbox_global": [x1, y1, x2, y2],  # Full image coordinates
            "class": int,
            "class_name": str,
            "confidence": float,
            "knee_id": int,
        },
        ...
    ],
    "kl_grade": int,  # 0-4
    "knee_crops": [np.ndarray, ...],  # If return_crops=True
}
```

##### `visualize(image_path, output_path=None)`

Run inference and create annotated visualization.

**Parameters:**
- `image_path` (str): Path to input image
- `output_path` (str, optional): Path to save visualization

**Returns:**
- `np.ndarray`: Annotated image (RGB)

##### `detect_knees(image)`

Step 1: Detect knee regions (used internally).

##### `detect_lesions(knee_image, knee_id=0)`

Step 2: Detect lesions in cropped knee (used internally).

---

## Examples

### Example 1: Batch Processing

```python
from pathlib import Path
from two_step_yolo_api import TwoStepYOLOInference

# Initialize
pipeline = TwoStepYOLOInference(
    knee_model_path="runs/detect/knee_detector/weights/best.pt",
    lesion_model_path="runs/detect/lesion_8class_balanced/weights/best.pt",
    knee_conf_threshold=0.75,
    lesion_conf_threshold=0.25,
)

# Process all images in directory
image_dir = Path("data/test_images")
results = []

for img_path in image_dir.glob("*.jpg"):
    result = pipeline.predict(str(img_path))
    results.append(result)
    print(f"{img_path.name}: KL{result['kl_grade']}")

# Save results
import json
with open("batch_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Example 2: Custom Confidence Threshold

```python
# Lower threshold for more detections
pipeline = TwoStepYOLOInference(
    knee_model_path="...",
    lesion_model_path="...",
    conf_threshold=0.15,  # Lower = more sensitive
    iou_threshold=0.5,
)

result = pipeline.predict("xray.jpg")
```

### Example 3: Get Cropped Knees

```python
result = pipeline.predict("xray.jpg", return_crops=True)

# Save cropped knees
for i, crop in enumerate(result["knee_crops"]):
    cv2.imwrite(f"knee_{i}.jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
```

---

## Performance

### Speed

| GPU | Knee Detection | Lesion Detection | Total |
|-----|----------------|------------------|-------|
| RTX 2080 Ti | ~50ms | ~80ms | ~130ms |
| RTX 3090 | ~30ms | ~50ms | ~80ms |
| CPU (i7) | ~500ms | ~800ms | ~1.3s |

### Accuracy

Depends on chosen lesion model. See evaluation results in `doc-training/`.

---

## Comparison with Other Approaches

| Approach | GPU Memory | Speed | Accuracy | Status |
|----------|------------|-------|----------|--------|
| **Two-Step YOLO** | 3-4GB | Fast | Good | ✅ **Ready** |
| CDT-CAD | 12-16GB | Slow | Better? | ⚠️ Needs large GPU |
| KIOCMIL-CADA | 6-8GB | Medium | Best | ✅ Available |

**Recommendation**: 
- Use **Two-Step YOLO** for fast inference on limited hardware
- Use **KIOCMIL-CADA** for best accuracy
- Use **CDT-CAD** if you have access to large GPU (A100, V100)

---

## Troubleshooting

### Issue: Low Detection Rate

**Solution**: Lower confidence threshold
```python
pipeline = TwoStepYOLOInference(..., conf_threshold=0.15)
```

### Issue: Too Many False Positives

**Solution**: Raise confidence threshold
```python
pipeline = TwoStepYOLOInference(..., conf_threshold=0.35)
```

### Issue: CUDA Out of Memory

**Solution**: Use CPU or smaller batch size
```python
pipeline = TwoStepYOLOInference(..., device="cpu")
```

### Issue: Wrong KL Grade

The `_determine_kl_grade()` method uses a simple heuristic (highest class detected). For better grading:

1. Implement custom logic based on lesion counts and types
2. Use ensemble of multiple lesion models
3. Add post-processing rules based on clinical guidelines

---

## Next Steps

1. **Evaluate all models**: Run `evaluate_all_models.py`
2. **Choose best model**: Based on mAP@50 and use case
3. **Integrate into production**: Deploy as REST API or CLI tool
4. **Compare with KIOCMIL-CADA**: For accuracy benchmarking

---

## Related Documentation

- [Training Guide](../training/TRAINING_README.md)
- [Evaluation Script](../training/evaluate_all_models.py)
- [CDT-CAD Migration](../training/CDT_CAD_MIGRATION_GUIDE.md)
- [KIOCMIL-CADA API](../../api_end_to_end/inference.py)
