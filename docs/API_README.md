# KLGrade API Documentation

## Overview

KLGrade API cung cấp dịch vụ phân loại tự động mức độ thoái hóa khớp gối (KL grading) từ ảnh X-quang sử dụng kiến trúc KIOCMIL-CADA (Context-Aware Deformable Attention).

## Features

- ✅ **Multi-knee detection**: Tự động phát hiện và phân tích nhiều đầu gối trong một ảnh
- ✅ **10-class classification**: Phân loại chi tiết (KL0-a/b đến KL4-a/b: a=Osteophyte, b=Joint space)
- ✅ **Lesion detection**: Nhận diện tổn thương khớp (joint space narrowing và osteophytes)
- ✅ **GradCAM visualization**: Heatmap attention maps hiển thị vùng quan trọng
- ✅ **Comprehensive logging**: Ghi log chi tiết cho monitoring và debugging
- ✅ **API documentation**: Swagger UI tự động

## Installation

### Requirements

```bash
pip install fastapi uvicorn python-multipart
pip install torch torchvision
pip install ultralytics opencv-python numpy
pip install pydantic
```

### Model Files

Đảm bảo bạn có các model files:

- KIOCMIL-CADA model: `runs/kiocmil_cada/cada_10class_balanced/best_acc_model.pt`
- Knee detection model: `runs/detect/my_knee_run_resplit/weights/best.pt`
- Lesion detection model: `runs/detect/my_knee_run_resplit/weights/best.pt`

## Quick Start

### 1. Start API Server

```bash
python scripts/deployment/kiocmil_api_server.py \
  --kiocmil-model runs/kiocmil_cada/cada_10class_balanced/best_acc_model.pt \
  --knee-model runs/detect/my_knee_run_resplit/weights/best.pt \
  --lesion-model runs/detect/my_knee_run_resplit/weights/best.pt \
  --host 0.0.0.0 \
  --port 8001
```

### 2. Access API Documentation

Mở browser và truy cập: `http://localhost:8001/docs`

### 3. Test API

```bash
python scripts/deployment/test_api.py \
  --url http://localhost:8001 \
  --image path/to/test_image.jpg \
  --output-dir test_outputs
```

## API Endpoints

### 1. POST `/predict`

Phân tích ảnh X-quang và trả về predictions với bounding boxes.

**Request:**

```bash
curl -X POST "http://localhost:8001/predict" \
  -F "file=@knee_xray.jpg" \
  -F "knee_conf=0.5" \
  -F "lesion_conf=0.5"
```

**Response:**

```json
{
  "status": "success",
  "filename": "knee_xray.jpg",
  "num_knees_detected": 2,
  "predictions": [
    {
      "knee_bbox": {
        "x1": 100,
        "y1": 150,
        "x2": 300,
        "y2": 450
      },
      "predicted_class": "KL2-a",
      "predicted_class_id": 4,
      "confidence": 0.87,
      "class_probabilities": {
        "KL0-a": 0.01,
        "KL0-b": 0.02,
        "KL1-a": 0.05,
        "KL1-b": 0.03,
        "KL2-a": 0.87,
        ...
      },
      "num_js_lesions": 3,
      "num_ost_lesions": 2
    }
  ],
  "processing_time_ms": 234.5
}
```

### 2. POST `/predict_visual`

Phân tích ảnh X-quang và trả về predictions kèm ảnh đã annotate và GradCAM heatmap.

**Request:**

```bash
curl -X POST "http://localhost:8001/predict_visual" \
  -F "file=@knee_xray.jpg" \
  -F "knee_conf=0.5" \
  -F "lesion_conf=0.5" \
  -F "include_gradcam=true"
```

**Response:**

```json
{
  "status": "success",
  "filename": "knee_xray.jpg",
  "num_knees_detected": 2,
  "predictions": [...],
  "annotated_image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "gradcam_image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "processing_time_ms": 456.7
}
```

### 3. GET `/health`

Kiểm tra trạng thái API server.

**Request:**

```bash
curl -X GET "http://localhost:8001/health"
```

**Response:**

```json
{
  "status": "ok",
  "pipeline_loaded": true,
  "model_type": "KIOCMIL-CADA"
}
```

### 4. GET `/model_info`

Lấy thông tin chi tiết về model.

**Request:**

```bash
curl -X GET "http://localhost:8001/model_info"
```

**Response:**

```json
{
  "model_type": "KIOCMIL-CADA",
  "num_classes": 10,
  "ctx_size": [384, 384],
  "patch_size": [224, 224],
  "device": "cuda",
  "class_names": [
    "KL0-a",
    "KL0-b",
    "KL1-a",
    "KL1-b",
    "KL2-a",
    "KL2-b",
    "KL3-a",
    "KL3-b",
    "KL4-a",
    "KL4-b"
  ]
}
```

## Python Client Example

```python
import requests
import base64
import cv2
import numpy as np

# Initialize client
api_url = "http://localhost:8001"

# Test health
response = requests.get(f"{api_url}/health")
print(response.json())

# Predict with visualization
with open("knee_xray.jpg", "rb") as f:
    files = {"file": ("knee_xray.jpg", f, "image/jpeg")}
    data = {
        "knee_conf": 0.5,
        "lesion_conf": 0.5,
        "include_gradcam": True
    }

    response = requests.post(
        f"{api_url}/predict_visual",
        files=files,
        data=data
    )

    result = response.json()

    # Print predictions
    for i, pred in enumerate(result["predictions"]):
        print(f"Knee {i+1}: {pred['predicted_class']} ({pred['confidence']:.2f})")
        # Example output: "Knee 1: KL2-a (0.87)"

    # Save annotated image
    if result.get("annotated_image_base64"):
        img_data = base64.b64decode(result["annotated_image_base64"])
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        cv2.imwrite("output_annotated.jpg", img)

    # Save GradCAM
    if result.get("gradcam_image_base64"):
        img_data = base64.b64decode(result["gradcam_image_base64"])
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        cv2.imwrite("output_gradcam.jpg", img)
```

## Class Definitions

### 10-Class System

Hệ thống phân loại 10 lớp dựa trên KL grade (0-4) và loại tổn thương:

- **a**: Osteophyte (gai xương)
- **b**: Joint space (khe khớp)

| Class ID | Class Name | Description                         |
| -------- | ---------- | ----------------------------------- |
| 0        | KL0-a      | KL Grade 0 - Osteophyte (gai xương) |
| 1        | KL0-b      | KL Grade 0 - Joint space (khe khớp) |
| 2        | KL1-a      | KL Grade 1 - Osteophyte             |
| 3        | KL1-b      | KL Grade 1 - Joint space            |
| 4        | KL2-a      | KL Grade 2 - Osteophyte             |
| 5        | KL2-b      | KL Grade 2 - Joint space            |
| 6        | KL3-a      | KL Grade 3 - Osteophyte             |
| 7        | KL3-b      | KL Grade 3 - Joint space            |
| 8        | KL4-a      | KL Grade 4 - Osteophyte             |
| 9        | KL4-b      | KL Grade 4 - Joint space            |

## Logging

Logs được lưu trong thư mục `logs/`:

- `klgrade_api_YYYYMMDD.log`: Log chi tiết với rotation
- Format: `timestamp | level | module | function:line | message`

## Error Handling

API trả về error responses theo format chuẩn:

```json
{
  "status": "error",
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": "Additional error details"
  }
}
```

Common error codes:

- `INVALID_IMAGE`: Không thể đọc file ảnh
- `PREDICTION_ERROR`: Lỗi trong quá trình inference
- `VISUALIZATION_ERROR`: Lỗi khi tạo visualization

## Performance

- **Prediction time**: ~200-500ms per image (GPU)
- **Visual prediction time**: ~400-700ms per image (GPU)
- **Throughput**: ~2-5 images/second

## Troubleshooting

### Server không khởi động

- Kiểm tra model files có tồn tại
- Kiểm tra CUDA/GPU availability
- Xem logs trong `logs/` directory

### Prediction chậm

- Đảm bảo đang dùng GPU (`device: cuda`)
- Giảm confidence thresholds
- Batch multiple requests

### Out of memory

- Giảm batch size
- Resize ảnh input nhỏ hơn
- Sử dụng CPU inference

## License

MIT License

## Contact

For support: support@klgrade.com
