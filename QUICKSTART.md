# KLGrade API - Quick Start Guide

## ✅ Verified Setup Process

### Bước 1: Cài đặt Môi trường (5-10 phút)

**Windows:**

```bash
cd e:\CaoHoc\thesis\code\web-knee\KLGrade
setup_env.bat
```

**Linux/Mac:**

```bash
cd /path/to/KLGrade
chmod +x setup_env.sh
./setup_env.sh
```

Script tự động:

- Tạo conda environment `klgrade_api` với Python 3.10
- Cài PyTorch 2.5.1+cu121 qua pip (tránh lỗi DLL)
- Cài tất cả dependencies (FastAPI, Ultralytics, OpenCV, etc.)
- Verify installation

### Bước 2: Kiểm tra Cài đặt

```bash
conda activate klgrade_api
test_setup.bat  # Windows
# hoặc ./test_setup.sh  # Linux/Mac
```

**Kết quả mong đợi:**

```
[1/3] Checking PyTorch installation...
  ✓ PyTorch: 2.5.1+cu121
  ✓ CUDA available: True

[2/3] Checking API dependencies...
  ✓ FastAPI and Uvicorn OK

[3/3] Checking ML/CV dependencies...
  ✓ OpenCV, Albumentations, Ultralytics OK

✓ Environment is ready!
```

### Bước 3: Chạy API Server

```bash
conda activate klgrade_api

python scripts/deployment/kiocmil_api_server.py ^
  --kiocmil-model runs/kiocmil_cada/cada_10class_balanced/best_acc_model.pt ^
  --knee-model runs/detect/my_knee_run_resplit/weights/best.pt ^
  --lesion-model runs/detect/my_knee_run_resplit/weights/best.pt ^
  --port 8001
```

Server chạy tại: **http://localhost:8001**

### Bước 4: Test API

**Option 1: Swagger UI (Khuyến nghị)**

1. Mở browser: http://localhost:8001/docs
2. Test `/health` → `{"status": "healthy"}`
3. Test `/predict_visual`:
   - Click "Try it out"
   - Upload ảnh từ `datasets/dataset_v0/images/`
   - Set `knee_conf`: 0.5, `lesion_conf`: 0.5
   - Check `include_gradcam`: true
   - Click "Execute"

**Option 2: Python Script**

```bash
python scripts/deployment/test_api.py ^
  --image datasets/dataset_v0/images/1.2.392.200036.9107.307.24972.20220821.103937.1020831.jpg ^
  --output-dir test_outputs
```

Kết quả lưu trong `test_outputs/`:

- Annotated images với bounding boxes
- GradCAM heatmaps
- JSON response

## 📊 API Response Format

```json
{
  "status": "success",
  "num_knees_detected": 2,
  "predictions": [
    {
      "knee_bbox": {"x1": 245, "y1": 156, "x2": 512, "y2": 678},
      "knee_confidence": 0.92,
      "predicted_class": "KL2-a",
      "confidence": 0.87,
      "lesions_summary": {"js": 4, "ost": 3, "total": 7},
      "lesions_by_class": {
        "KL2-a": [
          {"lesion_type": "ost", "score": 0.85, "bbox": {...}},
          {"lesion_type": "js", "score": 0.78, "bbox": {...}}
        ]
      }
    }
  ],
  "annotated_image_base64": "...",
  "gradcam_image_base64": "...",
  "processing_time_ms": 456.7
}
```

## 🔧 Troubleshooting

### PyTorch DLL Error

Nếu gặp `OSError: [WinError 182]`:

```bash
conda activate klgrade_api
fix_pytorch.bat
```

### CUDA không khả dụng

- API vẫn chạy được trên CPU (chậm hơn)
- Kiểm tra: `python -c "import torch; print(torch.cuda.is_available())"`

### Model file not found

- Kiểm tra đường dẫn model files
- Sử dụng đường dẫn tuyệt đối nếu cần

### Port đã được sử dụng

```bash
# Dùng port khác
python scripts/deployment/kiocmil_api_server.py ... --port 8002
```

## 📚 Tài liệu

- **Setup Chi tiết**: `README_SETUP.md`
- **API Documentation**: `docs/API_README.md`
- **Walkthrough**: Artifact `walkthrough.md`
- **Swagger UI**: http://localhost:8001/docs (khi server chạy)

## 🎯 Class Naming Convention

- `KLX-Y` format:
  - `X` = KL grade (0-4)
  - `Y` = Lesion type:
    - `a` = Osteophyte (gai xương)
    - `b` = Joint Space (hẹp khe khớp)

Ví dụ: `KL2-a` = KL grade 2 với Osteophyte
