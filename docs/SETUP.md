# KLGrade API - Hướng Dẫn Cài Đặt Chi Tiết

## Tổng Quan

API này cung cấp khả năng phân loại mức độ thoái hóa khớp gối (KL Grade) từ ảnh X-quang, bao gồm:

- Phát hiện các khớp gối
- Phát hiện và phân loại KL grade (KL0-4)
- Phát hiện các tổn thương (Osteophyte và Joint Space Narrowing)
- Trực quan hóa với GradCAM
- RESTful API với Swagger UI

## Yêu Cầu Hệ Thống

### Phần cứng

- **RAM**: Tối thiểu 8GB (khuyến nghị 16GB+)
- **GPU**: NVIDIA GPU với CUDA support (khuyến nghị)
  - Compute Capability ≥ 3.5
  - VRAM ≥ 4GB
- **Disk**: ~10GB trống cho môi trường và models

### Phần mềm

- **OS**: Windows 10/11, Linux, hoặc macOS
- **Conda/Miniconda**: Phiên bản mới nhất
- **CUDA**: 12.1 (nếu dùng GPU)
- **Python**: 3.10 (sẽ được cài tự động)

## Cài Đặt Nhanh

### 1. Clone Repository (nếu chưa có)

```bash
git clone <repository-url>
cd KLGrade
```

### 2. Chạy Setup Script

**Windows:**

```bash
scripts/setup/setup_env.bat
```

**Linux/Mac:**

```bash
chmod +x scripts/setup/setup_env.sh
./scripts/setup/setup_env.sh
```

### 3. Kích hoạt Environment

```bash
conda activate klgrade_api
```

### 4. Verify Installation

```bash
scripts/setup/test_setup.bat  # Windows
# hoặc
./scripts/setup/test_setup.sh  # Linux/Mac
```

## Chi Tiết Quá Trình Setup

### Script `setup_env.bat` thực hiện:

1. **Tạo base conda environment**

   ```
   conda create -n klgrade_api python=3.10 -y
   ```

2. **Cài PyTorch với CUDA 12.1 qua pip**

   ```
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

   > **Lưu ý**: Dùng pip thay vì conda để tránh lỗi DLL trên Windows

3. **Cài API dependencies**
   - FastAPI 0.104.1
   - Uvicorn 0.24.0
   - Pydantic 2.5.0
   - Python-multipart 0.0.6

4. **Cài ML/CV dependencies**
   - Ultralytics 8.0.200 (YOLO)
   - OpenCV 4.8.1.78
   - Albumentations 1.3.1
   - PyDICOM 2.4.3

5. **Cài scientific computing libraries**
   - NumPy 1.24.3
   - Scikit-learn 1.3.2
   - PyYAML 6.0.1
   - tqdm 4.66.1

6. **Verify installation**
   - Kiểm tra PyTorch import
   - Kiểm tra CUDA availability

## Cấu Trúc Thư Mục

```
KLGrade/
├── runs/
│   ├── kiocmil_cada/
│   │   └── cada_10class_balanced/
│   │       └── best_acc_model.pt          # Main KIOCMIL-CADA model
│   └── detect/
│       └── my_knee_run_resplit/
│           └── weights/
│               └── best.pt                # YOLO detector
├── scripts/
│   ├── deployment/
│   │   ├── kiocmil_api_server.py          # API server
│   │   └── test_api.py                    # Test script
│   ├── setup/
│   │   ├── setup_env.bat                  # Windows setup
│   │   ├── setup_env.sh                   # Linux/Mac setup
│   │   ├── test_setup.bat                 # Verification script
│   │   └── fix_pytorch.bat                # PyTorch fix
│   └── api/
│       ├── start_api.sh                   # Start API (Linux/Mac)
│       ├── start_api.bat                  # Start API (Windows)
│       └── start_api_server.sh            # API server launcher
├── src/
│   ├── api/
│   │   ├── response_schemas.py            # Pydantic models
│   │   ├── kiocmil_inference.py           # Inference logic
│   │   ├── gradcam.py                     # Visualization
│   │   └── logger_config.py               # Logging
│   └── models/
│       ├── cada/                          # CADA architecture
│       ├── yolo/                          # YOLO-based models
│       └── resnet/                        # ResNet-based models
├── docs/
│   ├── SETUP.md                           # This file
│   ├── QUICKSTART.md                      # Quick start guide
│   └── API_README.md                      # API documentation
├── environment_api.yml                    # Conda environment
└── requirements.txt                       # Python dependencies
```

## Khởi Động API Server

### Lệnh Đầy Đủ

```bash
conda activate klgrade_api

python scripts/deployment/kiocmil_api_server.py \
  --kiocmil-model runs/kiocmil_cada/cada_10class_balanced/best_acc_model.pt \
  --knee-model runs/detect/my_knee_run_resplit/weights/best.pt \
  --lesion-model runs/detect/my_knee_run_resplit/weights/best.pt \
  --port 8001 \
  --host 0.0.0.0
```

### Tham Số

- `--kiocmil-model`: Đường dẫn đến KIOCMIL-CADA model
- `--knee-model`: Đường dẫn đến YOLO knee detector
- `--lesion-model`: Đường dẫn đến YOLO lesion detector
- `--port`: Port để chạy API (mặc định: 8001)
- `--host`: Host address (mặc định: 0.0.0.0)

### Kiểm Tra Server

```bash
# Health check
curl http://localhost:8001/health

# Model info
curl http://localhost:8001/model_info
```

## Testing

### 1. Swagger UI (Interactive)

Mở browser: http://localhost:8001/docs

### 2. Python Test Script

```bash
python scripts/deployment/test_api.py \
  --url http://localhost:8001 \
  --image datasets/dataset_v0/images/sample.jpg \
  --output-dir test_outputs
```

### 3. cURL

```bash
curl -X POST "http://localhost:8001/predict" \
  -F "file=@path/to/image.jpg" \
  -F "knee_conf=0.5" \
  -F "lesion_conf=0.5"
```

## Troubleshooting

### 1. PyTorch DLL Error (Windows)

**Lỗi:** `OSError: [WinError 182] The operating system cannot run %1`

**Giải pháp:**

```bash
conda activate klgrade_api
scripts/setup/fix_pytorch.bat
```

Script này sẽ:

- Gỡ PyTorch cũ
- Cài lại PyTorch qua pip
- Verify installation

### 2. CUDA Not Available

**Kiểm tra:**

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Nếu False:**

- API vẫn chạy được trên CPU (chậm hơn)
- Kiểm tra NVIDIA driver: `nvidia-smi`
- Kiểm tra CUDA version compatibility

### 3. Model File Not Found

**Lỗi:** `FileNotFoundError: Model file not found`

**Giải pháp:**

- Kiểm tra đường dẫn model files
- Sử dụng đường dẫn tuyệt đối
- Đảm bảo model files đã được download/trained

### 4. Port Already in Use

**Lỗi:** `OSError: [Errno 98] Address already in use`

**Giải pháp:**

```bash
# Windows
netstat -ano | findstr :8001
taskkill /PID <PID> /F

# Linux/Mac
lsof -i :8001
kill -9 <PID>
```

Hoặc dùng port khác:

```bash
python scripts/deployment/kiocmil_api_server.py ... --port 8002
```

### 5. Out of Memory (GPU)

**Lỗi:** `CUDA out of memory`

**Giải pháp:**

- Giảm batch size (nếu process nhiều ảnh)
- Dùng CPU: Set `CUDA_VISIBLE_DEVICES=""`
- Upgrade GPU hoặc giảm resolution ảnh input

## Performance Optimization

### GPU Acceleration

- Đảm bảo CUDA available: `torch.cuda.is_available() == True`
- First inference sẽ chậm (model loading)
- Subsequent inferences nhanh hơn (~200-500ms)

### CPU Mode

- Set environment variable: `export CUDA_VISIBLE_DEVICES=""`
- Inference time: ~2-5 giây/ảnh

### Batch Processing

- Sử dụng `/predict` endpoint cho nhiều ảnh
- API hỗ trợ xử lý nhiều knees trong 1 ảnh

## Logging

Logs được lưu trong `logs/`:

```
logs/
├── api_2026-01-23.log
├── api_2026-01-24.log
└── ...
```

Log rotation: 10MB per file, giữ 5 files gần nhất

## Uninstall

```bash
# Xóa conda environment
conda env remove -n klgrade_api -y

# Xóa logs (optional)
rm -rf logs/
```

## Tài Liệu Bổ Sung

- **Quick Start**: `docs/QUICKSTART.md`
- **API Documentation**: `docs/API_README.md`
- **Two-Step YOLO API**: `docs/TWO_STEP_YOLO_API.md`
- **Model Evaluation**: `docs/MODEL_EVALUATION.md`
- **Swagger UI**: http://localhost:8001/docs (khi server chạy)

## Liên Hệ & Support

Nếu gặp vấn đề:

1. Kiểm tra section Troubleshooting
2. Xem logs trong `logs/`
3. Chạy `test_setup.bat` để verify environment
4. Tạo issue với thông tin chi tiết về lỗi
