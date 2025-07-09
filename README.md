# KLGrade

## Mô tả

Dự án KLGrade là hệ thống nhận diện và phân loại hình ảnh X-quang đầu gối sử dụng PyTorch và các mô hình detection/classification hiện đại.

## Yêu cầu hệ thống

- Python >= 3.10
- Windows 10/11 **hoặc** Linux (Ubuntu khuyến nghị)
- GPU NVIDIA (nếu muốn tăng tốc với CUDA) hoặc chỉ CPU

## Cài đặt môi trường

### 1. Tạo môi trường ảo (khuyến nghị)

```sh
python -m venv .venv
# Kích hoạt môi trường ảo:
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Windows CMD:
.\.venv\Scripts\activate.bat
# Linux/macOS:
source .venv/bin/activate
```

### 2. Cài đặt PyTorch

**Chọn lệnh phù hợp với hệ điều hành và phần cứng của bạn:**

- **Truy cập:** <https://pytorch.org/get-started/locally/> để lấy lệnh cài đặt phù hợp nhất.

#### Ví dụ

- **Windows/Linux với GPU (CUDA 11.8):**

  ```sh
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

- **Windows/Linux chỉ dùng CPU:**

  ```sh
  pip install torch torchvision torchaudio
  ```

### 3. Cài đặt các thư viện phụ thuộc

```sh
pip install --upgrade pip
pip install -r requirements.txt
```

## Cấu trúc thư mục

- `main.py`, `main_test.py`: Chạy train/test mô hình
- `dataset/`: Chứa ảnh và label
- `OAI-KL/`: Code base tham khảo Các script xử lý, đánh giá, visualize
- `config.py`: Cấu hình chung
- `model.py`: Định nghĩa model
- `early_stop.py`: EarlyStopping callback

## Hướng dẫn sử dụng

### Train mô hình

```sh
python main.py -m resnet_101 -i 224
```

### Test mô hình

```sh
python main_test.py -m resnet_101 -i 224
```

## Lưu ý

- Nếu gặp lỗi liên quan đến numpy, scikit-learn, opencv, hãy xóa và tạo lại môi trường ảo, sau đó cài lại đúng các phiên bản trong `requirements.txt`.
- Nếu dùng GPU, đảm bảo driver và CUDA toolkit đã cài đặt đúng.
- Nếu dùng Linux, nên sử dụng Python >=3.10 và pip mới nhất.

## Tham khảo

- [PyTorch Get Started](https://pytorch.org/get-started/locally/)
- [TorchVision Detection Models](https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection)
