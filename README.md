# KLGrade

## Mô tả

KLGrade là dự án phát hiện và định độ nặng thoái hóa khớp gối (Kellgren–Lawrence, KL) trên ảnh X‑quang. Hiện tại dự án tập trung vào bài toán Detection (dạng YOLO) trên ảnh crop vùng gối, cho phép huấn luyện và suy luận nhanh.

### Tổng quan dự án

- Mục tiêu: Xây dựng pipeline end‑to‑end để:
  - Tiền xử lý dữ liệu từ ảnh toàn chân → crop vùng gối → tái chiếu nhãn KL vào crop (YOLO format)
  - Tạo tập train/val (splits)
  - Huấn luyện mô hình phát hiện (mặc định Ultralytics YOLO) trên ảnh crop vùng gối
- Thành phần chính:
  - `pipeline/preprocess_knee_kl.py`: tiền xử lý 2 giai đoạn (crop gối + tái chiếu KL)
  - `pipeline/train_det.py`: chuẩn bị dataset cho YOLO (train.txt/val.txt/dataset.yaml) và gọi train
  - `pipeline/preview_det_dataset.py`: xem nhanh dữ liệu qua `dataset.py` (vẽ bbox)
  - `pipeline/models/`: kiến trúc backend detector có thể hoán đổi (mặc định `ultralytics`)
- Dữ liệu đầu vào: ảnh gốc và nhãn YOLO (knee, KL) theo định dạng `class cx cy w h` (normalized)
- Dữ liệu đầu ra cho train: `processed/knee/images`, `processed/knee/labels`, `splits/train.txt`, `splits/val.txt`, và `processed/det/dataset.yaml`

### Thay đổi mô hình (backend detector)

- Mặc định dùng Ultralytics YOLO. Có thể đổi sang backend khác mà không sửa logic train:
  - Chạy: `python pipeline/train_det.py --backend ultralytics --model yolov8n.pt`
  - Hoặc đổi mặc định trong `pipeline/models/__init__.py` (biến `DEFAULT_BACKEND`)
  - Thêm backend mới: tạo lớp kế thừa `BaseDetector`, viết factory `build_xxx_detector` và đăng ký vào `BACKENDS` (xem `pipeline/models/README.md`)

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

- `pipeline/`: Tiền xử lý, huấn luyện YOLO (detection-only)
- `dataset/`: Chứa ảnh và label
- `OAI-KL/`: Code base tham khảo
- `config.py`: Cấu hình chung (bao gồm `CLASSES`)

## Hướng dẫn sử dụng

### Thiết lập môi trường nhanh

```sh
python -m venv .venv
./.venv/Scripts/Activate.ps1   # Windows PowerShell
pip install -r requirements.txt
```

### Tiền xử lý dữ liệu (two-stage)

```powershell
python pipeline/preprocess_knee_kl.py `
  --img_dir dataset/dataset_v0/images `
  --knee_labels dataset/dataset_v0/labels-knee `
  --kl_labels dataset/dataset_v0/labels `
  --knee_size 512 `
  --lesion_size 512
```

Kết quả: `processed/knee/images` và `processed/knee/labels` (YOLO trên knee-crop).

### Tạo splits (train/val)

**Với labels chuẩn (CLASSES):**

```powershell
python check_dataset/split_dataset.py `
  --img_dir processed/knee/images `
  --label_dir processed/knee/labels `
  --out_dir splits `
  --train 0.7 `
  --val 0.15 `
  --test 0.15 `
  --seed 42
```

**Với labels mới (CLASSES_LABEL_NEW):**

1. Sửa `check_dataset/split_dataset.py` dòng 34-35:
   ```python
   # from config import CLASSES
   from config import CLASSES_LABEL_NEW as CLASSES
   ```

2. Chạy split:
   ```powershell
   python check_dataset/split_dataset.py `
     --img_dir processed/knee/images `
     --label_dir processed/knee/labels_new `
     --out_dir splits `
     --train 0.7 `
     --val 0.15 `
     --test 0.15 `
     --seed 42
   ```

**Lưu ý:**
- Script tự động đọc class_id từ labels (hỗ trợ cả format chuẩn 0-4 và format mới 0-9)
- Khi dùng `labels_new`, cần sửa import trong `split_dataset.py` để hiển thị đúng tên class (KL0-a, KL0-b, ...)
- Sinh `splits/train.txt`, `splits/val.txt`, `splits/test.txt` (tên theo stem của ảnh crop, bao gồm cả index `_{k_idx}`)
- Sử dụng multilabel stratified split để giữ phân phối class giữa các splits

### Huấn luyện detector

```powershell
pip install ultralytics
python pipeline/train_det.py `
  --img_dir processed/knee/images `
  --splits_dir splits `
  --model yolov8n.pt `
  --epochs 100 `
  --imgsz 640 `
  --batch 16
```

Sinh `processed/det/train.txt`, `processed/det/val.txt`, `processed/det/dataset.yaml` và kết quả trong `processed/det/runs/`.

### Xem nhanh dữ liệu (preview)

```powershell
python pipeline/preview_det_dataset.py `
  --img_dir processed/knee/images `
  --splits_dir splits `
  --split train `
  --n 5 `
  --out_dir check_vis/preview
```

## Pipeline phát hiện KL (detection)

Luồng chuẩn end‑to‑end: kiểm tra dữ liệu → tiền xử lý crop gối + tái chiếu KL → tạo splits → train → (tuỳ chọn) preview.

0. Kiểm tra dữ liệu (khuyến nghị)

   ```powershell
   python check_dataset/check_image_label.py --img_dir dataset/dataset_v0/images --label_dir dataset/dataset_v0/labels
   python check_dataset/visualize_yolo_boxes.py --img_dir dataset/dataset_v0/images --label_dir dataset/dataset_v0/labels --out_dir check_vis
   ```

1. Tiền xử lý 2 giai đoạn (knee ROI + KL)

   - Sinh knee crops và nhãn KL dạng YOLO trên crop: `processed/knee/images|labels`
   - Knee ROI được mở rộng/di chuyển bao trọn box KL giao nhau (thêm đệm 2 px)

2. Tạo splits

   - Chạy `check_dataset/split_dataset.py` với `--label_dir` tương ứng (labels hoặc labels_new)
   - Sinh `splits/train.txt`, `splits/val.txt`, `splits/test.txt` theo stem ảnh crop (bao gồm index `_{k_idx}`)

3. Train detector

   - Dùng `pipeline/train_det.py` (mặc định Ultralytics YOLO)
   - Sinh `processed/det/*.txt` và `processed/det/dataset.yaml`

4. Preview dataset (tuỳ chọn)
   - Vẽ bbox để kiểm tra nhanh qua `pipeline/preview_det_dataset.py`

## Lưu ý

- Nếu gặp lỗi liên quan đến numpy, scikit-learn, opencv, hãy xóa và tạo lại môi trường ảo, sau đó cài lại đúng các phiên bản trong [requirements.txt](requirements.txt).
- Nếu dùng GPU, đảm bảo driver và CUDA toolkit đã cài đặt đúng.
- Nếu dùng Linux, nên sử dụng Python >=3.10 và pip mới nhất.

## Tham khảo

- PyTorch: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
- Ultralytics YOLO: [https://docs.ultralytics.com/](https://docs.ultralytics.com/)
- OpenCV: [https://docs.opencv.org/](https://docs.opencv.org/)
- scikit-learn (train/val split): [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
- TorchVision Detection Models: [https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection](https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection)
- Kellgren–Lawrence grading (wiki): [https://en.wikipedia.org/wiki/Kellgren%E2%80%93Lawrence_grading_scale](https://en.wikipedia.org/wiki/Kellgren%E2%80%93Lawrence_grading_scale)
