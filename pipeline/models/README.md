## Detection backends (Backend detector)

Mục tiêu: cho phép thay mô hình detection khác nhau mà không sửa code huấn luyện. Tất cả backend đều tuân theo một giao diện chung và được chọn qua 1 chỗ duy nhất.

### Kiến trúc nhanh

- `base_det.py`: định nghĩa interface `BaseDetector` với các phương thức: `train`, `val`, `predict`, `export`.
- `yolo_ultralytics.py`: triển khai backend Ultralytics YOLO dưới dạng lớp `YoloUltralyticsDetector` và factory `build_yolo_detector`.
- `__init__.py`: registry `BACKENDS` và `DEFAULT_BACKEND`, hàm `build_detection_model(weights, backend=None)`.

### Dùng ngay

- Mặc định: Ultralytics.
- Chạy CLI và chọn backend:

```bash
python pipeline/train_det.py --backend ultralytics --model yolov8n.pt
```

- Đổi mặc định cho toàn dự án: mở `pipeline/models/__init__.py` và chỉnh 1 dòng:

```python
DEFAULT_BACKEND = "ultralytics"  # đổi tên backend ở đây
```

### Thêm backend mới (ví dụ: mmdet, detectron2, custom PyTorch)

1. Tạo file mới, ví dụ `my_backend.py`, cài đặt lớp detector kế thừa `BaseDetector`:

```python
from typing import Any
from .base_det import BaseDetector

class MyDetector(BaseDetector):
    def __init__(self, weights: str = "" ):
        # TODO: khởi tạo model của bạn (load weights, cfg, ...)
        self.weights = weights
        # self.model = ...

    def train(self, **kwargs: Any):
        # TODO: ánh xạ tham số kwargs sang API train của backend
        pass

    def val(self, **kwargs: Any):
        pass

    def predict(self, **kwargs: Any):
        pass

    def export(self, **kwargs: Any):
        pass

def build_my_detector(weights: str = "") -> MyDetector:
    return MyDetector(weights=weights)
```

2. Đăng ký backend trong `pipeline/models/__init__.py`:

```python
from .my_backend import build_my_detector  # new import

BACKENDS = {
    "ultralytics": build_yolo_detector,
    "mybackend": build_my_detector,      # new line
}
```

3. Chọn backend:

- Tạm thời cho 1 lần chạy: `--backend mybackend`
- Hoặc đổi mặc định: `DEFAULT_BACKEND = "mybackend"`

### Quy ước giao diện

Các phương thức cần hỗ trợ (ít nhất):

- `train(**kwargs)` — chấp nhận các tham số phổ biến như `data`, `epochs`, `imgsz`, `batch`, `device`, `project`, `name`, ... (hoặc ánh xạ hợp lý sang backend của bạn).
- `val(**kwargs)`
- `predict(**kwargs)`
- `export(**kwargs)`

Lưu ý: Giữ tên tham số phổ biến để `pipeline/train_det.py` không cần thay đổi khi đổi backend.
