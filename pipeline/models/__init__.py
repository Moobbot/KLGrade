from typing import Callable, Dict

from .yolo_ultralytics import YoloUltralyticsDetector, build_yolo_detector


# Registry cho các detector backend. Chỉ cần sửa đúng 1 chỗ: BACKENDS["default"]
BACKENDS: Dict[str, Callable[[str], object]] = {
    "ultralytics": build_yolo_detector,
}

# Chọn backend mặc định tại MỘT NƠI này
DEFAULT_BACKEND: str = "ultralytics"


def build_detection_model(weights: str = "yolov8n.pt", backend: str | None = None):
    """Factory khởi tạo detector theo backend.

    Thay đổi backend mặc định chỉ ở duy nhất biến DEFAULT_BACKEND ở trên.
    Khi thêm model mới, chỉ cần đăng ký vào BACKENDS.
    """
    chosen = (backend or DEFAULT_BACKEND).lower()
    if chosen not in BACKENDS:
        available = ", ".join(sorted(BACKENDS.keys()))
        raise ValueError(f"Unknown backend '{chosen}'. Available: {available}")
    return BACKENDS[chosen](weights)