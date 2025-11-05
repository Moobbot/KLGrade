"""
OOP wrapper cho Ultralytics YOLO để dùng trong bài toán DETECTION.

Cho phép gọi thống nhất qua các phương thức: train, val, predict, export.
"""
from typing import Any
from .base_det import BaseDetector


class YoloUltralyticsDetector(BaseDetector):
    def __init__(self, weights: str = "yolov8n.pt"):
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:  # pragma: no cover - phụ thuộc môi trường
            raise RuntimeError(
                "Ultralytics chưa được cài. Hãy chạy: pip install ultralytics\n" + str(e)
            )
        # Khởi tạo model YOLO từ checkpoint/weights
        self.weights = weights
        self.model = YOLO(weights)

    def train(self, **kwargs: Any):
        """Huấn luyện model với tham số của Ultralytics (data, epochs, imgsz, batch, device, project, name, ...)"""
        return self.model.train(**kwargs)

    def val(self, **kwargs: Any):
        """Đánh giá model."""
        return self.model.val(**kwargs)

    def predict(self, **kwargs: Any):
        """Suy luận/predict."""
        return self.model.predict(**kwargs)

    def export(self, **kwargs: Any):
        """Xuất model sang định dạng khác (onnx, openvino, etc.)."""
        return self.model.export(**kwargs)


def build_yolo_detector(weights: str = "yolov8n.pt") -> YoloUltralyticsDetector:
    """Factory trả về đối tượng detector OOP bọc Ultralytics YOLO."""
    return YoloUltralyticsDetector(weights=weights)


