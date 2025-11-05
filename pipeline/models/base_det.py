"""
Base detector interface để thống nhất cách gọi cho các mô hình detection.
"""
from typing import Any
from torch import nn


class BaseDetector(nn.Module):
    def __init__(self):
        super().__init__()

    # Các phương thức sau là giao diện chung; tuỳ mô hình cụ thể mà cài đặt
    def train(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        raise NotImplementedError

    def val(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def predict(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def export(self, *args: Any, **kwargs: Any):
        raise NotImplementedError


