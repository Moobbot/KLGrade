"""
Lớp tương thích cho bài toán DETECTION: re-export hàm build_detection_model
từ package pipeline.models để bên ngoài gọi thống nhất.
"""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pipeline.models import build_detection_model  # noqa: F401

