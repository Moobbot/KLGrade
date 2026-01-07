"""
Dataset loaders for object detection tasks.

Available datasets:
- YoloDataset: For YOLO11 (in parent dataset.py)
- CocoDataset: For DETR and transformer-based models
"""

from .coco_dataset import CocoDataset
from .converters import create_coco_json, yolo_to_coco_bbox
from .detr_transforms import (
    get_detr_processor,
    detr_collate_fn,
    detr_collate_fn_dynamic_padding,
)

__all__ = [
    "CocoDataset",
    "create_coco_json",
    "yolo_to_coco_bbox",
    "get_detr_processor",
    "detr_collate_fn",
    "detr_collate_fn_dynamic_padding",
]
