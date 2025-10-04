#!/usr/bin/env python3
"""
Dataset Module for YOLO Object Detection
=======================================

Module chứa class ImageDataset để xử lý ảnh và label files cho object detection.
Hỗ trợ nhiều cách khởi tạo và xử lý augmentation an toàn.

Features:
- Hỗ trợ khởi tạo từ DataFrame, CSV file, hoặc thư mục
- Xử lý augmentation với Albumentations
- Validation và clipping bbox coordinates
- Error handling robust
- Tương thích với torchvision detection models

Author: NgoTam
Date: 2025
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import pandas as pd
from config import IMG_SIZE

# Import utility functions
from utils import (
    yolo_to_xyxy,
    filter_valid_boxes,
    load_yolo_labels,
    validate_yolo_bboxes,
    clip_yolo_bboxes,
    post_augment_validation,
    create_dataset_dataframe
)


class ImageDataset(Dataset):
    """
    Flexible dataset for images and YOLO-format labels.

    Có thể khởi tạo từ:
        - pandas DataFrame (df)
        - CSV file path (df as str)
        - image và label folders (images_dir, labels_dir)

    Nếu không cung cấp transform, ảnh sẽ được resize về IMG_SIZE từ config.py.

    Attributes:
        data (list): Danh sách đường dẫn ảnh
        label (list): Danh sách đường dẫn label files
        transforms (A.Compose): Albumentations transform pipeline
    """

    def __init__(
        self,
        df=None,
        images_dir=None,
        labels_dir=None,
        transforms=None,
    ):
        """
        Khởi tạo ImageDataset

        Args:
            df (pd.DataFrame or str, optional): DataFrame hoặc đường dẫn CSV
            images_dir (str, optional): Đường dẫn thư mục ảnh
            labels_dir (str, optional): Đường dẫn thư mục labels
            transforms (A.Compose, optional): Albumentations transform pipeline
        """
        # Thiết lập transform mặc định nếu không được cung cấp
        if transforms is None:
            self.transforms = A.Compose(
                [A.Resize(IMG_SIZE, IMG_SIZE), ToTensorV2()],
                bbox_params=A.BboxParams(
                    format="yolo",
                    label_fields=["class_labels"],
                    min_visibility=0.0,  # Cho phép bbox có visibility thấp
                    clip=True,  # Tự động clip bbox về [0,1]
                ),
            )
        else:
            self.transforms = transforms

        # Khởi tạo dataset từ các nguồn khác nhau
        if isinstance(df, str):
            # Nếu df là string, coi như đường dẫn CSV file
            df = pd.read_csv(df)

        if df is not None:
            self.data = df["data"]  # Danh sách đường dẫn ảnh
            self.label = df["label"]  # Danh sách đường dẫn label files
        elif images_dir is not None and labels_dir is not None:
            df = create_dataset_dataframe(images_dir, labels_dir)
            self.data = df["data"]
            self.label = df["label"]
        else:
            raise ValueError(
                "You must provide either df (DataFrame or CSV path) or images_dir and labels_dir"
            )

    def __len__(self):
        """Trả về số lượng samples trong dataset"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Lấy một sample từ dataset

        Args:
            idx (int): Index của sample

        Returns:
            dict: Dictionary chứa 'image' và 'target' với 'boxes' và 'labels'
        """
        img_path = self.data[idx]
        label_path = self.label[idx]

        # Load ảnh với error handling
        image = self._load_image(img_path)
        boxes = self._load_labels(label_path)

        # Validation và clipping bbox trước khi augment
        is_valid, error_messages = validate_yolo_bboxes(
            boxes, class_range=(1, 5), min_size=0.001)
        if not is_valid:
            print(
                f"[WARNING] Invalid bboxes in {os.path.basename(img_path)}: {error_messages}")
        boxes = clip_yolo_bboxes(boxes)

        # Áp dụng augmentation nếu có
        if self.transforms:
            image, boxes, labels = self._apply_augmentation(
                image, boxes, img_path)
        else:
            # Không có transform, chuyển đổi sang tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            labels = (
                torch.zeros((0,), dtype=torch.int64)
                if boxes.shape[0] == 0
                else torch.tensor(boxes[:, 0], dtype=torch.int64)
            )
            boxes = (
                torch.zeros((0, 4), dtype=torch.float32)
                if boxes.shape[0] == 0
                else torch.tensor(boxes[:, 1:], dtype=torch.float32)
            )

        # Chuyển đổi sang format phù hợp với torchvision detection models
        return self._format_output(image, boxes, labels)

    def _load_image(self, img_path):
        """
        Load ảnh với error handling

        Args:
            img_path (str): Đường dẫn ảnh

        Returns:
            np.ndarray: Ảnh RGB
        """
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(
                f"Warning: Could not load image {img_path}, using dummy image")
            # Trả về ảnh dummy với zeros
            image = np.zeros((224, 224), dtype=np.uint8)

        # Chuyển đổi sang RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image

    def _load_labels(self, label_path):
        """
        Load labels từ file sử dụng utils function.

        Args:
            label_path (str): Đường dẫn label file

        Returns:
            np.ndarray: Array chứa bbox [class_id, x, y, w, h]
        """
        # 0-based sang 1-based cho training
        return load_yolo_labels(label_path, class_offset=1)

    def _apply_augmentation(self, image, boxes, img_path):
        """
        Áp dụng augmentation với error handling

        Args:
            image (np.ndarray): Ảnh input
            boxes (np.ndarray): Bbox coordinates
            img_path (str): Đường dẫn ảnh (để logging)

        Returns:
            tuple: (transformed_image, transformed_boxes, transformed_labels)
        """
        try:
            # Chuẩn bị data cho albumentations
            bboxes = boxes[:, 1:] if len(boxes) > 0 else []
            class_labels = boxes[:, 0] if len(boxes) > 0 else []

            # Áp dụng transform
            transformed = self.transforms(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels,
            )

            image = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["class_labels"]

            # Validate và fix bbox sau augmentation
            if len(boxes) > 0:
                boxes, labels = post_augment_validation(boxes, labels)

            return image, boxes, labels

        except Exception as e:
            print(
                f"[ERROR] Augmentation failed for {os.path.basename(img_path)}: {e}")
            print(f"[FALLBACK] Using empty boxes to continue training")

            # Fallback: sử dụng ảnh gốc với empty boxes
            try:
                transformed = self.transforms(
                    image=image,
                    bboxes=[],
                    class_labels=[],
                )
                return transformed["image"], np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
            except:
                # Nếu cả fallback cũng fail, trả về ảnh gốc
                image_tensor = torch.from_numpy(
                    image).permute(2, 0, 1).float() / 255.0
                return image_tensor, np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    def _format_output(self, image, boxes, labels):
        """
        Format output cho torchvision detection models

        Args:
            image (torch.Tensor): Ảnh tensor
            boxes (np.ndarray): Bbox coordinates
            labels (np.ndarray): Class labels

        Returns:
            dict: Dictionary với 'image' và 'target'
        """
        if len(boxes) > 0:
            # Chuyển đổi box sang xyxy format và lọc box hợp lệ
            boxes = (
                np.array(boxes, dtype=np.float32)
                if len(boxes) > 0
                else np.zeros((0, 4), dtype=np.float32)
            )

            # Lấy kích thước ảnh
            if torch.is_tensor(image):
                img_size = image.shape[1:3]  # (H, W)
            else:
                img_size = image.shape[:2]  # (H, W)
            img_size = (img_size[1], img_size[0])  # Chuyển thành (W, H)

            # Chuyển đổi sang xyxy format
            bboxes_xyxy, labels_np = yolo_to_xyxy(
                boxes, labels, img_size=img_size, clip=True, filter_invalid=True
            )

            # Convert sang tensor
            labels = (
                torch.tensor(labels_np, dtype=torch.int64)
                if not torch.is_tensor(labels_np)
                else labels_np.clone().detach().to(torch.int64)
            )
            bboxes = torch.tensor(bboxes_xyxy, dtype=torch.float32)
        else:
            labels = torch.zeros((0,), dtype=torch.int64)
            bboxes = torch.zeros((0, 4), dtype=torch.float32)

        return {
            "image": image.float(),
            "target": {"boxes": bboxes, "labels": labels},
        }
