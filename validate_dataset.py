#!/usr/bin/env python3
"""
Dataset Validation Script
========================

Script để kiểm tra và thống kê toàn bộ dataset để tránh lỗi trong quá trình training.
Bao gồm các chức năng:
- Kiểm tra tính hợp lệ của ảnh và label files
- Mô phỏng quá trình augment để phát hiện lỗi
- Trực quan hóa dataset và debug các vấn đề
- Tạo dataset sạch và báo cáo thống kê

Author: NgoTam
Date: 2025
"""

import os
import warnings
import glob
from collections import Counter, defaultdict
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import torch
from albumentations import pytorch as A_pytorch
from albumentations.pytorch import ToTensorV2
import albumentations as A
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Import ImageDataset để test
from dataset import ImageDataset
from utils import (
    clip_bbox_coordinates,
    validate_bbox_coordinates,
    load_yolo_labels,
    validate_yolo_bboxes,
    clip_yolo_bboxes,
    create_dataset_dataframe,
)


class DatasetValidator:
    """
    Class chính để kiểm tra và validate dataset

    Attributes:
        images_dir (str): Đường dẫn thư mục chứa ảnh
        labels_dir (str): Đường dẫn thư mục chứa label files
        stats (dict): Dictionary chứa các thống kê về dataset
    """

    def __init__(self, images_dir, labels_dir):
        """
        Khởi tạo DatasetValidator

        Args:
            images_dir (str): Đường dẫn thư mục ảnh
            labels_dir (str): Đường dẫn thư mục labels
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir

        # Khởi tạo dictionary thống kê
        self.stats = {
            "total_images": 0,  # Tổng số ảnh
            "valid_images": 0,  # Số ảnh hợp lệ
            "corrupted_images": 0,  # Số ảnh bị hỏng
            "missing_images": 0,  # Số ảnh thiếu
            "total_labels": 0,  # Tổng số label files
            "valid_labels": 0,  # Số label files hợp lệ
            "missing_labels": 0,  # Số label files thiếu
            "empty_labels": 0,  # Số label files rỗng
            "invalid_bboxes": 0,  # Số bbox không hợp lệ
            "class_distribution": Counter(),  # Phân bố class
            "bbox_size_stats": [],  # Thống kê kích thước bbox
            "image_size_stats": [],  # Thống kê kích thước ảnh
            "errors": [],  # Danh sách lỗi
            "augment_failed": 0,  # Số lần augment thất bại
            "augment_fail_list": [],  # Danh sách lỗi augment
            "dataset_test_passed": 0,  # Số lần test ImageDataset thành công
            "dataset_test_failed": 0,  # Số lần test ImageDataset thất bại
            "dataset_test_errors": [],  # Danh sách lỗi test ImageDataset
        }

    def simulate_multiple_augment_check(self, img_path, boxes):
        """
        Mô phỏng nhiều loại augment để phát hiện loại augment nào gây ra lỗi.

        Args:
            img_path (str): Đường dẫn ảnh cần kiểm tra
            boxes (list): Danh sách bbox YOLO format [class_id, x, y, w, h]

        Returns:
            tuple: (success, error_message, augment_type)
        """
        # Định nghĩa các loại augment để test
        augment_configs = {
            "Resize": A.Compose(
                [A.Resize(224, 224)],
                bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.0)
            ),
            "Resize_Rotate": A.Compose(
                [A.Resize(224, 224), A.Rotate(limit=15)],
                bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.0)
            ),
            "Resize_Flip": A.Compose(
                [A.Resize(224, 224), A.HorizontalFlip(p=0.5)],
                bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.0)
            ),
            "Resize_Brightness": A.Compose(
                [A.Resize(224, 224), A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2)],
                bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.0)
            ),
            "Resize_Noise": A.Compose(
                [A.Resize(224, 224), A.GaussNoise(var_limit=(10.0, 50.0))],
                bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.0)
            ),
        }

        try:
            # Đọc ảnh và chuyển đổi sang RGB
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            h, w = image.shape[:2]

            # Chuẩn bị boxes
            boxes = np.array(boxes, dtype=np.float32)
            class_labels = boxes[:, 0]
            boxes_clipped = clip_yolo_bboxes(boxes)
            boxes_to_transform = boxes_clipped[:, 1:].copy()

            # Test từng loại augment
            for aug_name, transform in augment_configs.items():
                try:
                    # Apply transform
                    transformed = transform(
                        image=image, bboxes=boxes_to_transform, class_labels=class_labels
                    )

                    # Kiểm tra kết quả
                    transformed_boxes = np.array(transformed["bboxes"])
                    transformed_labels = transformed["class_labels"]

                    # Kiểm tra length mismatch
                    if len(transformed_boxes) != len(transformed_labels):
                        return False, f"The lengths of bboxes and class_labels do not match. Got {len(transformed_boxes)} and {len(transformed_labels)} respectively.", aug_name

                    # Kiểm tra có bbox nào còn lại không
                    if len(transformed_boxes) == 0:
                        return False, "No bboxes after augmentation", aug_name

                except Exception as e:
                    return False, f"Error in {aug_name}: {str(e)}", aug_name

            # Nếu tất cả augment đều thành công
            return True, None, "All"

        except Exception as e:
            return False, f"General error: {str(e)}", "Unknown"

    def simulate_augment_check(self, img_path, boxes):
        """
        Mô phỏng quá trình augment để phát hiện lỗi giống như lúc training.

        Function này thực hiện các bước:
        1. Đọc ảnh và chuyển đổi sang RGB
        2. Pre-clip bbox coordinates để tránh lỗi albumentations
        3. Áp dụng transform (resize về 224x224)
        4. Post-clip bbox sau augmentation
        5. Kiểm tra tính hợp lệ của kết quả

        Args:
            img_path (str): Đường dẫn ảnh cần kiểm tra
            boxes (list): Danh sách bbox YOLO format [class_id, x, y, w, h]

        Returns:
            tuple: (success, error_message, augment_type)
                - success (bool): True nếu augment thành công, False nếu có lỗi
                - error_message (str): Thông báo lỗi chi tiết nếu có lỗi, None nếu thành công
                - augment_type (str): Loại augment gây ra lỗi (Resize, Rotate, etc.)
        """
        try:
            # Đọc ảnh và chuyển đổi sang RGB
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            h, w = image.shape[:2]

            # Chuẩn bị boxes và pre-clip để tránh lỗi
            boxes = np.array(boxes, dtype=np.float32)
            class_labels = boxes[:, 0]

            # Pre-clip boxes sử dụng utils function
            boxes_clipped = clip_yolo_bboxes(boxes)
            boxes_to_transform = boxes_clipped[:, 1:].copy()  # Chỉ lấy x, y, w, h

            # Tạo albumentations transform (chỉ resize, không dùng ToTensorV2)
            transform = A.Compose(
                [
                    A.Resize(224, 224),
                    # A.Rotate(limit=20),  # Có thể bật rotation nếu cần
                ],
                bbox_params=A.BboxParams(
                    format="yolo", label_fields=["class_labels"], min_visibility=0.0
                ),
            )

            # Apply transform
            transformed = transform(
                image=image, bboxes=boxes_to_transform, class_labels=class_labels
            )

            # Kiểm tra kết quả sau transform
            transformed_boxes = np.array(transformed["bboxes"])
            transformed_labels = transformed["class_labels"]

            # Kiểm tra length mismatch trước khi xử lý
            if len(transformed_boxes) != len(transformed_labels):
                return False, f"The lengths of bboxes and class_labels do not match. Got {len(transformed_boxes)} and {len(transformed_labels)} respectively.", "Resize"

            # Clip các tọa độ bbox về trong khoảng hợp lệ
            clipped_boxes = []
            clipped_labels = []

            for i, (x, y, w_box, h_box) in enumerate(transformed_boxes):
                class_id = transformed_labels[i]

                # Clip tọa độ center về [0, 1]
                x_clipped = np.clip(x, 0.0, 1.0)
                y_clipped = np.clip(y, 0.0, 1.0)

                # Clip width và height về (0, 1]
                w_clipped = np.clip(w_box, 1e-6, 1.0)
                h_clipped = np.clip(h_box, 1e-6, 1.0)

                # Đảm bảo bbox không vượt ra ngoài biên
                x_min = x_clipped - w_clipped / 2
                y_min = y_clipped - h_clipped / 2
                x_max = x_clipped + w_clipped / 2
                y_max = y_clipped + h_clipped / 2

                # Clip các góc bbox
                x_min = np.clip(x_min, 0.0, 1.0)
                y_min = np.clip(y_min, 0.0, 1.0)
                x_max = np.clip(x_max, 0.0, 1.0)
                y_max = np.clip(y_max, 0.0, 1.0)

                # Tính lại center và kích thước sau khi clip
                new_x = (x_min + x_max) / 2
                new_y = (y_min + y_max) / 2
                new_w = x_max - x_min
                new_h = y_max - y_min

                # Đảm bảo kích thước tối thiểu
                new_w = max(new_w, 1e-6)
                new_h = max(new_h, 1e-6)

                clipped_boxes.append([new_x, new_y, new_w, new_h])
                clipped_labels.append(class_id)

            # Kiểm tra xem có bbox nào còn lại không
            if len(clipped_boxes) == 0:
                return False, "No bboxes after clipping", "Resize"

            return True, None, "Resize"

        except Exception as e:
            return False, str(e), "Resize"

    def test_image_dataset(self, max_samples=10):
        """
        Test ImageDataset để kiểm tra xem dataset có hoạt động đúng không.

        Function này thực hiện:
        1. Tạo ImageDataset instance từ thư mục ảnh và labels
        2. Test một số samples ngẫu nhiên (mặc định 10 samples)
        3. Kiểm tra format output có đúng không
        4. Kiểm tra augmentation có hoạt động không
        5. Test với nhiều loại augmentation khác nhau (trừ bóp méo hình)
        6. Cập nhật thống kê test results

        Args:
            max_samples (int): Số lượng samples tối đa để test. Mặc định là 10.
        """
        print(f"\n=== TESTING IMAGEDATASET ===")
        print(f"Testing up to {max_samples} samples...")

        try:
            # Tạo ImageDataset instance
            dataset = ImageDataset(
                images_dir=self.images_dir, labels_dir=self.labels_dir
            )

            print(f"Dataset created successfully with {len(dataset)} samples")

            # Test một số samples
            test_samples = min(max_samples, len(dataset))
            test_indices = np.random.choice(len(dataset), test_samples, replace=False)

            for i, idx in enumerate(test_indices):
                try:
                    # Lấy sample từ dataset
                    sample = dataset[idx]

                    # Kiểm tra format output
                    if not isinstance(sample, dict):
                        raise ValueError(f"Sample {idx} is not a dictionary")

                    if "image" not in sample or "target" not in sample:
                        raise ValueError(
                            f"Sample {idx} missing 'image' or 'target' key"
                        )

                    if (
                        "boxes" not in sample["target"]
                        or "labels" not in sample["target"]
                    ):
                        raise ValueError(
                            f"Sample {idx} missing 'boxes' or 'labels' in target"
                        )

                    # Kiểm tra image tensor
                    image = sample["image"]
                    if not hasattr(image, "shape"):
                        raise ValueError(f"Sample {idx} image is not a tensor")

                    # Kiểm tra target format
                    target = sample["target"]
                    boxes = target["boxes"]
                    labels = target["labels"]

                    if not hasattr(boxes, "shape") or not hasattr(labels, "shape"):
                        raise ValueError(
                            f"Sample {idx} boxes or labels are not tensors"
                        )

                    # Kiểm tra shape consistency
                    if len(boxes) != len(labels):
                        raise ValueError(
                            f"Sample {idx} boxes and labels length mismatch: {len(boxes)} vs {len(labels)}"
                        )

                    # Kiểm tra bbox format (xyxy)
                    if len(boxes) > 0:
                        if boxes.shape[1] != 4:
                            raise ValueError(
                                f"Sample {idx} boxes should have 4 coordinates (xyxy), got {boxes.shape[1]}"
                            )

                        # Kiểm tra bbox coordinates hợp lệ
                        if torch.any(boxes < 0) or torch.any(boxes > 1):
                            print(
                                f"Warning: Sample {idx} has bbox coordinates outside [0,1]"
                            )

                    self.stats["dataset_test_passed"] += 1
                    print(f"✓ Sample {idx} passed all tests")

                except Exception as e:
                    self.stats["dataset_test_failed"] += 1
                    error_msg = f"Sample {idx} failed: {str(e)}"
                    self.stats["dataset_test_errors"].append(error_msg)
                    print(f"✗ {error_msg}")

            print(f"\nDataset Test Results:")
            print(f"  Passed: {self.stats['dataset_test_passed']}")
            print(f"  Failed: {self.stats['dataset_test_failed']}")

            if self.stats["dataset_test_errors"]:
                print(f"  Errors:")
                for error in self.stats["dataset_test_errors"][
                    :5
                ]:  # Chỉ hiển thị 5 lỗi đầu
                    print(f"    - {error}")
                if len(self.stats["dataset_test_errors"]) > 5:
                    print(
                        f"    ... and {len(self.stats['dataset_test_errors']) - 5} more errors"
                    )

        except Exception as e:
            print(f"Failed to create or test ImageDataset: {str(e)}")
            self.stats["dataset_test_failed"] += 1
            self.stats["dataset_test_errors"].append(
                f"Dataset creation failed: {str(e)}"
            )

    def test_multiple_augmentations(self, max_samples=5):
        """
        Test ImageDataset với nhiều loại augmentation khác nhau (trừ bóp méo hình).

        Function này test các augmentation strategies:
        1. No augmentation (chỉ resize)
        2. Color/Brightness augmentation
        3. Geometric augmentation (không bóp méo)
        4. Noise augmentation
        5. Mixed augmentation

        Args:
            max_samples (int): Số lượng samples tối đa để test mỗi augmentation
        """
        print(f"\n=== TESTING MULTIPLE AUGMENTATIONS ===")
        print(f"Testing {max_samples} samples per augmentation type...")

        # Định nghĩa các loại augmentation
        augmentation_configs = {
            "No Augmentation": A.Compose(
                [A.Resize(224, 224), ToTensorV2()],
                bbox_params=A.BboxParams(
                    format="yolo", label_fields=["class_labels"], min_visibility=0.0
                ),
            ),
            "Color Augmentation": A.Compose(
                [
                    A.Resize(224, 224),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=0.8
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=20,
                        sat_shift_limit=30,
                        val_shift_limit=20,
                        p=0.8,
                    ),
                    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                    ToTensorV2(),
                ],
                bbox_params=A.BboxParams(
                    format="yolo", label_fields=["class_labels"], min_visibility=0.0
                ),
            ),
            "Geometric Augmentation": A.Compose(
                [
                    A.Resize(224, 224),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.2),
                    A.Rotate(limit=15, p=0.7),  # Rotation nhẹ, không bóp méo
                    A.ShiftScaleRotate(
                        shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.6
                    ),
                    ToTensorV2(),
                ],
                bbox_params=A.BboxParams(
                    format="yolo", label_fields=["class_labels"], min_visibility=0.0
                ),
            ),
            "Noise Augmentation": A.Compose(
                [
                    A.Resize(224, 224),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                    A.MotionBlur(blur_limit=3, p=0.3),
                    ToTensorV2(),
                ],
                bbox_params=A.BboxParams(
                    format="yolo", label_fields=["class_labels"], min_visibility=0.0
                ),
            ),
            "Mixed Augmentation": A.Compose(
                [
                    A.Resize(224, 224),
                    A.OneOf(
                        [
                            A.RandomBrightnessContrast(
                                brightness_limit=0.2, contrast_limit=0.2
                            ),
                            A.HueSaturationValue(
                                hue_shift_limit=20,
                                sat_shift_limit=30,
                                val_shift_limit=20,
                            ),
                            A.RandomGamma(gamma_limit=(80, 120)),
                        ],
                        p=0.7,
                    ),
                    A.OneOf(
                        [
                            A.HorizontalFlip(p=0.5),
                            A.Rotate(limit=15),
                            A.ShiftScaleRotate(
                                shift_limit=0.1, scale_limit=0.1, rotate_limit=10
                            ),
                        ],
                        p=0.6,
                    ),
                    A.OneOf(
                        [
                            A.GaussNoise(var_limit=(10.0, 50.0)),
                            A.GaussianBlur(blur_limit=(3, 7)),
                        ],
                        p=0.4,
                    ),
                    ToTensorV2(),
                ],
                bbox_params=A.BboxParams(
                    format="yolo", label_fields=["class_labels"], min_visibility=0.0
                ),
            ),
        }

        # Test từng loại augmentation
        for aug_name, transform in augmentation_configs.items():
            print(f"\n--- Testing {aug_name} ---")

            try:
                # Tạo dataset với augmentation cụ thể
                dataset = ImageDataset(
                    images_dir=self.images_dir,
                    labels_dir=self.labels_dir,
                    transforms=transform,
                )

                # Test một số samples
                test_samples = min(max_samples, len(dataset))
                test_indices = np.random.choice(
                    len(dataset), test_samples, replace=False
                )

                aug_passed = 0
                aug_failed = 0
                aug_errors = []

                for i, idx in enumerate(test_indices):
                    try:
                        # Lấy sample từ dataset
                        sample = dataset[idx]

                        # Kiểm tra format output
                        if not isinstance(sample, dict):
                            raise ValueError(f"Sample {idx} is not a dictionary")

                        if "image" not in sample or "target" not in sample:
                            raise ValueError(
                                f"Sample {idx} missing 'image' or 'target' key"
                            )

                        # Kiểm tra image tensor
                        image = sample["image"]
                        if not hasattr(image, "shape"):
                            raise ValueError(f"Sample {idx} image is not a tensor")

                        # Kiểm tra target format
                        target = sample["target"]
                        boxes = target["boxes"]
                        labels = target["labels"]

                        if not hasattr(boxes, "shape") or not hasattr(labels, "shape"):
                            raise ValueError(
                                f"Sample {idx} boxes or labels are not tensors"
                            )

                        # Kiểm tra shape consistency
                        if len(boxes) != len(labels):
                            raise ValueError(
                                f"Sample {idx} boxes and labels length mismatch: {len(boxes)} vs {len(labels)}"
                            )

                        # Kiểm tra bbox format (xyxy)
                        if len(boxes) > 0:
                            if boxes.shape[1] != 4:
                                raise ValueError(
                                    f"Sample {idx} boxes should have 4 coordinates (xyxy), got {boxes.shape[1]}"
                                )

                            # Kiểm tra bbox coordinates hợp lệ
                            if torch.any(boxes < 0) or torch.any(boxes > 1):
                                print(
                                    f"Warning: Sample {idx} has bbox coordinates outside [0,1]"
                                )

                        aug_passed += 1
                        print(f"  ✓ Sample {idx} passed")

                    except Exception as e:
                        aug_failed += 1
                        error_msg = f"Sample {idx} failed: {str(e)}"
                        aug_errors.append(error_msg)
                        print(f"  ✗ Sample {idx} failed: {str(e)}")

                print(f"  {aug_name} Results: {aug_passed} passed, {aug_failed} failed")

                # Cập nhật thống kê tổng thể
                self.stats["dataset_test_passed"] += aug_passed
                self.stats["dataset_test_failed"] += aug_failed
                self.stats["dataset_test_errors"].extend(
                    [f"{aug_name}: {err}" for err in aug_errors]
                )

            except Exception as e:
                print(f"  Failed to create dataset with {aug_name}: {str(e)}")
                self.stats["dataset_test_failed"] += 1
                self.stats["dataset_test_errors"].append(
                    f"{aug_name} creation failed: {str(e)}"
                )

        print(f"\n=== AUGMENTATION TEST SUMMARY ===")
        print(f"Total passed: {self.stats['dataset_test_passed']}")
        print(f"Total failed: {self.stats['dataset_test_failed']}")

        if self.stats["dataset_test_errors"]:
            print(f"Errors by augmentation type:")
            error_counts = {}
            for error in self.stats["dataset_test_errors"]:
                aug_type = error.split(":")[0]
                error_counts[aug_type] = error_counts.get(aug_type, 0) + 1

            for aug_type, count in error_counts.items():
                print(f"  {aug_type}: {count} errors")

    def visualize_augment_debug(self, img_path, boxes, save_path=None, show=True):
        """
        Trực quan hóa ảnh và bbox trước và sau augment để debug.

        Function này tạo ra visualization với 2 subplot:
        - Subplot 1: Ảnh gốc với bbox gốc (màu đỏ)
        - Subplot 2: Ảnh sau augment với bbox sau augment
          + Màu xanh: bbox không bị clip
          + Màu cam: bbox bị clip (có dấu --)

        Ngoài ra còn in thông tin debug chi tiết về:
        - Kích thước ảnh gốc
        - Tọa độ bbox gốc và sau augment
        - Số lượng bbox bị clip

        Args:
            img_path (str): Đường dẫn ảnh cần debug
            boxes (list): Danh sách bbox YOLO format [class_id, x, y, w, h]
            save_path (str, optional): Đường dẫn lưu hình ảnh debug.
                                     Nếu None thì chỉ hiển thị không lưu.
        """
        try:
            # Đọc ảnh gốc
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            h, w = image.shape[:2]

            # Chuẩn bị boxes
            boxes = np.array(boxes, dtype=np.float32)
            original_boxes = boxes.copy()
            class_labels = boxes[:, 0]

            # Tạo figure với 2 subplot
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f"Augment Debug: {os.path.basename(img_path)}", fontsize=14)

            # Subplot 1: Ảnh gốc với bbox gốc
            axes[0].imshow(image)
            axes[0].set_title("Before Augment (Original)")
            axes[0].axis("off")

            # Vẽ bbox gốc
            for i, (class_id, x, y, w_box, h_box) in enumerate(original_boxes):
                # Convert từ YOLO format sang pixel coordinates
                x_pixel = (x - w_box / 2) * w
                y_pixel = (y - h_box / 2) * h
                w_pixel = w_box * w
                h_pixel = h_box * h

                # Tạo rectangle
                rect = patches.Rectangle(
                    (x_pixel, y_pixel),
                    w_pixel,
                    h_pixel,
                    linewidth=2,
                    edgecolor="red",
                    facecolor="none",
                )
                axes[0].add_patch(rect)
                axes[0].text(
                    x_pixel,
                    y_pixel - 5,
                    f"C{class_id}",
                    color="red",
                    fontsize=10,
                    fontweight="bold",
                )

            # Subplot 2: Ảnh sau augment với bbox sau augment
            try:
                # Albumentations transform
                transform = A.Compose(
                    [A.Resize(224, 224), ToTensorV2()],
                    bbox_params=A.BboxParams(
                        format="yolo", label_fields=["class_labels"], min_visibility=0.0
                    ),
                )

                # Apply transform
                transformed = transform(
                    image=image, bboxes=boxes[:, 1:], class_labels=class_labels
                )
                transformed_image = transformed["image"]
                transformed_boxes = np.array(transformed["bboxes"])
                transformed_labels = transformed["class_labels"]

                # Convert tensor back to numpy for visualization
                if hasattr(transformed_image, "numpy"):
                    transformed_image = transformed_image.numpy()
                if hasattr(transformed_image, "permute"):
                    transformed_image = transformed_image.permute(1, 2, 0)

                # Normalize image for display
                if transformed_image.max() <= 1.0:
                    transformed_image = (transformed_image * 255).astype(np.uint8)

                axes[1].imshow(transformed_image)
                axes[1].set_title("After Augment (Resized to 224x224)")
                axes[1].axis("off")

                # Sử dụng utils function để validate và clip bbox sau augment
                if len(transformed_boxes) > 0:
                    # Tạo bbox array với class labels
                    bbox_array = np.column_stack(
                        [transformed_labels, transformed_boxes]
                    )
                    clipped_bboxes = clip_yolo_bboxes(bbox_array)

                    clipped_boxes = clipped_bboxes[:, 1:].tolist()
                    clipped_labels = clipped_bboxes[:, 0].tolist()
                    clipped_count = len(transformed_boxes) - len(clipped_boxes)
                else:
                    clipped_boxes = []
                    clipped_labels = []
                    clipped_count = 0

                # Vẽ bbox sau augment
                for i, (x, y, w_box, h_box) in enumerate(clipped_boxes):
                    class_id = clipped_labels[i]

                    # Convert từ YOLO format sang pixel coordinates (224x224)
                    x_pixel = (x - w_box / 2) * 224
                    y_pixel = (y - h_box / 2) * 224
                    w_pixel = w_box * 224
                    h_pixel = h_box * 224

                    # Tạo rectangle - màu xanh cho bbox đã được clip
                    rect = patches.Rectangle(
                        (x_pixel, y_pixel),
                        w_pixel,
                        h_pixel,
                        linewidth=2,
                        edgecolor="green",
                        facecolor="none",
                    )
                    axes[1].add_patch(rect)
                    axes[1].text(
                        x_pixel,
                        y_pixel - 5,
                        f"C{class_id}",
                        color="green",
                        fontsize=10,
                        fontweight="bold",
                    )

                # Thêm thông tin về bbox bị clip
                if clipped_count > 0:
                    axes[1].text(
                        0.02,
                        0.98,
                        f"Clipped {clipped_count} bbox(es)",
                        transform=axes[1].transAxes,
                        fontsize=10,
                        color="orange",
                        bbox=dict(
                            boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7
                        ),
                        verticalalignment="top",
                    )

            except Exception as e:
                axes[1].text(
                    0.5,
                    0.5,
                    f"Augment Error:\n{str(e)}",
                    ha="center",
                    va="center",
                    transform=axes[1].transAxes,
                    fontsize=10,
                    color="red",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow"),
                )
                axes[1].set_title("After Augment (ERROR)")

            # In thông tin debug
            # print(f"\n=== DEBUG INFO for {os.path.basename(img_path)} ===")
            # print(f"Original image size: {w}x{h}")
            # print(f"Original boxes:")
            # for i, (class_id, x, y, w_box, h_box) in enumerate(original_boxes):
            #     print(
            #         f"  Box {i}: Class={class_id}, x={x:.6f}, y={y:.6f}, w={w_box:.6f}, h={h_box:.6f}"
            #     )

            # Thử augment và kiểm tra kết quả sử dụng utils functions
            try:
                transform = A.Compose(
                    [
                        A.Resize(224, 224),
                    ],
                    bbox_params=A.BboxParams(
                        format="yolo", label_fields=["class_labels"], min_visibility=0.0
                    ),
                )

                transformed = transform(
                    image=image, bboxes=boxes[:, 1:], class_labels=class_labels
                )
                transformed_boxes = np.array(transformed["bboxes"])
                transformed_labels = transformed["class_labels"]

                print(f"\nAfter augmentation:")

                # Sử dụng utils function để clip bbox
                if len(transformed_boxes) > 0:
                    bbox_array = np.column_stack(
                        [transformed_labels, transformed_boxes]
                    )
                    clipped_bboxes = clip_yolo_bboxes(bbox_array)
                    clipped_count = len(transformed_boxes) - len(clipped_bboxes)

                    # print(f"  Original boxes: {len(transformed_boxes)}")
                    # print(f"  Clipped boxes: {len(clipped_bboxes)}")
                    # print(f"  Removed boxes: {clipped_count}")

                    for i, box in enumerate(clipped_bboxes):
                        class_id, x, y, w, h = box
                        # print(
                        #     f"  Box {i}: Class={int(class_id)}, x={x:.6f}, y={y:.6f}, w={w:.6f}, h={h:.6f} ✓"
                        # )
                else:
                    print("  No boxes after augmentation")

            except Exception as e:
                print(f"Error during augmentation: {e}")

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"Debug visualization saved to: {save_path}")

            if show:
                plt.show()
            else:
                plt.close()  # Đóng figure để tiết kiệm memory

        except Exception as e:
            print(f"Error in visualize_augment_debug: {str(e)}")
            import traceback

            traceback.print_exc()

    def validate_image(self, img_path):
        """
        Kiểm tra tính hợp lệ của ảnh.

        Function này thực hiện các kiểm tra:
        1. File tồn tại không
        2. Có thể đọc được ảnh không (không bị corrupt)
        3. Kích thước ảnh có hợp lệ không (>= 50x50 pixels)
        4. Cập nhật thống kê về kích thước ảnh

        Args:
            img_path (str): Đường dẫn ảnh cần kiểm tra

        Returns:
            tuple: (is_valid, error_message)
                - is_valid (bool): True nếu ảnh hợp lệ, False nếu có lỗi
                - error_message (str): Thông báo lỗi chi tiết nếu có lỗi, None nếu hợp lệ
        """
        if not os.path.exists(img_path):
            self.stats["missing_images"] += 1
            return False, f"Missing image: {img_path}"

        try:
            # Đọc ảnh
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                self.stats["corrupted_images"] += 1
                return False, f"Corrupted image: {img_path}"

            # Kiểm tra kích thước ảnh
            h, w = image.shape
            self.stats["image_size_stats"].append((w, h))

            # Kiểm tra ảnh có quá nhỏ không
            if h < 50 or w < 50:
                return False, f"Image too small: {img_path} ({w}x{h})"

            self.stats["valid_images"] += 1
            return True, None

        except Exception as e:
            self.stats["corrupted_images"] += 1
            return False, f"Error reading image {img_path}: {str(e)}"

    def validate_label(self, label_path):
        """
        Kiểm tra tính hợp lệ của label file sử dụng utils functions.

        Args:
            label_path (str): Đường dẫn label file cần kiểm tra

        Returns:
            tuple: (is_valid, boxes, error_message)
                - is_valid (bool): True nếu label hợp lệ, False nếu có lỗi
                - boxes (list): Danh sách bbox YOLO format [class_id, x, y, w, h]
                - error_message (str): Thông báo lỗi chi tiết nếu có lỗi, None nếu hợp lệ
        """
        if not os.path.exists(label_path):
            self.stats["missing_labels"] += 1
            return False, [], f"Missing label: {label_path}"

        try:
            # Load labels sử dụng utils function
            boxes = load_yolo_labels(
                label_path, class_offset=0
            )  # Giữ nguyên 0-based cho validation

            # Kiểm tra file rỗng
            if len(boxes) == 0:
                self.stats["empty_labels"] += 1
                return True, [], None

            # Validate bboxes sử dụng utils function
            is_valid, error_messages = validate_yolo_bboxes(
                boxes, class_range=(0, 4), min_size=0.001
            )

            if not is_valid:
                self.stats["invalid_bboxes"] += 1
                return (
                    False,
                    [],
                    f"Invalid bboxes in {label_path}: {error_messages[0] if error_messages else 'Unknown error'}",
                )

            # Cập nhật thống kê
            for box in boxes:
                class_id, x, y, w, h = box
                self.stats["class_distribution"][int(class_id)] += 1
                self.stats["bbox_size_stats"].append((w, h))

            self.stats["valid_labels"] += 1
            return True, boxes.tolist(), None

        except Exception as e:
            return False, [], f"Error reading label {label_path}: {str(e)}"

    def validate_dataset(self):
        """
        Kiểm tra toàn bộ dataset.

        Function này thực hiện validation toàn diện:
        1. Lấy danh sách tất cả file ảnh (.jpg, .jpeg, .png)
        2. Với mỗi ảnh:
           - Kiểm tra tính hợp lệ của ảnh
           - Kiểm tra tính hợp lệ của label file tương ứng
           - Nếu có bbox, mô phỏng quá trình augment để phát hiện lỗi
        3. Cập nhật thống kê tổng thể về dataset
        4. Trả về dictionary chứa tất cả thống kê

        Returns:
            dict: Dictionary chứa thống kê chi tiết về dataset:
                - total_images: Tổng số ảnh
                - valid_images: Số ảnh hợp lệ
                - corrupted_images: Số ảnh bị hỏng
                - missing_images: Số ảnh thiếu
                - total_labels: Tổng số label files
                - valid_labels: Số label files hợp lệ
                - missing_labels: Số label files thiếu
                - empty_labels: Số label files rỗng
                - invalid_bboxes: Số bbox không hợp lệ
                - class_distribution: Phân bố class
                - bbox_size_stats: Thống kê kích thước bbox
                - image_size_stats: Thống kê kích thước ảnh
                - errors: Danh sách lỗi
                - augment_failed: Số lần augment thất bại
                - augment_fail_list: Danh sách lỗi augment
        """
        print("Bat dau kiem tra dataset...")
        print(f"Images directory: {self.images_dir}")
        print(f"Labels directory: {self.labels_dir}")
        print("-" * 60)

        # Lấy danh sách tất cả file ảnh
        image_files = [
            f
            for f in os.listdir(self.images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        self.stats["total_images"] = len(image_files)

        print(f"Tong so anh: {self.stats['total_images']}")

        # Kiểm tra từng ảnh và label tương ứng
        for img_file in tqdm(image_files, desc="Kiem tra anh va labels"):
            img_path = os.path.join(self.images_dir, img_file)
            label_file = os.path.splitext(img_file)[0] + ".txt"
            label_path = os.path.join(self.labels_dir, label_file)

            # Kiểm tra ảnh
            img_valid, img_error = self.validate_image(img_path)
            if not img_valid:
                self.stats["errors"].append(img_error)
                continue

            # Kiểm tra label
            label_valid, boxes, label_error = self.validate_label(label_path)
            if not label_valid:
                self.stats["errors"].append(label_error)
                continue

            # Nếu có boxes, thử augment
            if boxes and not label_error:
                aug_valid, aug_error, augment_type = self.simulate_multiple_augment_check(img_path, boxes)
                if not aug_valid:
                    self.stats["augment_failed"] += 1
                    self.stats["augment_fail_list"].append(f"{img_path}: {aug_error} | Augment: {augment_type}")
                    self.stats["errors"].append(
                        f"Augment failed: {img_path} - {aug_error} | Augment: {augment_type} | label: {label_path}"
                    )

            if label_error:  # Empty label
                continue

        # Số label = số ảnh hợp lệ
        self.stats["total_labels"] = self.stats["valid_images"]

        return self.stats

    def print_summary(self):
        """
        In tóm tắt kết quả kiểm tra dataset.

        Function này hiển thị báo cáo chi tiết về:
        1. Thống kê ảnh (tổng số, hợp lệ, bị hỏng, thiếu)
        2. Thống kê labels (tổng số, hợp lệ, thiếu, rỗng, bbox không hợp lệ)
        3. Phân bố class với phần trăm
        4. Thống kê kích thước ảnh (min, max, avg)
        5. Thống kê kích thước bbox (min, max, avg)
        6. Số lần augment thất bại và ví dụ lỗi
        7. Danh sách lỗi phát hiện (tối đa 10 lỗi đầu)
        8. Đánh giá tổng thể chất lượng dataset

        Không có parameters và không trả về giá trị.
        """
        print("\n" + "=" * 60)
        print("TOM TAT KET QUA KIEM TRA DATASET")
        print("=" * 60)

        print(f"Tong so anh: {self.stats['total_images']}")
        print(f"Anh hop le: {self.stats['valid_images']}")
        print(f"Anh bi hong: {self.stats['corrupted_images']}")
        print(f"Anh thieu: {self.stats['missing_images']}")

        print(f"\nTong so labels: {self.stats['total_labels']}")
        print(f"Labels hop le: {self.stats['valid_labels']}")
        print(f"Labels thieu: {self.stats['missing_labels']}")
        print(f"Labels rong: {self.stats['empty_labels']}")
        print(f"Bbox khong hop le: {self.stats['invalid_bboxes']}")

        # Thống kê class distribution
        print(f"\nPhan bo class:")
        for class_id in sorted(self.stats["class_distribution"].keys()):
            count = self.stats["class_distribution"][class_id]
            percentage = (count / sum(self.stats["class_distribution"].values())) * 100
            print(f"  Class {class_id}: {count} bboxes ({percentage:.1f}%)")

        # Thống kê kích thước ảnh
        if self.stats["image_size_stats"]:
            widths = [w for w, h in self.stats["image_size_stats"]]
            heights = [h for w, h in self.stats["image_size_stats"]]
            print(f"\nKich thuoc anh:")
            print(
                f"  Width: min={min(widths)}, max={max(widths)}, avg={np.mean(widths):.1f}"
            )
            print(
                f"  Height: min={min(heights)}, max={max(heights)}, avg={np.mean(heights):.1f}"
            )

        # Thống kê kích thước bbox
        if self.stats["bbox_size_stats"]:
            widths = [w for w, h in self.stats["bbox_size_stats"]]
            heights = [h for w, h in self.stats["bbox_size_stats"]]
            print(f"\nKich thuoc bbox:")
            print(
                f"  Width: min={min(widths):.4f}, max={max(widths):.4f}, avg={np.mean(widths):.4f}"
            )
            print(
                f"  Height: min={min(heights):.4f}, max={max(heights):.4f}, avg={np.mean(heights):.4f}"
            )

        print(f"\nAugment that bai: {self.stats['augment_failed']}")
        if self.stats["augment_fail_list"]:
            # Thống kê loại augment gây ra lỗi
            augment_type_counts = {}
            for err in self.stats["augment_fail_list"]:
                if "| Augment:" in err:
                    try:
                        aug_type = err.split("| Augment:")[1].strip()
                        augment_type_counts[aug_type] = augment_type_counts.get(aug_type, 0) + 1
                    except:
                        augment_type_counts["Unknown"] = augment_type_counts.get("Unknown", 0) + 1
                else:
                    augment_type_counts["Unknown"] = augment_type_counts.get("Unknown", 0) + 1
            
            if augment_type_counts:
                print("  Thong ke loai augment gay ra loi:")
                for aug_type, count in sorted(augment_type_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / len(self.stats["augment_fail_list"])) * 100
                    print(f"    {aug_type}: {count} loi ({percentage:.1f}%)")
            
            print("  Vi du loi augment:")
            for i, err in enumerate(self.stats["augment_fail_list"][:5]):
                print(f"    {i+1}. {err}")
            if len(self.stats["augment_fail_list"]) > 5:
                print(f"    ... va {len(self.stats['augment_fail_list']) - 5} loi khac")

        # Thống kê test ImageDataset
        print(f"\nImageDataset Test:")
        print(f"  Test thanh cong: {self.stats['dataset_test_passed']}")
        print(f"  Test that bai: {self.stats['dataset_test_failed']}")
        if self.stats["dataset_test_errors"]:
            print("  Vi du loi test:")
            for i, err in enumerate(self.stats["dataset_test_errors"][:3]):
                print(f"    {i+1}. {err}")
            if len(self.stats["dataset_test_errors"]) > 3:
                print(
                    f"    ... va {len(self.stats['dataset_test_errors']) - 3} loi khac"
                )

        # In lỗi nếu có
        if self.stats["errors"]:
            print(f"\nCAC LOI PHAT HIEN ({len(self.stats['errors'])} loi):")
            # Chỉ in 10 lỗi đầu
            for i, error in enumerate(self.stats["errors"][:10]):
                print(f"  {i+1}. {error}")
            if len(self.stats["errors"]) > 10:
                print(f"  ... va {len(self.stats['errors']) - 10} loi khac")

        # Đánh giá tổng thể
        print(f"\nDANH GIA TONG THE:")
        valid_ratio = (
            self.stats["valid_images"] / self.stats["total_images"]
            if self.stats["total_images"] > 0
            else 0
        )
        if valid_ratio >= 0.95:
            print("  Dataset chat luong tot (>=95% anh hop le)")
        elif valid_ratio >= 0.90:
            print("  Dataset chat luong kha (90-95% anh hop le)")
        else:
            print("  Dataset can cai thien (<90% anh hop le)")

    def create_clean_dataset(self, output_csv=None):
        """
        Tạo dataset sạch (loại bỏ các ảnh/label không hợp lệ) sử dụng utils function.

        Args:
            output_csv (str, optional): Đường dẫn file CSV để lưu dataset sạch.
                                      Nếu None, mặc định là "dataset_clean.csv"

        Returns:
            pd.DataFrame: DataFrame chứa dataset sạch với 2 cột:
                - data: Đường dẫn ảnh hợp lệ
                - label: Đường dẫn label file tương ứng
        """
        if output_csv is None:
            output_csv = "dataset_clean.csv"

        print(f"\nTao dataset sach...")

        # Sử dụng utils function để tạo dataset
        df_all = create_dataset_dataframe(self.images_dir, self.labels_dir)

        # Lọc dataset sạch
        clean_data = []
        clean_labels = []

        for idx, row in tqdm(
            df_all.iterrows(), total=len(df_all), desc="Tao dataset sach"
        ):
            img_path = row["data"]
            label_path = row["label"]

            # Kiểm tra ảnh và label
            img_valid, _ = self.validate_image(img_path)
            label_valid, _, _ = self.validate_label(label_path)

            if img_valid and label_valid:
                clean_data.append(img_path)
                clean_labels.append(label_path)

        # Tạo DataFrame và lưu
        df_clean = pd.DataFrame({"data": clean_data, "label": clean_labels})

        df_clean.to_csv(output_csv, index=False)
        print(f"Da tao dataset sach: {output_csv}")
        print(f"Dataset sach co {len(df_clean)} anh hop le")

        return df_clean

    def save_debug_info_to_csv(self, output_file="debug_errors.csv"):
        """
        Lưu thông tin debug và lỗi vào các file CSV.

        Args:
            output_file (str): Đường dẫn file CSV để lưu thông tin debug
        """
        print(f"\nLưu thông tin debug vào CSV: {output_file}")

        # Tạo DataFrame cho augment errors
        augment_errors = []
        for i, error_info in enumerate(self.stats["augment_fail_list"]):
            img_path = error_info.split(":")[0]
            error_msg = error_info.split(":", 1)[1] if ":" in error_info else error_info
            
            # Parse augment type từ error message
            augment_type = "Unknown"
            if "| Augment:" in error_msg:
                try:
                    augment_type = error_msg.split("| Augment:")[1].strip()
                    error_msg = error_msg.split("| Augment:")[0].strip()
                except:
                    pass

            # Lấy thông tin ảnh
            img_name = os.path.basename(img_path)
            label_file = os.path.splitext(img_name)[0] + ".txt"
            label_path = os.path.join(self.labels_dir, label_file)

            # Đọc thông tin ảnh
            try:
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    h, w = image.shape
                    img_size = f"{w}x{h}"
                else:
                    img_size = "Unknown"
            except:
                img_size = "Error reading"

            # Đọc thông tin label
            try:
                if os.path.exists(label_path):
                    with open(label_path, "r") as f:
                        lines = f.readlines()
                    num_boxes = len([line for line in lines if line.strip()])

                    # Đọc class distribution
                    classes = []
                    for line in lines:
                        if line.strip():
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                classes.append(int(float(parts[0])))
                    class_dist = (
                        ", ".join(map(str, sorted(set(classes)))) if classes else "None"
                    )
                else:
                    num_boxes = 0
                    class_dist = "No label file"
            except:
                num_boxes = 0
                class_dist = "Error reading"

            augment_errors.append(
                {
                    "STT": i + 1,
                    "Image_Name": img_name,
                    "Image_Path": img_path,
                    "Image_Size": img_size,
                    "Label_File": label_file,
                    "Num_Boxes": num_boxes,
                    "Class_Distribution": class_dist,
                    "Error_Type": "Augment Failed",
                    "Augment_Type": augment_type,
                    "Error_Message": error_msg,
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

        # Tạo DataFrame cho dataset test errors
        test_errors = []
        for i, error_info in enumerate(self.stats["dataset_test_errors"]):
            # Parse error info
            if ":" in error_info:
                error_type, error_msg = error_info.split(":", 1)
            else:
                error_type = "Unknown"
                error_msg = error_info

            test_errors.append(
                {
                    "STT": i + 1,
                    "Image_Name": "N/A",
                    "Image_Path": "N/A",
                    "Image_Size": "N/A",
                    "Label_File": "N/A",
                    "Num_Boxes": "N/A",
                    "Class_Distribution": "N/A",
                    "Error_Type": error_type.strip(),
                    "Augment_Type": "N/A",
                    "Error_Message": error_msg.strip(),
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

        # Tạo DataFrame cho general errors
        general_errors = []
        for i, error_info in enumerate(self.stats["errors"]):
            # Parse error info để lấy tên file
            img_name = "Unknown"
            if "dataset/images" in error_info:
                try:
                    img_name = error_info.split("dataset/images")[1].split()[0]
                    if img_name.startswith("\\") or img_name.startswith("/"):
                        img_name = img_name[1:]
                except:
                    pass

            general_errors.append(
                {
                    "STT": i + 1,
                    "Image_Name": img_name,
                    "Image_Path": "See error message",
                    "Image_Size": "N/A",
                    "Label_File": "N/A",
                    "Num_Boxes": "N/A",
                    "Class_Distribution": "N/A",
                    "Error_Type": "General Error",
                    "Augment_Type": "N/A",
                    "Error_Message": error_info,
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

        # Tạo summary sheet
        summary_data = {
            "Metric": [
                "Total Images",
                "Valid Images",
                "Corrupted Images",
                "Missing Images",
                "Total Labels",
                "Valid Labels",
                "Missing Labels",
                "Empty Labels",
                "Invalid Bboxes",
                "Augment Failed",
                "Dataset Test Passed",
                "Dataset Test Failed",
                "Total Errors",
            ],
            "Count": [
                self.stats["total_images"],
                self.stats["valid_images"],
                self.stats["corrupted_images"],
                self.stats["missing_images"],
                self.stats["total_labels"],
                self.stats["valid_labels"],
                self.stats["missing_labels"],
                self.stats["empty_labels"],
                self.stats["invalid_bboxes"],
                self.stats["augment_failed"],
                self.stats["dataset_test_passed"],
                self.stats["dataset_test_failed"],
                len(self.stats["errors"]),
            ],
        }

        # Tạo class distribution sheet
        class_dist_data = {
            "Class_ID": list(self.stats["class_distribution"].keys()),
            "Count": list(self.stats["class_distribution"].values()),
            "Percentage": [
                (count / sum(self.stats["class_distribution"].values())) * 100
                for count in self.stats["class_distribution"].values()
            ],
        }

        # Lưu vào CSV files thay vì Excel
        base_name = output_file.replace('.xlsx', '').replace('.csv', '')
        
        # Summary CSV
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{base_name}_summary.csv", index=False)
        
        # Class distribution CSV
        class_dist_df = pd.DataFrame(class_dist_data)
        class_dist_df.to_csv(f"{base_name}_class_distribution.csv", index=False)
        
        # Augment errors CSV
        if augment_errors:
            augment_df = pd.DataFrame(augment_errors)
            augment_df.to_csv(f"{base_name}_augment_errors.csv", index=False)
        
        # Test errors CSV
        if test_errors:
            test_df = pd.DataFrame(test_errors)
            test_df.to_csv(f"{base_name}_test_errors.csv", index=False)
        
        # General errors CSV
        if general_errors:
            general_df = pd.DataFrame(general_errors)
            general_df.to_csv(f"{base_name}_general_errors.csv", index=False)

        # Tạo báo cáo thống kê augment types
        augment_type_stats = {}
        for error in augment_errors:
            aug_type = error.get("Augment_Type", "Unknown")
            augment_type_stats[aug_type] = augment_type_stats.get(aug_type, 0) + 1

        # Tạo DataFrame cho augment type statistics
        if augment_type_stats:
            aug_type_data = {
                "Augment_Type": list(augment_type_stats.keys()),
                "Error_Count": list(augment_type_stats.values()),
                "Percentage": [
                    (count / sum(augment_type_stats.values())) * 100
                    for count in augment_type_stats.values()
                ],
            }
            aug_type_df = pd.DataFrame(aug_type_data)
            aug_type_df.to_csv(f"{base_name}_augment_type_stats.csv", index=False)

        print(f"✅ Đã lưu thông tin debug vào các file CSV:")
        print(f"   - Summary: {base_name}_summary.csv ({len(summary_data['Metric'])} metrics)")
        print(f"   - Class Distribution: {base_name}_class_distribution.csv ({len(class_dist_data['Class_ID'])} classes)")
        if augment_errors:
            print(f"   - Augment Errors: {base_name}_augment_errors.csv ({len(augment_errors)} errors)")
        if test_errors:
            print(f"   - Test Errors: {base_name}_test_errors.csv ({len(test_errors)} errors)")
        if general_errors:
            print(f"   - General Errors: {base_name}_general_errors.csv ({len(general_errors)} errors)")
        if augment_type_stats:
            print(f"   - Augment Type Stats: {base_name}_augment_type_stats.csv ({len(augment_type_stats)} types)")

    def test_clean_dataset(self, clean_csv_path="dataset_clean.csv", max_samples=5):
        """
        Test ImageDataset với dataset sạch đã được tạo.

        Function này thực hiện:
        1. Đọc dataset sạch từ CSV file
        2. Tạo ImageDataset instance từ dataset sạch
        3. Test một số samples để đảm bảo hoạt động đúng
        4. So sánh kết quả với test dataset gốc

        Args:
            clean_csv_path (str): Đường dẫn file CSV dataset sạch
            max_samples (int): Số lượng samples tối đa để test
        """
        print(f"\n=== TESTING CLEAN DATASET ===")
        print(f"Testing clean dataset from: {clean_csv_path}")

        if not os.path.exists(clean_csv_path):
            print(f"Clean dataset file not found: {clean_csv_path}")
            return

        try:
            # Tạo ImageDataset từ dataset sạch
            dataset = ImageDataset(df=clean_csv_path)

            print(f"Clean dataset created successfully with {len(dataset)} samples")

            # Test một số samples
            test_samples = min(max_samples, len(dataset))
            test_indices = np.random.choice(len(dataset), test_samples, replace=False)

            clean_test_passed = 0
            clean_test_failed = 0
            clean_test_errors = []

            for i, idx in enumerate(test_indices):
                try:
                    # Lấy sample từ dataset
                    sample = dataset[idx]

                    # Kiểm tra format output
                    if not isinstance(sample, dict):
                        raise ValueError(f"Clean sample {idx} is not a dictionary")

                    if "image" not in sample or "target" not in sample:
                        raise ValueError(
                            f"Clean sample {idx} missing 'image' or 'target' key"
                        )

                    # Kiểm tra image tensor
                    image = sample["image"]
                    if not hasattr(image, "shape"):
                        raise ValueError(f"Clean sample {idx} image is not a tensor")

                    # Kiểm tra target format
                    target = sample["target"]
                    boxes = target["boxes"]
                    labels = target["labels"]

                    if not hasattr(boxes, "shape") or not hasattr(labels, "shape"):
                        raise ValueError(
                            f"Clean sample {idx} boxes or labels are not tensors"
                        )

                    # Kiểm tra shape consistency
                    if len(boxes) != len(labels):
                        raise ValueError(
                            f"Clean sample {idx} boxes and labels length mismatch: {len(boxes)} vs {len(labels)}"
                        )

                    clean_test_passed += 1
                    print(f"✓ Clean sample {idx} passed all tests")

                except Exception as e:
                    clean_test_failed += 1
                    error_msg = f"Clean sample {idx} failed: {str(e)}"
                    clean_test_errors.append(error_msg)
                    print(f"✗ {error_msg}")

            print(f"\nClean Dataset Test Results:")
            print(f"  Passed: {clean_test_passed}")
            print(f"  Failed: {clean_test_failed}")

            if clean_test_errors:
                print(f"  Errors:")
                for error in clean_test_errors[:3]:
                    print(f"    - {error}")
                if len(clean_test_errors) > 3:
                    print(f"    ... and {len(clean_test_errors) - 3} more errors")

            # So sánh với test gốc
            print(f"\nComparison with original dataset:")
            print(
                f"  Original dataset test: {self.stats['dataset_test_passed']} passed, {self.stats['dataset_test_failed']} failed"
            )
            print(
                f"  Clean dataset test: {clean_test_passed} passed, {clean_test_failed} failed"
            )

        except Exception as e:
            print(f"Failed to create or test clean ImageDataset: {str(e)}")

    def plot_statistics(self, save_path=None):
        """
        Vẽ biểu đồ thống kê dataset.

        Function này tạo ra 4 subplot:
        1. Class Distribution: Bar chart phân bố số lượng bbox theo class
        2. Image Size Distribution: Scatter plot kích thước ảnh (width vs height)
        3. Bbox Size Distribution: Scatter plot kích thước bbox (width vs height)
        4. Error Summary: Bar chart tổng hợp các loại lỗi

        Nếu không có dữ liệu class distribution, function sẽ báo lỗi và return.

        Args:
            save_path (str, optional): Đường dẫn file để lưu biểu đồ.
                                     Nếu None thì chỉ hiển thị không lưu.
                                     File được lưu với DPI=300 và bbox_inches='tight'
        """
        if not self.stats["class_distribution"]:
            print("Không có dữ liệu để vẽ biểu đồ")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Dataset Statistics", fontsize=16)

        # Class distribution
        classes = list(self.stats["class_distribution"].keys())
        counts = list(self.stats["class_distribution"].values())
        axes[0, 0].bar(classes, counts)
        axes[0, 0].set_title("Class Distribution")
        axes[0, 0].set_xlabel("Class ID")
        axes[0, 0].set_ylabel("Number of Bboxes")

        # Image size distribution
        if self.stats["image_size_stats"]:
            widths = [w for w, h in self.stats["image_size_stats"]]
            heights = [h for w, h in self.stats["image_size_stats"]]
            axes[0, 1].scatter(widths, heights, alpha=0.5)
            axes[0, 1].set_title("Image Size Distribution")
            axes[0, 1].set_xlabel("Width")
            axes[0, 1].set_ylabel("Height")

        # Bbox size distribution
        if self.stats["bbox_size_stats"]:
            bbox_widths = [w for w, h in self.stats["bbox_size_stats"]]
            bbox_heights = [h for w, h in self.stats["bbox_size_stats"]]
            axes[1, 0].scatter(bbox_widths, bbox_heights, alpha=0.5)
            axes[1, 0].set_title("Bbox Size Distribution")
            axes[1, 0].set_xlabel("Bbox Width")
            axes[1, 0].set_ylabel("Bbox Height")

        # Error summary
        error_types = ["Valid", "Corrupted", "Missing", "Invalid Bbox"]
        error_counts = [
            self.stats["valid_images"],
            self.stats["corrupted_images"],
            self.stats["missing_images"],
            self.stats["invalid_bboxes"],
        ]
        axes[1, 1].bar(error_types, error_counts)
        axes[1, 1].set_title("Error Summary")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Bieu do da duoc luu: {save_path}")

        plt.show()

    def debug_augment_errors(
        self, max_debug=None, save_csv=True, save_images=True, output_dir="debug_images"
    ):
        """
        Debug các ảnh có lỗi augment.

        Function này thực hiện:
        1. Lặp qua danh sách lỗi augment từ stats['augment_fail_list']
        2. Với mỗi lỗi, parse đường dẫn ảnh và tìm label file tương ứng
        3. Đọc bbox từ label file
        4. Gọi visualize_augment_debug để hiển thị ảnh trước/sau augment
        5. Lưu hình ảnh debug với tên file có format: debug_augment_{index}_{image_name}.png
        6. Lưu thông tin debug vào Excel nếu được yêu cầu
        7. Giới hạn số lượng debug theo max_debug để tránh quá nhiều output

        Args:
            max_debug (int, optional): Số lượng ảnh tối đa để debug. 
                                     Nếu None, debug tất cả ảnh có lỗi.
                                     Mặc định là None (debug tất cả).
            save_csv (bool): Có lưu thông tin debug vào Excel không. Mặc định là True.
            save_images (bool): Có lưu ảnh debug vào folder không. Mặc định là True.
            output_dir (str): Thư mục để lưu ảnh debug. Mặc định là "debug_images".
        """
        print(f"\n=== DEBUGGING AUGMENT ERRORS ===")
        print(f"Found {len(self.stats['augment_fail_list'])} augment errors")
        
        # Nếu max_debug là None, debug tất cả
        if max_debug is None:
            max_debug = len(self.stats['augment_fail_list'])
            print(f"Will debug ALL {max_debug} images with errors")
        else:
            print(f"Will debug {min(max_debug, len(self.stats['augment_fail_list']))} images (limited by max_debug={max_debug})")

        # Tạo thư mục output nếu cần lưu ảnh
        if save_images:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Lưu ảnh debug vào thư mục: {output_dir}")

        debug_count = 0
        
        # Tạo progress bar nếu debug nhiều ảnh
        if max_debug > 10:
            from tqdm import tqdm
            error_list = tqdm(self.stats["augment_fail_list"][:max_debug], desc="Debugging images")
        else:
            error_list = self.stats["augment_fail_list"][:max_debug]
        
        for error_info in error_list:
            # Parse error info để lấy đường dẫn ảnh
            img_path = error_info.split(":")[0]
            
            # Parse augment type nếu có
            augment_type = "Unknown"
            if "| Augment:" in error_info:
                try:
                    augment_type = error_info.split("| Augment:")[1].strip()
                except:
                    pass

            # Tìm label file tương ứng
            label_file = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
            label_path = os.path.join(self.labels_dir, label_file)

            # Đọc boxes từ label
            try:
                boxes = []
                with open(label_path, "r") as f:
                    lines = f.readlines()

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) == 5:
                        class_id, x, y, w, h = map(float, parts)
                        boxes.append([int(class_id), x, y, w, h])

                if boxes:
                    print(
                        f"\nDebugging {debug_count + 1}/{max_debug}: {os.path.basename(img_path)} (Augment: {augment_type})"
                    )

                    if save_images:
                        # Lưu ảnh vào thư mục chỉ định
                        save_path = os.path.join(
                            output_dir,
                            f"debug_augment_{debug_count + 1}_{os.path.splitext(os.path.basename(img_path))[0]}.png",
                        )
                        self.visualize_augment_debug(
                            img_path, boxes, save_path, show=False
                        )
                    else:
                        # Chỉ hiển thị không lưu
                        self.visualize_augment_debug(img_path, boxes, None, show=True)

                    debug_count += 1

            except Exception as e:
                print(f"Error reading label for {img_path}: {e}")
                continue

        print(f"\n=== DEBUG COMPLETED ===")
        print(f"Successfully debugged {debug_count} images")
        if save_images:
            print(f"Debug images saved to: {output_dir}")
        
        # Lưu thông tin debug vào CSV nếu được yêu cầu
        if save_csv:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file = f"debug_errors_{timestamp}.csv"
            self.save_debug_info_to_csv(csv_file)


def debug_specific_image(
    img_path, labels_dir, save_images=True, output_dir="debug_images", show=True
):
    """
    Debug một ảnh cụ thể.

    Function này cho phép debug một ảnh đơn lẻ:
    1. Tìm label file tương ứng với ảnh (thay đổi extension từ .jpg/.png sang .txt)
    2. Kiểm tra label file có tồn tại không
    3. Đọc bbox từ label file
    4. Tạo DatasetValidator instance và gọi visualize_augment_debug
    5. Lưu hình ảnh debug với tên file có format: debug_specific_{image_name}.png

    Function này hữu ích khi muốn debug một ảnh cụ thể thay vì debug tất cả lỗi.

    Args:
        img_path (str): Đường dẫn ảnh cần debug
        labels_dir (str): Đường dẫn thư mục chứa label files
        save_images (bool): Có lưu ảnh debug vào folder không. Mặc định là True.
        output_dir (str): Thư mục để lưu ảnh debug. Mặc định là "debug_images".
        show (bool): Có hiển thị ảnh không. Mặc định là True.
    """
    print(f"DEBUGGING SPECIFIC IMAGE: {img_path}")
    print("=" * 60)

    # Tìm label file tương ứng
    label_file = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
    label_path = os.path.join(labels_dir, label_file)

    if not os.path.exists(label_path):
        print(f"Khong tim thay label file: {label_path}")
        return

    # Đọc boxes từ label
    try:
        boxes = []
        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 5:
                class_id, x, y, w, h = map(float, parts)
                boxes.append([int(class_id), x, y, w, h])

        if boxes:
            validator = DatasetValidator("", labels_dir)

            if save_images:
                # Tạo thư mục output nếu cần lưu ảnh
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(
                    output_dir,
                    f"debug_specific_{os.path.splitext(os.path.basename(img_path))[0]}.png",
                )
                validator.visualize_augment_debug(img_path, boxes, save_path, show=show)
            else:
                # Chỉ hiển thị không lưu
                validator.visualize_augment_debug(img_path, boxes, None, show=show)
        else:
            print("Khong co boxes trong label file")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


def main():
    """
    Main function - Entry point của script.

    Function này thực hiện workflow hoàn chỉnh để validate dataset:
    1. Kiểm tra thư mục ảnh và labels có tồn tại không
    2. Tạo DatasetValidator instance
    3. Chạy validate_dataset() để kiểm tra toàn bộ dataset
    4. In tóm tắt kết quả bằng print_summary()
    5. Tạo dataset sạch bằng create_clean_dataset()
    6. Vẽ biểu đồ thống kê bằng plot_statistics()
    7. Nếu có lỗi augment, hỏi user có muốn debug không
    8. In thông báo hoàn thành

    Không có parameters và không trả về giá trị.
    """
    print("DATASET VALIDATION TOOL")
    print("=" * 60)

    # Đường dẫn dataset
    images_dir = "dataset/images"
    labels_dir = "dataset/labels"

    # Kiểm tra thư mục tồn tại
    if not os.path.exists(images_dir):
        print(f"Khong tim thay thu muc anh: {images_dir}")
        return

    if not os.path.exists(labels_dir):
        print(f"Khong tim thay thu muc labels: {labels_dir}")
        return

    # Tạo validator
    validator = DatasetValidator(images_dir, labels_dir)

    # Kiểm tra dataset
    stats = validator.validate_dataset()

    # Test ImageDataset
    validator.test_image_dataset(max_samples=10)

    # Test multiple augmentations
    validator.test_multiple_augmentations(max_samples=3)

    # In tóm tắt
    validator.print_summary()

    # Tạo dataset sạch
    clean_df = validator.create_clean_dataset("dataset_clean.csv")

    # Test dataset sạch
    validator.test_clean_dataset("dataset_clean.csv", max_samples=10)

    # Vẽ biểu đồ (tùy chọn)
    try:
        validator.plot_statistics("dataset_statistics.png")
    except Exception as e:
        print(f"Khong the tao bieu do: {e}")

    # Debug augment errors nếu có
    if validator.stats["augment_failed"] > 0:
        print(
            f"\nCo {validator.stats['augment_failed']} loi augment. Ban co muon debug khong? (y/n): ",
            end="",
        )
        try:
            user_input = input().lower().strip()
            if user_input in ["y", "yes", ""]:
                validator.debug_augment_errors(
                    max_debug=None,  # Debug tất cả ảnh có lỗi
                    save_csv=True,
                    save_images=True,
                    output_dir="debug_images",
                )
        except:
            print("Khong the doc input, bo qua debug.")
    else:
        # Nếu không có lỗi augment, vẫn lưu thông tin tổng quan vào CSV
        print("\nLưu thông tin tổng quan vào CSV...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = f"dataset_summary_{timestamp}.csv"
        validator.save_debug_info_to_csv(csv_file)

    print("\nHoan thanh kiem tra dataset!")


# ============================================================================
# DEBUG FUNCTIONS WITH FLEXIBLE OPTIONS
# ============================================================================
# Các hàm debug mới hỗ trợ nhiều options:
# - save_images: Có lưu ảnh debug vào folder không (True/False)
# - output_dir: Thư mục để lưu ảnh debug (mặc định: "debug_images")
# - show: Có hiển thị ảnh không (True/False)
#
# Ví dụ sử dụng:
# 1. Debug với lưu ảnh và hiển thị:
#    debug_specific_image("path/to/image.jpg", "path/to/labels", save_images=True, show=True)
#
# 2. Debug chỉ lưu ảnh, không hiển thị (phù hợp cho server):
#    debug_specific_image("path/to/image.jpg", "path/to/labels", save_images=True, show=False)
#
# 3. Debug batch với lưu ảnh:
#    debug_batch_images("path/to/images", "path/to/labels", max_images=10, save_images=True, show=False)
#
# 4. Debug chỉ hiển thị, không lưu:
#    debug_specific_image("path/to/image.jpg", "path/to/labels", save_images=False, show=True)
# ============================================================================


def debug_batch_images(
    images_dir,
    labels_dir,
    max_images=5,
    save_images=True,
    output_dir="debug_images",
    show=False,
):
    """
    Debug nhiều ảnh cùng lúc với các options linh hoạt.

    Args:
        images_dir (str): Đường dẫn thư mục ảnh
        labels_dir (str): Đường dẫn thư mục labels
        max_images (int): Số lượng ảnh tối đa để debug
        save_images (bool): Có lưu ảnh debug vào folder không
        output_dir (str): Thư mục để lưu ảnh debug
        show (bool): Có hiển thị ảnh không
    """
    print(f"=== DEBUGGING BATCH IMAGES ===")
    print(f"Images directory: {images_dir}")
    print(f"Labels directory: {labels_dir}")
    print(f"Max images: {max_images}")
    print(f"Save images: {save_images}")
    print(f"Output directory: {output_dir}")
    print(f"Show images: {show}")
    print("=" * 50)

    # Tạo thư mục output nếu cần lưu ảnh
    if save_images:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    # Lấy danh sách ảnh
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    image_files = []

    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
        image_files.extend(glob.glob(os.path.join(images_dir, f"*{ext.upper()}")))

    image_files = image_files[:max_images]

    print(f"Found {len(image_files)} images to debug")

    for i, img_path in enumerate(image_files):
        print(f"\nDebugging {i+1}/{len(image_files)}: {os.path.basename(img_path)}")
        debug_specific_image(
            img_path,
            labels_dir,
            save_images=save_images,
            output_dir=output_dir,
            show=show,
        )

    print(f"\n=== BATCH DEBUG COMPLETED ===")
    if save_images:
        print(f"Debug images saved to: {output_dir}")


if __name__ == "__main__":
    import sys

    # Kiểm tra nếu có argument để debug ảnh cụ thể
    if len(sys.argv) > 1:
        if sys.argv[1] == "--debug" and len(sys.argv) > 2:
            img_path = sys.argv[2]
            labels_dir = "dataset/labels"
            debug_specific_image(img_path, labels_dir, save_images=True, show=False)
        else:
            print("Usage: python validate_dataset.py [--debug <image_path>]")
            print(
                "Example: python validate_dataset.py --debug dataset/images/example.jpg"
            )
    else:
        main()
