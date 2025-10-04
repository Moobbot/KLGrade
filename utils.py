#!/usr/bin/env python3
"""
Utility Functions for Object Detection
=====================================

Module chứa các utility functions để xử lý bounding boxes, 
chuyển đổi format coordinates và quản lý dataset.

Functions:
- yolo_to_xyxy: Chuyển đổi YOLO format sang XYXY format với validation
- filter_valid_boxes: Lọc bỏ các bbox không hợp lệ
- make_dataset_dataframe: Tạo DataFrame từ thư mục ảnh/labels
- save_dataset: Lưu dataset sang các format khác nhau
- validate_bbox_coordinates: Validate bbox coordinates
- clip_bbox_coordinates: Clip bbox về phạm vi hợp lệ
- calculate_bbox_area: Tính diện tích bbox
- calculate_bbox_center: Tính center point của bbox

Author: NgoTam
Date: 2025
"""

import os
import numpy as np
import pandas as pd


def yolo_to_xyxy(boxes_arr, labels=None, img_size=None, clip=True, filter_invalid=True):
    """
    Chuyển đổi bounding box từ YOLO format (x_center, y_center, w, h) 
    sang XYXY format (x_min, y_min, x_max, y_max) với validation và filtering.
    
    Args:
        boxes_arr (np.ndarray): Array shape (N, 4) chứa bbox YOLO format
        labels (np.ndarray, optional): Array shape (N,) chứa class labels
        img_size (tuple, optional): (W, H) để chuyển sang pixel coordinates
        clip (bool): Có clip coordinates về [0,1] hoặc [0, img_size] không
        filter_invalid (bool): Có lọc bỏ bbox không hợp lệ không
        
    Returns:
        tuple or np.ndarray:
            - Nếu có labels: (boxes_xyxy, labels_filtered)
            - Nếu không có labels: boxes_xyxy
    """
    # Xử lý trường hợp empty array
    if boxes_arr.shape[0] == 0:
        if labels is not None:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        else:
            return np.zeros((0, 4), dtype=np.float32)
    
    x_c, y_c, w, h = boxes_arr[:, 0], boxes_arr[:, 1], boxes_arr[:, 2], boxes_arr[:, 3]
    
    # Chuyển đổi từ center format sang corner format
    x_min = x_c - w / 2
    y_min = y_c - h / 2
    x_max = x_c + w / 2
    y_max = y_c + h / 2
    
    boxes_xyxy = np.stack([x_min, y_min, x_max, y_max], axis=1)

    if clip:
        if img_size is None:
            # Clip về [0, 1] với tolerance để xử lý floating point precision
            boxes_xyxy = np.clip(boxes_xyxy, 1e-8, 1.0 - 1e-8)
        else:
            W, H = img_size
            # Chuyển sang pixel coordinates
            boxes_xyxy[:, [0, 2]] *= W
            boxes_xyxy[:, [1, 3]] *= H
            # Clip về pixel boundaries
            boxes_xyxy = np.clip(boxes_xyxy, [0, 0, 0, 0], [W, H, W, H])
    
    # Validation và filtering nếu được yêu cầu
    if filter_invalid:
        if img_size is None:
            # Normalized coordinates - chỉ loại bỏ nếu hoàn toàn nằm ngoài [0,1]
            valid = (
                (boxes_xyxy[:, 2] > boxes_xyxy[:, 0]) &  # x_max > x_min
                (boxes_xyxy[:, 3] > boxes_xyxy[:, 1]) &  # y_max > y_min
                (boxes_xyxy[:, 2] > 0) &  # x_max > 0 (có phần trong hình)
                (boxes_xyxy[:, 3] > 0) &  # y_max > 0 (có phần trong hình)
                (boxes_xyxy[:, 0] < 1) &  # x_min < 1 (có phần trong hình)
                (boxes_xyxy[:, 1] < 1)    # y_min < 1 (có phần trong hình)
            )
        else:
            W, H = img_size
            # Pixel coordinates - chỉ loại bỏ nếu hoàn toàn nằm ngoài hình
            valid = (
                (boxes_xyxy[:, 2] > boxes_xyxy[:, 0]) &  # x_max > x_min
                (boxes_xyxy[:, 3] > boxes_xyxy[:, 1]) &  # y_max > y_min
                (boxes_xyxy[:, 2] > 0) &  # x_max > 0 (có phần trong hình)
                (boxes_xyxy[:, 3] > 0) &  # y_max > 0 (có phần trong hình)
                (boxes_xyxy[:, 0] < W) &  # x_min < W (có phần trong hình)
                (boxes_xyxy[:, 1] < H)    # y_min < H (có phần trong hình)
            )
        
        if labels is not None:
            labels_filtered = labels[valid]
            return boxes_xyxy[valid].astype(np.float32), labels_filtered
        else:
            return boxes_xyxy[valid].astype(np.float32)
    else:
        # Không filter, chỉ trả về kết quả
        if labels is not None:
            return boxes_xyxy.astype(np.float32), labels
        else:
            return boxes_xyxy.astype(np.float32)


def filter_valid_boxes(boxes_xyxy, labels=None):
    """
    Lọc bỏ các bbox không hợp lệ (x_max <= x_min hoặc y_max <= y_min)
    
    Args:
        boxes_xyxy (np.ndarray): Array shape (N, 4) chứa bbox XYXY format
        labels (np.ndarray, optional): Array shape (N,) chứa class labels
        
    Returns:
        tuple or np.ndarray: 
            - Nếu có labels: (boxes_xyxy_filtered, labels_filtered)
            - Nếu không có labels: boxes_xyxy_filtered
    """
    # Tạo mask cho các bbox hợp lệ
    valid = (boxes_xyxy[:, 2] > boxes_xyxy[:, 0]) & (boxes_xyxy[:, 3] > boxes_xyxy[:, 1])
    
    if labels is not None:
        labels_filtered = labels[valid]
        return boxes_xyxy[valid].astype(np.float32), labels_filtered
    else:
        return boxes_xyxy[valid]




def make_dataset_dataframe(images_dir, labels_dir, out_csv=None):
    """
    Tạo DataFrame từ thư mục ảnh và labels, có thể lưu thành CSV file.
    
    Args:
        images_dir (str): Đường dẫn thư mục chứa ảnh
        labels_dir (str): Đường dẫn thư mục chứa label files
        out_csv (str, optional): Đường dẫn file CSV để lưu kết quả
        
    Returns:
        pd.DataFrame: DataFrame chứa cột 'data' (đường dẫn ảnh) và 'label' (đường dẫn label)
    """
    # Lấy danh sách file ảnh
    image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
    data = []
    label = []
    
    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        label_file = img_file.replace(".jpg", ".txt")
        label_path = os.path.join(labels_dir, label_file)
        
        # Thêm vào danh sách
        data.append(img_path)
        if os.path.exists(label_path):
            label.append(label_path)
        else:
            label.append(None)  # Không có label file tương ứng
    
    # Tạo DataFrame
    df = pd.DataFrame({"data": data, "label": label})
    
    # Lưu CSV nếu được yêu cầu
    if out_csv is not None:
        df.to_csv(out_csv, index=False)
        print(f"Dataset DataFrame saved to: {out_csv}")
    
    return df


def save_dataset(dataset, out_path, fmt="pt"):
    """
    Lưu toàn bộ dataset (ảnh và labels) sang file với format được chọn.
    
    Args:
        dataset (ImageDataset): Instance của ImageDataset
        out_path (str): Đường dẫn file output
        fmt (str): Format file ('pt', 'pth', 'npz', 'h5')
        
    Raises:
        ImportError: Nếu cần h5py nhưng chưa cài đặt
        ValueError: Nếu format không được hỗ trợ
    """
    print(f"Saving dataset to {out_path} in {fmt} format...")
    
    # Thu thập tất cả data
    images = []
    targets = []
    
    for i in range(len(dataset)):
        item = dataset[i]
        images.append(item["image"].numpy())
        targets.append(item["target"])
    
    images = np.stack(images)
    targets = np.array(targets, dtype=object)
    
    # Lưu theo format được chọn
    if fmt in ["pt", "pth"]:
        import torch
        torch.save({"images": images, "targets": targets}, out_path)
        
    elif fmt == "npz":
        np.savez_compressed(out_path, images=images, targets=targets)
        
    elif fmt == "h5":
        try:
            import h5py
        except ImportError:
            raise ImportError("You need to install h5py to save in .h5 format")

        with h5py.File(out_path, "w") as f:
            f.create_dataset("images", data=images)
            grp = f.create_group("targets")
            for i, arr in enumerate(targets):
                grp.create_dataset(str(i), data=arr)
    else:
        raise ValueError("Only supports formats: pt, pth, npz, h5")
    
    print(f"Dataset saved successfully!")


def validate_bbox_coordinates(boxes, img_size=None, tolerance=1e-6):
    """
    Validate bbox coordinates để đảm bảo tính hợp lệ.
    
    Args:
        boxes (np.ndarray): Array shape (N, 4) chứa bbox coordinates
        img_size (tuple, optional): (W, H) kích thước ảnh
        tolerance (float): Tolerance cho floating point comparison
        
    Returns:
        tuple: (is_valid, error_messages)
            - is_valid (bool): True nếu tất cả bbox hợp lệ
            - error_messages (list): Danh sách thông báo lỗi
    """
    error_messages = []
    
    if len(boxes) == 0:
        return True, []
    
    # Kiểm tra shape
    if boxes.shape[1] != 4:
        error_messages.append(f"Invalid bbox shape: expected (N, 4), got {boxes.shape}")
        return False, error_messages
    
    # Kiểm tra coordinates
    if img_size is None:
        # Normalized coordinates [0, 1]
        if np.any(boxes < -tolerance) or np.any(boxes > 1 + tolerance):
            error_messages.append("Bbox coordinates out of [0, 1] range")
    else:
        W, H = img_size
        if np.any(boxes < -tolerance) or np.any(boxes[:, [0, 2]] > W + tolerance) or np.any(boxes[:, [1, 3]] > H + tolerance):
            error_messages.append(f"Bbox coordinates out of image bounds {img_size}")
    
    # Kiểm tra bbox validity (x_max > x_min, y_max > y_min)
    invalid_boxes = (boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])
    if np.any(invalid_boxes):
        invalid_count = np.sum(invalid_boxes)
        error_messages.append(f"{invalid_count} boxes have invalid dimensions (x_max <= x_min or y_max <= y_min)")
    
    return len(error_messages) == 0, error_messages


def clip_bbox_coordinates(boxes, img_size=None):
    """
    Clip bbox coordinates về trong phạm vi hợp lệ.
    
    Args:
        boxes (np.ndarray): Array shape (N, 4) chứa bbox coordinates
        img_size (tuple, optional): (W, H) kích thước ảnh
        
    Returns:
        np.ndarray: Array bbox đã được clip
    """
    if len(boxes) == 0:
        return boxes
    
    boxes_clipped = boxes.copy()
    
    if img_size is None:
        # Clip về [0, 1]
        boxes_clipped = np.clip(boxes_clipped, 1e-8, 1.0 - 1e-8)
    else:
        W, H = img_size
        boxes_clipped[:, [0, 2]] = np.clip(boxes_clipped[:, [0, 2]], 0, W)
        boxes_clipped[:, [1, 3]] = np.clip(boxes_clipped[:, [1, 3]], 0, H)
    
    return boxes_clipped


def calculate_bbox_area(boxes):
    """
    Tính diện tích của các bbox.
    
    Args:
        boxes (np.ndarray): Array shape (N, 4) chứa bbox XYXY format
        
    Returns:
        np.ndarray: Array shape (N,) chứa diện tích của từng bbox
    """
    if len(boxes) == 0:
        return np.array([])
    
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    areas = widths * heights
    
    return areas


def calculate_bbox_center(boxes):
    """
    Tính center point của các bbox.
    
    Args:
        boxes (np.ndarray): Array shape (N, 4) chứa bbox XYXY format
        
    Returns:
        np.ndarray: Array shape (N, 2) chứa (x_center, y_center) của từng bbox
    """
    if len(boxes) == 0:
        return np.zeros((0, 2))
    
    x_center = (boxes[:, 0] + boxes[:, 2]) / 2
    y_center = (boxes[:, 1] + boxes[:, 3]) / 2
    
    return np.stack([x_center, y_center], axis=1)


def load_yolo_labels(label_path, class_offset=0):
    """
    Load labels từ YOLO format file.
    
    Args:
        label_path (str): Đường dẫn label file
        class_offset (int): Offset để chuyển class_id (0-based sang 1-based)
        
    Returns:
        np.ndarray: Array chứa bbox [class_id, x, y, w, h]
    """
    boxes = []
    
    if label_path is not None and os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id, x, y, w, h = map(float, parts)
                    # Chuyển từ 0-based sang 1-based nếu cần
                    class_id = int(class_id) + class_offset
                    boxes.append([class_id, x, y, w, h])
    
    return (
        np.array(boxes, dtype=np.float32)
        if boxes
        else np.zeros((0, 5), dtype=np.float32)
    )


def validate_yolo_bboxes(boxes, class_range=(0, 4), min_size=0.001):
    """
    Validate YOLO format bboxes với class và size checks.
    
    Args:
        boxes (np.ndarray): Array bbox [class_id, x, y, w, h]
        class_range (tuple): (min_class, max_class) cho class_id hợp lệ
        min_size (float): Kích thước tối thiểu cho w, h
        
    Returns:
        tuple: (is_valid, error_messages)
    """
    error_messages = []
    
    if len(boxes) == 0:
        return True, []
    
    # Kiểm tra shape
    if boxes.shape[1] != 5:
        error_messages.append(f"Invalid bbox shape: expected (N, 5), got {boxes.shape}")
        return False, error_messages
    
    for i, box in enumerate(boxes):
        class_id, x, y, w, h = box
        
        # Kiểm tra class_id
        if class_id < class_range[0] or class_id > class_range[1]:
            error_messages.append(f"Box {i}: Invalid class_id {class_id}, expected {class_range[0]}-{class_range[1]}")
        
        # Kiểm tra coordinates
        if x < 0 or x > 1 or y < 0 or y > 1 or w <= 0 or w > 1 or h <= 0 or h > 1:
            error_messages.append(f"Box {i}: Invalid coordinates x:{x}, y:{y}, w:{w}, h:{h}")
        
        # Kiểm tra kích thước tối thiểu
        if w < min_size or h < min_size:
            error_messages.append(f"Box {i}: Too small w:{w}, h:{h}, min_size:{min_size}")
    
    return len(error_messages) == 0, error_messages


def clip_yolo_bboxes(boxes):
    """
    Clip YOLO format bboxes để đảm bảo tính hợp lệ.
    
    Args:
        boxes (np.ndarray): Array bbox [class_id, x, y, w, h]
        
    Returns:
        np.ndarray: Bbox đã được clip
    """
    if len(boxes) == 0:
        return boxes
    
    boxes_clipped = boxes.copy()
    
    # Clip center coordinates
    boxes_clipped[:, 1] = np.clip(boxes_clipped[:, 1], 0.0, 1.0)  # x
    boxes_clipped[:, 2] = np.clip(boxes_clipped[:, 2], 0.0, 1.0)  # y
    
    # Clip width và height
    boxes_clipped[:, 3] = np.clip(boxes_clipped[:, 3], 1e-6, 1.0)  # w
    boxes_clipped[:, 4] = np.clip(boxes_clipped[:, 4], 1e-6, 1.0)  # h
    
    # Đảm bảo bbox không vượt ra ngoài biên
    valid_boxes = []
    for box in boxes_clipped:
        class_id, x, y, w, h = box
        
        # Đảm bảo bbox hoàn toàn nằm trong [0, 1]
        x_min = max(0, x - w/2)
        y_min = max(0, y - h/2)
        x_max = min(1, x + w/2)
        y_max = min(1, y + h/2)
        
        # Tính lại center và kích thước
        new_x = (x_min + x_max) / 2
        new_y = (y_min + y_max) / 2
        new_w = x_max - x_min
        new_h = y_max - y_min
        
        # Đảm bảo kích thước tối thiểu
        new_w = max(new_w, 1e-6)
        new_h = max(new_h, 1e-6)
        
        # Kiểm tra bbox có hợp lệ không
        if (new_w > 0.001 and new_h > 0.001 and
            new_x >= 0 and new_y >= 0 and
            new_x + new_w <= 1.0 and new_y + new_h <= 1.0):
            valid_boxes.append([class_id, new_x, new_y, new_w, new_h])
    
    return (
        np.array(valid_boxes, dtype=np.float32)
        if valid_boxes
        else np.zeros((0, 5), dtype=np.float32)
    )


def post_augment_validation(boxes, labels):
    """
    Validate và fix bbox sau augmentation.
    
    Args:
        boxes (list): List bbox sau augmentation
        labels (list): List labels sau augmentation
        
    Returns:
        tuple: (valid_boxes, valid_labels)
    """
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    
    boxes_np = np.array(boxes)
    labels_np = np.array(labels)
    
    # Clip coordinates về [0, 1]
    boxes_clipped = np.clip(boxes_np, 1e-8, 1.0 - 1e-8)
    
    # Lọc bbox hợp lệ
    valid_indices = []
    for i, box in enumerate(boxes_clipped):
        x, y, w, h = box[0], box[1], box[2], box[3]
        
        # Kiểm tra bbox có hợp lệ không
        if (w > 0.001 and h > 0.001 and
            x >= 0 and y >= 0 and
            x + w <= 1.0 and y + h <= 1.0):
            valid_indices.append(i)
    
    if len(valid_indices) > 0:
        return boxes_clipped[valid_indices], labels_np[valid_indices]
    else:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)


def create_dataset_dataframe(images_dir, labels_dir, out_csv=None):
    """
    Tạo DataFrame từ thư mục ảnh và labels, có thể lưu thành CSV file.
    
    Args:
        images_dir (str): Đường dẫn thư mục chứa ảnh
        labels_dir (str): Đường dẫn thư mục chứa label files
        out_csv (str, optional): Đường dẫn file CSV để lưu kết quả
        
    Returns:
        pd.DataFrame: DataFrame chứa cột 'data' (đường dẫn ảnh) và 'label' (đường dẫn label)
    """
    # Lấy danh sách file ảnh
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    data = []
    label = []
    
    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        # Tìm label file tương ứng với extension khác nhau
        label_file = None
        for ext in ['.jpg', '.jpeg', '.png']:
            if img_file.lower().endswith(ext):
                label_file = img_file.replace(ext, '.txt')
                break
        
        if label_file:
            label_path = os.path.join(labels_dir, label_file)
        else:
            label_path = None
        
        # Thêm vào danh sách
        data.append(img_path)
        if label_path and os.path.exists(label_path):
            label.append(label_path)
        else:
            label.append(None)  # Không có label file tương ứng
    
    # Tạo DataFrame
    df = pd.DataFrame({"data": data, "label": label})
    
    # Lưu CSV nếu được yêu cầu
    if out_csv is not None:
        df.to_csv(out_csv, index=False)
        print(f"Dataset DataFrame saved to: {out_csv}")
    
    return df