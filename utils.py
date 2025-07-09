import numpy as np

def yolo_to_xyxy(boxes_arr):
    """
    Chuyển đổi bounding box từ YOLO (x_center, y_center, w, h) sang (x_min, y_min, x_max, y_max)
    boxes_arr: ndarray shape (N, 4)
    """
    x_c, y_c, w, h = (
        boxes_arr[:, 0],
        boxes_arr[:, 1],
        boxes_arr[:, 2],
        boxes_arr[:, 3],
    )
    x_min = x_c - w / 2
    y_min = y_c - h / 2
    x_max = x_c + w / 2
    y_max = y_c + h / 2
    return np.stack([x_min, y_min, x_max, y_max], axis=1)

def filter_valid_boxes(boxes_xyxy, labels=None):
    """
    Loại bỏ box không hợp lệ (x_max <= x_min hoặc y_max <= y_min)
    boxes_xyxy: ndarray shape (N, 4)
    labels: ndarray shape (N,) hoặc None
    Trả về: boxes_xyxy đã lọc, labels đã lọc (nếu có)
    """
    valid = (boxes_xyxy[:, 2] > boxes_xyxy[:, 0]) & (boxes_xyxy[:, 3] > boxes_xyxy[:, 1])
    if labels is not None:
        return boxes_xyxy[valid], labels[valid]
    return boxes_xyxy[valid] 