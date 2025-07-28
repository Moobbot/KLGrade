import os
import numpy as np
import pandas as pd


def yolo_to_xyxy(boxes_arr, clip=True, img_size=None):
    """
    Convert bounding box YOLO (x_center, y_center, w, h) sang (x_min, y_min, x_max, y_max)
    boxes_arr: ndarray (N,4)
    """
    x_c, y_c, w, h = boxes_arr[:, 0], boxes_arr[:, 1], boxes_arr[:, 2], boxes_arr[:, 3]
    x_min = x_c - w / 2
    y_min = y_c - h / 2
    x_max = x_c + w / 2
    y_max = y_c + h / 2
    boxes = np.stack([x_min, y_min, x_max, y_max], axis=1)
    # Clip về [0,1] hoặc pixel
    if clip:
        if img_size is None:
            boxes_xyxy = np.clip(boxes, 0.0, 1.0)
        else:
            W, H = img_size
            boxes_xyxy[:, [0, 2]] *= W
            boxes_xyxy[:, [1, 3]] *= H
            boxes_xyxy = np.clip(boxes_xyxy, [0, 0, 0, 0], [W, H, W, H])
    return boxes


def filter_valid_boxes(boxes_xyxy, labels=None):
    """
    Loại bỏ box không hợp lệ (x_max <= x_min hoặc y_max <= y_min)
    boxes_xyxy: ndarray shape (N, 4)
    labels: ndarray shape (N,) hoặc None
    Trả về: boxes_xyxy đã lọc, labels đã lọc (nếu có)
    """
    valid = (boxes_xyxy[:, 2] > boxes_xyxy[:, 0]) & (
        boxes_xyxy[:, 3] > boxes_xyxy[:, 1]
    )
    if labels is not None:
        labels = labels[valid]
        return boxes_xyxy.astype(np.float32), labels
    return boxes_xyxy[valid]


import numpy as np


def safe_yolo_to_xyxy(boxes_arr, labels=None, clip=True, img_size=None):
    """
    Chuyển YOLO (x_c, y_c, w, h) -> XYXY, đồng thời clip và lọc box không hợp lệ.
    boxes_arr: ndarray (N, 4)
    labels: ndarray (N,) hoặc None
    clip: True -> clip về [0,1] hoặc về pixel nếu img_size != None
    img_size: (W, H) nếu muốn chuyển về pixel. Nếu None -> giữ [0,1].
    Trả về: (boxes_xyxy, labels_filtered)
    """
    if boxes_arr.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32), (
            np.zeros((0,), dtype=np.int64) if labels is not None else None
        )
    boxes_xyxy = yolo_to_xyxy(boxes_arr, clip=clip, img_size=img_size)

    # Lọc box không hợp lệ
    valid = (boxes_xyxy[:, 2] > boxes_xyxy[:, 0]) & (
        boxes_xyxy[:, 3] > boxes_xyxy[:, 1]
    )
    boxes_xyxy = boxes_xyxy[valid]
    boxes_xyxy = filter_valid_boxes(boxes_xyxy, labels=labels)
    return boxes_xyxy


def make_dataset_dataframe(images_dir, labels_dir, out_csv=None):
    """
    Create a dataframe from image/label folders and optionally save to a csv file.
    """
    image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
    data = []
    label = []
    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        label_file = img_file.replace(".jpg", ".txt")
        label_path = os.path.join(labels_dir, label_file)
        if os.path.exists(label_path):
            data.append(img_path)
            label.append(label_path)
        else:
            data.append(img_path)
            label.append(None)
    df = pd.DataFrame({"data": data, "label": label})
    if out_csv is not None:
        df.to_csv(out_csv, index=False)
    return df


def save_dataset(dataset, out_path, fmt="pt"):
    """
    Save the entire dataset (images and labels) to a file in the selected format.
    Args:
        dataset: instance of ImageDataset
        out_path: output file path
        fmt: 'pt', 'pth', 'npz', 'h5'
    """
    images = []
    targets = []
    for i in range(len(dataset)):
        item = dataset[i]
        images.append(item["image"].numpy())
        targets.append(item["target"].numpy())
    images = np.stack(images)
    targets = np.array(targets, dtype=object)
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
