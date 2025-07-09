import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import pandas as pd
from config import IMG_SIZE

# Thêm import các hàm utils
from utils import yolo_to_xyxy, filter_valid_boxes


class ImageDataset(Dataset):
    """
    Flexible dataset for images and YOLO-format labels.
    Can be initialized from:
        - a pandas DataFrame (df)
        - a CSV file path (df as str)
        - image and label folders (images_dir, labels_dir)
    If no transform is provided, images will be resized to IMG_SIZE from config.py.
    """

    def __init__(
        self,
        df=None,
        images_dir=None,
        labels_dir=None,
        transforms=None,
        return_torchvision=False,  # Thêm tuỳ chọn trả về dạng torchvision
    ):
        self.return_torchvision = return_torchvision
        if transforms is None:
            self.transforms = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), ToTensorV2()])
        else:
            self.transforms = transforms
        if isinstance(df, str):
            # If df is a string, treat as CSV file path
            df = pd.read_csv(df)
        if df is not None:
            self.data = df["data"]  # list of image file paths
            self.label = df["label"]  # list of label file paths
        elif images_dir is not None and labels_dir is not None:
            self.data, self.label = self._make_dataset(images_dir, labels_dir)
        else:
            raise ValueError(
                "You must provide either df (DataFrame or CSV path) or images_dir and labels_dir"
            )

    def _make_dataset(self, images_dir, labels_dir):
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
        return data, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label_path = self.label[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x, y, w, h = map(float, parts)
                        boxes.append([class_id, x, y, w, h])
        boxes = (
            np.array(boxes, dtype=np.float32)
            if boxes
            else np.zeros((0, 5), dtype=np.float32)
        )
        # Albumentations expects 'bboxes' and 'class_labels' for augmentation
        if self.transforms:
            transformed = self.transforms(
                image=image,
                bboxes=boxes[:, 1:],
                class_labels=boxes[:, 0] if len(boxes) > 0 else [],
            )
            image = transformed["image"]
            # Re-combine class_id and bbox after augmentation
            if len(transformed["bboxes"]) > 0:
                boxes = np.hstack(
                    [
                        np.array(transformed["class_labels"]).reshape(-1, 1),
                        np.array(transformed["bboxes"]),
                    ]
                )
            else:
                boxes = np.zeros((0, 5), dtype=np.float32)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        if self.return_torchvision:
            # Trả về target dạng dict cho detection head torchvision
            if len(boxes) > 0:
                labels = torch.tensor(boxes[:, 0], dtype=torch.int64)
                # Chuyển đổi box sang xyxy và lọc box hợp lệ
                bboxes_xyxy = yolo_to_xyxy(boxes[:, 1:])
                bboxes_xyxy, labels_np = filter_valid_boxes(bboxes_xyxy, boxes[:, 0])
                labels = torch.tensor(labels_np, dtype=torch.int64)
                bboxes = torch.tensor(bboxes_xyxy, dtype=torch.float32)
            else:
                labels = torch.zeros((0,), dtype=torch.int64)
                bboxes = torch.zeros((0, 4), dtype=torch.float32)
            return {
                "image": image.float(),
                "target": {"boxes": bboxes, "labels": labels},
            }
        else:
            return {
                "image": image.float(),
                "target": torch.tensor(boxes, dtype=torch.float32),
            }


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
