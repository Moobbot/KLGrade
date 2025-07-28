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
            self.transforms = A.Compose(
                [A.Resize(IMG_SIZE, IMG_SIZE), ToTensorV2()],
                bbox_params=A.BboxParams(
                    format="yolo",
                    label_fields=["class_labels"],
                    min_visibility=0.1,
                    clip=True,
                ),
            )
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
        else:
            data.append(img_path)
            label.append(None)
        return data, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label_path = self.label[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        boxes = []
        if label_path is not None and os.path.exists(label_path):
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

        # Checkpoint: Trước augment
        if len(boxes) > 0:
            # print(f"[Checkpoint] Before augment - img: {img_path}")
            # print(f"  boxes (yolo): {boxes}")
            if np.any(boxes[:, 1:] < 0) or np.any(boxes[:, 1:] > 1):
                print(f"  [ERROR] Found box out of [0,1] before augment: {boxes}")
                raise Exception(f"Box out of range before augment: {boxes} in {img_path}")

        # Augmentation
        if self.transforms:
            try:
                transformed = self.transforms(
                    image=image,
                    bboxes=boxes[:, 1:] if len(boxes) > 0 else [],
                    class_labels=boxes[:, 0] if len(boxes) > 0 else [],
                )
            except Exception as e:
                print(f"[Checkpoint] Augment Exception - img: {img_path}")
                print(f"  boxes (yolo): {boxes}")
                raise
            image = transformed["image"]
            boxes = transformed['bboxes']
            labels = transformed['class_labels']
            # Checkpoint: Sau augment
            # print(f"[Checkpoint] After augment - img: {img_path}")
            # print(f"  boxes (yolo): {boxes}")
            # print(f"  labels: {labels}")
            # Clip bbox về [0, 1] và cảnh báo nếu có giá trị bị clip
            if len(boxes) > 0:
                boxes_np = np.array(boxes)
                boxes_clipped = np.clip(boxes_np, 0.0, 1.0)
                if not np.allclose(boxes_np, boxes_clipped, atol=1e-6):
                    print(f"[WARNING] Bbox clipped for image: {img_path}")
                    print(f"  Before clip: {boxes_np}")
                    print(f"  After  clip: {boxes_clipped}")
                boxes = boxes_clipped
            if len(boxes) > 0 and (np.any(np.array(boxes) < 0) or np.any(np.array(boxes) > 1)):
                print(f"  [ERROR] Found box out of [0,1] after augment: {boxes}")
                raise Exception(f"Box out of range after augment: {boxes} in {img_path}")
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            labels = torch.zeros((0,), dtype=torch.int64) if boxes.shape[0] == 0 else torch.tensor(
                boxes[:, 0], dtype=torch.int64)
            boxes = torch.zeros((0, 4), dtype=torch.float32) if boxes.shape[0] == 0 else torch.tensor(
                boxes[:, 1:], dtype=torch.float32)
        if self.return_torchvision:
            # Trả về target dạng dict cho detection head torchvision
            if len(boxes) > 0:
                # labels = torch.tensor(boxes[:, 0], dtype=torch.int64)
                # Chuyển đổi box sang xyxy và lọc box hợp lệ
                # import IPython; IPython.embed()
                bboxes_xyxy = yolo_to_xyxy(boxes)
                # Checkpoint: Sau khi chuyển sang xyxy
                # print(f"[Checkpoint] After yolo_to_xyxy - img: {img_path}")
                # print(f"  bboxes_xyxy: {bboxes_xyxy}")
                if np.any(bboxes_xyxy < 0) or np.any(bboxes_xyxy > 1):
                    print(f"  [ERROR] Found xyxy box out of [0,1]: {bboxes_xyxy}")
                    raise Exception(f"xyxy box out of range: {bboxes_xyxy} in {img_path}")
                bboxes_xyxy, labels_np = filter_valid_boxes(
                    bboxes_xyxy, labels)
                labels = torch.tensor(labels_np, dtype=torch.int64) if not torch.is_tensor(
                    labels_np) else labels_np.clone().detach().to(torch.int64)
                bboxes = torch.tensor(bboxes_xyxy, dtype=torch.float32)
                # print("Filtered bboxes:", bboxes)
                # print("Filtered labels:", labels)
            else:
                labels = torch.zeros((0,), dtype=torch.int64)
                bboxes = torch.zeros((0, 4), dtype=torch.float32)
            return {
                "image": image.float(),
                "target": {"boxes": bboxes, "labels": labels},
            }
        else:
            # Trả về [class_id, x_center, y_center, w, h] cho custom
            if boxes.shape[0] > 0:
                out = torch.cat([labels.unsqueeze(1).float(),
                                boxes], dim=1).to(torch.float32)
            else:
                out = torch.zeros((0, 5), dtype=torch.float32)
            return {
                "image": image.float(),
                "target": out,
            }

