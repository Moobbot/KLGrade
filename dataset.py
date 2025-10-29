import os
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class YoloDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_files = sorted(
            [
                f
                for f in os.listdir(img_dir)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ]
        )
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(
            self.label_dir, os.path.splitext(img_name)[0] + ".txt"
        )

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        # đọc nhãn YOLO
        boxes, labels = [], []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    c, x, y, bw, bh = map(float, line.strip().split())
                    x1 = (x - bw / 2) * w
                    y1 = (y - bh / 2) * h
                    x2 = (x + bw / 2) * w
                    y2 = (y + bh / 2) * h
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(c))

        # augment
        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, class_labels=labels)
            image = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["class_labels"]

        # convert tensor
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        return image_tensor, boxes_tensor, labels_tensor


# === Hàm visualize ===
def visualize_sample(image_tensor, boxes, labels, class_names=None):
    """
    Hiển thị ảnh và bbox sau augment
    image_tensor: torch.Tensor (C,H,W)
    boxes: list hoặc tensor [[x1,y1,x2,y2], ...]
    labels: list hoặc tensor
    class_names: danh sách tên lớp (tuỳ chọn)
    """
    image = image_tensor.permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image)
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        rect = patches.Rectangle(
            (x1, y1), width, height, linewidth=2, edgecolor="lime", facecolor="none"
        )
        ax.add_patch(rect)
        if class_names:
            lbl = class_names[label]
        else:
            lbl = str(label)
        ax.text(x1, y1 - 5, lbl, color="yellow", fontsize=10, backgroundcolor="black")
    plt.axis("off")
    plt.show()
