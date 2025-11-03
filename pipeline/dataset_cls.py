"""
Classification Dataset for cropped knee images.

Reads a CSV with columns: path,label and returns transformed image tensor and label.
Supports optional mixup/cutmix via external collate or wrapper.
"""
import os
import csv
import random
import sys
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


class ClassificationDataset(Dataset):
    def __init__(self, csv_path: str, transforms=None):
        self.df = pd.read_csv(csv_path)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["path"]
        label = int(row["label"])

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms:
            img = self.transforms(image=img)["image"]
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return {"image": img, "label": torch.tensor(label, dtype=torch.long)}
