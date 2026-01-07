"""
COCO Dataset Loader for DETR and transformer-based detection models.
"""

import json
from pathlib import Path
from typing import Optional, Dict, List, Callable
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image


class CocoDataset(Dataset):
    """
    COCO format dataset loader for DETR and transformer-based models.

    Args:
        coco_json_path: Path to COCO format JSON annotation file
        img_dir: Directory containing images
        processor: Optional image processor (e.g., DetrImageProcessor from transformers)
        transform: Optional custom transform function

    Example:
        >>> from transformers import DetrImageProcessor
        >>> processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        >>> dataset = CocoDataset(
        ...     coco_json_path="processed/coco/annotations_train.json",
        ...     img_dir="processed/knee/images",
        ...     processor=processor
        ... )
    """

    def __init__(
        self,
        coco_json_path: str,
        img_dir: str,
        processor: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        self.coco_json_path = Path(coco_json_path)
        self.img_dir = Path(img_dir)
        self.processor = processor
        self.transform = transform

        # Load COCO annotations
        with open(self.coco_json_path, "r", encoding="utf-8") as f:
            self.coco_data = json.load(f)

        # Create category mapping
        self.categories = {cat["id"]: cat for cat in self.coco_data["categories"]}
        self.num_classes = len(self.categories)

        # Create image id to annotations mapping
        self.img_to_anns = {}
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        # Store images info
        self.images = self.coco_data["images"]

        print(
            f"✅ Loaded COCO dataset: {len(self.images)} images, "
            f"{len(self.coco_data['annotations'])} annotations, "
            f"{self.num_classes} classes"
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a dictionary containing image and annotations.

        If processor is provided, returns processed format for DETR:
            {
                'pixel_values': tensor,
                'pixel_mask': tensor,
                'labels': {
                    'class_labels': tensor,
                    'boxes': tensor,
                    'image_id': int
                }
            }

        Otherwise returns raw data:
            {
                'image': PIL Image or numpy array,
                'annotations': list of dicts,
                'image_id': int
            }
        """
        # Get image info
        img_info = self.images[idx]
        img_id = img_info["id"]
        img_filename = img_info["file_name"]

        # Load image
        img_path = self.img_dir / img_filename
        image = Image.open(img_path).convert("RGB")

        # Get annotations for this image
        annotations = self.img_to_anns.get(img_id, [])

        # Prepare target dict
        target = {
            "image_id": img_id,
            "annotations": annotations,
        }

        # Extract boxes and labels
        boxes = []
        class_labels = []

        for ann in annotations:
            # COCO bbox format: [x, y, width, height]
            bbox = ann["bbox"]
            # Convert to [x_min, y_min, x_max, y_max]
            x, y, w, h = bbox
            boxes.append([x, y, x + w, y + h])
            class_labels.append(ann["category_id"])

        target["boxes"] = boxes
        target["class_labels"] = class_labels

        # Apply custom transform if provided
        if self.transform:
            image, target = self.transform(image, target)

        # Apply processor if provided (for DETR)
        if self.processor:
            # Prepare annotations in COCO format for DETR processor
            # Format: {'image_id': int, 'annotations': [{'bbox': [...], 'category_id': int, 'area': float, 'iscrowd': 0}, ...]}
            coco_format_annotations = []
            for ann in annotations:
                coco_format_annotations.append(
                    {
                        "bbox": ann["bbox"],  # Keep COCO format [x, y, w, h]
                        "category_id": ann["category_id"],
                        "area": ann.get(
                            "area", ann["bbox"][2] * ann["bbox"][3]
                        ),  # w * h
                        "iscrowd": ann.get("iscrowd", 0),
                    }
                )

            target_for_processor = {
                "image_id": img_id,
                "annotations": coco_format_annotations,
            }

            # Handle empty annotations
            if len(coco_format_annotations) == 0:
                target_for_processor["annotations"] = []

            # Process with DETR processor
            encoding = self.processor(
                images=image, annotations=target_for_processor, return_tensors="pt"
            )

            # Remove batch dimension
            pixel_values = encoding["pixel_values"].squeeze(0)
            pixel_mask = encoding["pixel_mask"].squeeze(0)

            # Get labels
            labels = encoding["labels"][0]

            return {
                "pixel_values": pixel_values,
                "pixel_mask": pixel_mask,
                "labels": labels,
            }
        else:
            # Return raw format
            return {"image": image, "target": target}

    def get_class_names(self) -> Dict[int, str]:
        """Get class id to name mapping."""
        return {cat["id"]: cat["name"] for cat in self.coco_data["categories"]}

    def get_image_info(self, idx: int) -> Dict:
        """Get image metadata."""
        return self.images[idx]


def visualize_coco_sample(
    dataset: CocoDataset, idx: int, save_path: Optional[str] = None
):
    """
    Visualize a sample from COCO dataset.

    Args:
        dataset: CocoDataset instance
        idx: Index of sample to visualize
        save_path: Optional path to save visualization
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches_lib

    # Get sample (raw format)
    temp_processor = dataset.processor
    dataset.processor = None  # Temporarily disable processor
    sample = dataset[idx]
    dataset.processor = temp_processor

    image = sample["image"]
    target = sample["target"]

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)

    # Get class names
    class_names = dataset.get_class_names()

    # Draw boxes
    boxes = target["boxes"]
    labels = target["class_labels"]

    for box, label in zip(boxes, labels):
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min

        # Draw rectangle
        rect = patches_lib.Rectangle(
            (x_min, y_min),
            width,
            height,
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect)

        # Draw label
        label_text = class_names.get(label, f"Class {label}")
        ax.text(
            x_min,
            y_min - 5,
            label_text,
            color="yellow",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
        )

    ax.set_title(f"Image ID: {target['image_id']} | Boxes: {len(boxes)}")
    ax.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    # Example usage - raw format
    dataset = CocoDataset(
        coco_json_path="processed/coco/annotations_train.json",
        img_dir="processed/knee/images",
    )

    print(f"\nDataset size: {len(dataset)}")
    print(f"Class names: {dataset.get_class_names()}")

    # Get a sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Image size: {sample['image'].size}")
    print(f"Num boxes: {len(sample['target']['boxes'])}")
