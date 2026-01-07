"""
Format conversion utilities for converting YOLO labels to COCO format.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import cv2
from tqdm import tqdm


def yolo_to_coco_bbox(
    yolo_bbox: List[float], img_width: int, img_height: int
) -> List[float]:
    """
    Convert YOLO format bbox to COCO format.

    Args:
        yolo_bbox: [x_center, y_center, width, height] (normalized 0-1)
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        COCO bbox: [x_min, y_min, width, height] in pixels
    """
    x_center, y_center, width, height = yolo_bbox

    # Convert from normalized to absolute coordinates
    x_center_abs = x_center * img_width
    y_center_abs = y_center * img_height
    width_abs = width * img_width
    height_abs = height * img_height

    # Convert from center format to top-left format
    x_min = x_center_abs - width_abs / 2
    y_min = y_center_abs - height_abs / 2

    return [x_min, y_min, width_abs, height_abs]


def coco_to_yolo_bbox(
    coco_bbox: List[float], img_width: int, img_height: int
) -> List[float]:
    """
    Convert COCO format bbox to YOLO format (for reference).

    Args:
        coco_bbox: [x_min, y_min, width, height] in pixels
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        YOLO bbox: [x_center, y_center, width, height] (normalized 0-1)
    """
    x_min, y_min, width, height = coco_bbox

    # Convert to center format
    x_center = x_min + width / 2
    y_center = y_min + height / 2

    # Normalize
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height

    return [x_center_norm, y_center_norm, width_norm, height_norm]


def create_coco_json(
    yolo_label_dir: str,
    img_dir: str,
    output_path: str,
    class_names: Dict[int, str],
    split_file: Optional[str] = None,
    info_dict: Optional[Dict] = None,
) -> str:
    """
    Create COCO format JSON file from YOLO format labels.

    Args:
        yolo_label_dir: Directory containing YOLO .txt label files
        img_dir: Directory containing images
        output_path: Path to save the COCO JSON file
        class_names: Dictionary mapping class_id to class_name (e.g., CLASSES or CLASSES_10_CLASS)
        split_file: Optional path to split file (train.txt/val.txt) to filter images
        info_dict: Optional info dictionary for COCO JSON metadata

    Returns:
        Path to created JSON file
    """
    label_dir = Path(yolo_label_dir)
    img_dir = Path(img_dir)
    output_path = Path(output_path)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load split file if provided
    image_stems = None
    if split_file and Path(split_file).exists():
        with open(split_file, "r", encoding="utf-8") as f:
            image_stems = set(line.strip() for line in f if line.strip())

    # Initialize COCO structure
    coco_data = {
        "info": info_dict
        or {
            "description": "KLGrade Dataset - Knee OA Detection",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().strftime("%Y-%m-%d"),
        },
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": [],
    }

    # Add categories
    for class_id, class_name in sorted(class_names.items()):
        coco_data["categories"].append(
            {"id": class_id, "name": class_name, "supercategory": "knee_oa"}
        )

    # Process images and annotations
    annotation_id = 1
    image_id = 1
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    # Get all images
    all_images = [f for f in img_dir.iterdir() if f.suffix.lower() in img_extensions]

    # Filter by split if provided
    if image_stems:
        all_images = [f for f in all_images if f.stem in image_stems]

    print(f"Processing {len(all_images)} images...")

    for img_file in tqdm(sorted(all_images)):
        # Read image to get dimensions
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"Warning: Cannot read image {img_file}")
            continue

        img_height, img_width = img.shape[:2]

        # Add image info
        image_info = {
            "id": image_id,
            "file_name": img_file.name,
            "width": img_width,
            "height": img_height,
        }
        coco_data["images"].append(image_info)

        # Load corresponding label file
        label_file = label_dir / f"{img_file.stem}.txt"

        if not label_file.exists():
            print(f"Warning: Label file not found for {img_file.name}")
            image_id += 1
            continue

        # Parse YOLO labels
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 5:
                    continue

                class_id = int(float(parts[0]))
                yolo_bbox = list(map(float, parts[1:5]))

                # Convert to COCO format
                coco_bbox = yolo_to_coco_bbox(yolo_bbox, img_width, img_height)

                # Calculate area
                area = coco_bbox[2] * coco_bbox[3]

                # Add annotation
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": coco_bbox,
                    "area": area,
                    "iscrowd": 0,
                }
                coco_data["annotations"].append(annotation)
                annotation_id += 1

        image_id += 1

    # Save to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=2)

    print(f"\n✅ COCO JSON created successfully!")
    print(f"   Path: {output_path}")
    print(f"   Images: {len(coco_data['images'])}")
    print(f"   Annotations: {len(coco_data['annotations'])}")
    print(f"   Categories: {len(coco_data['categories'])}")

    return str(output_path)


if __name__ == "__main__":
    # Example usage
    from src.config import CLASSES, CLASSES_10_CLASS

    # Create COCO JSON for labels (original 5 classes)
    create_coco_json(
        yolo_label_dir="processed/knee/labels",
        img_dir="processed/knee/images",
        output_path="processed/coco/annotations_train.json",
        class_names=CLASSES,
        split_file="splits/train.txt",
    )

    # Create COCO JSON for labels_new (10 classes)
    create_coco_json(
        yolo_label_dir="processed/knee/labels_new",
        img_dir="processed/knee/images",
        output_path="processed/coco/annotations_train_new.json",
        class_names=CLASSES_10_CLASS,
        split_file="splits/train.txt",
    )
