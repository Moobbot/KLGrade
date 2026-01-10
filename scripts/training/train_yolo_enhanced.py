"""
Enhanced YOLO Training Script with Advanced Preprocessing
Based on techniques from Yolo_Detection_XuongKhop_v2.ipynb

Features:
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Gaussian Blur for noise reduction
- Data balancing with flip augmentation
- Label scaling for resized images
"""

import os
import sys
import cv2
import argparse
import random
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ultralytics import YOLO
import wandb


def preprocess_image_clahe(image_path, target_size=(640, 640)):
    """
    Preprocess image with CLAHE and Gaussian Blur.
    
    Args:
        image_path: Path to input image
        target_size: Target size for resizing
    
    Returns:
        Preprocessed image (grayscale)
    """
    # Read image as grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image from {image_path}")
    
    # Store original size for label scaling
    original_size = image.shape[:2]
    
    # Resize image
    image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
    
    # Apply Gaussian Blur (noise reduction)
    image_blurred = cv2.GaussianBlur(image_resized, (5, 5), 0)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(image_blurred)
    
    return enhanced_image, original_size


def scale_bounding_box(bbox, original_size, new_size=(640, 640)):
    """
    Scale bounding box coordinates from original size to new size.
    
    Args:
        bbox: [x_center, y_center, width, height] in normalized format
        original_size: (height, width) of original image
        new_size: (width, height) of target image
    
    Returns:
        Scaled bounding box
    """
    x_center, y_center, width, height = bbox
    
    # Convert to pixel coordinates
    x_center_pixel = x_center * original_size[1]  # width
    y_center_pixel = y_center * original_size[0]  # height
    width_pixel = width * original_size[1]
    height_pixel = height * original_size[0]
    
    # Scale to new size
    x_scale = new_size[0] / original_size[1]
    y_scale = new_size[1] / original_size[0]
    
    x_center_scaled = x_center_pixel * x_scale
    y_center_scaled = y_center_pixel * y_scale
    width_scaled = width_pixel * x_scale
    height_scaled = height_pixel * y_scale
    
    # Convert back to normalized coordinates
    x_center_new = x_center_scaled / new_size[0]
    y_center_new = y_center_scaled / new_size[1]
    width_new = width_scaled / new_size[0]
    height_new = height_scaled / new_size[1]
    
    return [x_center_new, y_center_new, width_new, height_new]


def process_labels(label_path, original_size, new_size=(640, 640)):
    """
    Process and scale labels for resized images.
    
    Args:
        label_path: Path to label file
        original_size: (height, width) of original image
        new_size: (width, height) of target image
    
    Returns:
        List of scaled labels
    """
    with open(label_path, 'r') as file:
        lines = file.readlines()
    
    new_labels = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:5])
        
        scaled_bbox = scale_bounding_box(
            [x_center, y_center, width, height], 
            original_size, 
            new_size
        )
        
        new_labels.append(f"{class_id} {' '.join(map(str, scaled_bbox))}")
    
    return new_labels


def flip_image_and_labels(image, labels):
    """
    Flip image horizontally and adjust labels.
    
    Args:
        image: Input image
        labels: List of label strings
    
    Returns:
        Flipped image and adjusted labels
    """
    flipped_image = cv2.flip(image, 1)
    
    flipped_labels = []
    for label in labels:
        parts = label.split()
        class_id = parts[0]
        x_center, y_center, width, height = map(float, parts[1:5])
        
        # Flip x_center
        x_center_flipped = 1.0 - x_center
        
        flipped_labels.append(
            f"{class_id} {x_center_flipped:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )
    
    return flipped_image, flipped_labels


def count_class_distribution(label_dir, num_classes=5):
    """
    Count class distribution in dataset.
    
    Args:
        label_dir: Directory containing label files
        num_classes: Number of classes
    
    Returns:
        Dictionary with class counts
    """
    class_counts = {i: 0 for i in range(num_classes)}
    
    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            label_path = os.path.join(label_dir, label_file)
            with open(label_path, 'r') as f:
                for line in f:
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1
    
    return class_counts


def balance_dataset_with_flip(
    img_dir, 
    label_dir, 
    output_img_dir, 
    output_label_dir,
    num_classes=5,
    target_size=(640, 640)
):
    """
    Balance dataset by flipping augmentation for minority classes.
    
    Args:
        img_dir: Input image directory
        label_dir: Input label directory
        output_img_dir: Output image directory
        output_label_dir: Output label directory
        num_classes: Number of classes
        target_size: Target image size
    """
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    # Count class distribution
    print("\n📊 Analyzing class distribution...")
    class_counts = count_class_distribution(label_dir, num_classes)
    
    print("\nOriginal class distribution:")
    for class_id, count in class_counts.items():
        print(f"  Class {class_id}: {count} instances")
    
    max_count = max(class_counts.values())
    print(f"\nTarget count: {max_count} instances per class")
    
    # Process all images
    image_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"\n🔄 Processing {len(image_files)} images with CLAHE + Gaussian Blur...")
    
    for img_file in tqdm(image_files):
        img_path = os.path.join(img_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(label_dir, label_file)
        
        if not os.path.exists(label_path):
            continue
        
        # Preprocess image with CLAHE
        enhanced_image, original_size = preprocess_image_clahe(img_path, target_size)
        
        # Process labels
        new_labels = process_labels(label_path, original_size, target_size)
        
        # Save processed image (as grayscale PNG)
        output_img_path = os.path.join(output_img_dir, os.path.splitext(img_file)[0] + '.png')
        cv2.imwrite(output_img_path, enhanced_image)
        
        # Save processed labels
        output_label_path = os.path.join(output_label_dir, label_file)
        with open(output_label_path, 'w') as f:
            f.write('\n'.join(new_labels))
    
    # Augment minority classes
    print("\n🔄 Augmenting minority classes with flip...")
    
    current_counts = count_class_distribution(output_label_dir, num_classes)
    
    for class_id in range(num_classes):
        needed = max_count - current_counts[class_id]
        
        if needed <= 0:
            print(f"  Class {class_id}: Already balanced")
            continue
        
        print(f"  Class {class_id}: Need {needed} more instances")
        
        # Find images with this class
        images_with_class = []
        for label_file in os.listdir(output_label_dir):
            if not label_file.endswith('.txt'):
                continue
                
            label_path = os.path.join(output_label_dir, label_file)
            with open(label_path, 'r') as f:
                labels = f.readlines()
                if any(int(line.split()[0]) == class_id for line in labels):
                    img_file = os.path.splitext(label_file)[0] + '.png'
                    img_path = os.path.join(output_img_dir, img_file)
                    if os.path.exists(img_path):
                        images_with_class.append((img_path, label_path))
        
        if not images_with_class:
            print(f"    ⚠️  No images found for class {class_id}")
            continue
        
        # Augment by flipping
        random.shuffle(images_with_class)
        augmented = 0
        idx = 0
        
        while augmented < needed and idx < len(images_with_class) * 10:  # Max 10 rounds
            img_path, label_path = images_with_class[idx % len(images_with_class)]
            idx += 1
            
            # Read image and labels
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            with open(label_path, 'r') as f:
                labels = [line.strip() for line in f.readlines()]
            
            # Filter labels for current class only
            class_labels = [l for l in labels if int(l.split()[0]) == class_id]
            
            if not class_labels:
                continue
            
            # Flip image and labels
            flipped_image, flipped_labels = flip_image_and_labels(image, class_labels)
            
            # Save augmented data
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            aug_img_name = f"{base_name}_flip_{augmented}.png"
            aug_label_name = f"{base_name}_flip_{augmented}.txt"
            
            aug_img_path = os.path.join(output_img_dir, aug_img_name)
            aug_label_path = os.path.join(output_label_dir, aug_label_name)
            
            cv2.imwrite(aug_img_path, flipped_image)
            with open(aug_label_path, 'w') as f:
                f.write('\n'.join(flipped_labels))
            
            augmented += len(flipped_labels)
        
        print(f"    ✅ Augmented {augmented} instances")
    
    # Final count
    final_counts = count_class_distribution(output_label_dir, num_classes)
    print("\n📊 Final class distribution:")
    for class_id, count in final_counts.items():
        print(f"  Class {class_id}: {count} instances")


def train_yolo_enhanced(
    img_dir: str,
    label_dir: str,
    split_dir: str = None,
    num_classes: int = 5,
    model_name: str = "yolo11n.pt",
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = "0",
    project: str = "runs/detect",
    name: str = "yolo_enhanced",
    apply_preprocessing: bool = True,
    apply_balancing: bool = True,
):
    """
    Train YOLO with enhanced preprocessing.
    
    Args:
        img_dir: Image directory
        label_dir: Label directory
        split_dir: Split directory (train.txt, val.txt, test.txt)
        num_classes: Number of classes
        model_name: YOLO model name
        epochs: Training epochs
        batch_size: Batch size
        img_size: Image size
        device: Device (0 for GPU, cpu for CPU)
        project: Project directory
        name: Experiment name
        apply_preprocessing: Apply CLAHE + Gaussian preprocessing
        apply_balancing: Apply data balancing
    """
    print("=" * 70)
    print("Enhanced YOLO Training with CLAHE + Data Balancing")
    print("=" * 70)
    
    # Setup WandB
    wandb_project = os.getenv("WANDB_PROJECT", "KLGrade-Knee-OA")
    print(f"\n📊 Initializing WandB Project: {wandb_project}")
    
    wandb.init(
        project=wandb_project,
        name=name,
        config={
            "model": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "img_size": img_size,
            "num_classes": num_classes,
            "preprocessing": "CLAHE + Gaussian Blur" if apply_preprocessing else "None",
            "balancing": "Flip Augmentation" if apply_balancing else "None",
        }
    )
    
    # Create processed data directory
    processed_base = Path("processed") / "enhanced" / name
    processed_img_dir = processed_base / "images"
    processed_label_dir = processed_base / "labels"
    
    if apply_preprocessing or apply_balancing:
        print(f"\n🔄 Preprocessing data to {processed_base}")
        
        balance_dataset_with_flip(
            img_dir=img_dir,
            label_dir=label_dir,
            output_img_dir=str(processed_img_dir),
            output_label_dir=str(processed_label_dir),
            num_classes=num_classes,
            target_size=(img_size, img_size)
        )
        
        # Use processed data
        final_img_dir = str(processed_img_dir)
        final_label_dir = str(processed_label_dir)
    else:
        final_img_dir = img_dir
        final_label_dir = label_dir
    
    # Create YOLO dataset config
    dataset_yaml = Path(f"configs/yolo_enhanced_{name}.yaml")

    # If splits are provided, rewrite them to point at processed images
    train_ref = None
    val_ref = None
    test_ref = None

    if split_dir:
        split_dir_path = Path(split_dir)
        processed_split_dir = Path("processed") / "enhanced" / "splits" / split_dir_path.name / name
        processed_split_dir.mkdir(parents=True, exist_ok=True)

        def rewrite_split(in_file: Path, out_file: Path):
            if not in_file.exists():
                return None
            lines = [l.strip() for l in in_file.read_text().splitlines() if l.strip()]
            new_lines = []
            for p in lines:
                stem = Path(p).stem
                proc_img = Path(final_img_dir) / f"{stem}.png"
                if proc_img.exists():
                    new_lines.append(str(proc_img.resolve()))
            if not new_lines:
                return None
            out_file.write_text("\n".join(new_lines))
            return out_file.resolve()

        train_ref = rewrite_split(split_dir_path / "train.txt", processed_split_dir / "train.txt")
        val_ref = rewrite_split(split_dir_path / "val.txt", processed_split_dir / "val.txt")
        test_txt = split_dir_path / "test.txt"
        test_ref = rewrite_split(test_txt, processed_split_dir / "test.txt") if test_txt.exists() else None

    # Fallbacks when splits aren't provided or rewrite produced no files
    if not split_dir or not train_ref or not val_ref:
        # Use directories directly if we can't use splits
        train_ref = Path(final_img_dir).resolve()
        val_ref = Path(final_img_dir).resolve()
        test_ref = None

    yaml_lines = [
        "# Enhanced YOLO Dataset Config",
        f"nc: {num_classes}",
        f"names: {list(range(num_classes))}",
    ]

    # Use absolute references for train/val/test
    yaml_lines.append(f"train: {train_ref}")
    yaml_lines.append(f"val: {val_ref}")
    if test_ref:
        yaml_lines.append(f"test: {test_ref}")

    dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
    dataset_yaml.write_text("\n".join(yaml_lines) + "\n")

    print(f"\n✅ Dataset config created: {dataset_yaml}")
    
    # Load model
    print(f"\n📦 Loading YOLO model: {model_name}")
    model = YOLO(model_name)
    
    # Train
    print(f"\n🚀 Starting training...")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {img_size}")
    print(f"  Device: {device}")
    
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project,
        name=name,
        patience=50,
        save_period=10,
        plots=True,
    )
    
    # Validate
    print("\n📊 Running validation...")
    metrics = model.val()
    
    print(f"\n✅ Training completed!")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    
    wandb.finish()
    
    return results, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhanced YOLO training with CLAHE and data balancing"
    )
    
    parser.add_argument("--img_dir", type=str, required=True, help="Image directory")
    parser.add_argument("--label_dir", type=str, required=True, help="Label directory")
    parser.add_argument("--split_dir", type=str, default=None, help="Split directory")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of classes")
    parser.add_argument("--model", type=str, default="yolo11l.pt", help="YOLO model")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="0", help="Device")
    parser.add_argument("--project", type=str, default="runs/detect", help="Project dir")
    parser.add_argument("--name", type=str, default="yolo11l_enhanced", help="Experiment name")
    parser.add_argument("--no-preprocessing", action="store_true", help="Disable preprocessing")
    parser.add_argument("--no-balancing", action="store_true", help="Disable balancing")
    
    args = parser.parse_args()
    
    train_yolo_enhanced(
        img_dir=args.img_dir,
        label_dir=args.label_dir,
        split_dir=args.split_dir,
        num_classes=args.num_classes,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        apply_preprocessing=not args.no_preprocessing,
        apply_balancing=not args.no_balancing,
    )
