"""
DETR-specific transforms and collate functions.
"""

from typing import List, Dict, Optional, Tuple
import torch
from transformers import DetrImageProcessor


def get_detr_processor(
    model_name: str = "facebook/detr-resnet-50",
    size: Optional[Dict[str, int]] = None
) -> DetrImageProcessor:
    """
    Get DETR image processor from HuggingFace.
    
    Args:
        model_name: HuggingFace model name
        size: Optional dict with 'height' and 'width' keys
        
    Returns:
        DetrImageProcessor instance
    """
    processor = DetrImageProcessor.from_pretrained(model_name)
    
    if size:
        processor.size = size
    
    return processor


def detr_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DETR DataLoader.
    
    Handles batching of images with different sizes by padding.
    
    Args:
        batch: List of dicts from CocoDataset with keys:
            - 'pixel_values': torch.Tensor (C, H, W)
            - 'pixel_mask': torch.Tensor (H, W)
            - 'labels': dict with 'class_labels', 'boxes', etc.
            
    Returns:
        Batched dict with keys:
            - 'pixel_values': torch.Tensor (B, C, H_max, W_max)
            - 'pixel_mask': torch.Tensor (B, H_max, W_max)
            - 'labels': list of dicts (length B)
    """
    # Stack pixel values and masks
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    pixel_mask = torch.stack([item['pixel_mask'] for item in batch])
    
    # Collect labels (keep as list of dicts)
    labels = [item['labels'] for item in batch]
    
    return {
        'pixel_values': pixel_values,
        'pixel_mask': pixel_mask,
        'labels': labels
    }


def detr_collate_fn_dynamic_padding(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Advanced collate function with dynamic padding to the largest image in batch.
    
    This is useful when images have very different sizes and you want to minimize padding.
    
    Args:
        batch: List of dicts from CocoDataset
        
    Returns:
        Batched dict with dynamically padded tensors
    """
    # Find max height and width in batch
    max_h = max(item['pixel_values'].shape[1] for item in batch)
    max_w = max(item['pixel_values'].shape[2] for item in batch)
    
    # Pad each image to max size
    pixel_values_list = []
    pixel_mask_list = []
    
    for item in batch:
        pixel_val = item['pixel_values']
        pixel_m = item['pixel_mask']
        
        c, h, w = pixel_val.shape
        
        # Create padded tensors
        padded_val = torch.zeros(c, max_h, max_w, dtype=pixel_val.dtype)
        padded_mask = torch.zeros(max_h, max_w, dtype=pixel_m.dtype)
        
        # Copy original data
        padded_val[:, :h, :w] = pixel_val
        padded_mask[:h, :w] = pixel_m
        
        pixel_values_list.append(padded_val)
        pixel_mask_list.append(padded_mask)
    
    # Stack
    pixel_values = torch.stack(pixel_values_list)
    pixel_mask = torch.stack(pixel_mask_list)
    
    # Collect labels
    labels = [item['labels'] for item in batch]
    
    return {
        'pixel_values': pixel_values,
        'pixel_mask': pixel_mask,
        'labels': labels
    }


class DetrTransform:
    """
    Custom transform for DETR that can apply augmentations before processing.
    
    Example:
        >>> from albumentations import HorizontalFlip, Compose
        >>> augmentations = Compose([
        ...     HorizontalFlip(p=0.5)
        ... ], bbox_params={'format': 'pascal_voc', 'label_fields': ['class_labels']})
        >>> 
        >>> transform = DetrTransform(
        ...     processor=processor,
        ...     augmentations=augmentations
        ... )
    """
    
    def __init__(
        self,
        processor: DetrImageProcessor,
        augmentations: Optional[object] = None
    ):
        self.processor = processor
        self.augmentations = augmentations
    
    def __call__(self, image, target: Dict):
        """
        Apply augmentations and then process with DETR processor.
        
        Args:
            image: PIL Image
            target: Dict with 'boxes' and 'class_labels'
            
        Returns:
            Processed image and target
        """
        import numpy as np
        
        # Convert PIL to numpy for albumentations
        image_np = np.array(image)
        
        # Apply augmentations if provided
        if self.augmentations:
            boxes = target.get('boxes', [])
            class_labels = target.get('class_labels', [])
            
            if len(boxes) > 0:
                augmented = self.augmentations(
                    image=image_np,
                    bboxes=boxes,
                    class_labels=class_labels
                )
                
                image_np = augmented['image']
                target['boxes'] = augmented['bboxes']
                target['class_labels'] = augmented['class_labels']
        
        # Convert back to PIL for processor
        from PIL import Image
        image = Image.fromarray(image_np)
        
        return image, target


if __name__ == "__main__":
    # Example usage
    print("DETR Transforms Module")
    
    # Get processor
    processor = get_detr_processor()
    print(f"✅ Loaded DETR processor: {processor}")
    
    # Test collate function
    dummy_batch = [
        {
            'pixel_values': torch.randn(3, 480, 640),
            'pixel_mask': torch.ones(480, 640),
            'labels': {'class_labels': [0, 1], 'boxes': [[10, 20, 100, 120], [150, 200, 300, 350]]}
        },
        {
            'pixel_values': torch.randn(3, 512, 512),
            'pixel_mask': torch.ones(512, 512),
            'labels': {'class_labels': [2], 'boxes': [[50, 60, 200, 180]]}
        }
    ]
    
    batched = detr_collate_fn_dynamic_padding(dummy_batch)
    print(f"\n✅ Collate function test:")
    print(f"   Pixel values shape: {batched['pixel_values'].shape}")
    print(f"   Pixel mask shape: {batched['pixel_mask'].shape}")
    print(f"   Number of labels: {len(batched['labels'])}")
