#!/bin/bash
# Preprocess balanced knees_cropped dataset with all presets

echo "all" | PYTHONPATH=/home/ngoductam/KLGrade \
/home/ngoductam/miniconda3/envs/klgrade/bin/python -c "
import sys
from pathlib import Path
from tqdm import tqdm
import shutil

# Add project root
project_root = Path('/home/ngoductam/KLGrade')
sys.path.append(str(project_root))

from src.data.preprocessing import (
    load_image, save_image,
    get_basic_pipeline, get_v0_pipeline,
    get_v3_legacy_pipeline, get_notebook_pipeline,
)

# Paths
input_base = project_root / 'datasets/dataset_knees_cropped_balanced'
output_base = project_root / 'datasets/data_processed_balanced'

print('='*60)
print('PREPROCESSING BALANCED KNEES CROPPED DATASET')
print('='*60)
print(f'Input: {input_base}')
print(f'Output: {output_base}')
print()

# Presets
presets = {
    'resize_only': ('Resize only', get_basic_pipeline()),
    'blur_clahe2': ('Blur + CLAHE 2.0', get_v0_pipeline()),
    'sharp_clahe4': ('No Blur + CLAHE 4.0', get_v3_legacy_pipeline()),
    'blur_clahe2_notebook': ('Notebook (Blur + CLAHE 2.0)', get_notebook_pipeline()),
}

input_images = input_base / 'images'
image_files = list(input_images.glob('*.jpg')) + list(input_images.glob('*.png'))
print(f'Found {len(image_files)} images')

for preset_name, (desc, pipeline) in presets.items():
    print(f'\nProcessing {preset_name}...')
    
    # Create output dir
    output_img_dir = output_base / preset_name / 'images'
    output_img_dir.mkdir(parents=True, exist_ok=True)
    
    # Process images
    for img_path in tqdm(image_files, desc=preset_name):
        image = load_image(img_path, mode='grayscale')
        processed_img, _ = pipeline(image) if pipeline else (image, None)
        output_path = output_img_dir / img_path.name
        save_image(processed_img, output_path, format='png')
    
    # Copy all label directories
    for label_dir_name in ['labels', 'labels_new', 'labels_4_class', 'labels_8_class', 'labels-knee']:
        input_labels = input_base / label_dir_name
        if input_labels.exists():
            output_labels = output_base / preset_name / label_dir_name
            if output_labels.exists():
                shutil.rmtree(output_labels)
            shutil.copytree(input_labels, output_labels)
            print(f'  ✅ Copied {label_dir_name}: {len(list(output_labels.glob(\"*.txt\")))} files')

print()
print('='*60)
print('COMPLETE!')
print('='*60)
print(f'Output: {output_base}')
"
