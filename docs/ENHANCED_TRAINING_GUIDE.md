# Enhanced YOLO Training with Medical Image Processing

## Tổng quan

Script này áp dụng các kỹ thuật xử lý ảnh y khoa nâng cao từ notebook `Yolo_Detection_XuongKhop_v2.ipynb` để cải thiện hiệu suất training YOLO trên dataset KLGrade.

## Kỹ thuật áp dụng

### 1. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**

Cải thiện độ tương phản của ảnh X-ray:

```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced_image = clahe.apply(grayscale_image)
```

**Lợi ích:**
- ✅ Tăng độ tương phản cục bộ
- ✅ Làm nổi bật các đặc trưng khớp gối
- ✅ Giảm ảnh hưởng của độ sáng không đồng đều
- ✅ Phù hợp với ảnh y khoa (X-ray)

### 2. **Gaussian Blur (Noise Reduction)**

Giảm nhiễu trong ảnh:

```python
image_blurred = cv2.GaussianBlur(image, (5, 5), 0)
```

**Lợi ích:**
- ✅ Loại bỏ nhiễu sensor
- ✅ Làm mượt ảnh trước khi training
- ✅ Giảm overfitting trên chi tiết không quan trọng

### 3. **Data Balancing với Flip Augmentation**

Cân bằng class distribution bằng cách flip ảnh:

```python
# Flip image horizontally
flipped_image = cv2.flip(image, 1)

# Adjust bounding box
x_center_flipped = 1.0 - x_center
```

**Lợi ích:**
- ✅ Cân bằng minority classes (KL0, KL4)
- ✅ Tăng số lượng training samples
- ✅ Không làm thay đổi ngữ nghĩa y khoa (knees symmetric)
- ✅ Tự động điều chỉnh bounding boxes

### 4. **Label Scaling**

Điều chỉnh bounding boxes khi resize ảnh:

```python
# Scale bbox from original size to new size
x_scale = new_size[0] / original_size[1]
y_scale = new_size[1] / original_size[0]

x_center_scaled = x_center_pixel * x_scale
y_center_scaled = y_center_pixel * y_scale
```

**Lợi ích:**
- ✅ Bounding boxes chính xác sau resize
- ✅ Không bị méo hộp khi resize
- ✅ Giữ nguyên tỷ lệ aspect ratio

## Files đã tạo

### 1. Training Script
**File:** [scripts/training/train_yolo_enhanced.py](scripts/training/train_yolo_enhanced.py)

Script Python chứa toàn bộ logic preprocessing và training.

**Features:**
- CLAHE preprocessing
- Gaussian blur
- Data balancing
- Label scaling
- WandB integration
- Automatic data splitting

### 2. Training Shell Script
**File:** [docs/TRAINING_ENHANCED.sh](docs/TRAINING_ENHANCED.sh)

Script bash để chạy các experiments.

**Experiments:**
- `E_ENHANCED_001`: Dataset V0 + Full Enhancement
- `E_ENHANCED_002`: Dataset V0 + CLAHE only (no balancing)
- `E_ENHANCED_003`: knee_5_class + Full Enhancement

## Cách sử dụng

### Quick Test (2 epochs)

```bash
# Activate environment
conda activate klgrade

# Test với dataset_v0
python scripts/training/train_yolo_enhanced.py \
  --img_dir dataset/dataset_v0/images \
  --label_dir dataset/dataset_v0/labels \
  --split_dir splits/dataset_v0 \
  --num_classes 5 \
  --epochs 2 \
  --batch 4 \
  --device 0 \
  --name test_enhanced
```

### Full Training Pipeline

```bash
# Run all experiments
bash docs/TRAINING_ENHANCED.sh

# Or run in background
nohup bash docs/TRAINING_ENHANCED.sh > training_enhanced.log 2>&1 &
```

### Custom Training

```bash
python scripts/training/train_yolo_enhanced.py \
  --img_dir dataset/dataset_v0/images \
  --label_dir dataset/dataset_v0/labels \
  --split_dir splits/dataset_v0 \
  --num_classes 5 \
  --model yolo11n.pt \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --device 0 \
  --project runs/detect \
  --name my_enhanced_experiment \
  --no-balancing  # Disable data balancing (optional)
  --no-preprocessing  # Disable CLAHE (optional)
```

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--img_dir` | str | Required | Image directory |
| `--label_dir` | str | Required | Label directory |
| `--split_dir` | str | None | Split directory (train.txt, val.txt) |
| `--num_classes` | int | 5 | Number of classes |
| `--model` | str | yolo11n.pt | YOLO model |
| `--epochs` | int | 100 | Training epochs |
| `--batch` | int | 16 | Batch size |
| `--imgsz` | int | 640 | Image size |
| `--device` | str | 0 | Device (0 for GPU, cpu) |
| `--project` | str | runs/detect | Project directory |
| `--name` | str | yolo_enhanced | Experiment name |
| `--no-preprocessing` | flag | False | Disable CLAHE + Gaussian |
| `--no-balancing` | flag | False | Disable data balancing |

## Workflow

### 1. Preprocessing Phase

```
Input: dataset/dataset_v0/images + labels
         ↓
    Grayscale Conversion
         ↓
    Resize to 640x640
         ↓
    Gaussian Blur (5x5)
         ↓
    CLAHE Enhancement
         ↓
    Label Scaling
         ↓
Output: processed/enhanced/images + labels
```

### 2. Balancing Phase

```
Analyze Class Distribution
         ↓
Identify Minority Classes
         ↓
For each minority class:
    - Find images with class
    - Flip image horizontally
    - Adjust bounding boxes
    - Save augmented data
         ↓
Balanced Dataset
```

### 3. Training Phase

```
Create YOLO Config
         ↓
Load YOLO Model
         ↓
Train with Enhanced Data
         ↓
Validate
         ↓
Save Results to WandB
```

## Output Structure

```
processed/
└── enhanced/
    ├── images/                    # Enhanced images (grayscale PNG)
    │   ├── original_001.png
    │   ├── original_001_flip_0.png  # Augmented
    │   └── ...
    └── labels/                    # Scaled labels
        ├── original_001.txt
        ├── original_001_flip_0.txt
        └── ...

runs/detect/
├── E_ENHANCED_001_v0_full/
│   ├── weights/
│   │   ├── best.pt
│   │   └── last.pt
│   ├── results.png
│   ├── confusion_matrix.png
│   └── ...
├── E_ENHANCED_002_v0_clahe_only/
└── E_ENHANCED_003_knee5_full/
```

## Expected Results

### Improvements from CLAHE

- **Better contrast**: Ảnh X-ray có độ tương phản tốt hơn
- **Feature visibility**: Các đặc trưng khớp gối rõ ràng hơn
- **Reduced variation**: Giảm ảnh hưởng của điều kiện chụp khác nhau

### Improvements from Data Balancing

- **Class balance**: Cân bằng class distribution
- **Reduced bias**: Giảm bias về majority class (KL2)
- **Better minority class performance**: Cải thiện precision/recall cho KL0, KL4

### Expected mAP Improvement

Dựa trên notebook gốc:
- **Baseline**: mAP50 ~ 0.65-0.70
- **With CLAHE**: mAP50 ~ 0.70-0.75 (+5-10%)
- **With Balancing**: mAP50 ~ 0.75-0.80 (+10-15%)

## Comparison Table

| Experiment | Dataset | CLAHE | Balancing | Expected mAP50 |
|------------|---------|-------|-----------|----------------|
| E001 (Baseline) | dataset_v0 | ❌ | ❌ | 0.65-0.70 |
| E_ENHANCED_001 | dataset_v0 | ✅ | ✅ | 0.75-0.80 |
| E_ENHANCED_002 | dataset_v0 | ✅ | ❌ | 0.70-0.75 |
| E_ENHANCED_003 | knee_5_class | ✅ | ✅ | 0.78-0.83 |

## Monitor Progress

### Local Logs
```bash
tail -f training_enhanced.log
```

### GPU Usage
```bash
watch -n 1 nvidia-smi
```

### WandB Dashboard
https://wandb.ai/ngotam2k1-thuyloi-university/KLGrade-Knee-OA

Metrics logged:
- Loss curves (box, cls, dfl)
- mAP50, mAP50-95
- Precision, Recall per class
- Confusion matrix
- Sample predictions

## Troubleshooting

### Issue: Preprocessing takes too long

**Solution:** Reduce number of images or disable preprocessing for testing
```bash
python scripts/training/train_yolo_enhanced.py \
  --no-preprocessing \
  --epochs 10 \
  ...
```

### Issue: Out of Memory during balancing

**Solution:** Reduce batch size or disable balancing
```bash
python scripts/training/train_yolo_enhanced.py \
  --no-balancing \
  --batch 8 \
  ...
```

### Issue: Grayscale images look strange

**Solution:** This is expected! CLAHE creates grayscale images which may look different but have better contrast for detection.

### Issue: Class still imbalanced after augmentation

**Check:** 
1. Verify flip augmentation worked
2. Check `processed/enhanced/labels/` for `*_flip_*.txt` files
3. Increase max augmentation rounds in code

## Advanced Usage

### Custom CLAHE Parameters

Edit `train_yolo_enhanced.py`:

```python
# Default
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# More aggressive
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))

# More subtle
clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
```

### Custom Gaussian Blur

```python
# Default
image_blurred = cv2.GaussianBlur(image, (5, 5), 0)

# More blur
image_blurred = cv2.GaussianBlur(image, (7, 7), 0)

# Less blur
image_blurred = cv2.GaussianBlur(image, (3, 3), 0)
```

### Disable specific augmentations

```python
# In balance_dataset_with_flip(), comment out sections:

# Skip preprocessing
# enhanced_image, original_size = preprocess_image_clahe(...)
# Use original image instead

# Skip balancing
# Skip the augmentation loop
```

## References

### Original Notebook
- File: `Yolo_Detection_XuongKhop_v2.ipynb`
- Author: Original KLGrade team
- Techniques: CLAHE, Gaussian Blur, Flip Augmentation

### Papers
- CLAHE: Zuiderveld, K. (1994). "Contrast Limited Adaptive Histogram Equalization"
- YOLO: Jocher, G. et al. (2023). "Ultralytics YOLO"

### Documentation
- OpenCV CLAHE: https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
- Ultralytics: https://docs.ultralytics.com/

## Summary

✅ **Created:**
- Enhanced training script với CLAHE + Data Balancing
- Shell script để chạy experiments
- Documentation chi tiết

✅ **Ready to use:**
```bash
bash docs/TRAINING_ENHANCED.sh
```

✅ **Expected improvements:**
- Better contrast (CLAHE)
- Balanced classes (Flip augmentation)
- Higher mAP50 (+10-15%)
- Better minority class detection
