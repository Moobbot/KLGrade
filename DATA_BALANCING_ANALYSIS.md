# Data Balancing & Augmentation Analysis

## Overview
Analyzing `check_data.py` (from Jupyter notebook) to identify:
1. What's already covered by existing preprocessing
2. What needs to be added (data balancing, augmentation)
3. How to integrate or separate these pipelines

---

## Functions in check_data.py

### Already Covered by Current Preprocessing

1. **`load_grayscale_image()`** ✅
   - **Current**: `src/data/preprocessing/core/base.py::load_image(mode='grayscale')`
   - Status: Already implemented

2. **`resize_image()`** ✅
   - **Current**: `src/data/preprocessing/core/base.py::resize_image()`
   - Status: Already implemented

3. **`gaussian_blur()`** ✅
   - **Current**: `src/data/preprocessing/core/blur.py::gaussian_blur()`
   - Status: Already implemented

4. **`apply_clahe()`** ✅
   - **Current**: `src/data/preprocessing/core/clahe.py::clahe_equalization()`
   - Status: Already implemented

5. **`preprocess_image()`** ✅
   - **Current**: Composed via `Pipeline` in `src/data/preprocessing/pipeline.py`
   - Status: Already implemented via modular composition

6. **`save_processed_data()`** ✅
   - **Current**: `src/data/preprocessing/core/base.py::save_image()`
   - Status: Already implemented

### NOT Covered - Data Balancing & Augmentation

7. **`remove_class_0_from_labels()`** ❌ NEW
   - Purpose: Filter out class 0 and remap classes
   - Current status: Partially covered in `prepare_knee_crops.py` (filter_class_0)
   - **Recommendation**: Create general utility function

8. **`check_data()`** ❌ NEW
   - Purpose: Validate image-label pairs, move unmatched files
   - Current status: Not implemented
   - **Recommendation**: Create data validation utility

9. **`count_labels()`** ❌ NEW
   - Purpose: Count samples per class for balancing
   - Current status: Part of `analyze_knee_dataset.py` but not reusable
   - **Recommendation**: Create utility function

10. **`flip_image_and_labels()`** ❌ NEW
    - Purpose: Horizontal flip with label adjustment
    - Current status: Not implemented
    - **Recommendation**: Add to augmentation module

11. **`balance_data()`** ❌ NEW
    - Purpose: Oversample minority classes via flipping
    - Current status: Not implemented
    - **Recommendation**: Create separate data balancing module

12. **`scale_bounding_box()`** ❌ NEW
    - Purpose: Adjust bounding boxes after resize
    - Current status: Similar logic in `knee_crop.py` but not general
    - **Recommendation**: Create utility for YOLO label transformation

13. **`process_labels()`** ❌ NEW
    - Purpose: Adjust all labels after image transformation
    - Current status: Not implemented
    - **Recommendation**: Part of data balancing module

14. **`data_split()`** ❌ NEW
    - Purpose: Train/val/test split
    - Current status: Similar in `scripts/split_dataset.py`
    - **Recommendation**: Verify existing implementation

---

## Gap Analysis

### Current Preprocessing (Implemented)
✅ **Pixel-level transformations**:
- Load, resize, save
- Blur (Gaussian, median, bilateral)
- CLAHE
- Brightness, contrast adjustments
- Flip augmentation (basic)

### Missing Components (From check_data.py)

❌ **Data Validation**:
- Check image-label correspondence
- Move unmatched files
- Validate label format

❌ **Data Balancing**:
- Count samples per class
- Oversample minority classes
- Balance via augmentation

❌ **YOLO-Specific Utilities**:
- Bounding box scaling after resize
- Label remapping (class filtering)
- YOLO format validation

❌ **Augmentation for Balancing**:
- Flip with label adjustment
- Generate copies until balanced
- Track augmentation statistics

---

## Recommended Architecture

### Option 1: Separate Data Balancing Module

```
src/data/
├── preprocessing/          # Existing (pixel transformations)
├── balancing/              # NEW - Data balancing
│   ├── __init__.py
│   ├── validators.py       # check_data, validate pairs
│   ├── sampler.py          # count_labels, balance_data
│   ├── augmentor.py        # flip_image_and_labels
│   └── yolo_utils.py       # scale_bounding_box, process_labels
└── splitting/              # Train/val/test split
    ├── __init__.py
    └── stratified.py       # data_split with stratification
```

### Option 2: Add to Existing Preprocessing

```
src/data/preprocessing/
├── core/
│   ├── ...existing...
│   ├── validation.py       # NEW - data validation
│   └── balancing.py        # NEW - data balancing
```

**Recommendation**: **Option 1** - Separate module
- Data balancing is logically different from preprocessing
- Preprocessing = pixel transformations
- Balancing = dataset-level operations
- Easier to maintain and test separately

---

## Integration Points

### Current Workflow
```
1. prepare_knee_crops.py → Crop + filter classes
2. preprocess_production.py → Apply pixel transformations
3. ??? → Data balancing (MISSING)
4. ??? → Train/val/test split
5. Training
```

### Proposed Workflow
```
1. prepare_knee_crops.py → Crop + filter classes
2. validate_dataset.py → Check image-label pairs (NEW)
3. balance_dataset.py → Oversample minority classes (NEW)
4. preprocess_production.py → Apply pixel transformations
5. split_dataset.py → Train/val/test split (exists but verify)
6. Training
```

---

## Next Steps

1. **Create `src/data/balancing/` module** with:
   - validators.py (check_data)
   - sampler.py (count_labels, balance_data)
   - augmentor.py (flip_image_and_labels)
   - yolo_utils.py (bbox scaling, label processing)

2. **Create scripts**:
   - `scripts/validate_dataset.py` - Check data integrity
   - `scripts/balance_dataset.py` - Balance classes via augmentation
   - Verify `scripts/split_dataset.py` - Stratified splitting

3. **Update workflow** to include balancing step

4. **Documentation**: Update PREPROCESSING_WORKFLOW.md

---

## Priority Functions to Implement

### High Priority (Core Balancing)
1. `count_labels()` - Count samples per class
2. `balance_data()` - Oversample minority classes
3. `flip_image_and_labels()` - Augmentation for balancing

### Medium Priority (Validation)
4. `check_data()` - Validate image-label pairs
5. `scale_bounding_box()` - YOLO label adjustment

### Low Priority (Already Exists)
6. `data_split()` - Verify existing implementation
7. Image preprocessing functions - Already covered

---

## Comparison with Existing Code

### `scripts/run_full_pipeline.sh` includes:
- Knee cropping ✅
- Class filtering ✅
- Train/val/test split ✅
- **Missing**: Data balancing, validation

### `src/data/preprocessing.py` (legacy) includes:
- `balance_dataset_with_flip()` - Similar to `balance_data()`!
- This suggests balancing WAS implemented but not migrated to new modular architecture

**Action**: Review legacy `preprocessing.py` to extract balancing logic
