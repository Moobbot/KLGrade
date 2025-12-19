# Dataset Analysis Report

**Created on**: 20/12/2025 

## Class Distribution

- **Total Classes**: 5
- **Total Instances**: 3127
- **Imbalance Ratio**: 13.71:1

### Per-Class Counts

| Class | Count |
|-------|-------|
| 0 | 99 |
| 1 | 794 |
| 2 | 1357 |
| 3 | 580 |
| 4 | 297 |

## Bounding Box Statistics

### Normalized Dimensions

- **Width**: μ=0.0713, σ=0.0625
- **Height**: μ=0.0634, σ=0.0320
- **Area**: μ=0.0054, σ=0.0071
- **Aspect Ratio**: μ=1.1474, median=0.9486

## Object Size Categories

- **Small (<1% of image)**: 2732 (87.4%)
- **Medium (1-5%)**: 390 (12.5%)
- **Large (>5%)**: 5 (0.2%)

## Recommendations

⚠️ **Class Imbalance Detected**: Consider using weighted loss or data augmentation for minority classes.

⚠️ **Many Small Objects**: Consider using feature pyramid networks or multi-scale training.

📊 **Anchor Box Suggestions**: Use k-means clustering on bbox dimensions to optimize anchor boxes for YOLO.

