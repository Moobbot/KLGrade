# Dataset Analysis Report

**Created on**: 20/12/2025 

## Class Distribution

- **Total Classes**: 7
- **Total Instances**: 3069
- **Imbalance Ratio**: 24.13:1

### Per-Class Counts

| Class | Count |
|-------|-------|
| 0 | 89 |
| 1 | 771 |
| 2 | 1303 |
| 3 | 54 |
| 4 | 521 |
| 5 | 59 |
| 6 | 272 |

## Bounding Box Statistics

### Normalized Dimensions

- **Width**: μ=0.0669, σ=0.0540
- **Height**: μ=0.0629, σ=0.0319
- **Area**: μ=0.0050, σ=0.0063
- **Aspect Ratio**: μ=1.1027, median=0.9448

## Object Size Categories

- **Small (<1% of image)**: 2732 (89.0%)
- **Medium (1-5%)**: 332 (10.8%)
- **Large (>5%)**: 5 (0.2%)

## Recommendations

⚠️ **Class Imbalance Detected**: Consider using weighted loss or data augmentation for minority classes.

⚠️ **Many Small Objects**: Consider using feature pyramid networks or multi-scale training.

📊 **Anchor Box Suggestions**: Use k-means clustering on bbox dimensions to optimize anchor boxes for YOLO.

