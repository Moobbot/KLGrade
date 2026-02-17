# Result of Evaluation

## Summary
Evaluation of 8 trained YOLO models completed successfully.

**Top 3 Models:**
1. **Knee Detection**: mAP@50-95 = **0.732** (Precision: 0.995)
2. **8-class Cropped (Balanced)**: mAP@50-95 = **0.667** (**Best Lesion Model**)
3. **5-class Cropped (Balanced)**: mAP@50-95 = **0.571**

## Detailed Metrics

| Model | mAP@50 | mAP@50-95 | Best for... |
|-------|--------|-----------|-------------|
| Knee Detection | 0.995 | 0.732 | Step 1 (ROI Extraction) |
| 8-class Cropped | 0.763 | 0.667 | Step 2 (Lesion Detection) |
| 5-class Cropped | 0.734 | 0.571 | Alternative |
| 10-class Cropped | 0.633 | 0.571 | Fine-grained analysis |
| 10-class Full | 0.681 | 0.527 | Single-stage fallback |
| 8-class Full | 0.657 | 0.514 | Single-stage fallback |
| 4-class Cropped | 0.584 | 0.348 | Not recommended |
| 4-class Full | 0.388 | 0.182 | Not recommended |

## Recommendation
Deploy the **Two-Step Pipeline** using **Knee Detector** + **8-class Lesion Detector**.
