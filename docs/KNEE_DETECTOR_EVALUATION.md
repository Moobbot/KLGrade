# Knee Detector Evaluation & Selection

**Date:** 2026-02-17
**Selected Model:** YOLO11N (Nano)
**Comparison Target:** YOLO11L (Large)

## 🏆 Executive Summary

After rigorous evaluation and comparison, **YOLO11N (Nano)** has been selected as the production knee detector model.

Contrary to common assumptions that "larger is better," the Nano model demonstrated:
- **Equal Accuracy**: Matches the Large model in mAP@50 (99.48%).
- **Better Localization**: Slightly outperforms Large model in mAP@50-95 (+0.1%).
- **Superior Speed**: **5x faster** inference (1.8ms vs 9.4ms).
- **Efficiency**: **9x smaller** model size (5.3MB vs 49MB).

## 📊 Model Comparison (Test Set)

| Metric | YOLO11N (Selected) | YOLO11L (Comparison) | Difference | Status |
| :--- | :---: | :---: | :---: | :--- |
| **mAP@50** | **99.48%** | 99.48% | = | ✅ Excellent |
| **mAP@50-95** | **82.56%** | 82.46% | **+0.10%** | ✅ Superior |
| **Precision** | **99.61%** | 98.97% | **+0.64%** | ✅ Superior |
| **Recall** | 98.68% | **98.84%** | -0.16% | ⚠️ Comparable |
| **Inference Time** | **1.87 ms** | 9.36 ms | **5x Faster** | 🚀 Outstanding |
| **Model Size** | **5.3 MB** | 49.6 MB | **9x Smaller** | 💾 Efficient |
| **Parameters** | **2.6M** | 25.3M | **10x Fewer** | ⚡ Efficient |

## 🧠 Analysis: Why YOLO11N Wins?

1.  **Task Complexity**: Knee detection in X-rays is a "single large object detection" task with relatively simple features. The 2.6M parameters of YOLO11N are sufficient to saturate the performance.
2.  **Overfitting**: The massive 25M parameters of YOLO11L likely led to slight overfitting or memorization of training noise, whereas YOLO11N's constrained capacity forced it to learn more robust, generalizable features.
3.  **Diminishing Returns**: Increasing model size by 10x provided NO accuracy benefit, only computational cost.

## 📈 Selected Model Performance (YOLO11N)

### Complete Dataset Evaluation
Evaluated on **1,461 images** across all splits:

| Split | Images | mAP@50 | mAP@50-95 | Precision | Recall |
|-------|--------|--------|-----------|-----------|--------|
| **Train** | 1,022 | **99.48%** | 87.86% | 99.74% | 99.52% |
| **Val** | 219 | **99.50%** | 83.03% | 99.63% | 99.76% |
| **Test** | 220 | **99.48%** | 82.56% | 99.61% | 98.68% |
| **AVERAGE** | 1,461 | **99.49%** | **84.48%** | **99.66%** | **99.32%** |

### Speed Performance (RTX 2080 Ti)
- **Preprocess**: 0.6ms
- **Inference**: 1.8ms
- **Postprocess**: 0.5ms
- **Total**: ~3ms per image (**~333 FPS**)

## ✅ Conclusion & Recommendation

The **YOLO11N** model currently stored at `runs/detect/knee_yolo11n_*/weights/best.pt` is **RECOMMENDED** for deployment in the Two-Step Pipeline.

### Key Benefits for Pipeline:
1.  **Low Latency**: Adds negligible delay (<3ms) to the total pipeline processing time.
2.  **Resource Friendly**: Consumes minimal VRAM, leaving resources available for the heavier Step 2 (Lesion Classification/Grading) model.
3.  **High Reliability**: >99% mAP ensures the second step almost always receives a valid knee crop.

### Usage
```python
from ultralytics import YOLO

# Load the efficient Nano model
model = YOLO('runs/detect/knee_yolo11n_20260217_134003/weights/best.pt')

# Inference
results = model('image.jpg')
```
