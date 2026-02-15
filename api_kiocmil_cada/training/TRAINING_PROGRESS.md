# Priority Training Progress Summary

## Status: ✅ Completed

**Started:** 2026-02-12 10:10
**Completed:** 2026-02-15 13:45
**Total Duration:** ~3 days (including evaluation)

## Final Results (Top Models)

| Rank | Model | mAP@50 | mAP@50-95 | Notes |
|------|-------|--------|-----------|-------|
| 1 | **Knee Detection** | **0.995** | **0.732** | Excellent ROI extraction |
| 2 | **8-class Cropped (Balanced)** | **0.763** | **0.667** | **BEST Lesion Detector** |
| 3 | 5-class Cropped (Balanced) | 0.734 | 0.571 | Good backup |
| 4 | 10-class Full X-ray (Balanced) | 0.681 | 0.527 | Best single-stage model |

## Training Queue Status

### Priority 1: ✅ Completed
- **Knee Detection**
- **Result:** mAP@50-95: 0.732 (Exceeded expectation of 0.727)
- **Status:** Integrated into Two-Step Pipeline

### Priority 2: ✅ Completed
- **8-class Cropped (Balanced)**
- **Result:** mAP@50-95: 0.667 (Matches expectation)
- **Status:** **SELECTED** for Final Pipeline

### Priority 3: ✅ Completed
- **5-class Cropped (Balanced)**
- **Result:** mAP@50-95: 0.571

### Priority 4: ✅ Completed
- **10-class Cropped (Balanced)**
- **Result:** mAP@50-95: 0.571

### Priority 5: ✅ Completed
- **10-class Full X-ray (Balanced)**
- **Result:** mAP@50-95: 0.527

### Priority 6-8: ✅ Completed
- 8-class Full X-ray, 4-class Cropped/Full
- **Status:** Lower performance, not recommended.

## Next Steps

1.  **Deployment**: The Two-Step Pipeline is now finalized using:
    -   Step 1: `runs/detect/knee_detector`
    -   Step 2: `runs/detect/lesion_8class_balanced`
    
2.  **API Usage**:
    ```bash
    python api_kiocmil_cada/inference/two_step_yolo_api.py --image test.jpg
    ```

3.  **Documentation**: Updated `TWO_STEP_YOLO_API.md` with final metrics.
