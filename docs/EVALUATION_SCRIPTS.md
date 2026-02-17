# Evaluation Scripts Organization

## Current Evaluation Scripts

### Knee Detection Evaluation

1. **`evaluate_knee.py`** ✅ **Primary Script**
   - **Purpose**: Flexible knee detector evaluation with CLI arguments
   - **Features**:
     - Argparse for flexible configuration
     - Error handling
     - JSON output
     - Per-class metrics
     - Configurable split (train/val/test)
   - **Usage**:
     ```bash
     python scripts/evaluation/evaluate_knee.py \
         --model runs/detect/knee_detector/weights/best.pt \
         --split test
     ```

2. **`evaluate_knee_full.py`** ✅ **Comprehensive Evaluation**
   - **Purpose**: Evaluate on ALL splits (train + val + test)
   - **Features**:
     - Evaluates all 3 splits automatically
     - Summary table
     - Average metrics
     - JSON output
   - **Usage**:
     ```bash
     python scripts/evaluation/evaluate_knee_full.py \
         --model runs/detect/knee_detector/weights/best.pt \
         --output docs/results.json
     ```

3. **`compare_knee_detectors.py`** ✅ **Model Comparison**
   - **Purpose**: Compare multiple knee detector models
   - **Features**:
     - Side-by-side comparison
     - All splits evaluation
     - JSON output
   - **Usage**:
     ```bash
     python scripts/evaluation/compare_knee_detectors.py
     ```

### Other Evaluation Scripts

4. **`evaluate_cdt_cad.py`** - CDT-CAD model evaluation
5. **`evaluate_detr.py`** - DETR model evaluation
6. **`evaluate_yolo_standalone.py`** - Standalone YOLO evaluation
7. **`evaluate_all_models.py`** - Batch evaluation of all models
8. **`error_analysis.py`** - Error analysis tools
9. **`visualize_predictions.py`** - Prediction visualization
10. **`summarize_experiments.py`** - Experiment summary

## Recommendations

### For Single Split Evaluation:
```bash
python scripts/evaluation/evaluate_knee.py --model <path> --split test
```

### For Complete Evaluation (All Splits):
```bash
python scripts/evaluation/evaluate_knee_full.py --model <path>
```

### For Model Comparison:
```bash
python scripts/evaluation/compare_knee_detectors.py
```

## Training Scripts

### Knee Detector Training

**`scripts/training/train_knee_detector_yolo11n.sh`** ✅
- YOLO11N training with timestamp-based naming
- Prevents folder name conflicts
- Auto-evaluation after training
- Output: `runs/detect/knee_yolo11n_YYYYMMDD_HHMMSS/`
