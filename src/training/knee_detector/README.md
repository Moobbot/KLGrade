# Knee Detection Training

This directory contains scripts for training the knee detection model (Step 1 of the two-step pipeline).

## Files

- `src/training/knee_detector/train.py` - Main training script for knee detection
- `src/training/knee_detector/prepare_data.py` - Data preparation utilities
- `src/training/knee_detector/evaluate.py` - Early evaluation script (deprecated, use `scripts/evaluation/evaluate_knee_full.py`)

## Usage

```bash
# Train knee detector
python api_two_step_yolo/training/knee/train_knee_detector.py --data <dataset.yaml> --epochs 100

# Evaluate knee detector
python api_two_step_yolo/training/knee/evaluate_knee_detector.py --model <model_path>
```
