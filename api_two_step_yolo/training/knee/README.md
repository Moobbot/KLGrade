# Knee Detection Training

This directory contains scripts for training the knee detection model (Step 1 of the two-step pipeline).

## Files

- `train_knee_detector.py` - Main training script for knee detection
- `prepare_knee_data.py` - Data preparation utilities
- `evaluate_knee_detector.py` - Evaluation script for knee detector

## Usage

```bash
# Train knee detector
python api_two_step_yolo/training/knee/train_knee_detector.py --data <dataset.yaml> --epochs 100

# Evaluate knee detector
python api_two_step_yolo/training/knee/evaluate_knee_detector.py --model <model_path>
```
