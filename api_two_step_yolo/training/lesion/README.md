# Lesion Detection Training

This directory contains scripts for training the lesion detection models (Step 2 of the two-step pipeline).

## Files

- `train_lesion_detector.py` - Main training script for lesion detection
- `prepare_lesion_data.py` - Data preparation utilities
- `evaluate_lesion_detector.py` - Evaluation script for lesion detector
- `train_lesion_sequential.py` - Sequential training for multiple configurations
- `train_lesion_sequential.sh` - Shell script wrapper for sequential training

## Usage

```bash
# Train lesion detector
python api_two_step_yolo/training/lesion/train_lesion_detector.py --data <dataset.yaml> --epochs 100

# Train multiple configurations sequentially
python api_two_step_yolo/training/lesion/train_lesion_sequential.py

# Evaluate lesion detector
python api_two_step_yolo/training/lesion/evaluate_lesion_detector.py --model <model_path>
```
