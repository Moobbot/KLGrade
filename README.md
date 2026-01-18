# KLGrade - Auto-Grading Knee Osteoarthritis

An end-to-end pipeline for Knee Osteoarthritis grading from X-ray images using Deep Learning (YOLOv8 / MIL).

---

## 📚 Documentation
The project documentation is organized into three main pillars:

1.  **[Data Pipeline](docs/DATA_PIPELINE.md)** (`docs/DATA_PIPELINE.md`)
    *   From raw X-rays to processed, balanced, and split datasets.
    *   Covers Label Generation, Cropping, Balancing, and Preprocessing.

2.  **[Dataset Reference](docs/DATASETS.md)** (`docs/DATASETS.md`)
    *   Reference guide for consuming the generated datasets.
    *   Details on folder structure, class formats (4/5/8/10-class), and statistics.

3.  **[Training & Evaluation](docs/TRAINING.md)** (`docs/TRAINING.md`)
    *   How to train models (YOLO, KIOCMIL).
    *   How to evaluate performance and run inference.

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
conda create -n klgrade python=3.10 -y
conda activate klgrade
pip install -r requirements.txt
```

### 2. Prepare Data (The Pipeline)
If you have raw data in `datasets/dataset_v0`, run the full pipeline to generate everything:
```bash
# Runs Steps 1-5 (Label Gen -> Crop -> Balance -> Preprocess -> Split)
# See docs/DATA_PIPELINE.md for details
bash scripts/pipelines/step_1_generate_labels.sh
bash scripts/pipelines/step_2_crop_knees.sh
bash scripts/pipelines/step_3_balance.sh
bash scripts/pipelines/step_4_preprocess_balanced.sh
bash scripts/pipelines/step_5_create_splits.sh
```

### 3. Train a Model
Train a baseline YOLOv8 model on the 5-class balanced dataset:
```bash
yolo detect train \
    data=configs/yolo_5_class_baseline.yaml \
    epochs=100 \
    batch=16 \
    imgsz=640 \
    name=quickstart_run
```

### 4. Evaluate
```bash
python scripts/evaluation/evaluate_yolo_standalone.py \
    --model runs/detect/quickstart_run/weights/best.pt \
    --data configs/yolo_5_class_baseline.yaml \
    --split test
```

---

## 📂 Project Structure
```
KLGrade/
├── datasets/               # Data (Raw, Cropped, Balanced)
├── docs/                   # Documentation (Data, Training, Logs)
├── configs/                # Training configurations (YAML)
├── scripts/                # Utility scripts
│   ├── pipelines/          # End-to-end pipeline steps
│   ├── training/           # Training runners
│   └── ...
├── src/                    # Source code (Models, Datasets)
└── runs/                   # Experiment outputs
```
