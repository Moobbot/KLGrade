# KLGrade - Auto-Grading Knee Osteoarthritis

An end-to-end pipeline for Knee Osteoarthritis grading from X-ray images using Deep Learning.

**Models**: KIOCMIL CADA (Context-Aware Deformable Attention) | YOLO Detection

---

## 📚 Documentation

Comprehensive documentation organized by topic:

### Core Documentation

1. **[Documentation Index](docs/README.md)** - Complete navigation guide
2. **[Data Pipeline](docs/guides/DATA_PIPELPE.md)** - From raw X-rays to processed datasets
3. **[Dataset Reference](docs/DATASETS.md)** - Dataset structure and statistics
4. **[Training Index](docs/TRAINING_INDEX.md)** - Central hub for all training docs

### Model-Specific Training

- **[KIOCMIL CADA Training](docs/KIOCMIL_CADA_TRAINING.md)** - Complete CADA training guide
- **[CADA Results](docs/experiments/CADA_COMPLETE_RESULTS.md)** - Evaluation results (20 experiments)

### Tools & Analysis

- **[Dataset Analysis Tools](tools/dataset_analysis/README.md)** - Dataset validation & visualization
- **[Visualization Organization](docs/reference/VISUALIZATION_ORGANIZATION.md)** - Viz code structure

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
conda create -n klgrade python=3.10 -y
conda activate klgrade
pip install -r requirements.txt
```

### 2. Prepare Data

From raw X-rays in `datasets/dataset_v0`, run the pipeline:

```bash
# Complete pipeline (recommended)
bash scripts/pipelines/run_full_pipeline.sh

# Or step-by-step:
bash scripts/pipelines/step_1_crop_knees.sh
bash scripts/pipelines/step_2_generate_labels.sh
bash scripts/pipelines/step_3_balance.sh
bash scripts/pipelines/step_4_preprocess_balanced.sh
bash scripts/pipelines/step_5_create_splits.sh
```

### 3. Train KIOCMIL CADA Model

Train the best-performing model (5-class balanced):

```bash
python src/training/train_kiocmil_cada.py \
    --num_classes 5 \
    --balanced \
    --epochs 100 \
    --batch_size 16 \
    --experiment_name cada_5class_balanced
```

Or use training scripts:

```bash
bash scripts/training/run_train_4_5_class.sh
```

### 4. Evaluate

```bash
bash scripts/training/evaluate_all_cada_experiments.sh
```

---

````

### Quick Test
```bash
# Verify installation
scripts/setup/test_setup.sh  # Linux/Mac
# or
scripts/setup/test_setup.bat  # Windows
````

📖 **Full Setup Guide**: [docs/SETUP.md](docs/SETUP.md)  
⚡ **Quick Start Guide**: [docs/QUICKSTART.md](docs/QUICKSTART.md)

## 📁 Project Structure

```
KLGrade/
├── src/                        # Source code
│   ├── models/                 # Model architectures (organized by type)
│   │   ├── cada/              # CADA architecture
│   │   ├── yolo/              # YOLO-based models
│   │   ├── resnet/            # ResNet-based models
│   │   └── end_to_end/        # End-to-end models
│   ├── training/              # Training scripts (organized by architecture)
│   │   ├── cada/              # CADA training
│   │   ├── yolo/              # YOLO training
│   │   └── resnet/            # ResNet training
│   ├── datasets/              # Dataset implementations
│   ├── api/                   # API inference logic
│   └── utils/                 # Utility functions
│
├── scripts/                   # Automation scripts
│   ├── setup/                # Environment setup
│   ├── api/                  # API launchers
│   ├── training/             # Training automation
│   └── deployment/           # Deployment scripts
│
├── docs/                      # Documentation
│   ├── SETUP.md              # Setup guide
│   ├── QUICKSTART.md         # Quick start
│   ├── TWO_STEP_YOLO_API.md  # Two-step API docs
│   ├── MODEL_EVALUATION.md   # Evaluation results
│   └── archive/              # Historical docs
│
├── data_processing/          # Data preparation pipelines
├── tools/                    # Development tools
├── tests/                    # Test suites
└── runs/                     # Training outputs & checkpoints
```

## 🏗️ Architectures

### 1. KIOCMIL-CADA (Recommended)

**Best accuracy** with context-aware deformable attention mechanism.

- **Accuracy**: Highest (target: 60-65% on 10-class)
- **Speed**: Medium (~500ms per image)
- **Use case**: Production, research

📖 [Architecture Details](src/models/cada/README.md)

### 2. Two-Step YOLO Pipeline

**Fast and efficient** detection + classification pipeline.

- **Accuracy**: Good (mAP@50-95: 0.667 on 8-class)
- **Speed**: Fast (~200ms per image)
- **Use case**: Real-time applications, edge devices

📖 [API Documentation](docs/TWO_STEP_YOLO_API.md)

### 3. YOLO-Based Models

YOLO11L backbone for improved feature extraction.

📖 [Model Details](src/models/yolo/README.md)

### 4. ResNet-Based Models

Lightweight baseline models.

📖 [Model Details](src/models/resnet/README.md)

## 🎓 Training

### CADA Model

```bash
python src/training/cada/train.py \
    --train_img_dir processed/knee_10_class/images \
    --train_knee_label_dir processed/knee_10_class/labels_knee \
    --train_lesion_label_dir processed/knee_10_class/labels_lesion \
    --epochs 100 \
    --batch_size 16
```

### YOLO Model

```bash
python src/training/yolo/train_v3.py \
    --img_dir dataset/dataset_v0/images \
    --epochs 50 \
    --batch_size 4
```

📖 **Training Guide**: [docs/TRAINING.md](docs/TRAINING.md)

## 🔬 Evaluation

```bash
# CADA Evaluation
python src/training/cada/evaluate.py \
    --checkpoint runs/kiocmil_cada/best_model.pt \
    --split_file splits/knee_10_class/test.txt

# YOLO Evaluation
python src/training/evaluate_kiocmil.py \
    --checkpoint runs/kiocmil_yolo/best_model.pt
```

📊 **Evaluation Results**: [docs/MODEL_EVALUATION.md](docs/MODEL_EVALUATION.md)

## 🌐 API Deployment

### KIOCMIL-CADA API

```bash
python scripts/deployment/kiocmil_api_server.py \
    --kiocmil-model runs/kiocmil_cada/best_acc_model.pt \
    --knee-model runs/detect/knee_detector/weights/best.pt \
    --port 8001
```

### Two-Step YOLO API

```bash
python scripts/api/start_two_step_api.py \
    --knee-model runs/detect/knee_detector/weights/best.pt \
    --lesion-model runs/detect/lesion_8class/weights/best.pt \
    --port 8002
```

**Swagger UI**: http://localhost:8001/docs

📖 **API Documentation**:

- [KIOCMIL-CADA API](docs/API_README.md)
- [Two-Step YOLO API](docs/TWO_STEP_YOLO_API.md)

## 📊 Performance Comparison

| Model      | Accuracy | Speed    | Memory | Complexity |
| ---------- | -------- | -------- | ------ | ---------- |
| **CADA**   | ⭐⭐⭐⭐ | ⭐⭐     | High   | Complex    |
| **YOLO**   | ⭐⭐⭐   | ⭐⭐⭐   | Medium | Medium     |
| **ResNet** | ⭐⭐     | ⭐⭐⭐⭐ | Low    | Simple     |

## 🛠️ Development

### Environment

- Python 3.10
- PyTorch 2.5.1+cu121
- CUDA 12.1 (optional, for GPU)

### Dependencies

```bash
pip install -r requirements.txt
```

### Testing

```bash
# Run tests
pytest tests/

# Lint code
flake8 src/
```

## 📚 Documentation

- **[Setup Guide](docs/SETUP.md)** - Detailed installation instructions
- **[Quick Start](docs/QUICKSTART.md)** - Get started in 5 minutes
- **[Training Guide](docs/TRAINING.md)** - How to train models
- **[API Documentation](docs/TWO_STEP_YOLO_API.md)** - API usage
- **[Model Evaluation](docs/MODEL_EVALUATION.md)** - Performance metrics
- **[YOLO Results](docs/YOLO_EVALUATION_RESULTS.md)** - YOLO evaluation

### Architecture-Specific Docs

- [CADA Models](src/models/cada/README.md)
- [YOLO Models](src/models/yolo/README.md)
- [ResNet Models](src/models/resnet/README.md)
- [End-to-End Models](src/models/end_to_end/README.md)

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

[Add your license here]

## 🙏 Acknowledgments

- YOLO by Ultralytics
- PyTorch team
- [Add other acknowledgments]

## 📧 Contact

[Add contact information]

---

**Note**: This project is under active development. For the latest updates, check the [changelog](CHANGELOG.md) or recent commits.
