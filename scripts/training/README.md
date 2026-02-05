# Training Scripts

This directory contains all training scripts for the KLGrade project, organized by training approach.

## 📁 Folder Structure

```
scripts/training/
├── end_to_end/              # End-to-end KIOCMIL + Detection
│   ├── train_end_to_end.py
│   ├── visualize_end_to_end_*.sh
│   └── README.md
│
├── kiocmil_two_step/        # Two-step: Detection → Classification
│   ├── run_all_cada_experiments.sh
│   ├── train_kiocmil*.sh
│   └── README.md
│
└── yolo_detection/          # YOLO Detection Only
    ├── train_yolo*.py
    ├── train_knee_detection.py
    └── README.md
```

## 🎯 Which Approach to Use?

### End-to-End (`end_to_end/`)
**Use when:**
- Want joint optimization of detection + classification
- Need single model deployment
- Have sufficient GPU memory (~8 GiB)

**Pros:**
- ✅ Single model, simpler deployment
- ✅ Joint learning may improve both tasks
- ✅ Faster inference (one forward pass)

**Cons:**
- ❌ Harder to debug
- ❌ Requires more memory
- ❌ Less mature (experimental)

### Two-Step (`kiocmil_two_step/`)
**Use when:**
- Want modular, debuggable pipeline
- Need flexibility to swap components
- Following proven KIOCMIL approach

**Pros:**
- ✅ Modular: train detection/classification separately
- ✅ Easier debugging
- ✅ Proven results (42.47% accuracy)
- ✅ Flexible: swap models independently

**Cons:**
- ❌ Two models to deploy
- ❌ Slower inference (two forward passes)

### YOLO Detection Only (`yolo_detection/`)
**Use when:**
- Only need detection (no classification)
- Training detection for two-step pipeline
- Evaluating detection performance

## 🚀 Quick Start

### End-to-End Training
```bash
cd /path/to/KLGrade
./scripts/training/end_to_end/train_end_to_end_test.sh
```

### Two-Step Training
```bash
# Step 1: Train YOLO detection
python scripts/training/yolo_detection/train_yolo.py --data datasets/dataset_v0/data.yaml

# Step 2: Train KIOCMIL classification
./scripts/training/kiocmil_two_step/run_all_cada_experiments.sh
```

### Visualization
```bash
# End-to-end model visualization
./scripts/training/end_to_end/visualize_end_to_end_optimized.sh
```

## 📊 Performance Comparison

| Approach | Val Accuracy | GPU Memory | Inference Speed | Maturity |
|----------|--------------|------------|-----------------|----------|
| **Two-Step** | **42.47%** | ~4 GiB | ~30ms | ✅ Proven |
| **End-to-End** | 35.85% | ~8 GiB | ~15ms | ⚠️ Experimental |

## 📚 Documentation

Each folder contains its own README with:
- Detailed script descriptions
- Usage examples
- Best practices
- Performance metrics

**Read the folder-specific READMEs for more details:**
- [`end_to_end/README.md`](end_to_end/README.md)
- [`kiocmil_two_step/README.md`](kiocmil_two_step/README.md)
- [`yolo_detection/README.md`](yolo_detection/README.md)

## 🔧 Common Tasks

### Train from Scratch
```bash
# Two-step (recommended)
./scripts/training/kiocmil_two_step/run_all_cada_experiments.sh

# End-to-end (experimental)
./scripts/training/end_to_end/train_end_to_end_test.sh
```

### Evaluate Models
```bash
# Two-step
./scripts/training/kiocmil_two_step/evaluate_all_cada_experiments.sh

# End-to-end
python scripts/training/end_to_end/test_end_to_end_inference.py
```

### Visualize Predictions
```bash
# End-to-end (optimized)
./scripts/training/end_to_end/visualize_end_to_end_optimized.sh
```

## 📝 Notes

- **Recommended**: Start with two-step approach (proven results)
- **Experimental**: End-to-end approach shows promise but needs more work
- **Memory**: End-to-end requires more GPU memory (~8 GiB vs ~4 GiB)
- **Debugging**: Two-step is easier to debug and iterate on

## 🎯 Next Steps

1. ✅ Folder structure organized
2. ✅ Scripts categorized by approach
3. 🔲 Load real images in end-to-end training
4. 🔲 Full training runs with optimized settings
5. 🔲 Comprehensive evaluation and comparison
