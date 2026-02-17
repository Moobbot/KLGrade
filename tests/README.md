# Tests

This directory contains test scripts for the KLGrade project.

## Test Scripts

### API Tests

- **`test_api.py`** - Test API server functionality

```bash
# Start API server first
python api_two_step_yolo/inference/server.py

# Run test
conda run -n klgrade python tests/test_api.py
```

### Inference Tests

- **`test_v0_inference.py`** - Test two-step YOLO inference

```bash
python tests/test_v0_inference.py \
    --image datasets/dataset_v0/images/sample.jpg
```

---

## Running Tests

### Prerequisites
```bash
# Activate environment
conda activate klgrade

# Ensure models are available
ls runs/detect/knee_detector/weights/best.pt
ls runs/detect/lesion_detector/weights/best.pt
```

### Run All Tests
```bash
# From project root
cd /home/ngoductam/KLGrade

# Run individual tests
python tests/test_api.py
python tests/test_v0_inference.py
```

---

## Notes

- All tests should be run from project root
- Tests require trained models in `runs/detect/`
- API tests require server to be running
- Use `--help` flag for test-specific options
