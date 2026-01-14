# Visualization Summary - Preprocessing Comparisons
Date: 2026-01-14

## Generated Visualizations

### Full X-rays (dataset_v0)
Location: `datasets/data_examples/`

1. **comparison_raw_vs_processed.png**
   - Multi-sample overview (3 samples)
   - 5 preprocessing methods side-by-side
   - Methods: Raw, Basic (Resize), v0 (Blur+CLAHE2.0), v3 Legacy (Sharp+CLAHE4.0), Notebook
   
2. **comparison_detailed.png**
   - Single-image detailed analysis
   - Includes histograms and statistics
   - 4 methods: Raw, Basic, v0, v3 Legacy
   - Statistics: Mean, Std, Range

### Cropped Knees (knees_cropped)
Location: `datasets/data_examples/knees_cropped/`

1. **comparison_raw_vs_processed.png**
   - Multi-sample overview (3 knee crops)
   - 5 preprocessing methods side-by-side
   - Methods: Raw, Basic (Resize), v0 (Blur+CLAHE2.0), v3 Legacy (Sharp+CLAHE4.0), Notebook
   
2. **comparison_detailed.png**
   - Single knee crop detailed analysis
   - Includes histograms and statistics
   - 4 methods: Raw, Basic, v0, v3 Legacy
   - Statistics: Mean, Std, Range

## Scripts

- `examples/preprocessing_comparison.py` - Full X-rays comparisons
- `examples/preprocessing_comparison_knees.py` - Cropped knees comparisons

## Usage

```bash
# Generate comparisons for full X-rays
PYTHONPATH=/home/ngoductam/KLGrade \
python examples/preprocessing_comparison.py

# Generate comparisons for cropped knees
PYTHONPATH=/home/ngoductam/KLGrade \
python examples/preprocessing_comparison_knees.py
```

## Purpose

These visualizations help evaluate and choose the best preprocessing method by:
- Comparing visual quality across methods
- Analyzing pixel intensity distributions
- Understanding statistical properties of each method
- Identifying artifacts or issues in preprocessing
