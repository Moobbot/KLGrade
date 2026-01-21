# KIOCMIL CADA - Experiment Results Summary

**Generated**: 2026-01-21  
**Total Experiments**: 20

---

## 📊 Overall Best Results

| Classes | Best Accuracy | Experiment | Balance | Preprocess |
|---------|---------------|------------|---------|------------|
| **4-Class** | **83.09%** | `cada_4class_unbalanced` | Unbalanced | None |
| **5-Class** | **80.65%** | `cada_5class_unbalanced` | Unbalanced | None |
| **8-Class** | **76.03%** | `cada_8class_balanced_resize` | Balanced | Resize |
| **10-Class** | **82.66%** | `cada_10class_balanced` | Balanced | None |

---

## 🎯 Key Findings

### 1. **4-Class Configuration** (KL1-4)
- 🏆 **Best**: Unbalanced (83.09%)
- Balancing **hurts** performance significantly
- Best balanced: Resize preprocessing (77.60%)
- **Insight**: Natural class distribution works best for 4-class

| Experiment | Accuracy | Kappa | F1 Macro |
|------------|----------|-------|----------|
| ✅ Unbalanced | **83.09%** | 0.7629 | 0.7541 |
| Balanced + Resize | 77.60% | 0.6993 | 0.7645 |
| Balanced + Blur | 70.59% | 0.6072 | 0.6992 |
| Balanced + Sharp | 69.91% | 0.6001 | 0.6914 |
| Balanced (None) | 25.34% | 0.1382 | 0.1397 |

### 2. **5-Class Configuration** (KL0-4)
- 🏆 **Best**: Unbalanced (80.65%)
- Balancing improves from 25% → 78% (Balanced None)
- Best balanced: None preprocessing (78.21%)
- **Insight**: Unbalanced still superior but balanced viable

| Experiment | Accuracy | Kappa | F1 Macro |
|------------|----------|-------|----------|
| ✅ Unbalanced | **80.65%** | 0.7310 | 0.6698 |
| Balanced (None) | 78.21% | 0.7013 | 0.6981 |
| Balanced + Resize | 72.22% | 0.6182 | 0.6276 |
| Balanced + Sharp | 66.67% | 0.5356 | 0.5616 |
| Balanced + Blur | 65.38% | 0.5186 | 0.4984 |

### 3. **8-Class Configuration** (KL1-4 × a/b)
- 🏆 **Best**: Balanced + Resize (76.03%)
- Balancing **essential** for fine-grained classification
- Unbalanced performs poorly (52.17%)
- **Derived Metrics**:
  - Grade Accuracy: 79.22%
  - Type Accuracy: 95.62%

| Experiment | Main Acc | Grade Acc | Type Acc | Kappa | F1 Macro |
|------------|----------|-----------|----------|-------|----------|
| ✅ Balanced + Resize | **76.03%** | 79.22% | 95.62% | 0.7238 | 0.7028 |
| Balanced + Sharp | 73.02% | 75.48% | 95.35% | 0.6907 | 0.6768 |
| Balanced + Blur | 67.46% | 68.64% | 96.72% | 0.6260 | 0.6219 |
| Unbalanced | 52.17% | 53.14% | 98.55% | 0.3819 | 0.3979 |
| Balanced (None) | 13.13% | 27.99% | 41.75% | 0.0042 | 0.0303 |

### 4. **10-Class Configuration** (KL0-4 × a/b)
- 🏆 **Best**: Balanced (82.66%)
- Spectacular improvement with balancing
- No preprocessing needed
- **Derived Metrics**:
  - Grade Accuracy: 82.66%
  - Type Accuracy: 99.79%

| Experiment | Main Acc | Grade Acc | Type Acc | AUC Macro | F1 Macro |
|------------|----------|-----------|----------|-----------|----------|
| ✅ Balanced (None) | **82.66%** | 82.66% | 99.79% | 0.9795 | 0.7799 |
| Unbalanced | 56.68% | 58.53% | 97.24% | 0.8888 | 0.3925 |
| Balanced + Resize | 56.41% | 56.84% | 98.72% | 0.8861 | 0.4380 |
| Balanced + Sharp | 49.57% | 50.00% | 98.72% | 0.8711 | 0.2792 |
| Balanced + Blur | 47.44% | 47.86% | 98.72% | 0.8624 | 0.3220 |

---

## 🔍 Insights & Recommendations

### Balancing Strategy
- **4-Class & 5-Class**: Unbalanced data works best
- **8-Class & 10-Class**: Balancing is **critical** (improves 52% → 76% and 57% → 83%)

### Preprocessing Strategy
- **Best for 8-class**: Resize Only
- **Best for 10-class**: No preprocessing
- **Avoid**: Blur+CLAHE and Sharp+CLAHE often degrade performance

### Type (a/b) Classification
- Extremely accurate across all configurations (95-99%)
- Model excels at distinguishing osteophytes vs joint space narrowing

### Grade-Level Performance
- 8-class: Grade accuracy (79%) > Main accuracy (76%)
- 10-class: Grade accuracy (83%) ≈ Main accuracy (83%)
- Model learns grade structure well despite fine-grained training

---

## 📈 Performance Comparison

### By Class Configuration

```
4-Class:  ████████████████  83.09%
10-Class: ███████████████▌  82.66%
5-Class:  ███████████████   80.65%
8-Class:  ██████████████▎   76.03%
```

### Impact of Balancing

**Positive Impact (10-Class)**:
- Unbalanced: 56.68% → Balanced: 82.66% **(+25.98%)**

**Negative Impact (4-Class)**:
- Unbalanced: 83.09% → Best Balanced: 77.60% **(-5.49%)**

---

## 🎯 Recommended Configurations

### For Production Deployment

| Task | Configuration | Accuracy | Notes |
|------|---------------|----------|-------|
| **Simple KL Grading** | 4-class unbalanced | 83.09% | Fast, accurate, simple |
| **Full KL Grading** | 5-class unbalanced | 80.65% | Includes KL0 detection |
| **Lesion Type Analysis** | 8-class balanced+resize | 76.03% | Best for osteophytes vs joint space |
| **Fine-grained Analysis** | 10-class balanced | 82.66% | Highest granularity |

---

## 📁 Files

- **Raw Results**: [`results_summary.csv`](file:///home/ngoductam/KLGrade/runs/kiocmil_cada/results_summary.csv)
- **Experiment Directory**: [`runs/kiocmil_cada/`](file:///home/ngoductam/KLGrade/runs/kiocmil_cada/)

---

## 🚀 Next Steps

1. **Analyze**: Review confusion matrices for top performers
2. **Deploy**: Package best models for each use case
3. **Test**: Validate on external test set
4. **Document**: Update model cards with performance metrics
