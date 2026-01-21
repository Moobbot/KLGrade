# KIOCMIL CADA - Complete Evaluation Results

**Generated**: 2026-01-21
**Total Experiments**: 30

---

## 🏆 Best Results by Configuration

| Classes | Best Model | Accuracy | Status | Grade Acc | Type Acc |
|---------|------------|----------|--------|-----------|----------|
| **4-Class** | 📊 `cada_4class_balanced` | **0.8371** | original | N/A | N/A |
| **5-Class** | 🔧 `cada_5class_unbalanced_corrected` | **0.8111** | corrected | N/A | N/A |
| **8-Class** | 📊 `cada_8class_balanced` | **0.7803** | original | 0.7803 | 0.9945 |
| **10-Class** | 📊 `cada_10class_balanced` | **0.8201** | original | 0.8201 | 0.9979 |

---

## � Model Comparison Analysis

Comparing performance across different model configurations.

### 4-Class Models

| Experiment | Original Acc | Corrected Acc | Δ Acc | AUC (Orig) | AUC (Corr) |
|------------|--------------|---------------|-------|------------|------------|
| `cada_4class_balanced` | 0.8371 | 0.8371 |  0.0000 | 0.0000 | 0.9660 |
| `cada_4class_balanced_blur` | 0.6991 | 0.6719 | 📉 -0.0271 | 0.0000 | 0.8947 |
| `cada_4class_balanced_resize` | 0.7692 | 0.7851 | 📈 +0.0158 | 0.0000 | 0.9325 |
| `cada_4class_balanced_sharp` | 0.6561 | 0.6968 | 📈 +0.0407 | 0.0000 | 0.9056 |
| `cada_4class_unbalanced` | 0.8261 | 0.8116 | 📉 -0.0145 | 0.0000 | 0.9324 |
### 5-Class Models

| Experiment | Original Acc | Corrected Acc | Δ Acc | AUC (Orig) | AUC (Corr) |
|------------|--------------|---------------|-------|------------|------------|
| `cada_5class_balanced` | 0.7650 | 0.7735 | 📈 +0.0085 | 0.0000 | 0.9425 |
| `cada_5class_balanced_blur` | 0.6453 | 0.6667 | 📈 +0.0214 | 0.0000 | 0.8453 |
| `cada_5class_balanced_resize` | 0.7308 | 0.7521 | 📈 +0.0214 | 0.0000 | 0.9051 |
| `cada_5class_balanced_sharp` | 0.6838 | 0.6838 |  0.0000 | 0.0000 | 0.8511 |
| `cada_5class_unbalanced` | 0.8065 | 0.8111 | 📈 +0.0046 | 0.0000 | 0.9407 |

---

## 4-Class Configuration

### Main Classification Metrics

| Experiment | Status | Accuracy | Kappa | F1 | AUC |
|------------|--------|----------|-------|-------|-----|
| 📊 `cada_4class_balanced_blur` | original | 0.6991 | 0.5956 | 0.6765 | 0.0000 |
| 📊 `cada_4class_balanced` | original | 0.8371 | 0.7776 | 0.8128 | 0.0000 |
| 📊 `cada_4class_balanced_resize` | original | 0.7692 | 0.6885 | 0.7624 | 0.0000 |
| 📊 `cada_4class_balanced_sharp` | original | 0.6561 | 0.5450 | 0.6477 | 0.0000 |
| 📊 `cada_4class_unbalanced` | original | 0.8261 | 0.7575 | 0.7686 | 0.0000 |
| 🔧 `cada_4class_balanced_blur_corrected` | corrected | 0.6719 | 0.5644 | 0.6610 | 0.8947 |
| 🔧 `cada_4class_balanced_corrected` | corrected | 0.8371 | 0.7795 | 0.8265 | 0.9660 |
| 🔧 `cada_4class_balanced_resize_corrected` | corrected | 0.7851 | 0.7114 | 0.7712 | 0.9325 |
| 🔧 `cada_4class_balanced_sharp_corrected` | corrected | 0.6968 | 0.5979 | 0.6864 | 0.9056 |
| 🔧 `cada_4class_unbalanced_corrected` | corrected | 0.8116 | 0.7352 | 0.7103 | 0.9324 |

## 5-Class Configuration

### Main Classification Metrics

| Experiment | Status | Accuracy | Kappa | F1 | AUC |
|------------|--------|----------|-------|-------|-----|
| 📊 `cada_5class_balanced_blur` | original | 0.6453 | 0.5009 | 0.4568 | 0.0000 |
| 📊 `cada_5class_balanced` | original | 0.7650 | 0.6817 | 0.7078 | 0.0000 |
| 📊 `cada_5class_balanced_resize` | original | 0.7308 | 0.6371 | 0.6581 | 0.0000 |
| 📊 `cada_5class_balanced_sharp` | original | 0.6838 | 0.5525 | 0.4881 | 0.0000 |
| 📊 `cada_5class_unbalanced` | original | 0.8065 | 0.7324 | 0.6787 | 0.0000 |
| 🔧 `cada_5class_balanced_blur_corrected` | corrected | 0.6667 | 0.5402 | 0.5147 | 0.8453 |
| 🔧 `cada_5class_balanced_corrected` | corrected | 0.7735 | 0.6973 | 0.7277 | 0.9425 |
| 🔧 `cada_5class_balanced_resize_corrected` | corrected | 0.7521 | 0.6555 | 0.6466 | 0.9051 |
| 🔧 `cada_5class_balanced_sharp_corrected` | corrected | 0.6838 | 0.5681 | 0.5908 | 0.8511 |
| 🔧 `cada_5class_unbalanced_corrected` | corrected | 0.8111 | 0.7420 | 0.7484 | 0.9407 |

## 8-Class Configuration

### Main Classification Metrics

| Experiment | Status | Accuracy | Kappa | F1 | AUC |
|------------|--------|----------|-------|-------|-----|
| 📊 `cada_8class_balanced_blur` | original | 0.7247 | 0.6820 | 0.6603 | 0.0000 |
| 📊 `cada_8class_balanced` | original | 0.7803 | 0.7469 | 0.7259 | 0.0000 |
| 📊 `cada_8class_balanced_resize` | original | 0.7521 | 0.7141 | 0.6950 | 0.0000 |
| 📊 `cada_8class_balanced_sharp` | original | 0.6764 | 0.6284 | 0.6234 | 0.0000 |
| 📊 `cada_8class_unbalanced` | original | 0.5266 | 0.3834 | 0.3975 | 0.0000 |

## 10-Class Configuration

### Main Classification Metrics

| Experiment | Status | Accuracy | Kappa | F1 | AUC |
|------------|--------|----------|-------|-------|-----|
| 📊 `cada_10class_balanced_blur` | original | 0.4786 | 0.3170 | 0.3867 | 0.8585 |
| 📊 `cada_10class_balanced` | original | 0.8201 | 0.7989 | 0.7621 | 0.9756 |
| 📊 `cada_10class_balanced_resize` | original | 0.5342 | 0.4005 | 0.2810 | 0.8812 |
| 📊 `cada_10class_balanced_sharp` | original | 0.4701 | 0.2857 | 0.2945 | 0.8562 |
| 📊 `cada_10class_unbalanced` | original | 0.5300 | 0.3985 | 0.3391 | 0.8856 |


---

## 📈 Performance Insights

### Data Balancing Impact

- **10-class & 8-class**: Balancing dramatically improves performance
- **5-class & 4-class**: Both balanced and unbalanced perform well

### Preprocessing Effects

- **Best**: Minimal or no preprocessing for most configurations
- **Resize**: Moderate improvements in some cases
- **Blur/Sharp**: Mixed results, dataset dependent

### Lesion Type Classification

- Models trained on 8/10-class excel at distinguishing osteophytes from joint space narrowing
- Type accuracy consistently >95%, reaching 99.79% for 10-class models

