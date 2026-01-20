# KIOCMIL CADA Experiment Report

Generated on: 2026-01-20 05:59:23

## Summary Table

| Experiment | Num_Classes | Accuracy | Kappa | AUC_Macro | F1_Macro | Precision_Macro | Recall_Macro | Derived_Acc | Derived_Kappa | Derived_F1 | Derived_Prec | Derived_Rec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cada_10class_balanced | 10 | 0.8266 | 0.8063 | 0.9795 | 0.7799 | 0.8068 | 0.7851 | 0.8266 | 0.7833 | 0.8258 | 0.8382 | 0.8272 |
| cada_10class_balanced_blur | 10 | 0.4744 | 0.3040 | 0.8624 | 0.3220 | 0.3075 | 0.4036 | 0.4786 | 0.2606 | 0.3713 | 0.4461 | 0.3667 |
| cada_10class_balanced_resize | 10 | 0.5641 | 0.4377 | 0.8861 | 0.4380 | 0.4630 | 0.4566 | 0.5684 | 0.4011 | 0.4223 | 0.6378 | 0.4421 |
| cada_10class_balanced_sharp | 10 | 0.4957 | 0.3520 | 0.8711 | 0.2792 | 0.3188 | 0.3611 | 0.5000 | 0.3092 | 0.3320 | 0.2994 | 0.3781 |
| cada_10class_unbalanced | 10 | 0.5668 | 0.4509 | 0.8888 | 0.3925 | 0.3958 | 0.4404 | 0.5853 | 0.4381 | 0.5261 | 0.5485 | 0.5438 |
| cada_8class_balanced | 8 | 0.1313 | 0.0042 | 0.0000 | 0.0303 | 0.0282 | 0.1017 | 0.2799 | -0.0125 | 0.1007 | 0.2877 | 0.1911 |
| cada_8class_balanced_blur | 8 | 0.6746 | 0.6260 | 0.0000 | 0.6219 | 0.6584 | 0.6330 | 0.6864 | 0.5828 | 0.6837 | 0.6982 | 0.6946 |
| cada_8class_balanced_resize | 8 | 0.7603 | 0.7238 | 0.0000 | 0.7028 | 0.7277 | 0.7100 | 0.7922 | 0.7221 | 0.7867 | 0.7959 | 0.7903 |
| cada_8class_balanced_sharp | 8 | 0.7302 | 0.6907 | 0.0000 | 0.6768 | 0.7108 | 0.6833 | 0.7548 | 0.6740 | 0.7525 | 0.7624 | 0.7608 |
| cada_8class_unbalanced | 8 | 0.5217 | 0.3819 | 0.0000 | 0.3979 | 0.3623 | 0.4593 | 0.5314 | 0.3465 | 0.4302 | 0.4005 | 0.4789 |
| cada_5class_balanced | 5 | 0.7821 | 0.7013 | 0.0000 | 0.6981 | 0.7571 | 0.6769 | - | - | - | - | - |
| cada_5class_balanced_blur | 5 | 0.6538 | 0.5186 | 0.0000 | 0.4984 | 0.5580 | 0.5077 | - | - | - | - | - |
| cada_5class_balanced_resize | 5 | 0.7222 | 0.6182 | 0.0000 | 0.6276 | 0.6927 | 0.6088 | - | - | - | - | - |
| cada_5class_balanced_sharp | 5 | 0.6667 | 0.5356 | 0.0000 | 0.5616 | 0.7357 | 0.5425 | - | - | - | - | - |
| cada_5class_unbalanced | 5 | 0.8065 | 0.7310 | 0.0000 | 0.6698 | 0.7007 | 0.6658 | - | - | - | - | - |
| cada_4class_balanced | 4 | 0.2534 | 0.1382 | 0.0000 | 0.1397 | 0.2586 | 0.1720 | - | - | - | - | - |
| cada_4class_balanced_blur | 4 | 0.7059 | 0.6072 | 0.0000 | 0.6992 | 0.7016 | 0.7046 | - | - | - | - | - |
| cada_4class_balanced_resize | 4 | 0.7760 | 0.6993 | 0.0000 | 0.7645 | 0.7900 | 0.7683 | - | - | - | - | - |
| cada_4class_balanced_sharp | 4 | 0.6991 | 0.6001 | 0.0000 | 0.6914 | 0.7065 | 0.6994 | - | - | - | - | - |
| cada_4class_unbalanced | 4 | 0.8309 | 0.7629 | 0.0000 | 0.7541 | 0.8183 | 0.7572 | - | - | - | - | - |


## Analysis by Class Configuration

### 10-Class Experiments

| Experiment | Num_Classes | Accuracy | Kappa | AUC_Macro | F1_Macro | Precision_Macro | Recall_Macro | Derived_Acc | Derived_Kappa | Derived_F1 | Derived_Prec | Derived_Rec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cada_10class_balanced | 10 | 0.8266 | 0.8063 | 0.9795 | 0.7799 | 0.8068 | 0.7851 | 0.8266 | 0.7833 | 0.8258 | 0.8382 | 0.8272 |
| cada_10class_balanced_blur | 10 | 0.4744 | 0.3040 | 0.8624 | 0.3220 | 0.3075 | 0.4036 | 0.4786 | 0.2606 | 0.3713 | 0.4461 | 0.3667 |
| cada_10class_balanced_resize | 10 | 0.5641 | 0.4377 | 0.8861 | 0.4380 | 0.4630 | 0.4566 | 0.5684 | 0.4011 | 0.4223 | 0.6378 | 0.4421 |
| cada_10class_balanced_sharp | 10 | 0.4957 | 0.3520 | 0.8711 | 0.2792 | 0.3188 | 0.3611 | 0.5000 | 0.3092 | 0.3320 | 0.2994 | 0.3781 |
| cada_10class_unbalanced | 10 | 0.5668 | 0.4509 | 0.8888 | 0.3925 | 0.3958 | 0.4404 | 0.5853 | 0.4381 | 0.5261 | 0.5485 | 0.5438 |

**🏆 Best 10-Class Model:** `cada_10class_balanced` (Acc: 0.8266)

### 8-Class Experiments

| Experiment | Num_Classes | Accuracy | Kappa | AUC_Macro | F1_Macro | Precision_Macro | Recall_Macro | Derived_Acc | Derived_Kappa | Derived_F1 | Derived_Prec | Derived_Rec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cada_8class_balanced | 8 | 0.1313 | 0.0042 | 0.0000 | 0.0303 | 0.0282 | 0.1017 | 0.2799 | -0.0125 | 0.1007 | 0.2877 | 0.1911 |
| cada_8class_balanced_blur | 8 | 0.6746 | 0.6260 | 0.0000 | 0.6219 | 0.6584 | 0.6330 | 0.6864 | 0.5828 | 0.6837 | 0.6982 | 0.6946 |
| cada_8class_balanced_resize | 8 | 0.7603 | 0.7238 | 0.0000 | 0.7028 | 0.7277 | 0.7100 | 0.7922 | 0.7221 | 0.7867 | 0.7959 | 0.7903 |
| cada_8class_balanced_sharp | 8 | 0.7302 | 0.6907 | 0.0000 | 0.6768 | 0.7108 | 0.6833 | 0.7548 | 0.6740 | 0.7525 | 0.7624 | 0.7608 |
| cada_8class_unbalanced | 8 | 0.5217 | 0.3819 | 0.0000 | 0.3979 | 0.3623 | 0.4593 | 0.5314 | 0.3465 | 0.4302 | 0.4005 | 0.4789 |

**🏆 Best 8-Class Model:** `cada_8class_balanced_resize` (Acc: 0.7603)

### 5-Class Experiments

| Experiment | Num_Classes | Accuracy | Kappa | AUC_Macro | F1_Macro | Precision_Macro | Recall_Macro | Derived_Acc | Derived_Kappa | Derived_F1 | Derived_Prec | Derived_Rec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cada_5class_balanced | 5 | 0.7821 | 0.7013 | 0.0000 | 0.6981 | 0.7571 | 0.6769 | - | - | - | - | - |
| cada_5class_balanced_blur | 5 | 0.6538 | 0.5186 | 0.0000 | 0.4984 | 0.5580 | 0.5077 | - | - | - | - | - |
| cada_5class_balanced_resize | 5 | 0.7222 | 0.6182 | 0.0000 | 0.6276 | 0.6927 | 0.6088 | - | - | - | - | - |
| cada_5class_balanced_sharp | 5 | 0.6667 | 0.5356 | 0.0000 | 0.5616 | 0.7357 | 0.5425 | - | - | - | - | - |
| cada_5class_unbalanced | 5 | 0.8065 | 0.7310 | 0.0000 | 0.6698 | 0.7007 | 0.6658 | - | - | - | - | - |

**🏆 Best 5-Class Model:** `cada_5class_unbalanced` (Acc: 0.8065)

### 4-Class Experiments

| Experiment | Num_Classes | Accuracy | Kappa | AUC_Macro | F1_Macro | Precision_Macro | Recall_Macro | Derived_Acc | Derived_Kappa | Derived_F1 | Derived_Prec | Derived_Rec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cada_4class_balanced | 4 | 0.2534 | 0.1382 | 0.0000 | 0.1397 | 0.2586 | 0.1720 | - | - | - | - | - |
| cada_4class_balanced_blur | 4 | 0.7059 | 0.6072 | 0.0000 | 0.6992 | 0.7016 | 0.7046 | - | - | - | - | - |
| cada_4class_balanced_resize | 4 | 0.7760 | 0.6993 | 0.0000 | 0.7645 | 0.7900 | 0.7683 | - | - | - | - | - |
| cada_4class_balanced_sharp | 4 | 0.6991 | 0.6001 | 0.0000 | 0.6914 | 0.7065 | 0.6994 | - | - | - | - | - |
| cada_4class_unbalanced | 4 | 0.8309 | 0.7629 | 0.0000 | 0.7541 | 0.8183 | 0.7572 | - | - | - | - | - |

**🏆 Best 4-Class Model:** `cada_4class_unbalanced` (Acc: 0.8309)


## YOLO Detection Performance (Knee/Lesion Localization)

Performance of the YOLO models used to generate the bounding boxes for the above experiments.

| Model | mAP50 | mAP50-95 | Precision | Recall |
| --- | --- | --- | --- | --- |
| my_knee_run_resplit | 0.9870 | 0.6940 | 0.9812 | 0.9853 |
| train5 | 0.5729 | 0.3891 | 0.5825 | 0.6430 |
| train3 | 0.3308 | 0.1626 | 0.3041 | 0.4236 |
| E005_4_class_baseline | 0.3210 | 0.1176 | 0.3669 | 0.3706 |
| E006_8class_baseline | 0.3173 | 0.1137 | 0.4186 | 0.3778 |
| E001_5class_baseline | 0.2971 | 0.1194 | 0.3359 | 0.4216 |
| E004_10class_baseline | 0.2827 | 0.1009 | 0.2727 | 0.4289 |
| E002_5class_conservative | 0.0341 | 0.0094 | 0.0042 | 0.2206 |
