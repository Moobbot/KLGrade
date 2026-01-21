# KIOCMIL CADA Experiment Report

Generated on: 2026-01-21 22:09:23

## Summary Table

| Experiment | Num_Classes | Accuracy | Kappa | AUC_Macro | F1_Macro | Precision_Macro | Recall_Macro | Derived_Acc | Derived_Kappa | Derived_F1 | Derived_Prec | Derived_Rec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cada_10class_balanced | 10 | 0.8201 | 0.7989 | 0.9756 | 0.7621 | 0.7971 | 0.7714 | 0.8201 | 0.7748 | 0.8194 | 0.8308 | 0.8180 |
| cada_10class_balanced_blur | 10 | 0.4786 | 0.3170 | 0.8585 | 0.3867 | 0.3987 | 0.4625 | 0.4872 | 0.2764 | 0.3942 | 0.4995 | 0.3821 |
| cada_10class_balanced_resize | 10 | 0.5342 | 0.4005 | 0.8812 | 0.2810 | 0.2464 | 0.3385 | 0.5427 | 0.3664 | 0.3580 | 0.3218 | 0.4053 |
| cada_10class_balanced_sharp | 10 | 0.4701 | 0.2857 | 0.8562 | 0.2945 | 0.3122 | 0.3574 | 0.4701 | 0.2210 | 0.2845 | 0.3031 | 0.3122 |
| cada_10class_unbalanced | 10 | 0.5300 | 0.3985 | 0.8856 | 0.3391 | 0.3177 | 0.4090 | 0.5438 | 0.3734 | 0.4100 | 0.4022 | 0.4451 |
| cada_8class_balanced | 8 | 0.7803 | 0.7469 | 0.0000 | 0.7259 | 0.7478 | 0.7431 | 0.7803 | 0.7079 | 0.7794 | 0.7859 | 0.7900 |
| cada_8class_balanced_blur | 8 | 0.7247 | 0.6820 | 0.0000 | 0.6603 | 0.6774 | 0.6680 | 0.7411 | 0.6532 | 0.7359 | 0.7417 | 0.7402 |
| cada_8class_balanced_resize | 8 | 0.7521 | 0.7141 | 0.0000 | 0.6950 | 0.7233 | 0.7037 | 0.7785 | 0.7039 | 0.7731 | 0.7804 | 0.7788 |
| cada_8class_balanced_sharp | 8 | 0.6764 | 0.6284 | 0.0000 | 0.6234 | 0.6679 | 0.6343 | 0.7156 | 0.6191 | 0.6998 | 0.7064 | 0.7074 |
| cada_8class_unbalanced | 8 | 0.5266 | 0.3834 | 0.0000 | 0.3975 | 0.3783 | 0.4530 | 0.5314 | 0.3433 | 0.4279 | 0.4099 | 0.4725 |
| cada_5class_balanced | 5 | 0.7650 | 0.6817 | 0.0000 | 0.7078 | 0.7309 | 0.6924 | - | - | - | - | - |
| cada_5class_balanced_blur | 5 | 0.6453 | 0.5009 | 0.0000 | 0.4568 | 0.5964 | 0.4820 | - | - | - | - | - |
| cada_5class_balanced_blur_corrected | 5 | 0.6667 | 0.5402 | 0.8453 | 0.5147 | 0.5589 | 0.5278 | - | - | - | - | - |
| cada_5class_balanced_corrected | 5 | 0.7735 | 0.6973 | 0.9425 | 0.7277 | 0.7674 | 0.7135 | - | - | - | - | - |
| cada_5class_balanced_resize | 5 | 0.7308 | 0.6371 | 0.0000 | 0.6581 | 0.6988 | 0.6478 | - | - | - | - | - |
| cada_5class_balanced_resize_corrected | 5 | 0.7521 | 0.6555 | 0.9051 | 0.6466 | 0.8369 | 0.6219 | - | - | - | - | - |
| cada_5class_balanced_sharp | 5 | 0.6838 | 0.5525 | 0.0000 | 0.4881 | 0.5824 | 0.5074 | - | - | - | - | - |
| cada_5class_balanced_sharp_corrected | 5 | 0.6838 | 0.5681 | 0.8511 | 0.5908 | 0.6400 | 0.5804 | - | - | - | - | - |
| cada_5class_unbalanced | 5 | 0.8065 | 0.7324 | 0.0000 | 0.6787 | 0.7220 | 0.6627 | - | - | - | - | - |
| cada_5class_unbalanced_corrected | 5 | 0.8111 | 0.7420 | 0.9407 | 0.7484 | 0.7848 | 0.7265 | - | - | - | - | - |
| cada_4class_balanced | 4 | 0.8371 | 0.7776 | 0.0000 | 0.8128 | 0.8411 | 0.8121 | - | - | - | - | - |
| cada_4class_balanced_blur | 4 | 0.6991 | 0.5956 | 0.0000 | 0.6765 | 0.7234 | 0.6871 | - | - | - | - | - |
| cada_4class_balanced_blur_corrected | 4 | 0.6719 | 0.5644 | 0.8947 | 0.6610 | 0.6894 | 0.6787 | - | - | - | - | - |
| cada_4class_balanced_corrected | 4 | 0.8371 | 0.7795 | 0.9660 | 0.8265 | 0.8358 | 0.8261 | - | - | - | - | - |
| cada_4class_balanced_resize | 4 | 0.7692 | 0.6885 | 0.0000 | 0.7624 | 0.7678 | 0.7596 | - | - | - | - | - |
| cada_4class_balanced_resize_corrected | 4 | 0.7851 | 0.7114 | 0.9325 | 0.7712 | 0.7705 | 0.7753 | - | - | - | - | - |
| cada_4class_balanced_sharp | 4 | 0.6561 | 0.5450 | 0.0000 | 0.6477 | 0.6896 | 0.6672 | - | - | - | - | - |
| cada_4class_balanced_sharp_corrected | 4 | 0.6968 | 0.5979 | 0.9056 | 0.6864 | 0.7061 | 0.7057 | - | - | - | - | - |
| cada_4class_unbalanced | 4 | 0.8261 | 0.7575 | 0.0000 | 0.7686 | 0.8075 | 0.7645 | - | - | - | - | - |
| cada_4class_unbalanced_corrected | 4 | 0.8116 | 0.7352 | 0.9324 | 0.7103 | 0.7906 | 0.7289 | - | - | - | - | - |


## Analysis by Class Configuration

### 10-Class Experiments

| Experiment | Num_Classes | Accuracy | Kappa | AUC_Macro | F1_Macro | Precision_Macro | Recall_Macro | Derived_Acc | Derived_Kappa | Derived_F1 | Derived_Prec | Derived_Rec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cada_10class_balanced | 10 | 0.8201 | 0.7989 | 0.9756 | 0.7621 | 0.7971 | 0.7714 | 0.8201 | 0.7748 | 0.8194 | 0.8308 | 0.8180 |
| cada_10class_balanced_blur | 10 | 0.4786 | 0.3170 | 0.8585 | 0.3867 | 0.3987 | 0.4625 | 0.4872 | 0.2764 | 0.3942 | 0.4995 | 0.3821 |
| cada_10class_balanced_resize | 10 | 0.5342 | 0.4005 | 0.8812 | 0.2810 | 0.2464 | 0.3385 | 0.5427 | 0.3664 | 0.3580 | 0.3218 | 0.4053 |
| cada_10class_balanced_sharp | 10 | 0.4701 | 0.2857 | 0.8562 | 0.2945 | 0.3122 | 0.3574 | 0.4701 | 0.2210 | 0.2845 | 0.3031 | 0.3122 |
| cada_10class_unbalanced | 10 | 0.5300 | 0.3985 | 0.8856 | 0.3391 | 0.3177 | 0.4090 | 0.5438 | 0.3734 | 0.4100 | 0.4022 | 0.4451 |

**🏆 Best 10-Class Model:** `cada_10class_balanced` (Acc: 0.8201)

### 8-Class Experiments

| Experiment | Num_Classes | Accuracy | Kappa | AUC_Macro | F1_Macro | Precision_Macro | Recall_Macro | Derived_Acc | Derived_Kappa | Derived_F1 | Derived_Prec | Derived_Rec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cada_8class_balanced | 8 | 0.7803 | 0.7469 | 0.0000 | 0.7259 | 0.7478 | 0.7431 | 0.7803 | 0.7079 | 0.7794 | 0.7859 | 0.7900 |
| cada_8class_balanced_blur | 8 | 0.7247 | 0.6820 | 0.0000 | 0.6603 | 0.6774 | 0.6680 | 0.7411 | 0.6532 | 0.7359 | 0.7417 | 0.7402 |
| cada_8class_balanced_resize | 8 | 0.7521 | 0.7141 | 0.0000 | 0.6950 | 0.7233 | 0.7037 | 0.7785 | 0.7039 | 0.7731 | 0.7804 | 0.7788 |
| cada_8class_balanced_sharp | 8 | 0.6764 | 0.6284 | 0.0000 | 0.6234 | 0.6679 | 0.6343 | 0.7156 | 0.6191 | 0.6998 | 0.7064 | 0.7074 |
| cada_8class_unbalanced | 8 | 0.5266 | 0.3834 | 0.0000 | 0.3975 | 0.3783 | 0.4530 | 0.5314 | 0.3433 | 0.4279 | 0.4099 | 0.4725 |

**🏆 Best 8-Class Model:** `cada_8class_balanced` (Acc: 0.7803)

### 5-Class Experiments

| Experiment | Num_Classes | Accuracy | Kappa | AUC_Macro | F1_Macro | Precision_Macro | Recall_Macro | Derived_Acc | Derived_Kappa | Derived_F1 | Derived_Prec | Derived_Rec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cada_5class_balanced | 5 | 0.7650 | 0.6817 | 0.0000 | 0.7078 | 0.7309 | 0.6924 | - | - | - | - | - |
| cada_5class_balanced_blur | 5 | 0.6453 | 0.5009 | 0.0000 | 0.4568 | 0.5964 | 0.4820 | - | - | - | - | - |
| cada_5class_balanced_blur_corrected | 5 | 0.6667 | 0.5402 | 0.8453 | 0.5147 | 0.5589 | 0.5278 | - | - | - | - | - |
| cada_5class_balanced_corrected | 5 | 0.7735 | 0.6973 | 0.9425 | 0.7277 | 0.7674 | 0.7135 | - | - | - | - | - |
| cada_5class_balanced_resize | 5 | 0.7308 | 0.6371 | 0.0000 | 0.6581 | 0.6988 | 0.6478 | - | - | - | - | - |
| cada_5class_balanced_resize_corrected | 5 | 0.7521 | 0.6555 | 0.9051 | 0.6466 | 0.8369 | 0.6219 | - | - | - | - | - |
| cada_5class_balanced_sharp | 5 | 0.6838 | 0.5525 | 0.0000 | 0.4881 | 0.5824 | 0.5074 | - | - | - | - | - |
| cada_5class_balanced_sharp_corrected | 5 | 0.6838 | 0.5681 | 0.8511 | 0.5908 | 0.6400 | 0.5804 | - | - | - | - | - |
| cada_5class_unbalanced | 5 | 0.8065 | 0.7324 | 0.0000 | 0.6787 | 0.7220 | 0.6627 | - | - | - | - | - |
| cada_5class_unbalanced_corrected | 5 | 0.8111 | 0.7420 | 0.9407 | 0.7484 | 0.7848 | 0.7265 | - | - | - | - | - |

**🏆 Best 5-Class Model:** `cada_5class_unbalanced_corrected` (Acc: 0.8111)

### 4-Class Experiments

| Experiment | Num_Classes | Accuracy | Kappa | AUC_Macro | F1_Macro | Precision_Macro | Recall_Macro | Derived_Acc | Derived_Kappa | Derived_F1 | Derived_Prec | Derived_Rec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cada_4class_balanced | 4 | 0.8371 | 0.7776 | 0.0000 | 0.8128 | 0.8411 | 0.8121 | - | - | - | - | - |
| cada_4class_balanced_blur | 4 | 0.6991 | 0.5956 | 0.0000 | 0.6765 | 0.7234 | 0.6871 | - | - | - | - | - |
| cada_4class_balanced_blur_corrected | 4 | 0.6719 | 0.5644 | 0.8947 | 0.6610 | 0.6894 | 0.6787 | - | - | - | - | - |
| cada_4class_balanced_corrected | 4 | 0.8371 | 0.7795 | 0.9660 | 0.8265 | 0.8358 | 0.8261 | - | - | - | - | - |
| cada_4class_balanced_resize | 4 | 0.7692 | 0.6885 | 0.0000 | 0.7624 | 0.7678 | 0.7596 | - | - | - | - | - |
| cada_4class_balanced_resize_corrected | 4 | 0.7851 | 0.7114 | 0.9325 | 0.7712 | 0.7705 | 0.7753 | - | - | - | - | - |
| cada_4class_balanced_sharp | 4 | 0.6561 | 0.5450 | 0.0000 | 0.6477 | 0.6896 | 0.6672 | - | - | - | - | - |
| cada_4class_balanced_sharp_corrected | 4 | 0.6968 | 0.5979 | 0.9056 | 0.6864 | 0.7061 | 0.7057 | - | - | - | - | - |
| cada_4class_unbalanced | 4 | 0.8261 | 0.7575 | 0.0000 | 0.7686 | 0.8075 | 0.7645 | - | - | - | - | - |
| cada_4class_unbalanced_corrected | 4 | 0.8116 | 0.7352 | 0.9324 | 0.7103 | 0.7906 | 0.7289 | - | - | - | - | - |

**🏆 Best 4-Class Model:** `cada_4class_balanced` (Acc: 0.8371)


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
