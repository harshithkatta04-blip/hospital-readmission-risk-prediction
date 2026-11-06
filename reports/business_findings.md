# Hospital Readmission Risk Prediction
## Business Findings Report

### Business Objective

Investigate whether a Multi-Layer Perceptron (MLP) can identify
hospital encounters at increased risk of readmission within 30 days,
potentially supporting prioritization of discharge follow-up reviews.

### Dataset and Methodology

The project uses the UCI Diabetes 130-US Hospitals (1999-2008) dataset.
After excluding death-related discharges, 100,114 eligible encounters
remained. Approximately 11.34% were positive for 30-day readmission.

A patient-grouped train/validation/test split reduced patient overlap
between evaluation partitions. Preprocessing was fitted on training
data, with diagnosis grouping, healthcare-utilization features,
categorical encoding, and numerical scaling.

Several approaches were compared, including baseline, class-weighted,
undersampled, compact, and reduced-feature MLPs, plus Logistic Regression.

### Selected Model

The selected model uses 21 raw features transformed into 230 inputs,
with Dense layers of 64 and 32 units, Dropout, L2 regularization,
balanced class weights, and a sigmoid output.

The final classification threshold (0.612904) was selected using
validation data to target a 20% alert budget.

### Untouched Test Results

| Metric | Result |
|---|---:|
| Encounters | 14,998 |
| Accuracy | 77.64% |
| Precision | 21.94% |
| Recall | 37.96% |
| F1-score | 0.2781 |
| ROC-AUC | 0.6651 |
| Average Precision | 0.2214 |
| Alert rate | 19.63% |

### Error Analysis

- True positives: 646
- False positives: 2,298
- False negatives: 1,056
- True negatives: 10,998

The model identified 646 of 1,702 actual readmissions.
However, 1,056 readmissions were missed, and most alerts were false
positives. This limits both sensitivity and operational efficiency.

### Key Findings

1. **Healthcare utilization matters:** Previous hospital visits were
   among the strongest predictors in the validation analysis.
2. **Class imbalance changes model behavior:** Class weighting increased
   recall at the default threshold but increased false alerts.
3. **More complexity did not solve the problem:** Alternative ANN
   architectures offered limited improvements in ranking performance.

### Business Recommendations

Hospitals could research risk-ranked, staff-reviewed discharge follow-up
workflows, but should not use the current system to make autonomous
patient-care decisions. Operational workload and missed cases must be
considered together.

Before deployment, perform external and temporal validation, assess
calibration and subgroup fairness, audit feature availability at
discharge, and obtain clinical approval.

### Limitations and Conclusion

The historical dataset may not represent current clinical practice.
The model exhibits moderate discrimination, low precision, and many
missed readmissions. The validation-selected alert budget is illustrative,
not a clinically approved threshold.

This project demonstrates a reproducible educational ANN workflow
and an evidence-based assessment of its practical limitations.
