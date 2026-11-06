
@'
# Hospital Readmission Risk Prediction using ANN

## Project Overview

An end-to-end machine learning project that investigates
30-day hospital readmission risk using historical diabetic
patient encounter data.

The objective is to explore whether an Artificial Neural
Network can rank encounters by readmission risk to support
research into discharge follow-up prioritization.

This is an educational research prototype, not a clinically
validated decision-support system.

## Dataset

Source: UCI Diabetes 130-US Hospitals (1999-2008)

- Original encounters: 101,766
- Eligible encounters after exclusions: 100,114
- Target: Readmission within 30 days
- Positive class prevalence: Approximately 11.34%

### Patient-Grouped Data Splitting

| Dataset | Encounters |
|---|---:|
| Training | 69,944 |
| Validation | 15,172 |
| Test | 14,998 |

Patients were separated across splits to reduce
patient-level leakage.

## Methodology

1. Business problem definition
2. Dataset inspection and data-quality analysis
3. Exploratory data analysis
4. Missing-value treatment
5. ICD-9 diagnosis grouping
6. Healthcare utilization feature engineering
7. Patient-grouped train/validation/test splitting
8. Training-only preprocessing
9. Baseline ANN and alternative model development
10. Class imbalance handling
11. Regularization and feature ablation
12. Validation-based model and threshold selection
13. Final evaluation on untouched test data

## Experiments

Models investigated:

- Baseline MLP
- Class-weighted regularized MLP
- Undersampled MLP
- Reduced-feature weighted MLP
- Compact MLP
- Logistic Regression benchmark

The comparison considered ROC-AUC, Average Precision,
precision, recall, and top-K alert budgets.

## Final Selected Model

Architecture:

Input (230 encoded features)
    -> Dense(64, ReLU)
    -> Dropout(0.2)
    -> Dense(32, ReLU)
    -> Dropout(0.2)
    -> Dense(1, Sigmoid)

- 21 selected raw input features
- L2 regularization
- Balanced class weights
- Adam optimizer
- Frozen validation-selected threshold: 0.612904

## Final Test Results

Evaluated once on 14,998 previously unseen test encounters.

| Metric | Result |
|---|---:|
| Accuracy | 77.64% |
| Precision | 21.94% |
| Recall | 37.96% |
| F1-score | 0.2781 |
| ROC-AUC | 0.6651 |
| Average Precision | 0.2214 |
| Alert Rate | 19.63% |

### Confusion Matrix

| | Predicted No | Predicted Yes |
|---|---:|---:|
| Actual No | 10,998 | 2,298 |
| Actual Yes | 1,056 | 646 |

The model identified 646 of 1,702 actual readmissions,
while generating 2,298 false-positive alerts.

## Key Findings

- Class weighting increased sensitivity at the default
  classification threshold but also increased false alerts.
- Undersampling changed classification behavior without
  substantially improving overall ranking performance.
- Feature reduction modestly improved validation performance.
- More complex ANN architectures did not provide large gains.
- Previous healthcare utilization was an important predictor.
- Moderate discrimination limited readmission detection.

## Limitations

- Historical observational dataset
- No temporal or external hospital validation
- Moderate ROC-AUC and low precision
- A substantial number of readmissions remain undetected
- Potential probability calibration issues
- Clinical fairness and operational safety require assessment

The model must not be used independently for clinical decisions.

## Project Structure

```text
hospital-readmission-risk-prediction/
|-- notebooks/
|   |-- 01_hospital_readmission_mlp.ipynb
|-- src/
|   |-- preprocessing.py
|   |-- model.py
|   |-- evaluation.py
|-- models/
|-- reports/
|   |-- figures/
|   |-- business_findings.md
|   |-- final_test_evaluation.json
|-- requirements.txt
|-- .python-version
|-- .gitignore
|-- README.md
```

## Installation

Use Python 3.14.4 and create a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Open the notebook in VS Code or Jupyter using the
project environment.

The notebook contains the modeling experiments.
Raw data and serialized trained model artifacts are
not committed to the repository.

## Reproducibility

Random seed: 42

The notebook records model experimentation, selection,
evaluation, and conclusions. Reproduction requires
obtaining the source dataset and supplying it at the
path expected by the notebook.

## Conclusion

This project demonstrates a leakage-conscious end-to-end
ANN workflow, including class imbalance experiments,
feature engineering, operational threshold selection,
and honest evaluation of model limitations.

Further external validation, calibration, fairness analysis,
and clinical review are required before deployment.
'@ | Set-Content README.md -Encoding utf8
