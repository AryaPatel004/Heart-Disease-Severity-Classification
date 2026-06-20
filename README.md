# Heart Disease Severity Classification
---

## Project Overview

This project addresses the multi-class ordinal classification task of
predicting heart disease severity levels (0–4) from clinical patient data.
Rather than binary disease detection (present/absent), the goal is finer-grained
severity stratification to better support clinical decision-making.

The project uses the **UCI Heart Disease dataset** (303 patient records,
13 clinical features) collected from the Cleveland Clinic Foundation.
Five machine learning models were trained, tuned, and evaluated.

---

## Problem Statement

- **Task:** Predict heart disease severity level (0 = No Disease → 4 = Most Severe)
- **Challenges:**
  - Severe class imbalance: 54.1% of samples are Class 0 (no disease)
  - Ordinal structure: misclassifying a Class 4 patient as Class 0 is
    clinically far worse than a one-level error
  - Missing data: features `ca` and `thal` contain small proportions
    of missing values requiring imputation

---

## Dataset

| Severity Level | Label       | Count | Proportion |
|----------------|-------------|-------|------------|
| 0              | No Disease  | 164   | 54.1%      |
| 1              | Mild        | 55    | 18.2%      |
| 2              | Moderate    | 36    | 11.9%      |
| 3              | Serious     | 35    | 11.6%      |
| 4              | Severe      | 13    | 4.3%       |

**13 Clinical Features:** age, sex, cp (chest pain type), trestbps
(resting blood pressure), chol (cholesterol), fbs (fasting blood sugar),
restecg (resting ECG), thalach (max heart rate), exang (exercise angina),
oldpeak (ST depression), slope, ca (major vessels), thal (thalassemia)

---

## Models Implemented

### 1. Baseline – Logistic Regression (`Baseline_Model_LR.ipynb` / `LR.ipynb`)
- Multinomial softmax logistic regression
- `class_weight='balanced'` to handle class imbalance
- Two imputation strategies compared: Simple (median/mode) vs. KNN
- 5-fold stratified cross-validation

### 2. LSTM Deep Learning (`LSTM.ipynb`)
- Stacked LSTM architecture (64 units → 32 units)
- Labels remapped from 5 classes to 3: No Disease, Mild-Moderate, Severe-Critical
- SMOTE oversampling applied to training set only
- Hyperparameter tuning via Keras Tuner (Hyperband)
- EarlyStopping and ReduceLROnPlateau callbacks

### 3. Ordinal Logistic Regression (`ordinal_regression.ipynb`)
- Uses `mord.LogisticAT` (All-Threshold) — ordinal-aware classifier
- Preserves all 5 severity levels (no class collapsing)
- Hyperparameter tuning via RandomizedSearchCV scored by Quadratic Weighted Kappa
- Isotonic probability calibration via CalibratedClassifierCV
- Shannon entropy used for uncertainty analysis

### 4. Random Forest (`RF.ipynb`)
- RandomForestClassifier with `class_weight='balanced'`, 100 estimators
- Hyperparameter tuning via RandomizedSearchCV scored by Cohen's Kappa
- Probability calibration via CalibratedClassifierCV (isotonic regression)
- Permutation feature importance and partial dependence analysis

### 5. Gradient Boosting (`GB.ipynb`)
- GradientBoostingClassifier with RobustScaler for numerical features
- Hyperparameter tuning via RandomizedSearchCV (40 iterations, 5-fold CV)
- Probability calibration via CalibratedClassifierCV (sigmoid method)
- Uncertainty analysis via prediction entropy

---

## Results Summary

| Model                  | Accuracy | Macro F1 | Quadratic Weighted Kappa |
|------------------------|----------|----------|--------------------------|
| Baseline (LR)          | 0.50     | 0.34     | 0.58                     |
| LSTM                   | **0.76** | **0.69** | N/A*                     |
| Ordinal Regression     | 0.70     | 0.46     | **0.776**                |
| Random Forest          | 0.70     | 0.53     | **0.783**                |
| Gradient Boosting      | 0.59     | 0.29     | 0.39                     |

*LSTM used 3-class remapping; QWK not directly comparable to 5-class models.

**Key Takeaway:** The LSTM model achieved the highest accuracy (76%) and
macro F1, while Ordinal Regression and Random Forest achieved the highest
Quadratic Weighted Kappa scores, indicating the best ordinal agreement
with true severity levels.

---

## Evaluation Metrics

- **Macro-Averaged F1:** Equal weight to all classes regardless of support;
  penalizes poor minority-class recall
- **Quadratic Weighted Kappa (QWK):** Severity-aware metric; penalizes
  predictions proportional to squared distance from true class
- **Severe-Critical Recall:** Most clinically important metric —
  a false negative here means a missed severe diagnosis

---

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/heart-disease-severity-classification.git
   cd heart-disease-severity-classification

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Download the UCI Heart Disease dataset (repository ID: 45) and place it in the data/ folder.
4. Run any notebook inside the notebooks/ folder using Jupyter:
   ```bash
   jupyter notebook

---

## Requirements

Key libraries used:

- **Macro-Averaged F1:** Equal weight to all classes regardless of support;
  penalizes poor minority-class recall
- **scikit-learn**
- **tensorflow / keras**
- **keras-tuner**
- **imbalanced-learn (SMOTE)**
- **mord (ordinal regression)**
- **pandas, numpy, matplotlib, seaborn**

See requirements.txt for the full list with pinned versions.

---

## References

- UCI Heart Disease Dataset: https://archive.ics.uci.edu/dataset/45/heart+disease
- Detrano, R. et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease.

---
