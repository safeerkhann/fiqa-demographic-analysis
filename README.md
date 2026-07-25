# FIQA-Demographic-Analysis

## Predicting Face Image Quality Assessment (FIQA) Scores from Facial Attributes and Demographic Characteristics

### Overview

This project investigates whether Face Image Quality Assessment (FIQA) scores can be predicted from facial attributes and demographic information.

The analysis combines facial attribute annotations from the DiveFace dataset with image quality scores generated using the CR-FIQA framework. Statistical analyses and machine learning models are used to investigate which facial and image characteristics are associated with FIQA scores and whether these relationships remain consistent across demographic groups.

The complete workflow includes data preparation, exploratory analysis, predictive modeling, statistical inference, model explainability, and demographic consistency analysis.

---

# Research Objectives

The primary research question is:

> **To what extent can FIQA scores be predicted from facial attributes, image characteristics, and demographic information?**

The project further investigates:

- Which facial and image characteristics are associated with FIQA scores
- Which characteristics remain significant after adjustment for other predictors
- Which machine learning models best predict FIQA scores
- Which features are most important for nonlinear prediction
- Whether feature–FIQA relationships remain consistent across demographic groups

---

# Dataset

## DiveFace

The project uses the DiveFace dataset containing

- facial images
- demographic group labels
- facial attribute annotations
- multiple identities per demographic group

Demographic groups:

- Asian Woman
- Asian Man
- Black Woman
- Black Man
- Caucasian Woman
- Caucasian Man

---

## CR-FIQA

Image quality scores are generated using the CR-FIQA framework.

The resulting FIQA scores are merged with the DiveFace annotations to create the final analysis dataset.

---

# Repository Structure

```text
fiqa-demographic-analysis/
│
├── notebooks/
│   ├── 00_colab_setup.ipynb
│   ├── 01_extract_and_merge_cr_fiqa_scores.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_ml_models.ipynb
│   ├── 04_model_optimization.ipynb
│   ├── 05_rq2_correlation_analysis.ipynb
│   ├── 06_rq3_regression_analysis.ipynb
│   ├── 07_model_explainability.ipynb
│   └── 08_demographic_consistency_analysis.ipynb
│
├── models/
├── results/
├── README.md
├── LICENSE
├── requirements.txt
└── .gitignore
```

---

# Workflow

## 00 — Environment Setup

- Project setup
- Dependency installation
- Google Drive configuration
- CR-FIQA setup

---

## 01 — CR-FIQA Score Extraction

- Load pretrained CR-FIQA model
- Generate FIQA scores
- Merge scores with DiveFace annotations
- Export merged dataset

---

## 02 — Exploratory Data Analysis

- Dataset overview
- Missing-value analysis
- Distribution analysis
- Demographic overview
- Correlation exploration
- Visualizations

---

## 03 — Machine Learning Models

- Identity-aware train/test split
- Feature preprocessing
- Baseline models
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest
- Gradient Boosting
- XGBoost

Evaluation metrics

- RMSE
- MAE
- R²

---

## 04 — Model Optimization

- Hyperparameter optimization
- Cross-validation
- Model comparison
- Best-model selection
- Performance diagnostics

---

## 05 — RQ2 Correlation Analysis

Statistical analysis of feature–FIQA relationships.

Includes

- Pearson correlation
- Spearman correlation
- Point-biserial analysis
- Mann–Whitney U tests
- Cohen's d
- False Discovery Rate correction

---

## 06 — RQ3 Regression Analysis

Adjusted statistical analysis.

Includes

- Multiple Linear Regression
- HC3 robust inference
- Variance Inflation Factor (VIF)
- Residual diagnostics
- Cook's distance
- Sensitivity analyses
- Comparison with RQ2

---

## 07 — Model Explainability

Explainability of the final optimized prediction model.

Includes

- Permutation Importance
- SHAP
- Global explanations
- Local explanations
- Feature ranking
- Comparison with RQ2 and RQ3

---

## 08 — Demographic Consistency Analysis

Evaluation of feature–FIQA relationships across demographic groups.

Includes

- Group-wise correlations
- Group-wise regression models
- Interaction tests
- Male-only facial hair analysis
- Demographic consistency evaluation

---

# Machine Learning Models

The following regression models are evaluated:

- Mean Baseline
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest
- Gradient Boosting
- XGBoost

The strongest model is further optimized and interpreted using permutation importance and SHAP.

---

# Required Data

The following files are **not included** in this repository:

```text
DiveFace_subset/
DiveFace_subset_annotations.pkl
181952backbone.pth
```

Place these files inside the project directory before executing the notebooks.

---

# Environment

Developed using

- Python 3
- Google Colab
- PyTorch
- NumPy
- pandas
- scikit-learn
- XGBoost
- SHAP
- statsmodels
- SciPy
- Matplotlib

---
## Installation

Clone the repository:

```bash
git clone https://github.com/<username>/fiqa-demographic-analysis.git
cd fiqa-demographic-analysis
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```
# Reproducibility

The machine learning experiments use an **identity-aware train/test split**, ensuring that images of the same identity never appear in both training and test sets.

All subsequent notebooks reuse the identical split to ensure reproducible statistical analyses and model explanations.

---

# Results

The project produces:

- optimized machine learning models
- statistical association analyses
- adjusted regression models
- model explainability using SHAP
- demographic consistency analyses
- publication-ready tables and figures

---

# License

MIT License
