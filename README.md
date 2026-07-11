# FIQA-Demographic-Analysis

## Predicting Face Image Quality Assessment (FIQA) Scores from Facial Attributes and Demographic Information

### Overview

This project investigates whether Face Image Quality Assessment (FIQA) scores can be predicted from facial attributes and demographic characteristics.

The analysis combines facial attribute annotations from the DiveFace dataset with quality scores generated using the CR-FIQA framework.

The project explores the relationship between image quality, facial characteristics, and demographic information through exploratory data analysis and machine learning.

---

## Research Objective

The primary research question is:

> To what extent can FIQA scores be predicted from facial attributes and demographic characteristics?

Additional objectives include:

- Identifying facial attributes most strongly associated with FIQA scores
- Investigating differences in FIQA score distributions across demographic groups
- Comparing machine learning models for FIQA score prediction
- Analyzing the importance of demographic and facial attributes in FIQA prediction

---

## Dataset

The project uses:

### DiveFace

A large-scale facial image dataset containing:

- Demographic group labels
- Facial attribute annotations
- Multiple identities per demographic group

Demographic groups:

- Asian Woman
- Asian Man
- Black Woman
- Black Man
- Caucasian Woman
- Caucasian Man

### CR-FIQA

CR-FIQA is used to generate image quality scores for all images.

The resulting FIQA scores are merged with the DiveFace annotations to create the final analysis dataset.

---

## Repository Structure

```text
fiqa-demographic-analysis/
│
├── notebooks/
│   ├── 00_colab_setup.ipynb
│   ├── 01_extract_and_merge_cr_fiqa_scores.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_ml_models.ipynb
│   ├── 04_model_interpretation.ipynb     (planned)
│   └── 05_bias_analysis.ipynb            (planned)
│
├── README.md
├── LICENSE
└── .gitignore
```

---

## Workflow

### 1. Environment Setup

**00_colab_setup.ipynb**

- Mount Google Drive
- Verify project files
- Download CR-FIQA repository
- Install required dependencies

---

### 2. CR-FIQA Score Extraction

**01_extract_and_merge_cr_fiqa_scores.ipynb**

- Load pretrained CR-FIQA model
- Generate FIQA scores for all images
- Merge FIQA scores with DiveFace annotations
- Create the final merged dataset

---

### 3. Exploratory Data Analysis

**02_exploratory_data_analysis.ipynb**

- Descriptive statistics
- Distribution analysis
- Correlation analysis
- Demographic group analysis
- Scatterplots and visualizations

---

### 4. Machine Learning Models

**03_ml_models.ipynb**

- Identity-aware train-test split
- Feature preprocessing
- One-hot encoding of demographic groups
- Model training and evaluation

Models evaluated:

- Mean Baseline
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regression
- Gradient Boosting Regression
- XGBoost Regression

Evaluation metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)

---

### 5. Model Interpretation (Planned)

**04_model_interpretation.ipynb**

- Linear model coefficient analysis
- Feature importance analysis
- Comparison of model explanations
- Interpretation of demographic effects

---

### 6. Demographic Bias Analysis (Planned)

**05_bias_analysis.ipynb**

- Group-wise performance evaluation
- Error analysis across demographic groups
- Fairness assessment
- Discussion of demographic effects on FIQA prediction

---

## Current Results

The evaluated machine learning models produced the following test performance:

| Model | Test RMSE | Test R² |
|---------|---------:|---------:|
| XGBoost | 0.2917 | 0.2346 |
| Gradient Boosting | 0.2932 | 0.2268 |
| Random Forest | 0.2946 | 0.2194 |
| Lasso Regression | 0.3272 | 0.0369 |
| Ridge Regression | 0.3276 | 0.0348 |
| Linear Regression | 0.3277 | 0.0342 |
| Mean Baseline | 0.3335 | -0.0004 |

### Key Findings

- Tree-based models substantially outperform linear models.
- The relationship between facial attributes and FIQA scores appears to be nonlinear.
- XGBoost achieved the best predictive performance.
- Facial attributes contain useful information for FIQA prediction, but explain only part of the total score variation.

---

## Required Data

The following files are required but are not included in this repository:

```text
DiveFace_subset/
DiveFace_subset_annotations.pkl
181952backbone.pth
```

Place these files inside the project directory before running the notebooks.

---

## Environment

The project was developed using:

- Python 3
- Google Colab
- PyTorch
- NumPy
- pandas
- scikit-learn
- XGBoost
- OpenCV
- Matplotlib

---

## Reproducibility

To ensure a realistic evaluation setup, images belonging to the same identity are never split across training and test sets.

An identity-aware train-test split based on the `cls` identity column is used throughout the machine learning experiments.

---

## License

MIT License
