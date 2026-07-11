# Fiqa-demographic-analysis
## Predicting FIQA Scores from Facial Attributes

### Overview

This project investigates whether Face Image Quality Assessment (FIQA) scores can be predicted from facial attributes and demographic information.

The analysis is based on the DiveFace dataset and the CR-FIQA framework. Facial attribute annotations are combined with CR-FIQA quality scores to explore which factors influence perceived face image quality.

The project includes:

- CR-FIQA score extraction
- Data integration and preprocessing
- Exploratory data analysis (EDA)
- Correlation analysis between facial attributes and FIQA scores
- Demographic group analysis
- Predictive modeling of FIQA scores using machine learning methods

---

## Research Objective

The main objective is to answer the following question:

> To what extent can FIQA scores be predicted from facial attributes and demographic characteristics?

The project further investigates:

- Which facial attributes are most strongly associated with FIQA scores
- Whether FIQA score distributions differ across demographic groups
- How accurately machine learning models can predict FIQA scores

---

## Repository Structure

```text
fiqa-demographic-analysis/
│
├── notebooks/
│   ├── 00_colab_setup.ipynb
│   ├── 01_extract_and_merge_cr_fiqa_scores.ipynb
│   └── 02_exploratory_data_analysis.ipynb
│
├── README.md
├── LICENSE
└── .gitignore
```

---

## Workflow

### 1. Environment Setup

`00_colab_setup.ipynb`

- Mounts Google Drive
- Validates project files
- Downloads the CR-FIQA repository
- Installs required dependencies

### 2. CR-FIQA Score Extraction

`01_extract_and_merge_cr_fiqa_scores.ipynb`

- Loads the CR-FIQA model
- Computes FIQA scores for all images
- Merges scores with DiveFace annotations
- Creates the final merged dataset

### 3. Exploratory Data Analysis

`02_exploratory_data_analysis.ipynb`

- Descriptive statistics
- Distribution analysis
- Correlation analysis
- Demographic group analysis
- Scatterplots and visualizations

---

## Required Data

The following files are required but are not included in this repository:

```text
DiveFace_subset/
DiveFace_subset_annotations.pkl
181952backbone.pth
```

Place these files inside your project directory before running the notebooks.

---

## Environment

The project was developed using:

- Python 3
- Google Colab
- PyTorch
- NumPy
- pandas
- scikit-learn
- OpenCV
- Matplotlib

---

## Current Status

Completed:

- CR-FIQA score extraction
- Annotation merge
- Exploratory data analysis

Planned:

- Linear Regression
- Random Forest Regression
- Feature Importance Analysis
- Model Comparison

---

## Dataset

This repository does not contain the DiveFace dataset or model checkpoints.

Please obtain the dataset and pretrained weights from their respective sources.

---

## License

MIT License
