# FIQA Demographic Analysis

## Predicting Face Image Quality Assessment Scores from Facial Attributes and Demographic Characteristics

## Overview

This project investigates the extent to which Face Image Quality Assessment (FIQA) scores can be predicted from facial attributes, image characteristics, and demographic information.

Facial attribute annotations from the DiveFace dataset are combined with image-quality scores generated using the CR-FIQA framework. The resulting dataset is analyzed through exploratory data analysis, statistical inference, machine learning, model optimization, explainability methods, and demographic consistency tests.

The main project workflow is implemented as standalone Python scripts. The scripts are designed to be portable and do not assume a fixed operating system, Google Drive location, project directory, checkpoint filename, or CR-FIQA installation path.

---

## Research Objectives

The primary research question is:

> **To what extent can FIQA scores be predicted from facial attributes, image characteristics, and demographic information?**

The project further investigates:

- Which facial and image characteristics are associated with FIQA scores
- Which characteristics remain significant after adjustment for other predictors
- Which machine learning models best predict FIQA scores
- Which features are most important for nonlinear prediction
- Whether feature–FIQA relationships remain consistent across demographic groups

---

## Dataset

### DiveFace

The project uses a subset of the DiveFace dataset containing:

- facial images
- demographic group labels
- facial attribute annotations
- multiple images per identity

The demographic groups are:

- Asian Woman
- Asian Man
- Black Woman
- Black Man
- Caucasian Woman
- Caucasian Man

The DiveFace images and annotations are not included in this repository.

### CR-FIQA

Image-quality scores are generated using the CR-FIQA framework.

A compatible CR-FIQA checkpoint is selected by the user during project setup. The repository does not require a fixed checkpoint filename or a specific local installation path.

The generated FIQA scores are merged with the DiveFace annotation table to create the dataset used by all subsequent analysis scripts.

---

## Repository Structure

```text
fiqa-demographic-analysis/
│
├── scripts/
│   ├── setup_project.py
│   ├── extract_scores.py
│   ├── pipeline_common.py
│   ├── run_eda.py
│   ├── train_models.py
│   ├── optimize_models.py
│   ├── correlation_analysis.py
│   ├── regression_analysis.py
│   ├── explainability.py
│   ├── demographic_consistency.py
│   ├── run_pipeline.py
│   └── STANDALONE_PIPELINE.md
│
├── config/
│   └── config.example.json
│
├── notebooks/
│   └── archive/
│
├── models/
│   └── .gitkeep
│
├── results/
│   └── .gitkeep
│
├── README.md
├── LICENSE
├── requirements.txt
└── .gitignore
```

The scripts form the reproducible project pipeline. The notebooks are retained only as archived exploratory material and are not required to execute the analysis.

---

## Workflow

### 1. Project Setup

`scripts/setup_project.py` validates all user-supplied paths and creates a runtime configuration file.

The setup is dynamic and does not hardcode:

- a Google Drive path
- a project directory
- an image directory
- an annotation filename
- a checkpoint filename
- a CR-FIQA repository path

The script creates:

```text
config/runtime_config.json
```

This local runtime file is ignored by Git because it may contain machine-specific paths.

### 2. CR-FIQA Score Extraction

`scripts/extract_scores.py`:

- loads a user-selected CR-FIQA-compatible backbone
- loads the selected checkpoint
- preprocesses the facial images
- generates CR-FIQA scores
- records unreadable images
- merges the scores with the annotation table
- exports the merged analysis dataset

### 3. Exploratory Data Analysis

`scripts/run_eda.py` produces:

- dataset summaries
- missing-value diagnostics
- feature distributions
- demographic group summaries
- target-score distributions
- exploratory figures and tables

### 4. Machine Learning Models

`scripts/train_models.py` performs:

- identity-aware train/test splitting
- feature preprocessing
- baseline evaluation
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest
- Gradient Boosting
- optional XGBoost
- RMSE, MAE, and R² evaluation
- model and prediction export

The identity-aware split ensures that images belonging to the same identity do not appear in both the training and test sets.

### 5. Model Optimization

`scripts/optimize_models.py`:

- reuses the saved train/test split
- performs identity-aware cross-validation
- tunes supported nonlinear models
- compares optimized models
- selects and saves the best model
- exports predictions and model diagnostics

### 6. Correlation Analysis

`scripts/correlation_analysis.py` investigates feature–FIQA relationships using:

- Pearson correlation
- Spearman correlation
- point-biserial correlation
- Welch tests
- Mann–Whitney U tests
- Cohen's d
- false discovery rate correction

### 7. Regression Analysis

`scripts/regression_analysis.py` performs adjusted statistical analysis using:

- multiple linear regression
- HC3 robust standard errors
- false discovery rate correction
- variance inflation factors
- residual diagnostics
- Cook's distance
- sensitivity analyses
- comparison with the correlation analysis

### 8. Model Explainability

`scripts/explainability.py` explains the best optimized model using:

- permutation importance
- optional SHAP analysis
- global feature rankings
- model-performance summaries
- comparison with the correlation and regression results

### 9. Demographic Consistency Analysis

`scripts/demographic_consistency.py` evaluates whether feature effects remain stable across demographic groups.

It includes:

- group-wise correlation analysis
- group-wise regression models
- common-predictor comparisons
- feature-by-group interaction tests
- male-only facial-hair analyses
- demographic consistency summaries

Age is excluded from the demographic consistency feature analysis, while facial-hair variables are evaluated only for male demographic groups.

### 10. Full Pipeline Execution

`scripts/run_pipeline.py` executes the analysis stages in sequence after score extraction has been completed.

---

## Required External Data

The following resources are not included in this repository:

- DiveFace image directory
- DiveFace annotation file
- compatible CR-FIQA checkpoint
- CR-FIQA source repository

Large datasets, checkpoints, trained models, runtime configurations, and generated outputs are excluded through `.gitignore`.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/safeerkhann/fiqa-demographic-analysis.git
cd fiqa-demographic-analysis
```

Create and activate a virtual environment.

Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Configuration

Run the setup script with paths matching your local environment:

```bash
python scripts/setup_project.py \
  --project-dir /path/to/project-data \
  --images /path/to/diveface-images \
  --annotations /path/to/annotations.pkl \
  --checkpoint /path/to/compatible-checkpoint.pth \
  --cr-fiqa-dir /path/to/CR-FIQA \
  --create-output-dirs
```

The generated configuration is saved by default as:

```text
config/runtime_config.json
```

A user may select any compatible CR-FIQA checkpoint. The corresponding architecture must be passed to the score-extraction script.

---

## Usage

### Smoke Test

Before processing the full dataset, run score extraction on a small number of images:

```bash
python scripts/extract_scores.py \
  --config config/runtime_config.json \
  --architecture iresnet100 \
  --batch-size 16 \
  --limit 10
```

The architecture must match the selected checkpoint.

### Full Score Extraction

```bash
python scripts/extract_scores.py \
  --config config/runtime_config.json \
  --architecture iresnet100 \
  --batch-size 32
```

### Run the Remaining Pipeline

Quick execution:

```bash
python scripts/run_pipeline.py \
  --config config/runtime_config.json \
  --optimization-mode quick
```

Quick execution without optional XGBoost and SHAP stages:

```bash
python scripts/run_pipeline.py \
  --config config/runtime_config.json \
  --optimization-mode quick \
  --skip-xgboost \
  --skip-shap
```

Full model optimization:

```bash
python scripts/run_pipeline.py \
  --config config/runtime_config.json \
  --optimization-mode full
```

Each script may also be executed individually. Available arguments can be inspected with:

```bash
python scripts/<script_name>.py --help
```

---

## Output Structure

By default, generated files are written below the configured results and models directories.

Example:

```text
results/
├── 01_score_extraction/
├── 02_eda/
├── 03_ml_models/
├── 04_model_optimization/
├── 05_correlation_analysis/
├── 06_regression_analysis/
├── 07_explainability/
└── 08_demographic_consistency/

models/
├── 03_ml_models/
└── 04_model_optimization/
```

Generated outputs may include:

- CR-FIQA score files
- merged analysis datasets
- train/test split assignments
- trained models
- optimized models
- prediction files
- statistical result tables
- feature-importance rankings
- SHAP outputs
- demographic consistency tables
- publication-ready figures

---

## Reproducibility

The project uses an identity-aware train/test split so that images from the same identity are not distributed across both the training and test sets.

The split assignment is saved and reused by later modeling and explainability stages. This prevents the optimization and explanation scripts from creating inconsistent evaluation partitions.

Additional reproducibility measures include:

- centralized path handling through the runtime configuration
- shared feature definitions in `pipeline_common.py`
- fixed random seeds where applicable
- saved model artifacts
- saved prediction tables
- explicit command-line arguments
- reusable standalone scripts

Because some machine-learning libraries and GPU operations may contain nondeterministic components, exact numerical equality across all hardware and software environments cannot always be guaranteed.

---

## Main Evaluation Metrics

The predictive models are evaluated using:

- Root Mean Squared Error
- Mean Absolute Error
- coefficient of determination, R²

Statistical analyses additionally report:

- correlation coefficients
- adjusted regression coefficients
- robust confidence intervals
- false-discovery-rate-adjusted p-values
- effect sizes
- variance inflation factors
- interaction-test results

---

## Dependencies

The project is based on:

- Python 3
- PyTorch
- NumPy
- pandas
- SciPy
- scikit-learn
- statsmodels
- Matplotlib
- OpenCV
- XGBoost
- SHAP
- joblib
- tqdm

The exact installation requirements are listed in `requirements.txt`.

---

## Notes

- Do not commit datasets, annotation files, model checkpoints, local runtime configurations, or generated model files.
- `config/config.example.json` may be committed as a path template.
- `config/runtime_config.json` should remain local.
- Checkpoint architecture and the `--architecture` argument must match.
- Run the smoke test before starting full CR-FIQA extraction.
- The archived notebooks are not part of the required execution path.

---

## License

This project is licensed under the MIT License.
