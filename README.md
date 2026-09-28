# Brazil Tourism Forecasting Benchmark

**MBA Thesis — Data Science & Analytics | USP ESALQ**
**Author:** Gabriel Moraes Magalhães | **Advisor:** Édipo Menezes Da Silva

---

## Overview

A comparative benchmarking study of time series forecasting techniques applied to international tourist arrivals in Brazil. The goal is to identify the best-performing predictive approach — balancing accuracy, computational cost, and explainability — to support strategic planning in the tourism sector.

## Research Objective

Compare three classes of forecasting models on the task of predicting monthly international tourist arrivals to Brazil:

| Approach | Model |
|---|---|
| Statistical baseline | ARIMA |
| Machine Learning | XGBoost |
| Deep Learning | PatchTST (Patch Time Series Transformer) |

Models are evaluated on: **RMSE**, **MAE**, **MAPE**, and **R²**, plus qualitative criteria (explainability, scalability, training time).

## Dataset

- **Source:** [Estimativas de Chegadas de Turistas Internacionais ao Brasil](https://dados.gov.br/dados/conjuntos-dados/estimativas-de-chegadas-de-turistas-internacionais-ao-brasil) — Ministério do Turismo
- **Period:** 1989 – present, updated as new source files are released
- **Granularity:** Monthly
- **Features:** year, month, access route (air/land/river/sea), destination state, country of origin, arrival count

Notebook `2` splits the monthly series into three consecutive blocks:

| Split | Months | Used for |
|---|---|---|
| train | everything before the validation year | fitting the models |
| validation | the 12 months before the test year | comparing and selecting models (notebooks 5 and 7) |
| test | the **last 12 months** of the data | the final evaluation only (final notebook) |

The EDA and feature engineering notebooks see train + validation; the test set is only read
once, by the final notebook, after the model has been selected. Adding a new source CSV to
`data/raw/` and re-running the notebooks in order rolls all three blocks forward
automatically — no dates need to be edited by hand.

## Setup

The project uses [uv](https://docs.astral.sh/uv/) for dependency management (Python 3.13+):

```
uv sync                     # install dependencies into .venv
uv run task mlflow          # start the local MLflow tracking server (Docker required)
uv run jupyter lab          # open the notebooks
```

Notebooks 3, 5 and 6 import feature-engineering and target-transformation helpers from the
companion repository of *Modern Time Series Forecasting with Python* (Joseph & Tackes, 2E),
expected as a sibling folder `../Modern-Time-Series-Forecasting-with-Python-2E` (overridable
with the `MTSF_REPO_PATH` environment variable).

## Project Structure

```
├── data/
│   ├── raw/                   # Original source CSVs + the unified raw parquet
│   └── processed/             # cleaned data, train/validation/test splits, engineered features
├── documents/
│   ├── notebooks/             # notebooks exported to PDF (task export_pdf)
│   ├── notes/                 # study notes
│   └── paper_development/     # thesis-related documents
├── notebooks/
│   ├── 0_data_extraction.ipynb            # unify the raw CSVs into one parquet
│   ├── 1_data_cleaning.ipynb              # standardize categories, build the date index
│   ├── 2_train_test_split.ipynb           # chronological train / validation / test split
│   ├── 3_exploratory_data_analysis.ipynb  # trend, seasonality, heteroscedasticity, transforms
│   ├── 4_working_with_pandemic_data.ipynb # COVID-gap imputation methods
│   ├── 5_arima_model_testing.ipynb        # SARIMAX baseline (statistical model)
│   ├── 6_feature_engineering.ipynb        # lags, rolling/seasonal windows, calendar features
│   └── 7_ml_model_testing.ipynb           # Linear/Ridge/Lasso/LightGBM/XGBoost benchmark
├── scripts/
│   ├── export_notebooks_pdf.py            # task export_pdf
│   ├── mlflow_purge.py                    # task mlflow_purge
│   └── verify_turismo_gov.ps1
├── docker-compose.yml                     # local MLflow tracking server
└── README.md
```

## Methodology

1. **Preprocessing:** encoding/category standardization, missing value treatment, chronological train/validation/test split
2. **Feature Engineering:** lags, rolling and seasonal-rolling windows, EWMA, calendar (categorical and Fourier) and elapsed-time features, domain (COVID) flags
3. **Modeling:**
   - Statistical baseline: SARIMAX, with `none`/`log`/AutoML (`AutoStationaryTransformer`) target processing
   - Machine Learning: Linear Regression, Ridge, Lasso, LightGBM and XGBoost, with recursive and direct multi-step strategies
   - Deep Learning: PatchTST (planned)
4. **Evaluation:** RMSE, MAE, MAPE, R², MASE and Forecast Bias against naive and seasonal-naive baselines, tracked per experiment run in MLflow. Models are selected on the validation year; the final notebook (planned) refits the selected models and scores them once on the test year

## Experiment Tracking

Experiments are tracked with [MLflow](https://mlflow.org/), running locally via Docker Compose
(tracking server + SQLite backend + local artifact store):

```
uv run task mlflow          # start the tracking server (http://localhost:5000)
uv run task mlflow_logs     # follow the server logs
uv run task mlflow_stop     # stop the server
uv run task mlflow_purge    # permanently delete every experiment, run and registered model
```

`MLFLOW_TRACKING_URI` and related settings are read from a local `.env` file (see the
`MLFLOW_*` variables referenced across the notebooks).

## Exporting Notebooks to PDF

Notebooks can be exported to PDF (rendered by Chromium, no LaTeX installation required):

```
uv run task export_pdf                              # every notebook in notebooks/
uv run task export_pdf notebooks/5_*.ipynb           # selected notebooks
```

PDFs are written to `documents/notebooks/` and reflect the outputs already saved in each
notebook (notebooks are not re-executed).

## Timeline

| Activity | Mar | Apr | May | Jun | Jul | Aug | Sep |
|---|---|---|---|---|---|---|---|
| Literature review | X | | | | | | |
| Research proposal | X | | | | | | |
| Data processing | | X | | | | | |
| Data analysis | | X | X | X | X | | |
| Preliminary results | | | X | X | X | | |
| Final writing | | | | | X | X | |
| Submission | | | | | | X | X |
| Presentation | | | | | | | X |

## References

- Géron, A. (2021). *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*. 2nd ed. Alta Books.
- Gujarati, D. N. (2019). *Econometrics: Principles, Theory and Practical Applications*. Saraivauni.
- Joseph, M.; Tackes, J. (2024). *Modern Time Series Forecasting with Python*. Packt Publishing.
- Nielsen, A. (2021). *Practical Time Series Analysis*. Alta Books.
- Ministério do Turismo. (2026). Dataset: Estimativas de chegadas de turistas internacionais ao Brasil.
