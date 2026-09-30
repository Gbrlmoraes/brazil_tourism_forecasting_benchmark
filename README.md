# Brazil Tourism Forecasting Benchmark

**MBA Thesis — Data Science & Analytics | USP ESALQ**
**Author:** Gabriel Moraes Magalhães | **Advisor:** Édipo Menezes Da Silva

---

## Overview

A comparative benchmarking study of time series forecasting techniques applied to international tourist arrivals in Brazil. The goal is to identify the best-performing predictive approach — balancing accuracy, computational cost, and explainability — to support strategic planning in the tourism sector.

## Research Objective

Compare three classes of forecasting models on the task of predicting total monthly international tourist arrivals to Brazil, 12 months ahead:

| Approach | Models | Notebook |
|---|---|---|
| Statistical | SARIMAX (with COVID regressors) | 5 |
| Machine Learning, local | Ridge, Lasso, LightGBM and XGBoost trained on the total series | 7 |
| Machine Learning, global | the same four families trained on a panel of related series (access route × arrival region × origin subdivision), forecasting the total bottom-up | 8 |
| Foundation model (transformer) | TimesFM 3.0 (Google Research), zero-shot: no training on this series | 9 |

All approaches share one evaluation protocol, so their results are directly comparable:

- **Selection metric:** the mean **seasonal MASE** over four 12-month validation windows (see
  *Methodology*), against the seasonal-naive forecast ("same month last year").
- **Also reported:** RMSE, MAE, MAPE, R² and Forecast Bias on the validation year, the worst
  validation window, and (TimesFM) the coverage of the 80% forecast interval.
- **Qualitative criteria:** explainability (feature importance), scalability and computational
  cost (fit time).

## Dataset

- **Source:** [Estimativas de Chegadas de Turistas Internacionais ao Brasil](https://dados.gov.br/dados/conjuntos-dados/estimativas-de-chegadas-de-turistas-internacionais-ao-brasil) — Ministério do Turismo
- **Period:** 1989 – present, updated as new source files are released
- **Granularity:** Monthly
- **Features:** year, month, access route (air/land/river/sea), destination state, country of origin, arrival count

Notebook `2` splits the monthly series into three consecutive blocks:

| Split | Months | Used for |
|---|---|---|
| train | everything before the validation year | fitting the models |
| validation | the 12 months before the test year | comparing and selecting models (notebooks 5, 7, 8 and 9), together with three earlier yearly windows cut from the training data |
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

Notebooks 3–9 import feature-engineering, forecasting and target-transformation helpers
from the companion repository of *Modern Time Series Forecasting with Python* (Joseph &
Tackes, 2E), expected as a sibling folder `../Modern-Time-Series-Forecasting-with-Python-2E`
(overridable with the `MTSF_REPO_PATH` environment variable).

Notebook 9 downloads the TimesFM 3.0 weights (`google/timesfm-3.0-pytorch`) from Hugging Face
on its first run and runs them on the CPU. The TimesFM 3.0 weights are licensed for research
and non-commercial use only (the code is Apache-2.0).

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
│   ├── 7_ml_model_testing.ipynb           # Ridge/Lasso/LightGBM/XGBoost benchmark (local)
│   ├── 8_global_ml_model_testing.ipynb    # global models on route × region × origin series
│   ├── 9_timesfm_model_testing.ipynb      # TimesFM 3.0 zero-shot foundation model
│   ├── 10_ensembling_and_stacking.ipynb   # mean, median, inverse-MASE mean and stacking
│   └── 11_validation_overview.ipynb       # validation overview of the selected approaches
├── scripts/
│   ├── export_notebooks_pdf.py            # task export_pdf
│   ├── mlflow_purge.py                    # task mlflow_purge
│   └── verify_turismo_gov.ps1
├── docker-compose.yml                     # local MLflow tracking server
└── README.md
```

## Methodology

1. **Preprocessing** (notebooks 0–2): unification of the yearly source files, category standardization, and a chronological train / validation / test split that rolls forward with the data.
2. **Exploratory analysis** (notebooks 3–4): trend, seasonality, heteroscedasticity and stationarity tests, target transformations, and a comparison of methods to impute the COVID gap.
3. **Feature engineering** (notebook 6): lags chosen from the ACF/PACF, rolling and seasonal-rolling windows, EWMA, calendar (categorical and Fourier) and elapsed-time features, and COVID flags, with a leakage check.
4. **Modeling.** Every notebook runs a full grid of choices and logs it to MLflow:
   - **SARIMAX** (notebook 5): orders × target processing (`none`, `log`, AutoML `AutoStationaryTransformer`) × training window, with COVID and recovery regressors.
   - **Local Machine Learning** (notebook 7): Ridge and Lasso (penalty chosen by temporal cross-validation), LightGBM and XGBoost × target processing (including a seasonal log difference, `log yₜ − log yₜ₋₁₂`, so the trees forecast growth instead of levels) × training window × COVID handling (keep or remove 2020–2022) × multi-step strategy (recursive or single-model direct) × feature set. The best configuration of every tree family × training window is then tuned with Optuna.
   - **Global Machine Learning** (notebook 8): the same families trained on one panel of 19 component series plus a remainder and the total, with static series features; the total is forecast bottom-up (the sum of the components and the remainder) and compared with the direct forecast.
   - **Foundation model** (notebook 9): TimesFM 3.0, zero-shot on the total series, testing what the model reads: context window, COVID handling (keep, remove or mask as missing), target processing, calendar and COVID covariates, and symmetric averaging.
5. **Evaluation:** a repeated holdout over four non-overlapping 12-month validation windows, two before the pandemic and the two most recent years (the last one is the validation set; all dates follow the data). Configurations are ranked by the **mean seasonal MASE** over these windows, with the worst window, RMSE, MAE, MAPE, R² and Forecast Bias also reported, against naive and seasonal-naive baselines. The choices are compared with p95 heatmaps (the value that only the best 5% of the configurations sharing two choices beat). The pandemic is taken to start in April 2020.
6. **Ensembling and stacking** (notebook 10): the best configuration of each approach is combined by the mean, the median, an inverse-MASE weighted mean, a Huber-regression stack (free weights) and linear, Ridge and Lasso stacks (non-negative weights; Ridge and Lasso penalties chosen by nested cross-validation), with the stacking weights learned leaving one validation origin out and everything evaluated on the same validation origins.
7. **Validation overview** (notebook 11): the selected configuration of every approach, the best ensemble and the best stacking, compared overall, before vs after the pandemic, and in flat vs rapid-growth validation years (evaluation only, no re-selection).
8. **Final evaluation** (planned): the selected models are refit on train + validation and scored once on the test year.

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

- Das, A.; Kong, W.; Sen, R.; Zhou, Y. (2024). A decoder-only foundation model for time-series forecasting. *Proceedings of the 41st International Conference on Machine Learning (ICML)*. [arXiv:2310.10688](https://arxiv.org/abs/2310.10688). Code and weights: [google-research/timesfm](https://github.com/google-research/timesfm).
- Géron, A. (2021). *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*. 2nd ed. Alta Books.
- Gujarati, D. N. (2019). *Econometrics: Principles, Theory and Practical Applications*. Saraivauni.
- Joseph, M.; Tackes, J. (2024). *Modern Time Series Forecasting with Python*. Packt Publishing.
- Nielsen, A. (2021). *Practical Time Series Analysis*. Alta Books.
- Ministério do Turismo. (2026). Dataset: Estimativas de chegadas de turistas internacionais ao Brasil.
