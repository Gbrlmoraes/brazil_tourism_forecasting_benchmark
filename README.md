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
- **Period:** 1989 – 2025
- **Granularity:** Monthly
- **Features:** year, month, access route (air/land/river/sea), continent, country of origin, arrival count

## Project Structure

```
├── data/
│   ├── raw/                   # Original source files
│   └── processed/             # train/val/test splits
├── notebooks/
│   ├── 0_data_extraction.ipynb
│   ├── 1_data_cleaning.ipynb
│   ├── 2_train_test_split.ipynb
│   ├── 3_exploratory_data_analysis.ipynb
│   └── 4_...
├── scripts/
│   └── utils.py
└── README.md
```

## Methodology

1. **Preprocessing:** missing value treatment, outlier detection, chronological train/val/test split
2. **Feature Engineering:** lags, rolling means, seasonality indicators
3. **Modeling:** training and hyperparameter tuning of ARIMA, XGBoost, and PatchTST under identical hardware conditions
4. **Evaluation:** standardized error metrics + engineering trade-off analysis

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
