"""Forecasting pipelines of notebooks 5, 7, 8 and 9, refittable at any origin.

The final notebook (12) retrains the configuration each notebook selected on the train
and validation data and forecasts the test year. The code of each module is a port of
its notebook, with the history and the origin passed explicitly instead of read from
globals:

| Module | Notebook | Approach |
|---|---|---|
| `sarimax` | 5 | SARIMAX with COVID regressors |
| `local_ml` | 7 | Ridge / Lasso / LightGBM / XGBoost on the total series |
| `global_ml` | 8 | the same families on a panel, total forecast bottom-up |
| `timesfm_zero_shot` | 9 | TimesFM 3.0, zero-shot |

Notebook 12 checks that every module reproduces the validation forecasts logged by its
notebook before it forecasts the test year.
"""
