"""Constants and helpers shared by every pipeline (the values of notebooks 5-9)."""

import os
import sys
from pathlib import Path

import pandas as pd

COVID_START = pd.Timestamp('2020-04-01')  # first pandemic month: April 2020
COVID_END = pd.Timestamp('2021-12-31')
POST_COVID_START = pd.Timestamp('2022-01-01')
COVID_RECOVERY_END = pd.Timestamp('2022-06-30')
# seasonal differences distorted by the pandemic collapse and rebound
MASE_EXCLUDED = (COVID_START, pd.Timestamp('2022-12-31'))

TARGET = 'arrival_count'
FREQ = 'ME'
SEASONAL_PERIOD = 12
HORIZON = 12
RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def use_book_repo():
    """Put the companion repository of the book on `sys.path` (as the notebooks do).

    Joseph, M. & Tackes, J. "Modern Time Series Forecasting with Python", 2nd ed.
    (Packt, 2024) — MIT License
    github.com/PacktPublishing/Modern-Time-Series-Forecasting-with-Python-2E
    """
    path = os.getenv(
        'MTSF_REPO_PATH',
        str(PROJECT_ROOT.parent / 'Modern-Time-Series-Forecasting-with-Python-2E'),
    )
    if path not in sys.path:
        sys.path.insert(0, path)
    return path


def forecast_dates(origin, horizon=HORIZON):
    """The `horizon` month ends after `origin` (the last month a model sees)."""
    return pd.date_range(origin + pd.offsets.MonthEnd(1), periods=horizon, freq=FREQ)


def seasonal_mase_scale(history):
    """In-sample MAE of the seasonal naive (y_t - y_{t-12}), pandemic pairs excluded."""
    diffs = (history - history.shift(SEASONAL_PERIOD)).abs()
    in_pandemic = history.index.to_series().between(*MASE_EXCLUDED)
    touches = in_pandemic | in_pandemic.shift(SEASONAL_PERIOD, fill_value=False)
    return float(diffs[~touches.to_numpy()].mean())
