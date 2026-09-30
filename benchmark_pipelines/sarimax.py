"""SARIMAX pipeline of notebook 5 (data selection, target processing, COVID flags)."""

import re
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .common import (
    COVID_END,
    COVID_RECOVERY_END,
    COVID_START,
    FREQ,
    HORIZON,
    POST_COVID_START,
    SEASONAL_PERIOD,
    forecast_dates,
    use_book_repo,
)

EXOG_VARIANTS = {
    'covid_exog': ['is_covid'],
    'covid_recovery_exog': ['is_covid', 'is_covid_recovery'],
}
LAST_YEARS = re.compile(r'last_(\d+)y_(covid_exog|covid_recovery_exog)')


@dataclass(frozen=True)
class SarimaxConfig:
    order: tuple
    seasonal_order: tuple
    transform: str
    data_selection: str

    @classmethod
    def from_params(cls, params):
        """The configuration of a notebook 5 MLflow run (its logged parameters)."""
        p, d, q, P, D, Q, s = (int(params[k]) for k in 'p d q P D Q s'.split())
        return cls(
            (p, d, q), (P, D, Q, s), params['transform'], params['data_selection']
        )

    @property
    def label(self):
        """The run name of notebook 5."""
        p, d, q = self.order
        P, D, Q, s = self.seasonal_order
        return (
            f'SARIMAX({p},{d},{q})({P},{D},{Q})[{s}] '
            f'({self.data_selection} | {self.transform})'
        )


class IdentityTarget:
    """No target transformation."""

    @staticmethod
    def transform(y):
        return y

    @staticmethod
    def inverse_transform(y):
        return y


class LogTarget:
    """Natural log of the arrival counts; inverse is exp."""

    @staticmethod
    def transform(y):
        return np.log(y)

    @staticmethod
    def inverse_transform(y):
        return np.exp(y)


def fit_auto_stationary(train_series):
    """Fit a fresh AutoStationaryTransformer (Mann-Kendall trend, Guerrero Box-Cox)."""
    use_book_repo()
    from src.transforms.target_transformations import (  # noqa: PLC0415
        AutoStationaryTransformer,
    )

    transformer = AutoStationaryTransformer(
        confidence=0.05,
        seasonal_period=SEASONAL_PERIOD,
        trend_check_params={'mann_kendall': True},
        detrender_params={'degree': 1},
        deseasonalizer_params={},
        box_cox_params={'optimization': 'guerrero'},
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return transformer.fit(train_series, freq=FREQ)


def fit_target_transform(name, train_series):
    if name == 'none':
        return IdentityTarget()
    if name == 'log':
        return LogTarget()
    if name == 'auto_stationary':
        return fit_auto_stationary(train_series)
    raise ValueError(f'unknown target transform: {name}')


def covid_flags(dates):
    """The exogenous regressors of notebook 5, known in advance for any month."""
    dates = pd.DatetimeIndex(dates).to_series()
    return pd.DataFrame({
        'is_covid': dates.between(COVID_START, COVID_END).astype(float),
        'is_covid_recovery': dates.between(POST_COVID_START, COVID_RECOVERY_END).astype(
            float
        ),
    })


def training_selection(data_selection, series, origin):
    """Training window and exogenous columns of a data selection at `origin`."""
    history = series.loc[:origin]
    match = LAST_YEARS.fullmatch(data_selection)
    if match:
        years, variant = int(match[1]), match[2]
        window = history[history.index >= origin - pd.DateOffset(years=years)]
        # before the pandemic the flags are all zero: no regressor
        exog = EXOG_VARIANTS[variant] if origin >= COVID_START else None
        return window, exog
    if data_selection in {'post_covid', 'post_covid_recovery_exog'}:
        if origin <= POST_COVID_START:
            raise ValueError(f'{data_selection} does not exist at {origin:%Y-%m}')
        window = history[history.index >= POST_COVID_START]
        exog = (
            ['is_covid_recovery']
            if data_selection == 'post_covid_recovery_exog'
            else None
        )
        return window, exog
    raise ValueError(f'unknown data selection: {data_selection}')


def forecast(config, series, origin, horizon=HORIZON):
    """Fit `config` on `series` up to `origin`; forecast the next `horizon` months."""
    window, exog = training_selection(config.data_selection, series, origin)
    transformer = fit_target_transform(config.transform, window)
    future = forecast_dates(origin, horizon)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fit = SARIMAX(
            transformer.transform(window),
            exog=covid_flags(window.index)[exog] if exog else None,
            order=config.order,
            seasonal_order=config.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        transformed = np.asarray(
            fit.forecast(
                steps=horizon, exog=covid_flags(future)[exog] if exog else None
            )
        )
    predicted = transformer.inverse_transform(pd.Series(transformed, index=future))
    return pd.Series(np.asarray(predicted, dtype=float), index=future)
