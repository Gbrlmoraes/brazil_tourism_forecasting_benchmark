"""TimesFM zero-shot pipeline of notebook 9 (context, target processing, covariates).

TimesFM 3.0 is not trained on this series: a configuration only chooses what the model
reads. Das, A., Kong, W., Sen, R. & Zhou, Y. "A decoder-only foundation model for
time-series forecasting", ICML 2024 (arXiv:2310.10688); google-research/timesfm.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd

from .common import (
    COVID_END,
    COVID_RECOVERY_END,
    COVID_START,
    HORIZON,
    POST_COVID_START,
    SEASONAL_PERIOD,
    forecast_dates,
)

WINDOWS = {
    'all': None,
    **{f'last_{years}y': years for years in [14, 12, 10, 8, 6, 4]},
    'post_covid': POST_COVID_START,
}
REMOVED = (pd.Timestamp('2020-01-31'), pd.Timestamp('2022-12-31'))  # three full years
MASKED = (COVID_START, pd.Timestamp('2022-12-31'))  # pandemic and rebound
MEAN_DAYS_IN_MONTH = 365.25 / 12


class Config(NamedTuple):
    window: str
    covid_rows: str
    transform: str
    covariates: str
    symmetric: bool

    @classmethod
    def from_params(cls, params):
        """The configuration of a notebook 9 MLflow run (its logged parameters)."""
        return cls(
            params['window'],
            params['covid_rows'],
            params['transform'],
            params['covariates'],
            params['symmetric'] == 'True',
        )

    @property
    def label(self):
        return (
            f'{self.window} | {self.covid_rows} | {self.transform} | '
            f'{self.covariates} | {"symmetric" if self.symmetric else "plain"}'
        )


def load_model(model_id):
    """The pretrained TimesFM 3.0 forecaster (weights from Hugging Face)."""
    from timesfm3 import TimesFM3Forecaster  # noqa: PLC0415

    return TimesFM3Forecaster.from_pretrained(model_id)


def window_start(window, series, origin):
    spec = WINDOWS[window]
    if spec is None:
        return series.index[0]
    if isinstance(spec, pd.Timestamp):
        return spec
    return origin - pd.DateOffset(years=spec)


def window_exists(window, origin):
    spec = WINDOWS[window]
    return not isinstance(spec, pd.Timestamp) or origin > spec


def effective(config, origin):
    """The configuration run at `origin` (pre-COVID: no rows to drop, no flags)."""
    if origin >= REMOVED[0]:
        covid_rows = config.covid_rows
    else:
        covid_rows = 'keep'
    covariates = config.covariates
    if origin < COVID_START:
        covariates = {'covid': 'none', 'calendar+covid': 'calendar'}.get(
            covariates, covariates
        )
    return config._replace(covid_rows=covid_rows, covariates=covariates)


class SeasonalLogDiffTarget:
    """z_t = log(y_t) - log(y_{t-12}), positional; inverse from the last 12 months."""

    def __init__(self, history, period=SEASONAL_PERIOD):
        self.period = period
        self.log_history = np.log(history.astype(float)).to_numpy()
        self.last_date = history.index[-1]

    def transform(self, y):
        log_y = np.log(y.astype(float))
        return log_y - log_y.shift(self.period)

    def inverse_transform(self, z, dates):
        dates = pd.DatetimeIndex(dates)
        steps = (dates.year - self.last_date.year) * 12 + (
            dates.month - self.last_date.month
        )
        assert ((steps >= 1) & (steps <= self.period)).all(), 'up to one season ahead'
        base = self.log_history[len(self.log_history) - self.period + steps - 1]
        # z may be (horizon,) or (horizon, n_quantiles)
        return np.exp(
            np.asarray(z, dtype=float) + base.reshape(-1, *([1] * (np.ndim(z) - 1)))
        )


def make_context(config, series, origin):
    """Context values (model scale, NaN where masked) and the fitted inverse."""
    history = series.loc[window_start(config.window, series, origin) : origin]
    if config.covid_rows == 'remove':
        history = history[~history.index.to_series().between(*REMOVED).to_numpy()]
    if config.transform == 'log':
        values = np.log(history.astype(float))
        inverse = lambda z, dates: np.exp(np.asarray(z, dtype=float))  # noqa: E731
    elif config.transform == 'seasonal_log_diff':
        target = SeasonalLogDiffTarget(history)
        values = target.transform(history).iloc[SEASONAL_PERIOD:]
        inverse = target.inverse_transform
    else:
        values = history.astype(float)
        inverse = lambda z, dates: np.asarray(z, dtype=float)  # noqa: E731
    if config.covid_rows == 'mask':
        values = values.where(~values.index.to_series().between(*MASKED))
    return values, inverse


def make_covariates(covariates, context_dates, future_dates):
    """Past-and-future covariates, shape (n_covariates, context + horizon), or None."""
    if covariates == 'none':
        return None
    dates = pd.DatetimeIndex(context_dates).append(pd.DatetimeIndex(future_dates))
    dates_series = dates.to_series()
    rows = []
    if 'calendar' in covariates:
        month = dates.month.to_numpy()
        rows += [
            np.sin(2 * np.pi * month / 12),
            np.cos(2 * np.pi * month / 12),
            dates.days_in_month.to_numpy() - MEAN_DAYS_IN_MONTH,
        ]
    if 'covid' in covariates:
        rows += [
            dates_series.between(COVID_START, COVID_END).to_numpy(),
            dates_series.between(POST_COVID_START, COVID_RECOVERY_END).to_numpy(),
        ]
    return np.asarray(rows, dtype=np.float32)


def forecast(model, config, series, origin, horizon=HORIZON):
    """Median, 0.1 and 0.9 quantile forecasts of the `horizon` months after `origin`."""
    if not window_exists(config.window, origin):
        raise ValueError(f'window {config.window} does not exist at {origin:%Y-%m}')
    config = effective(config, origin)
    quantiles = list(model.config.quantiles)
    context, inverse = make_context(config, series, origin)
    future = forecast_dates(origin, horizon)
    output = model.predict(
        context.to_numpy(dtype=np.float32),
        horizon=horizon,
        past_future_covariates=make_covariates(
            config.covariates, context.index, future
        ),
        return_quantiles=True,
        use_symmetric_averaging=config.symmetric,
        make_positive=config.transform == 'none',
    )
    median = inverse(output.forecast, future)
    bands = inverse(output.quantiles, future)
    return (
        pd.Series(median, index=future),
        pd.Series(bands[:, quantiles.index(0.1)], index=future),
        pd.Series(bands[:, quantiles.index(0.9)], index=future),
    )
