"""Local machine learning pipeline of notebook 7 (one model on the total series).

A port of notebook 7, sections 1-4: target processing, training windows, COVID rows,
the feature builder, the model families and the two multi-step strategies. `forecast`
fits one configuration at one origin and returns the 12-month forecast in arrivals.
"""

import json
import math
import warnings
from typing import NamedTuple

import numpy as np
import pandas as pd
from category_encoders import OneHotEncoder
from lightgbm import LGBMRegressor, early_stopping
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from .common import (
    COVID_END,
    COVID_RECOVERY_END,
    COVID_START,
    FREQ,
    HORIZON,
    POST_COVID_START,
    RANDOM_STATE,
    SEASONAL_PERIOD,
    TARGET,
    forecast_dates,
    use_book_repo,
)

use_book_repo()
# Helpers credited to: Joseph, M. & Tackes, J. "Modern Time Series Forecasting with
# Python", 2nd ed. (Packt, 2024) — MIT License
from src.feature_engineering.autoregressive_features import (  # noqa: E402
    add_ewma,
    add_lags,
    add_rolling_features,
    add_seasonal_rolling_features,
)
from src.feature_engineering.temporal_features import (  # noqa: E402
    add_fourier_features,
    add_temporal_features,
)
from src.forecasting.ml_forecasting import (  # noqa: E402
    FeatureConfig,
    MissingValueConfig,
    MLForecast,
    ModelConfig,
)
from src.transforms.target_transformations import (  # noqa: E402
    AutoStationaryTransformer,
)

Y = 'y'  # model scale (after the target processing)
TS_ID = 'series_id'
SERIES_ID = 'brazil_total'


# ── target processing ────────────────────────────────────────────────────────
def fit_auto_stationary(train_series):
    """Fit a fresh AutoStationaryTransformer (Mann-Kendall trend, Guerrero Box-Cox)."""
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


class SeasonalLogDiffTarget:
    """Seasonal difference of the log: z_t = log(y_t) - log(y_{t-12}).

    Positional, like the book's `AdditiveDifferencingTransformer(diff_gap=12)`. The
    inverse takes y_{t-12} from the last `period` months of the fitted history, so it is
    exact for forecasts up to `period` months after the end of that history.
    """

    def __init__(self, train_series, period=SEASONAL_PERIOD):
        self.period = period
        self.log_history = np.log(train_series.astype(float)).to_numpy()
        self.last_date = train_series.index[-1]

    def transform(self, y):
        log_y = np.log(y.astype(float))
        return log_y - log_y.shift(self.period)

    def inverse_transform(self, z):
        dates = pd.DatetimeIndex(z.index)
        steps = (dates.year - self.last_date.year) * 12 + (
            dates.month - self.last_date.month
        )
        assert ((steps >= 1) & (steps <= self.period)).all(), 'up to one season ahead'
        base = self.log_history[len(self.log_history) - self.period + steps - 1]
        return pd.Series(np.exp(np.asarray(z, dtype=float) + base), index=z.index)


def fit_target_transform(name, train_series):
    if name == 'none':
        return IdentityTarget()
    if name == 'log':
        return LogTarget()
    if name == 'auto_stationary':
        return fit_auto_stationary(train_series)
    if name == 'seasonal_log_diff':
        return SeasonalLogDiffTarget(train_series)
    raise ValueError(f'unknown target transform: {name}')


# ── training windows and COVID rows ──────────────────────────────────────────
WINDOWS = {
    'all': None,
    **{f'last_{years}y': years for years in [14, 12, 10, 8, 6, 4]},
    'post_covid': POST_COVID_START,
}
REMOVED_START = pd.Timestamp('2020-01-31')
REMOVED_END = pd.Timestamp('2022-12-31')


def window_start(window, series, origin):
    """First training month of a window (counted back from the origin)."""
    spec = WINDOWS[window]
    if spec is None:
        return series.index[0]
    if isinstance(spec, pd.Timestamp):
        return spec
    return origin - pd.DateOffset(years=spec)


def window_exists(window, origin):
    spec = WINDOWS[window]
    return not isinstance(spec, pd.Timestamp) or origin > spec


def effective_covid_rows(covid_rows, origin):
    """Before the pandemic there is nothing to remove: `remove` equals `keep`."""
    return 'keep' if origin < REMOVED_START else covid_rows


def training_series(series, covid_rows, origin):
    history = series.loc[:origin]
    if covid_rows == 'remove':
        removed = history.index.to_series().between(REMOVED_START, REMOVED_END)
        return history[~removed.to_numpy()]
    return history


def make_history(transformed, original):
    """History table on the model scale (`y`) with the original target alongside."""
    return pd.DataFrame({
        'date': transformed.index,
        TS_ID: SERIES_ID,
        Y: transformed.to_numpy(),
        TARGET: original.reindex(transformed.index).to_numpy(),
    })


def make_setup(transform_name, covid_rows, series, origin):
    """Transformer and model-scale history of one origin, target processing, mode."""
    train_part = training_series(series, covid_rows, origin)
    transformer = fit_target_transform(transform_name, train_part)
    return {
        'origin': origin,
        'transformer': transformer,
        'history': make_history(transformer.transform(train_part), train_part),
    }


# ── feature builder (notebook 6 parameters) ──────────────────────────────────
GROUP_ORDER = [
    'lags',
    'growth',
    'rolling',
    'domain',
    'cal_cat',
    'cal_fourier',
    'elapsed',
]
TARGET_DERIVED = ['lags', 'growth', 'rolling']


def load_feature_spec(path):
    """Parameters of `feature_groups.json` (notebook 6) and the seasonal lags."""
    with open(path, encoding='utf-8') as f:
        spec = json.load(f)
    return {
        'params': spec['parameters'],
        'seasonal_lags': [
            lag
            for lag in spec['strategies']['recursive']['lags']
            if lag >= SEASONAL_PERIOD
        ],
    }


def lags_for_shift(shift, spec):
    """The two most recent legal lags plus the seasonal lags that are still legal."""
    return sorted(
        {shift, shift + 1} | {lag for lag in spec['seasonal_lags'] if lag >= shift}
    )


def growth_lags_for_shift(shift):
    return sorted({shift} | ({SEASONAL_PERIOD} if SEASONAL_PERIOD >= shift else set()))


def is_spliced(df):
    """True when the removed COVID years are missing from the history."""
    return (
        not df['date'].between(REMOVED_START, REMOVED_END).any()
        and (df['date'] > REMOVED_END).any()
    )


def add_domain_features(df, lags):
    df['days_in_month'] = df['date'].dt.days_in_month
    if is_spliced(df):
        # a lag crosses the gap when the row is after it and the lagged row before it
        df['is_pre_gap'] = (df['date'] < REMOVED_START).astype(int)
        df, pre_gap_lags = add_lags(df, lags=lags, column='is_pre_gap', ts_id=TS_ID)
        after_gap = (df['date'] > REMOVED_END).astype(int)
        flags = []
        for lag, column in zip(lags, pre_gap_lags):
            flag = f'crosses_gap_lag_{lag}'
            df[flag] = (df[column].fillna(1) * after_gap).astype(int)
            flags.append(flag)
        return df, ['days_in_month', *flags]
    df['is_covid'] = df['date'].between(COVID_START, COVID_END).astype(int)
    df['is_covid_recovery'] = (
        df['date'].between(POST_COVID_START, COVID_RECOVERY_END).astype(int)
    )
    df, covid_lags = add_lags(df, lags=lags, column='is_covid', ts_id=TS_ID)
    df[covid_lags] = df[covid_lags].fillna(0).astype(int)
    return df, ['is_covid', 'is_covid_recovery', 'days_in_month', *covid_lags]


def build_features(df, shift, spec):
    """Build every feature group for a model that knows the target up to `shift` back.

    `df` has `date`, `series_id`, `y` (model scale) and `arrival_count` (arrivals).
    Future rows have NaN targets. Returns the feature table and {group: [columns]}.
    """
    params = spec['params']
    df = df.copy()
    groups = {}
    lags = lags_for_shift(shift, spec)

    # target-derived groups (book helpers)
    df, groups['lags'] = add_lags(df, lags=lags, column=Y, ts_id=TS_ID)
    df['arrivals_mom_growth'] = df[TARGET] / df[TARGET].shift(1) - 1
    df, groups['growth'] = add_lags(
        df, lags=growth_lags_for_shift(shift), column='arrivals_mom_growth', ts_id=TS_ID
    )
    df, rolling = add_rolling_features(
        df,
        rolls=params['rolling_windows'],
        column=Y,
        agg_funcs=params['rolling_aggs'],
        ts_id=TS_ID,
        n_shift=shift,
    )
    df, seasonal = add_seasonal_rolling_features(
        df,
        seasonal_periods=[params['seasonal_period']],
        rolls=params['seasonal_windows'],
        column=Y,
        agg_funcs=params['seasonal_aggs'],
        ts_id=TS_ID,
        n_shift=math.ceil(shift / params['seasonal_period']),  # counted in seasons
    )
    df, ewma = add_ewma(
        df,
        column=Y,
        alphas=None,
        spans=params['ewma_spans'],
        ts_id=TS_ID,
        n_shift=shift,
    )
    groups['rolling'] = rolling + seasonal + ewma

    # calendar groups (book helpers); only the month and elapsed time are used
    df, _ = add_temporal_features(
        df,
        field_name='date',
        frequency=FREQ,
        add_elapsed=True,
        prefix='cal',
        drop=False,
    )
    df['cal_Elapsed'] = df['cal_Elapsed'].astype('int64')
    df, groups['cal_fourier'] = add_fourier_features(
        df,
        column_to_encode='cal_Month',
        max_value=12,
        n_fourier_terms=params['fourier_terms'],
    )
    df['cal_Month'] = pd.Categorical(df['cal_Month'], categories=range(1, 13))
    groups['cal_cat'] = ['cal_Month']
    groups['elapsed'] = ['cal_Elapsed']

    df, groups['domain'] = add_domain_features(df, lags)
    return df, groups


def training_rows(features, groups, start, end):
    """Rows of the training window with a known target and complete target features."""
    target_derived = [c for g in TARGET_DERIVED for c in groups[g]]
    rows = features[features['date'].between(start, end)]
    return rows.dropna(subset=[Y, *target_derived])


def canonical(feature_set):
    """Order the groups of a feature set consistently."""
    return tuple(g for g in GROUP_ORDER if g in feature_set)


def set_label(feature_set):
    return '+'.join(canonical(feature_set))


def make_feature_config(groups, feature_set):
    continuous = [
        col for g in canonical(feature_set) if g != 'cal_cat' for col in groups[g]
    ]
    categorical = groups['cal_cat'] if 'cal_cat' in feature_set else []
    return FeatureConfig(
        date='date',
        target=Y,
        original_target=TARGET,
        continuous_features=continuous,
        categorical_features=categorical,
        index_cols=['date'],
    )


def make_missing_config(groups):
    return MissingValueConfig(
        bfill_columns=groups['lags'] + groups['growth'] + groups['rolling']
    )


# ── model families ───────────────────────────────────────────────────────────
FAMILY_NAMES = {
    'ridge': 'Ridge',
    'lasso': 'Lasso',
    'lightgbm': 'LightGBM',
    'xgboost': 'XGBoost',
}
TREE_FAMILIES = {'lightgbm', 'xgboost'}

CV_SPLITS = 5
CV_TEST_SIZE = 12
RIDGE_ALPHAS = np.logspace(-3, 3, 13)

N_TREES = 2000
LGBM_PARAMS = {
    'n_estimators': N_TREES,
    'learning_rate': 0.02,
    'num_leaves': 8,
    'min_child_samples': 10,
    'subsample': 0.8,
    'subsample_freq': 1,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'n_jobs': 1,
    'verbose': -1,
}
XGB_PARAMS = {
    'n_estimators': N_TREES,
    'learning_rate': 0.02,
    'max_depth': 3,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    'enable_categorical': True,
    'random_state': RANDOM_STATE,
    'n_jobs': 1,
}
EARLY_STOPPING_ROUNDS = 100
EARLY_STOPPING_HOLDOUT = 12  # the last 12 months of the training window
MIN_ROWS_FOR_EARLY_STOPPING = 3 * EARLY_STOPPING_HOLDOUT
SHORT_WINDOW_TREES = 300


class MLForecastFixed(MLForecast):
    """Book MLForecast with the predict-time column check on the *input* columns."""

    def fit(self, X, y, is_transformed=False, fit_kwargs=None):
        self._input_features = X.columns.tolist()
        return super().fit(
            X, y, is_transformed=is_transformed, fit_kwargs=fit_kwargs or {}
        )

    def predict(self, X):
        missing = set(self._input_features) - set(X.columns)
        assert not missing, f'features missing at prediction time: {missing}'
        X = X[self._input_features].copy()
        if self.model_config.fill_missing:
            X = self.missing_config.impute_missing_values(X)
        if self.model_config.encode_categorical:
            X = self._cat_encoder.transform(X)
        if self.model_config.normalize:
            scaled = self._continuous_feats + self._encoded_categorical_features
            X[scaled] = self._scaler.transform(X[scaled])
        y_pred = pd.Series(
            self._model.predict(X[self._train_features]).ravel(),
            index=X.index,
            name=f'{self.model_config.name}',
        )
        if self.target_transformer is not None:
            y_pred = self.target_transformer.inverse_transform(y_pred)
        return y_pred


def temporal_cv(n_rows):
    """TimeSeriesSplit for the linear penalty search, shrunk for short windows."""
    n_splits = max(2, min(CV_SPLITS, n_rows // CV_TEST_SIZE - 1))
    test_size = min(CV_TEST_SIZE, n_rows // (n_splits + 1))
    return TimeSeriesSplit(n_splits=n_splits, test_size=test_size)


def make_estimator(family, n_rows, tree_params=None):
    if family == 'ridge':
        return RidgeCV(alphas=RIDGE_ALPHAS, cv=temporal_cv(n_rows))
    if family == 'lasso':
        return LassoCV(alphas=30, cv=temporal_cv(n_rows), max_iter=20000)
    if family == 'lightgbm':
        return LGBMRegressor(**{**LGBM_PARAMS, **(tree_params or {})})
    return XGBRegressor(**{**XGB_PARAMS, **(tree_params or {})})


def make_model_config(family, categorical, n_rows, tree_params=None):
    """A fresh ModelConfig (book) for one fit."""
    is_tree = family in TREE_FAMILIES
    encode = categorical and not is_tree
    return ModelConfig(
        model=make_estimator(family, n_rows, tree_params),
        name=FAMILY_NAMES[family],
        normalize=not is_tree,
        fill_missing=not is_tree,
        encode_categorical=encode,
        categorical_encoder=OneHotEncoder(cols=['cal_Month']) if encode else None,
    )


def feature_matrix(feature_config, rows, feature_set):
    """`get_X_y` (book) with the columns in a fixed, sorted order."""
    X, y, _ = feature_config.get_X_y(rows, categorical='cal_cat' in feature_set)
    return X[sorted(X.columns)], y


def best_n_estimators(  # noqa: PLR0913, PLR0917
    family, feature_config, missing_config, X, y, categorical, tree_params
):
    """Early stopping on the last training months (never on the forecast window)."""
    holdout = EARLY_STOPPING_HOLDOUT
    probe_config = make_model_config(family, categorical, len(X) - holdout, tree_params)
    eval_set = [(X.iloc[-holdout:], y.iloc[-holdout:])]
    if family == 'lightgbm':
        fit_kwargs = {
            'eval_set': eval_set,
            'callbacks': [early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        }
    else:
        probe_config.model.set_params(early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        fit_kwargs = {'eval_set': eval_set, 'verbose': False}
    probe = MLForecastFixed(
        model_config=probe_config,
        feature_config=feature_config,
        missing_config=missing_config,
    )
    probe.fit(X.iloc[:-holdout], y.iloc[:-holdout], fit_kwargs=fit_kwargs)
    if family == 'lightgbm':
        return probe._model.best_iteration_ or N_TREES
    return probe._model.best_iteration + 1


def fit_model(  # noqa: PLR0913, PLR0917
    family, feature_set, feature_config, missing_config, train, tree_params=None
):
    """Chapter 8's `evaluate_model`, fit part (MLForecast built from the arguments)."""
    categorical = 'cal_cat' in feature_set
    X, y = feature_matrix(feature_config, train, feature_set)
    y = y[Y]
    model_config = make_model_config(family, categorical, len(X), tree_params)
    if family in TREE_FAMILIES:
        n_trees = (
            best_n_estimators(
                family, feature_config, missing_config, X, y, categorical, tree_params
            )
            if len(X) >= MIN_ROWS_FOR_EARLY_STOPPING
            else SHORT_WINDOW_TREES
        )
        model_config.model.set_params(n_estimators=n_trees)
    model = MLForecastFixed(
        model_config=model_config,
        feature_config=feature_config,
        missing_config=missing_config,
    )
    model.fit(X, y)
    return model


def predict_rows(model, feature_config, rows, feature_set):
    X, _ = feature_matrix(feature_config, rows, feature_set)
    return model.predict(X)


# ── configurations and multi-step strategies ─────────────────────────────────
class Config(NamedTuple):
    family: str
    transform: str
    window: str
    covid_rows: str
    strategy: str
    feature_set: tuple

    @classmethod
    def from_params(cls, params):
        """The configuration of a notebook 7 MLflow run (its logged parameters)."""
        return cls(
            params['family'],
            params['transform'],
            params['window'],
            params['covid_rows'],
            params['strategy'],
            canonical(params['feature_set'].split('+')),
        )

    @property
    def label(self):
        return (
            f'{FAMILY_NAMES[self.family]} | {self.transform} | {self.window} | '
            f'{self.covid_rows} | {self.strategy} | {set_label(self.feature_set)}'
        )


def to_arrivals(setup, predicted_z):
    """Inverse-transform model-scale predictions (date-indexed Series) to arrivals."""
    return pd.Series(
        np.asarray(setup['transformer'].inverse_transform(predicted_z), dtype=float),
        index=predicted_z.index,
    )


def with_future_rows(history, dates):
    future = pd.DataFrame({'date': dates, TS_ID: SERIES_ID, Y: np.nan, TARGET: np.nan})
    return pd.concat([history, future], ignore_index=True)


def fit_for_shift(config, setup, frame, shift, context):  # noqa: PLR0917
    features, groups = build_features(frame, shift, context['spec'])
    feature_config = make_feature_config(groups, config.feature_set)
    model = fit_model(
        config.family,
        config.feature_set,
        feature_config,
        make_missing_config(groups),
        training_rows(features, groups, context['start'], setup['origin']),
        context['tree_params'],
    )
    return model, feature_config, features


def forecast_recursive(config, setup, future_dates, context):
    model, feature_config, _ = fit_for_shift(
        config, setup, setup['history'], 1, context
    )
    extended = setup['history'].copy()
    predictions = []
    for date in future_dates:
        extended = with_future_rows(extended, [date])
        features, _ = build_features(extended, 1, context['spec'])
        row = features.iloc[[-1]]
        pred = predict_rows(model, feature_config, row, config.feature_set).iloc[0]
        last = extended.index[-1]
        extended.loc[last, Y] = pred  # feed the prediction back
        extended.loc[last, TARGET] = to_arrivals(
            setup, pd.Series([pred], index=[date])
        ).iloc[0]
        predictions.append(pred)
    return pd.Series(predictions, index=future_dates)


def forecast_direct_single(config, setup, future_dates, context):
    shift = len(future_dates)
    extended = with_future_rows(setup['history'], future_dates)
    model, feature_config, features = fit_for_shift(
        config, setup, extended, shift, context
    )
    rows = features[features['date'].isin(future_dates)]
    predictions = predict_rows(model, feature_config, rows, config.feature_set)
    return pd.Series(predictions.to_numpy(), index=future_dates)


FORECASTERS = {
    'recursive': forecast_recursive,
    'direct_single': forecast_direct_single,
}


def forecast(config, tree_params, series, origin, spec, horizon=HORIZON):  # noqa: PLR0913, PLR0917
    """Fit `config` on `series` up to `origin`; the next `horizon` months in arrivals.

    `tree_params` are the Optuna parameters of a tuned tree configuration (else None)
    and `spec` the feature parameters of notebook 6 (`load_feature_spec`).
    """
    if not window_exists(config.window, origin):
        raise ValueError(f'window {config.window} does not exist at {origin:%Y-%m}')
    covid_rows = effective_covid_rows(config.covid_rows, origin)
    setup = make_setup(config.transform, covid_rows, series, origin)
    context = {
        'spec': spec,
        'start': window_start(config.window, series, origin),
        'tree_params': tree_params,
    }
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        predicted_z = FORECASTERS[config.strategy](
            config, setup, forecast_dates(origin, horizon), context
        )
    predicted = to_arrivals(setup, predicted_z)
    if not np.all(np.isfinite(predicted)):
        raise ValueError('non-finite forecast after the inverse transform')
    return predicted
