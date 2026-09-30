"""Global machine learning pipeline of notebook 8 (one model on a panel).

A port of notebook 8, sections 1-5: the panel of series, the per-series target
processing, the panel feature builder, the model families and the two multi-step
strategies. `forecast` fits one configuration at one origin and returns the bottom-up
forecast of the total (the sum of the component and remainder forecasts) and the
forecast of every series.
"""

import json
import math
import warnings
from dataclasses import dataclass
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

Y = 'y'  # model scale (after the target processing)
TS_ID = 'series_id'

# ── the panel of series ──────────────────────────────────────────────────────
ROUTES = {'Aérea': 'air', 'Terrestre': 'land'}
N_SUBDIVISIONS = 3
EXCLUDED_REGIONS = ['Outros']
MIN_MONTHLY_ARRIVALS = 1000
STATIC = ['route', 'region', 'subdivision', 'level']
SUBDIVISION_LABELS = {'Western / Eastern Europe': 'Europe'}


@dataclass(frozen=True)
class Panel:
    frame: pd.DataFrame  # long: date, series_id, arrival_count (contiguous per series)
    wide: pd.DataFrame  # one column per series
    static: dict  # {series_id: {route, region, subdivision, level}}
    series: list
    bottom_up: list  # components + remainder: their sum is the total
    static_categories: dict

    @property
    def first_date(self):
        return self.wide.index[0]


def component_static(route, region, subdivision):
    return {
        'route': ROUTES[route],
        'region': region.replace('Região ', ''),
        'subdivision': SUBDIVISION_LABELS.get(subdivision, subdivision),
        'level': 'component',
    }


def build_panel(rows, rules_end):  # noqa: PLR0914
    """The panel of notebook 8 from the raw `rows`.

    The selection rules (top subdivisions, minimum monthly volume) are computed from the
    rows up to `rules_end` (notebook 8: the end of `train.parquet`), so the panel is the
    one the configuration was selected on.
    """
    rule_rows = rows[rows['date'] <= rules_end]
    dates = pd.date_range(rows['date'].min(), rows['date'].max(), freq=FREQ)
    top_subdivisions = (
        rule_rows
        .groupby('country_cultural_subdivision')[TARGET]
        .sum()
        .nlargest(N_SUBDIVISIONS)
        .index.tolist()
    )
    candidates = rows[
        rows['access_route'].isin(ROUTES)
        & rows['country_cultural_subdivision'].isin(top_subdivisions)
        & ~rows['state_region'].isin(EXCLUDED_REGIONS)
    ]
    components = (
        candidates
        .pivot_table(
            index=['access_route', 'state_region', 'country_cultural_subdivision'],
            columns='date',
            values=TARGET,
            aggfunc='sum',
        )
        .reindex(columns=dates, fill_value=0)
        .fillna(0)
    )
    recent_mean = components.loc[:, :rules_end].iloc[:, -12:].mean(axis=1)
    components = components[recent_mean >= MIN_MONTHLY_ARRIVALS]

    total = rows.groupby('date')[TARGET].sum().reindex(dates)
    remainder = total - components.sum()

    series_static, series_values = {}, {}
    for key, values in components.iterrows():
        static = component_static(*key)
        series_id = f'{static["route"]}·{static["region"]}·{static["subdivision"]}'
        series_static[series_id] = static
        series_values[series_id] = values
    for series_id, values in [('remainder', remainder), ('total', total)]:
        series_static[series_id] = {col: series_id for col in STATIC[:-1]} | {
            'level': series_id
        }
        series_values[series_id] = values

    series = list(series_values)
    components_ids = [s for s in series if series_static[s]['level'] == 'component']
    frame = pd.concat(
        [
            pd.DataFrame({'date': dates, TS_ID: series_id, TARGET: values.to_numpy()})
            for series_id, values in series_values.items()
        ],
        ignore_index=True,
    )
    wide = frame.pivot_table(index='date', columns=TS_ID, values=TARGET, aggfunc='sum')[
        series
    ]
    bottom_up = [*components_ids, 'remainder']
    assert np.allclose(wide[bottom_up].sum(axis=1), wide['total'])
    return Panel(
        frame=frame,
        wide=wide,
        static=series_static,
        series=series,
        bottom_up=bottom_up,
        static_categories={
            col: sorted({static[col] for static in series_static.values()})
            for col in STATIC
        },
    )


# ── windows, COVID rows and per-series target processing ─────────────────────
WINDOWS = {'all': None, 'last_10y': 10, 'last_6y': 6}
REMOVED_START = pd.Timestamp('2020-01-31')
REMOVED_END = pd.Timestamp('2022-12-31')


def window_start(window, panel, origin):
    years = WINDOWS[window]
    return panel.first_date if years is None else origin - pd.DateOffset(years=years)


def effective_covid_rows(covid_rows, origin):
    """Before the pandemic there is nothing to remove: `remove` equals `keep`."""
    return 'keep' if origin < REMOVED_START else covid_rows


def training_panel(panel, covid_rows, origin):
    """The panel up to the origin, without the removed years for `remove`."""
    part = panel.frame[panel.frame['date'] <= origin]
    if covid_rows == 'remove':
        part = part[~part['date'].between(REMOVED_START, REMOVED_END)]
    return part.reset_index(drop=True)


class Log1pTarget:
    """log(1 + y); inverse exp(z) - 1."""

    @staticmethod
    def transform(y):
        return np.log1p(y.astype(float))

    @staticmethod
    def inverse_transform(z):
        return np.expm1(np.asarray(z, dtype=float))


class SeasonalLog1pDiffTarget:
    """z_t = log(1 + y_t) - log(1 + y_{t-12}), positional (notebook 7, with log1p)."""

    def __init__(self, history, period=SEASONAL_PERIOD):
        self.period = period
        self.log_history = np.log1p(history.astype(float)).to_numpy()
        self.last_date = history.index[-1]

    def transform(self, y):
        log_y = np.log1p(y.astype(float))
        return log_y - log_y.shift(self.period)

    def inverse_transform(self, z, dates):
        dates = pd.DatetimeIndex(dates)
        steps = (dates.year - self.last_date.year) * 12 + (
            dates.month - self.last_date.month
        )
        assert ((steps >= 1) & (steps <= self.period)).all(), 'up to one season ahead'
        base = self.log_history[len(self.log_history) - self.period + steps - 1]
        return np.expm1(np.asarray(z, dtype=float) + base)


class MeanScaleTarget:
    """y / m, with m the mean of the last 12 months of the fitted history."""

    def __init__(self, history):
        self.scale = max(float(history.iloc[-SEASONAL_PERIOD:].mean()), 1.0)

    def transform(self, y):
        return y.astype(float) / self.scale

    def inverse_transform(self, z):
        return np.asarray(z, dtype=float) * self.scale


def fit_target_transform(name, history):
    if name == 'log':
        return Log1pTarget()
    if name == 'seasonal_log_diff':
        return SeasonalLog1pDiffTarget(history)
    if name == 'mean_scale':
        return MeanScaleTarget(history)
    raise ValueError(f'unknown target transform: {name}')


def inverse(transformer, z, dates):
    if isinstance(transformer, SeasonalLog1pDiffTarget):
        return transformer.inverse_transform(z, dates)
    return transformer.inverse_transform(z)


def make_setup(panel, transform_name, covid_rows, origin):
    """Per-series transformers and the model-scale panel history of one origin."""
    part = training_panel(panel, covid_rows, origin)
    transformers, history = {}, []
    for series_id, frame in part.groupby(TS_ID, sort=False):
        values = frame.set_index('date')[TARGET]
        transformers[series_id] = fit_target_transform(transform_name, values)
        history.append(
            frame.assign(**{Y: transformers[series_id].transform(values).to_numpy()})
        )
    return {
        'origin': origin,
        'transformers': transformers,
        'history': pd.concat(history, ignore_index=True)[['date', TS_ID, Y, TARGET]],
    }


# ── panel feature builder (notebook 6 parameters) ────────────────────────────
GROUP_ORDER = [
    'lags',
    'growth',
    'rolling',
    'domain',
    'static',
    'cal_cat',
    'cal_fourier',
]
TARGET_DERIVED = ['lags', 'growth', 'rolling']
CATEGORICAL_GROUPS = ['static', 'cal_cat']


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


def add_ewma_per_series(df, column, spans, n_shift):
    """`add_ewma` (book) computed inside each series, with the book's column names."""
    by_series = df.groupby(TS_ID)[column]
    added = {
        f'{column}_ewma_span_{span}': by_series.transform(
            lambda x, span=span: x.shift(n_shift).ewm(span=span, adjust=False).mean()
        )
        for span in spans
    }
    return df.assign(**added), list(added)


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


def add_static_features(df, panel):
    for col in STATIC:
        values = df[TS_ID].map(lambda s, col=col: panel.static[s][col])
        df[col] = pd.Categorical(values, categories=panel.static_categories[col])
    return df, list(STATIC)


def build_features(df, shift, spec, panel):
    """Every feature group, per series, for a model knowing the target `shift` back."""
    params = spec['params']
    df = df.sort_values([TS_ID, 'date']).reset_index(drop=True)
    groups = {}
    lags = lags_for_shift(shift, spec)

    # target-derived groups (book helpers, per series)
    df, groups['lags'] = add_lags(df, lags=lags, column=Y, ts_id=TS_ID)
    previous = df.groupby(TS_ID)[TARGET].shift(1)
    df['arrivals_mom_growth'] = (df[TARGET] / previous - 1).replace(
        [np.inf, -np.inf], np.nan
    )
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
    df, ewma = add_ewma_per_series(df, Y, params['ewma_spans'], shift)
    groups['rolling'] = rolling + seasonal + ewma

    # calendar groups (book helpers)
    df, _ = add_temporal_features(
        df,
        field_name='date',
        frequency=FREQ,
        add_elapsed=True,
        prefix='cal',
        drop=False,
    )
    df, groups['cal_fourier'] = add_fourier_features(
        df,
        column_to_encode='cal_Month',
        max_value=12,
        n_fourier_terms=params['fourier_terms'],
    )
    df['cal_Month'] = pd.Categorical(df['cal_Month'], categories=range(1, 13))
    groups['cal_cat'] = ['cal_Month']

    df, groups['domain'] = add_domain_features(df, lags)
    df, groups['static'] = add_static_features(df, panel)
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


def categorical_columns(groups, feature_set):
    return [col for g in CATEGORICAL_GROUPS if g in feature_set for col in groups[g]]


def make_feature_config(groups, feature_set):
    continuous = [
        col
        for g in canonical(feature_set)
        if g not in CATEGORICAL_GROUPS
        for col in groups[g]
    ]
    return FeatureConfig(
        date='date',
        target=Y,
        original_target=TARGET,
        continuous_features=continuous,
        categorical_features=categorical_columns(groups, feature_set),
        index_cols=['date', TS_ID],
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
CV_TEST_MONTHS = 12
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
EARLY_STOPPING_MONTHS = 12  # the last 12 months of every series in the window
MIN_MONTHS_FOR_EARLY_STOPPING = 3 * EARLY_STOPPING_MONTHS
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
        return pd.Series(
            self._model.predict(X[self._train_features]).ravel(),
            index=X.index,
            name=f'{self.model_config.name}',
        )


def temporal_cv(n_months, n_series):
    """TimeSeriesSplit on month-sorted rows: folds of ~12 months of every series."""
    n_splits = max(2, min(CV_SPLITS, n_months // CV_TEST_MONTHS - 1))
    test_months = min(CV_TEST_MONTHS, n_months // (n_splits + 1))
    return TimeSeriesSplit(n_splits=n_splits, test_size=test_months * n_series)


def make_estimator(family, n_months, n_series, tree_params=None):
    if family == 'ridge':
        return RidgeCV(alphas=RIDGE_ALPHAS, cv=temporal_cv(n_months, n_series))
    if family == 'lasso':
        return LassoCV(alphas=30, cv=temporal_cv(n_months, n_series), max_iter=20000)
    if family == 'lightgbm':
        return LGBMRegressor(**{**LGBM_PARAMS, **(tree_params or {})})
    return XGBRegressor(**{**XGB_PARAMS, **(tree_params or {})})


def make_model_config(family, categorical, shape, tree_params=None):  # noqa: PLR0917
    """A fresh ModelConfig (book) for one fit; `shape` = (months, series)."""
    is_tree = family in TREE_FAMILIES
    encode = bool(categorical) and not is_tree
    return ModelConfig(
        model=make_estimator(family, *shape, tree_params),
        name=FAMILY_NAMES[family],
        normalize=not is_tree,
        fill_missing=not is_tree,
        encode_categorical=encode,
        categorical_encoder=OneHotEncoder(cols=categorical) if encode else None,
    )


def feature_matrix(feature_config, rows):
    """`get_X_y` (book) with the columns in a fixed, sorted order."""
    X, y, _ = feature_config.get_X_y(rows, categorical=True)
    return X[sorted(X.columns)], y


def best_n_estimators(  # noqa: PLR0913, PLR0917
    family, feature_config, missing_config, X, y, tree_params
):
    """Early stopping on the last 12 months of every series (never on the forecast)."""
    months = X.index.get_level_values('date')
    holdout = months > np.sort(months.unique())[-EARLY_STOPPING_MONTHS - 1]
    shape = (months[~holdout].nunique(), X.index.get_level_values(TS_ID).nunique())
    probe_config = make_model_config(
        family, feature_config.categorical_features, shape, tree_params
    )
    eval_set = [(X[holdout], y[holdout])]
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
    probe.fit(X[~holdout], y[~holdout], fit_kwargs=fit_kwargs)
    if family == 'lightgbm':
        return probe._model.best_iteration_ or N_TREES
    return probe._model.best_iteration + 1


def fit_model(family, feature_config, missing_config, train, tree_params=None):  # noqa: PLR0917
    """Chapter 8's `evaluate_model` (fit part) on month-sorted panel rows."""
    train = train.sort_values(['date', TS_ID])
    X, y = feature_matrix(feature_config, train)
    y = y[Y]
    shape = (train['date'].nunique(), train[TS_ID].nunique())
    model_config = make_model_config(
        family, feature_config.categorical_features, shape, tree_params
    )
    if family in TREE_FAMILIES:
        n_trees = (
            best_n_estimators(family, feature_config, missing_config, X, y, tree_params)
            if shape[0] >= MIN_MONTHS_FOR_EARLY_STOPPING
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


def predict_rows(model, feature_config, rows):
    X, _ = feature_matrix(feature_config, rows)
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
        """The configuration of a notebook 8 MLflow run (its logged parameters)."""
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
    """Inverse-transform a long frame (date, series_id, z) to arrivals, per series."""
    out = predicted_z.copy()
    for series_id, frame in predicted_z.groupby(TS_ID):
        out.loc[frame.index, TARGET] = inverse(
            setup['transformers'][series_id], frame['z'], frame['date']
        )
    return out


def with_future_rows(history, future_dates, series):
    future = pd.DataFrame(
        [(d, s) for s in series for d in future_dates], columns=['date', TS_ID]
    ).assign(**{Y: np.nan, TARGET: np.nan})
    return pd.concat([history, future], ignore_index=True)


def fit_for_shift(config, setup, frame, shift, context):  # noqa: PLR0917
    features, groups = build_features(frame, shift, context['spec'], context['panel'])
    feature_config = make_feature_config(groups, config.feature_set)
    model = fit_model(
        config.family,
        feature_config,
        make_missing_config(groups),
        training_rows(features, groups, context['start'], setup['origin']),
        context['tree_params'],
    )
    return model, feature_config, features


def predicted_frame(model, feature_config, rows):
    z = predict_rows(model, feature_config, rows)
    return pd.DataFrame({
        'date': z.index.get_level_values('date'),
        TS_ID: z.index.get_level_values(TS_ID),
        'z': z.to_numpy(),
    })


def forecast_recursive(config, setup, future_dates, context):
    model, feature_config, _ = fit_for_shift(
        config, setup, setup['history'], 1, context
    )
    extended = setup['history']
    steps = []
    for date in future_dates:
        extended = with_future_rows(extended, [date], context['panel'].series)
        features, _ = build_features(extended, 1, context['spec'], context['panel'])
        step = to_arrivals(
            setup,
            predicted_frame(model, feature_config, features[features['date'] == date]),
        )
        # feed the predictions back: model scale and arrivals, per series
        fed = step.set_index(TS_ID)
        is_new = extended['date'] == date
        extended.loc[is_new, Y] = extended.loc[is_new, TS_ID].map(fed['z']).to_numpy()
        extended.loc[is_new, TARGET] = (
            extended.loc[is_new, TS_ID].map(fed[TARGET]).to_numpy()
        )
        steps.append(step)
    return pd.concat(steps, ignore_index=True)


def forecast_direct_single(config, setup, future_dates, context):
    extended = with_future_rows(setup['history'], future_dates, context['panel'].series)
    model, feature_config, features = fit_for_shift(
        config, setup, extended, len(future_dates), context
    )
    rows = features[features['date'].isin(future_dates)]
    return to_arrivals(setup, predicted_frame(model, feature_config, rows))


FORECASTERS = {
    'recursive': forecast_recursive,
    'direct_single': forecast_direct_single,
}


def forecast(config, tree_params, panel, origin, spec, horizon=HORIZON):  # noqa: PLR0913, PLR0917
    """Fit `config` on the panel up to `origin` and forecast the next `horizon` months.

    Returns the bottom-up forecast of the total (Series, arrivals) and the forecast
    table of every series (index = month, one column per series).
    """
    covid_rows = effective_covid_rows(config.covid_rows, origin)
    setup = make_setup(panel, config.transform, covid_rows, origin)
    context = {
        'spec': spec,
        'panel': panel,
        'start': window_start(config.window, panel, origin),
        'tree_params': tree_params,
    }
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        predicted = FORECASTERS[config.strategy](
            config, setup, forecast_dates(origin, horizon), context
        )
    table = predicted.pivot_table(
        index='date', columns=TS_ID, values=TARGET, aggfunc='first'
    )[panel.series]
    if not np.all(np.isfinite(table.to_numpy())):
        raise ValueError('non-finite forecast after the inverse transform')
    return table[panel.bottom_up].sum(axis=1), table
