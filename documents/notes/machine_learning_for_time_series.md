# Machine Learning for Time Series
Notes from:
- Modern Time Series Forecasting with Python (Manu Joseph & Jeffery Taylor) - Chapters 5–10

## Time Series Forecasting as a Regression
- Time series forecasting, by definition, is an extrapolation problem, whereas regression is, most of the time, an interpolation problem.
- Its possible to create temporal features for each target T, by:
    - Time delay embedding: Encoding fixed lags for each target as features. Ex: lag_1 (T - 1), lag_2 (T - 2), lag_n (T - n)
    - Temporal embedding: Including features that capture the time passage. Ex: numeric sequences of Fourier terms to represent seasonality
-  A way to increase "features" and data to a time series, without needing to increase its length or probability of overfitting, is to increase its "width" by predicting multiple related time series at once
