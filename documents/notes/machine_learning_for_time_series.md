# Machine Learning for Time Series
Notes from:
- Modern Time Series Forecasting with Python (Manu Joseph & Jeffery Taylor) - Chapters 5–10

## Time Series Forecasting as a Regression
- Time series forecasting, by definition, is an extrapolation problem, whereas regression is, most of the time, an interpolation problem.
- It is possible to create temporal features for each target T, by:
    - Time delay embedding: Encoding fixed lags for each target as features. Ex: lag_1 (T - 1), lag_2 (T - 2), lag_n (T - n)
    - Temporal embedding: Including features that capture the time passage. Ex: numeric sequences of Fourier terms to represent seasonality
-  A way to increase "features" and data to a time series, without needing to increase its length or probability of overfitting, is to increase its "width" by predicting multiple related time series at once

## Feature Engineering for TS Forecasting
- It is a common practice to join train and test sets in the FE step.
- Fourier terms are not always the best method to represent seasonality, a categorical approach, like adding calendar features (Ex: Month, Quarter, etc) could be more effective depending on the dataset, the choice should be empirical.

## Target Transformations for TS Forecasting
- Detect non-statinarity in TS:
    - mean change over time
    - variance change over time
    - TS has periodic changes in mean
    - TS has a unit root
    - The first 3 can almost be ascertained using simple visual inspection, but unit roots are more difficult to understand

### Detect and correcting unit roots
- The Augmented Dickey-Fuller (ADF) test checks for unit roots. The H0 is that the series has a unit root and by extention non-stationarity.
- Ways to make a TS stationary include:
    - Differencing transform:
        - Change the domain of observation to the domain of change in observations.
        - Gets rid of the unit roots, stabilizes mean and variance and can reduce trend and seasonality
        - We lose the scale of the TS, but it is possible to do the inverse transformation by saving the last value of the TS
        - Other differencing operator can be division

### Detect and correcting for trends
- Trends can have two "flavors":
    - Deterministics trends can be perfectly modeled in function of time and are constant
    - Stochastic trends cannot be explained by a simple linear fit, it inherently depends on the previous value of the TS.
- Its possible to adapt the ADF test to detect the trend type.