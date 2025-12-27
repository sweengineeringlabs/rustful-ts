"""
rustful-ts: High-performance time series prediction algorithms

This library provides fast time series forecasting powered by Rust.
All algorithms accept and return NumPy arrays.

Example:
    >>> import numpy as np
    >>> from rustful_ts import Arima
    >>>
    >>> data = np.array([10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
    >>> model = Arima(1, 1, 1)
    >>> model.fit(data)
    >>> forecast = model.predict(5)
    >>> print(forecast)
"""

from rustful_ts._rustful_ts import (
    # Algorithms
    Arima,
    SimpleExponentialSmoothing,
    Holt,
    HoltWinters,
    SimpleMovingAverage,
    WeightedMovingAverage,
    LinearRegression,
    SeasonalLinearRegression,
    TimeSeriesKNN,
    # Metrics
    mae,
    mse,
    rmse,
    mape,
    # Preprocessing
    normalize,
    standardize,
    difference,
)

__version__ = "0.1.0"

__all__ = [
    # Algorithms
    "Arima",
    "SimpleExponentialSmoothing",
    "Holt",
    "HoltWinters",
    "SimpleMovingAverage",
    "WeightedMovingAverage",
    "LinearRegression",
    "SeasonalLinearRegression",
    "TimeSeriesKNN",
    # Metrics
    "mae",
    "mse",
    "rmse",
    "mape",
    # Preprocessing
    "normalize",
    "standardize",
    "difference",
]
