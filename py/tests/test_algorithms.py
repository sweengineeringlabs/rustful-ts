"""Tests for rustful-ts algorithms."""

import numpy as np
import pytest
from rustful_ts import (
    Arima,
    SimpleExponentialSmoothing,
    Holt,
    HoltWinters,
    SimpleMovingAverage,
    WeightedMovingAverage,
    LinearRegression,
    SeasonalLinearRegression,
    TimeSeriesKNN,
    mae,
    rmse,
)


class TestArima:
    def test_create(self):
        model = Arima(1, 1, 1)
        assert not model.is_fitted()

    def test_fit_predict(self):
        data = np.array([10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40])
        model = Arima(1, 1, 0)
        model.fit(data)
        assert model.is_fitted()

        forecast = model.predict(5)
        assert len(forecast) == 5
        assert all(f > 40 for f in forecast)  # Should continue upward trend

    def test_params(self):
        model = Arima(2, 1, 3)
        assert model.params() == (2, 1, 3)


class TestExponentialSmoothing:
    def test_ses(self):
        data = np.array([10, 12, 11, 13, 12, 14, 13, 15])
        model = SimpleExponentialSmoothing(0.3)
        model.fit(data)
        forecast = model.predict(3)
        # SES produces flat forecasts
        assert np.allclose(forecast[0], forecast[1])
        assert np.allclose(forecast[1], forecast[2])

    def test_holt(self):
        data = np.array([10, 12, 14, 16, 18, 20, 22, 24])
        model = Holt(0.3, 0.1)
        model.fit(data)
        forecast = model.predict(3)
        # Should follow upward trend
        assert forecast[0] < forecast[1] < forecast[2]

    def test_holt_winters(self):
        # Generate seasonal data
        data = np.array([
            100 + 20 * np.sin(i * np.pi / 6) + i * 0.5
            for i in range(36)
        ])
        model = HoltWinters(0.3, 0.1, 0.2, 12)
        model.fit(data)
        forecast = model.predict(12)
        assert len(forecast) == 12
        assert len(model.seasonal_components()) == 12


class TestMovingAverage:
    def test_sma(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        model = SimpleMovingAverage(3)
        model.fit(data)

        smoothed = model.smoothed_values()
        assert len(smoothed) == 5  # n - window + 1
        assert np.isclose(smoothed[0], 2.0)  # avg(1,2,3)

    def test_wma_linear(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        model = WeightedMovingAverage.linear(3)
        model.fit(data)
        smoothed = model.smoothed_values()
        assert len(smoothed) == 3


class TestLinearRegression:
    def test_perfect_fit(self):
        data = np.array([10, 12, 14, 16, 18, 20])
        model = LinearRegression()
        model.fit(data)

        assert np.isclose(model.slope(), 2.0)
        assert np.isclose(model.intercept(), 10.0)
        assert model.r_squared() > 0.99

        forecast = model.predict(3)
        assert np.isclose(forecast[0], 22.0)
        assert np.isclose(forecast[1], 24.0)

    def test_seasonal(self):
        data = np.array([10 + i * 0.5 + [2, -1, 0, -1][i % 4] for i in range(24)])
        model = SeasonalLinearRegression(4)
        model.fit(data)
        factors = model.seasonal_factors()
        assert len(factors) == 4


class TestKNN:
    def test_periodic_data(self):
        data = np.array([np.sin(i * 0.2) * 10 + 50 for i in range(100)])
        model = TimeSeriesKNN(5, 10)
        model.fit(data)

        assert model.is_fitted()
        assert model.n_patterns() > 0

        forecast = model.predict(5)
        assert len(forecast) == 5

    def test_manhattan_distance(self):
        data = np.array([np.sin(i * 0.2) * 10 + 50 for i in range(100)])
        model = TimeSeriesKNN(5, 10, metric="manhattan")
        model.fit(data)
        forecast = model.predict(3)
        assert len(forecast) == 3


class TestMetrics:
    def test_mae(self):
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.isclose(mae(actual, predicted), 0.0)

        predicted = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        assert np.isclose(mae(actual, predicted), 1.0)

    def test_rmse(self):
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([2.0, 3.0, 4.0])
        assert np.isclose(rmse(actual, predicted), 1.0)
