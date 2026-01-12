//! Linear Regression implementation.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Linear Regression output values.
#[derive(Debug, Clone, Copy)]
pub struct LinearRegressionOutput {
    /// Slope of the regression line
    pub slope: f64,
    /// Y-intercept of the regression line
    pub intercept: f64,
    /// Predicted value at the current point
    pub value: f64,
    /// R-squared (coefficient of determination)
    pub r_squared: f64,
}

/// Linear Regression.
///
/// Fits a least-squares regression line over a rolling window.
/// Useful for trend identification and forecasting.
///
/// The primary output is the regression value (predicted price).
/// Secondary output is the slope (trend direction/strength).
#[derive(Debug, Clone)]
pub struct LinearRegression {
    period: usize,
}

impl LinearRegression {
    /// Create a new LinearRegression indicator.
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Calculate linear regression for a single window.
    fn regression_window(&self, window: &[f64]) -> LinearRegressionOutput {
        let n = window.len() as f64;

        // Calculate sums for least squares
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;

        for (i, &y) in window.iter().enumerate() {
            let x = i as f64;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }

        // Calculate slope and intercept
        let denominator = n * sum_xx - sum_x * sum_x;
        let (slope, intercept) = if denominator.abs() < 1e-10 {
            (0.0, sum_y / n)
        } else {
            let slope = (n * sum_xy - sum_x * sum_y) / denominator;
            let intercept = (sum_y - slope * sum_x) / n;
            (slope, intercept)
        };

        // Predicted value at the end of the window
        let value = intercept + slope * (n - 1.0);

        // Calculate R-squared
        let mean_y = sum_y / n;
        let mut ss_tot = 0.0;
        let mut ss_res = 0.0;

        for (i, &y) in window.iter().enumerate() {
            let x = i as f64;
            let y_pred = intercept + slope * x;
            ss_tot += (y - mean_y).powi(2);
            ss_res += (y - y_pred).powi(2);
        }

        let r_squared = if ss_tot.abs() < 1e-10 {
            1.0 // Perfect fit if no variation
        } else {
            1.0 - (ss_res / ss_tot)
        };

        LinearRegressionOutput {
            slope,
            intercept,
            value,
            r_squared,
        }
    }

    /// Calculate regression values (predicted prices).
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period || self.period == 0 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &data[start..=i];
            let reg = self.regression_window(window);
            result.push(reg.value);
        }

        result
    }

    /// Calculate slope values.
    pub fn calculate_slope(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period || self.period == 0 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &data[start..=i];
            let reg = self.regression_window(window);
            result.push(reg.slope);
        }

        result
    }

    /// Calculate full regression output.
    pub fn calculate_full(&self, data: &[f64]) -> Vec<Option<LinearRegressionOutput>> {
        let n = data.len();
        if n < self.period || self.period == 0 {
            return vec![None; n];
        }

        let mut result: Vec<Option<LinearRegressionOutput>> = vec![None; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &data[start..=i];
            let reg = self.regression_window(window);
            result.push(Some(reg));
        }

        result
    }

    /// Calculate R-squared values.
    pub fn calculate_r_squared(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period || self.period == 0 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &data[start..=i];
            let reg = self.regression_window(window);
            result.push(reg.r_squared);
        }

        result
    }
}

impl TechnicalIndicator for LinearRegression {
    fn name(&self) -> &str {
        "LinearRegression"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close);
        let slopes = self.calculate_slope(&data.close);
        Ok(IndicatorOutput::dual(values, slopes))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        2 // value and slope
    }
}

impl SignalIndicator for LinearRegression {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let slopes = self.calculate_slope(&data.close);
        let last = slopes.last().copied().unwrap_or(f64::NAN);

        if last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Positive slope = uptrend = bullish
        // Negative slope = downtrend = bearish
        if last > 0.0 {
            Ok(IndicatorSignal::Bullish)
        } else if last < 0.0 {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let slopes = self.calculate_slope(&data.close);
        let signals = slopes
            .iter()
            .map(|&slope| {
                if slope.is_nan() {
                    IndicatorSignal::Neutral
                } else if slope > 0.0 {
                    IndicatorSignal::Bullish
                } else if slope < 0.0 {
                    IndicatorSignal::Bearish
                } else {
                    IndicatorSignal::Neutral
                }
            })
            .collect();
        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_regression() {
        let lr = LinearRegression::new(5);
        // Perfect linear data
        let data: Vec<f64> = (0..10).map(|i| 100.0 + i as f64 * 2.0).collect();
        let values = lr.calculate(&data);
        let slopes = lr.calculate_slope(&data);

        // Check that regression values match the data for linear input
        for i in 4..10 {
            assert!(!values[i].is_nan());
            assert!((slopes[i] - 2.0).abs() < 1e-10); // Slope should be 2
        }
    }

    #[test]
    fn test_r_squared_perfect_fit() {
        let lr = LinearRegression::new(5);
        // Perfect linear data should have R-squared = 1
        let data: Vec<f64> = (0..10).map(|i| 10.0 + i as f64).collect();
        let r_sq = lr.calculate_r_squared(&data);

        for i in 4..10 {
            assert!((r_sq[i] - 1.0).abs() < 1e-10);
        }
    }
}
