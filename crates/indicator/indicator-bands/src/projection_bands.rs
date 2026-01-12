//! Projection Bands implementation.
//!
//! Bands based on linear regression projections of high and low prices.

use indicator_spi::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};

/// Projection Bands.
///
/// Projection Bands use linear regression to project the price channel:
/// - Upper Band: Linear regression of highs projected forward
/// - Lower Band: Linear regression of lows projected forward
/// - Middle Band: (Upper + Lower) / 2
///
/// These bands provide a dynamic channel that follows the trend direction
/// and adapts to the slope of price movement.
#[derive(Debug, Clone)]
pub struct ProjectionBands {
    /// Period for the linear regression calculation.
    period: usize,
}

impl ProjectionBands {
    /// Create a new Projection Bands indicator.
    ///
    /// # Arguments
    /// * `period` - Period for linear regression (typically 14)
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Create with default parameters (14-period).
    pub fn default_params() -> Self {
        Self::new(14)
    }

    /// Calculate linear regression value at the end of a window.
    fn linear_regression_value(&self, data: &[f64]) -> f64 {
        let n = data.len() as f64;
        if n < 2.0 {
            return f64::NAN;
        }

        // Calculate sums for linear regression
        // y = a + bx where x is the index (1 to n)
        let sum_x: f64 = (1..=data.len()).map(|x| x as f64).sum();
        let sum_y: f64 = data.iter().sum();
        let sum_xy: f64 = data
            .iter()
            .enumerate()
            .map(|(i, &y)| (i + 1) as f64 * y)
            .sum();
        let sum_x2: f64 = (1..=data.len()).map(|x| (x as f64).powi(2)).sum();

        // Calculate slope (b) and intercept (a)
        let denominator = n * sum_x2 - sum_x.powi(2);
        if denominator.abs() < 1e-10 {
            return f64::NAN;
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denominator;
        let intercept = (sum_y - slope * sum_x) / n;

        // Linear regression value at the last point
        intercept + slope * n
    }

    /// Calculate minimum value within a window relative to regression line.
    fn min_below_regression(&self, data: &[f64]) -> f64 {
        let n = data.len() as f64;
        if n < 2.0 {
            return f64::NAN;
        }

        // Calculate linear regression parameters
        let sum_x: f64 = (1..=data.len()).map(|x| x as f64).sum();
        let sum_y: f64 = data.iter().sum();
        let sum_xy: f64 = data
            .iter()
            .enumerate()
            .map(|(i, &y)| (i + 1) as f64 * y)
            .sum();
        let sum_x2: f64 = (1..=data.len()).map(|x| (x as f64).powi(2)).sum();

        let denominator = n * sum_x2 - sum_x.powi(2);
        if denominator.abs() < 1e-10 {
            return f64::NAN;
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denominator;
        let intercept = (sum_y - slope * sum_x) / n;

        // Find minimum deviation below regression line
        let mut min_diff = 0.0f64;
        for (i, &val) in data.iter().enumerate() {
            let regression_val = intercept + slope * (i + 1) as f64;
            let diff = val - regression_val;
            if diff < min_diff {
                min_diff = diff;
            }
        }

        // Return regression value at end plus minimum deviation
        intercept + slope * n + min_diff
    }

    /// Calculate maximum value within a window relative to regression line.
    fn max_above_regression(&self, data: &[f64]) -> f64 {
        let n = data.len() as f64;
        if n < 2.0 {
            return f64::NAN;
        }

        // Calculate linear regression parameters
        let sum_x: f64 = (1..=data.len()).map(|x| x as f64).sum();
        let sum_y: f64 = data.iter().sum();
        let sum_xy: f64 = data
            .iter()
            .enumerate()
            .map(|(i, &y)| (i + 1) as f64 * y)
            .sum();
        let sum_x2: f64 = (1..=data.len()).map(|x| (x as f64).powi(2)).sum();

        let denominator = n * sum_x2 - sum_x.powi(2);
        if denominator.abs() < 1e-10 {
            return f64::NAN;
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denominator;
        let intercept = (sum_y - slope * sum_x) / n;

        // Find maximum deviation above regression line
        let mut max_diff = 0.0f64;
        for (i, &val) in data.iter().enumerate() {
            let regression_val = intercept + slope * (i + 1) as f64;
            let diff = val - regression_val;
            if diff > max_diff {
                max_diff = diff;
            }
        }

        // Return regression value at end plus maximum deviation
        intercept + slope * n + max_diff
    }

    /// Calculate Projection Bands (middle, upper, lower).
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();

        if n < self.period || self.period < 2 {
            return (
                vec![f64::NAN; n],
                vec![f64::NAN; n],
                vec![f64::NAN; n],
            );
        }

        let mut upper = vec![f64::NAN; self.period - 1];
        let mut lower = vec![f64::NAN; self.period - 1];
        let mut middle = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let high_window = &high[start..=i];
            let low_window = &low[start..=i];

            let upper_val = self.max_above_regression(high_window);
            let lower_val = self.min_below_regression(low_window);

            upper.push(upper_val);
            lower.push(lower_val);

            if upper_val.is_nan() || lower_val.is_nan() {
                middle.push(f64::NAN);
            } else {
                middle.push((upper_val + lower_val) / 2.0);
            }
        }

        (middle, upper, lower)
    }

    /// Calculate projection oscillator (position within bands normalized to -1 to 1).
    pub fn oscillator(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let (_, upper, lower) = self.calculate(high, low, close);

        close
            .iter()
            .zip(upper.iter().zip(lower.iter()))
            .map(|(&c, (&u, &l))| {
                if u.is_nan() || l.is_nan() {
                    f64::NAN
                } else {
                    let range = u - l;
                    if range.abs() < 1e-10 {
                        0.0
                    } else {
                        // Normalize to -1 to 1 (where 0 is middle)
                        let mid = (u + l) / 2.0;
                        2.0 * (c - mid) / range
                    }
                }
            })
            .collect()
    }
}

impl Default for ProjectionBands {
    fn default() -> Self {
        Self::default_params()
    }
}

impl TechnicalIndicator for ProjectionBands {
    fn name(&self) -> &str {
        "ProjectionBands"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.high.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.high.len(),
            });
        }

        let (middle, upper, lower) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_projection_bands() {
        let pb = ProjectionBands::new(10);
        let n = 30;
        let high: Vec<f64> = (0..n).map(|i| 105.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0).collect();

        let (middle, upper, lower) = pb.calculate(&high, &low, &close);

        assert_eq!(middle.len(), n);
        assert_eq!(upper.len(), n);
        assert_eq!(lower.len(), n);

        // Check bands after warmup
        for i in 10..n {
            if !middle[i].is_nan() && !upper[i].is_nan() && !lower[i].is_nan() {
                assert!(upper[i] >= middle[i], "Upper should be >= middle");
                assert!(lower[i] <= middle[i], "Lower should be <= middle");
            }
        }
    }

    #[test]
    fn test_projection_bands_trending() {
        // Upward trending data
        let pb = ProjectionBands::new(10);
        let high: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 + 5.0).collect();
        let low: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 - 5.0).collect();
        let close: Vec<f64> = (0..30).map(|i| 100.0 + i as f64).collect();

        let (middle, upper, lower) = pb.calculate(&high, &low, &close);

        // Verify bands follow the trend
        for i in 15..29 {
            if !middle[i].is_nan() && !middle[i + 1].is_nan() {
                assert!(middle[i + 1] > middle[i], "Middle should trend upward");
            }
        }
    }

    #[test]
    fn test_projection_oscillator() {
        let pb = ProjectionBands::new(10);
        let n = 30;
        let high: Vec<f64> = (0..n).map(|i| 105.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0).collect();

        let osc = pb.oscillator(&high, &low, &close);

        // Oscillator should be bounded roughly around -1 to 1
        for i in 10..n {
            if !osc[i].is_nan() {
                assert!(osc[i] >= -2.0 && osc[i] <= 2.0, "Oscillator out of expected range");
            }
        }
    }

    #[test]
    fn test_projection_default() {
        let pb = ProjectionBands::default();
        assert_eq!(pb.period, 14);
    }
}
