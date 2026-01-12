//! Standard Error Bands implementation.
//!
//! Bands based on linear regression and standard error.

use indicator_spi::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};

/// Standard Error Bands.
///
/// Bands constructed around a linear regression line using standard error:
/// - Middle Band: Linear regression value
/// - Upper Band: Linear regression + (multiplier x Standard Error)
/// - Lower Band: Linear regression - (multiplier x Standard Error)
///
/// Standard Error Bands adapt to the trend and volatility of the price,
/// providing dynamic support and resistance levels.
#[derive(Debug, Clone)]
pub struct StandardErrorBands {
    /// Period for the linear regression calculation.
    period: usize,
    /// Multiplier for the standard error.
    multiplier: f64,
}

impl StandardErrorBands {
    /// Create a new Standard Error Bands indicator.
    ///
    /// # Arguments
    /// * `period` - Period for linear regression (typically 21)
    /// * `multiplier` - Standard error multiplier (typically 2.0)
    pub fn new(period: usize, multiplier: f64) -> Self {
        Self { period, multiplier }
    }

    /// Create with default parameters (21-period, 2.0 multiplier).
    pub fn default_params() -> Self {
        Self::new(21, 2.0)
    }

    /// Calculate linear regression value and standard error for a window.
    fn linear_regression_with_error(&self, data: &[f64]) -> (f64, f64) {
        let n = data.len() as f64;
        if n < 2.0 {
            return (f64::NAN, f64::NAN);
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
            return (f64::NAN, f64::NAN);
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denominator;
        let intercept = (sum_y - slope * sum_x) / n;

        // Linear regression value at the last point
        let lr_value = intercept + slope * n;

        // Calculate standard error of the estimate
        let residuals_squared: f64 = data
            .iter()
            .enumerate()
            .map(|(i, &y)| {
                let predicted = intercept + slope * (i + 1) as f64;
                (y - predicted).powi(2)
            })
            .sum();

        let std_error = (residuals_squared / (n - 2.0)).sqrt();

        (lr_value, std_error)
    }

    /// Calculate Standard Error Bands (middle, upper, lower).
    pub fn calculate(&self, close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();

        if n < self.period || self.period < 2 {
            return (
                vec![f64::NAN; n],
                vec![f64::NAN; n],
                vec![f64::NAN; n],
            );
        }

        let mut middle = vec![f64::NAN; self.period - 1];
        let mut upper = vec![f64::NAN; self.period - 1];
        let mut lower = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &close[start..=i];

            let (lr_value, std_error) = self.linear_regression_with_error(window);

            if lr_value.is_nan() || std_error.is_nan() {
                middle.push(f64::NAN);
                upper.push(f64::NAN);
                lower.push(f64::NAN);
            } else {
                middle.push(lr_value);
                upper.push(lr_value + self.multiplier * std_error);
                lower.push(lr_value - self.multiplier * std_error);
            }
        }

        (middle, upper, lower)
    }

    /// Calculate just the linear regression line.
    pub fn linear_regression(&self, close: &[f64]) -> Vec<f64> {
        let (middle, _, _) = self.calculate(close);
        middle
    }

    /// Calculate just the standard error values.
    pub fn standard_error(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();

        if n < self.period || self.period < 2 {
            return vec![f64::NAN; n];
        }

        let mut errors = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &close[start..=i];

            let (_, std_error) = self.linear_regression_with_error(window);
            errors.push(std_error);
        }

        errors
    }
}

impl Default for StandardErrorBands {
    fn default() -> Self {
        Self::default_params()
    }
}

impl TechnicalIndicator for StandardErrorBands {
    fn name(&self) -> &str {
        "StandardErrorBands"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let (middle, upper, lower) = self.calculate(&data.close);
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
    fn test_std_error_bands() {
        let seb = StandardErrorBands::new(10, 2.0);
        // Use data with some variance (not perfectly linear)
        let close: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 * 0.5 + (i as f64 * 0.5).sin() * 2.0).collect();

        let (middle, upper, lower) = seb.calculate(&close);

        assert_eq!(middle.len(), 30);
        assert_eq!(upper.len(), 30);
        assert_eq!(lower.len(), 30);

        // Check bands after warmup
        for i in 10..30 {
            if !middle[i].is_nan() {
                assert!(upper[i] >= middle[i], "Upper should be >= middle");
                assert!(lower[i] <= middle[i], "Lower should be <= middle");
            }
        }
    }

    #[test]
    fn test_std_error_bands_trending() {
        // Perfect linear trend should have very small standard error
        let seb = StandardErrorBands::new(10, 2.0);
        let close: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();

        let (middle, upper, lower) = seb.calculate(&close);

        // For perfect linear data, standard error should be very small
        // So upper and lower should be very close to middle
        for i in 10..20 {
            if !middle[i].is_nan() {
                let band_width = upper[i] - lower[i];
                assert!(band_width < 0.001, "Band width should be very small for linear data");
            }
        }
    }

    #[test]
    fn test_std_error_bands_default() {
        let seb = StandardErrorBands::default();
        assert_eq!(seb.period, 21);
        assert!((seb.multiplier - 2.0).abs() < 1e-10);
    }
}
