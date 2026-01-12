//! Correlation implementation.

use crate::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};

/// Correlation (Pearson Correlation Coefficient).
///
/// Measures the linear correlation between two series.
/// Values range from -1 (perfect negative) to +1 (perfect positive).
/// 0 indicates no linear correlation.
///
/// When used as a TechnicalIndicator, it computes autocorrelation
/// with a lag of 1 by default, or correlation between close and volume.
#[derive(Debug, Clone)]
pub struct Correlation {
    period: usize,
    /// Mode: true for close-volume correlation, false for price autocorrelation
    close_volume: bool,
}

impl Correlation {
    /// Create a new Correlation indicator (close vs volume).
    pub fn new(period: usize) -> Self {
        Self {
            period,
            close_volume: true,
        }
    }

    /// Create correlation between close price and volume.
    pub fn close_volume(period: usize) -> Self {
        Self {
            period,
            close_volume: true,
        }
    }

    /// Create price autocorrelation (lag 1).
    pub fn price_autocorr(period: usize) -> Self {
        Self {
            period,
            close_volume: false,
        }
    }

    /// Calculate Pearson correlation between two series.
    fn pearson(x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return f64::NAN;
        }

        let n = x.len() as f64;

        let mean_x: f64 = x.iter().sum::<f64>() / n;
        let mean_y: f64 = y.iter().sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;

        for (xi, yi) in x.iter().zip(y.iter()) {
            let dx = xi - mean_x;
            let dy = yi - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        let denominator = (var_x * var_y).sqrt();
        if denominator.abs() < 1e-10 {
            0.0 // No variation in one or both series
        } else {
            cov / denominator
        }
    }

    /// Calculate correlation between two arbitrary series.
    pub fn calculate_between(&self, series1: &[f64], series2: &[f64]) -> Vec<f64> {
        let n = series1.len().min(series2.len());
        if n < self.period || self.period == 0 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window1 = &series1[start..=i];
            let window2 = &series2[start..=i];
            let corr = Self::pearson(window1, window2);
            result.push(corr);
        }

        result
    }

    /// Calculate price autocorrelation with lag.
    pub fn calculate_autocorr(&self, data: &[f64], lag: usize) -> Vec<f64> {
        let n = data.len();
        if n < self.period + lag || self.period == 0 || lag == 0 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period + lag - 1];

        for i in (self.period + lag - 1)..n {
            let start = i + 1 - self.period;
            let series_current = &data[start..=i];
            let series_lagged = &data[(start - lag)..=(i - lag)];
            let corr = Self::pearson(series_current, series_lagged);
            result.push(corr);
        }

        result
    }
}

impl TechnicalIndicator for Correlation {
    fn name(&self) -> &str {
        "Correlation"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let required = if self.close_volume {
            self.period
        } else {
            self.period + 1 // Need one extra for lag
        };

        if data.close.len() < required {
            return Err(IndicatorError::InsufficientData {
                required,
                got: data.close.len(),
            });
        }

        let values = if self.close_volume {
            self.calculate_between(&data.close, &data.volume)
        } else {
            self.calculate_autocorr(&data.close, 1)
        };

        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        if self.close_volume {
            self.period
        } else {
            self.period + 1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correlation_perfect_positive() {
        let corr = Correlation::new(5);
        // Perfectly correlated series
        let series1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let series2 = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0];
        let result = corr.calculate_between(&series1, &series2);

        for i in 4..result.len() {
            assert!((result[i] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_correlation_perfect_negative() {
        let corr = Correlation::new(5);
        // Perfectly negatively correlated series
        let series1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let series2 = vec![7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let result = corr.calculate_between(&series1, &series2);

        for i in 4..result.len() {
            assert!((result[i] - (-1.0)).abs() < 1e-10);
        }
    }

    #[test]
    fn test_autocorrelation() {
        let corr = Correlation::price_autocorr(5);
        // Trending data should have positive autocorrelation
        let data: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        let result = corr.calculate_autocorr(&data, 1);

        for i in 5..result.len() {
            assert!(result[i] > 0.9); // Should be highly autocorrelated
        }
    }
}
