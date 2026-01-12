//! Omega Ratio implementation.
//!
//! Measures the probability-weighted ratio of gains versus losses.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result, TechnicalIndicator,
};

/// Omega Ratio indicator.
///
/// Considers the entire distribution of returns, not just mean and variance.
/// Measures the probability-weighted sum of gains versus losses relative
/// to a threshold return.
///
/// Formula: Sum of gains above threshold / Sum of losses below threshold
///
/// - Omega > 1: More gains than losses relative to threshold
/// - Omega = 1: Equal gains and losses
/// - Omega < 1: More losses than gains
///
/// Higher values indicate better performance.
#[derive(Debug, Clone)]
pub struct OmegaRatio {
    /// Rolling window period for calculation.
    period: usize,
    /// Threshold return (minimum acceptable return).
    threshold: f64,
}

impl OmegaRatio {
    /// Create a new Omega Ratio indicator with zero threshold.
    ///
    /// # Arguments
    /// * `period` - Rolling window period for calculation
    pub fn new(period: usize) -> Self {
        Self {
            period,
            threshold: 0.0,
        }
    }

    /// Create a new Omega Ratio with custom threshold.
    ///
    /// # Arguments
    /// * `period` - Rolling window period
    /// * `threshold` - Minimum acceptable return (per period)
    pub fn with_threshold(period: usize, threshold: f64) -> Self {
        Self { period, threshold }
    }

    /// Calculate returns from price series.
    fn calculate_returns(prices: &[f64]) -> Vec<f64> {
        if prices.len() < 2 {
            return Vec::new();
        }
        prices
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect()
    }

    /// Calculate Omega Ratio for a window of returns.
    fn calculate_omega(&self, returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return f64::NAN;
        }

        let mut gains_sum = 0.0;
        let mut losses_sum = 0.0;

        for &r in returns {
            let excess = r - self.threshold;
            if excess > 0.0 {
                gains_sum += excess;
            } else {
                losses_sum += excess.abs();
            }
        }

        if losses_sum == 0.0 {
            if gains_sum > 0.0 {
                f64::INFINITY // All gains, no losses
            } else {
                f64::NAN // No movement
            }
        } else {
            gains_sum / losses_sum
        }
    }

    /// Calculate Omega Ratio values.
    pub fn calculate(&self, prices: &[f64]) -> Vec<f64> {
        let n = prices.len();
        if n < self.period + 1 {
            return vec![f64::NAN; n];
        }

        let returns = Self::calculate_returns(prices);
        let mut result = vec![f64::NAN; self.period];

        for i in (self.period - 1)..returns.len() {
            let start = i + 1 - self.period;
            let window = &returns[start..=i];
            let omega = self.calculate_omega(window);
            result.push(omega);
        }

        result
    }
}

impl TechnicalIndicator for OmegaRatio {
    fn name(&self) -> &str {
        "OmegaRatio"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_omega_basic() {
        let omega = OmegaRatio::new(20);
        // Generate prices with some volatility
        let prices: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64) * 0.1 + ((i as f64) * 0.5).sin() * 2.0)
            .collect();
        let result = omega.calculate(&prices);

        // Should have valid values after warm-up period
        assert!(!result[30].is_nan());
    }

    #[test]
    fn test_omega_uptrend() {
        let omega = OmegaRatio::new(20);
        // Strong uptrend - should have high omega
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 1.0).collect();
        let result = omega.calculate(&prices);

        // With strong uptrend and zero threshold, omega should be infinity or very high
        let last = result.last().unwrap();
        assert!(last.is_infinite() || *last > 10.0);
    }

    #[test]
    fn test_omega_downtrend() {
        let omega = OmegaRatio::new(20);
        // Strong downtrend - should have low omega
        let prices: Vec<f64> = (0..50).map(|i| 200.0 - (i as f64) * 1.0).collect();
        let result = omega.calculate(&prices);

        // With strong downtrend, omega should be 0 or very low
        let last = result.last().unwrap();
        assert!(*last < 0.1 || *last == 0.0);
    }

    #[test]
    fn test_omega_with_threshold() {
        let omega_low = OmegaRatio::with_threshold(20, 0.0);
        let omega_high = OmegaRatio::with_threshold(20, 0.005); // 0.5% per day

        let prices: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64) * 0.3)
            .collect();

        let result_low = omega_low.calculate(&prices);
        let result_high = omega_high.calculate(&prices);

        // Higher threshold should result in lower omega
        assert!(result_low[30] > result_high[30]);
    }
}
