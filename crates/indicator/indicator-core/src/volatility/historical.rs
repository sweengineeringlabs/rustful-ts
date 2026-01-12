//! Historical Volatility (HV) implementation.
//!
//! Historical volatility measures the dispersion of returns over a specified period.

use crate::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};

/// Historical Volatility (HV).
///
/// Calculates the standard deviation of logarithmic returns, typically annualized.
/// This is the most common measure of realized volatility used in options pricing.
///
/// Formula:
/// 1. Calculate log returns: ln(close[i] / close[i-1])
/// 2. Calculate standard deviation of returns over period
/// 3. Annualize: multiply by sqrt(trading_days_per_year)
#[derive(Debug, Clone)]
pub struct HistoricalVolatility {
    /// Lookback period for volatility calculation.
    period: usize,
    /// Number of trading days per year for annualization (typically 252).
    trading_days: f64,
    /// Whether to annualize the volatility.
    annualize: bool,
}

impl HistoricalVolatility {
    /// Create a new Historical Volatility indicator.
    ///
    /// # Arguments
    /// * `period` - Lookback period (commonly 20, 30, or 60)
    pub fn new(period: usize) -> Self {
        Self {
            period,
            trading_days: 252.0,
            annualize: true,
        }
    }

    /// Create with custom trading days for annualization.
    pub fn with_trading_days(period: usize, trading_days: f64) -> Self {
        Self {
            period,
            trading_days,
            annualize: true,
        }
    }

    /// Create without annualization (returns raw standard deviation).
    pub fn without_annualization(period: usize) -> Self {
        Self {
            period,
            trading_days: 252.0,
            annualize: false,
        }
    }

    /// Calculate log returns from price data.
    fn log_returns(data: &[f64]) -> Vec<f64> {
        let mut returns = Vec::with_capacity(data.len().saturating_sub(1));
        for i in 1..data.len() {
            if data[i - 1] > 0.0 && data[i] > 0.0 {
                returns.push((data[i] / data[i - 1]).ln());
            } else {
                returns.push(f64::NAN);
            }
        }
        returns
    }

    /// Calculate historical volatility values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period + 1 || self.period == 0 {
            return vec![f64::NAN; n];
        }

        let returns = Self::log_returns(data);
        let returns_len = returns.len();

        // Need period returns for first valid value
        let mut result = vec![f64::NAN; self.period];

        for i in (self.period - 1)..returns_len {
            let start = i + 1 - self.period;
            let window = &returns[start..=i];

            // Check for NaN values in window
            let valid_returns: Vec<f64> = window.iter()
                .filter(|x| !x.is_nan())
                .copied()
                .collect();

            if valid_returns.len() < self.period {
                result.push(f64::NAN);
                continue;
            }

            // Calculate mean
            let mean: f64 = valid_returns.iter().sum::<f64>() / valid_returns.len() as f64;

            // Calculate variance (using population variance for consistency)
            let variance: f64 = valid_returns.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / valid_returns.len() as f64;

            let std_dev = variance.sqrt();

            // Annualize if requested
            let volatility = if self.annualize {
                std_dev * self.trading_days.sqrt()
            } else {
                std_dev
            };

            result.push(volatility);
        }

        result
    }
}

impl TechnicalIndicator for HistoricalVolatility {
    fn name(&self) -> &str {
        "HistoricalVolatility"
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
    fn test_historical_volatility() {
        let hv = HistoricalVolatility::new(20);
        // Generate sample price data with some volatility
        let data: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0 + (i as f64 * 0.05))
            .collect();

        let result = hv.calculate(&data);

        assert_eq!(result.len(), 50);

        // First 20 values should be NaN
        for i in 0..20 {
            assert!(result[i].is_nan());
        }

        // Volatility should be positive
        for i in 20..50 {
            assert!(result[i] > 0.0, "Volatility should be positive at index {}", i);
        }
    }

    #[test]
    fn test_log_returns() {
        let data = vec![100.0, 105.0, 103.0, 108.0];
        let returns = HistoricalVolatility::log_returns(&data);

        assert_eq!(returns.len(), 3);
        assert!((returns[0] - (105.0_f64 / 100.0).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_non_annualized() {
        let hv_ann = HistoricalVolatility::new(10);
        let hv_raw = HistoricalVolatility::without_annualization(10);

        let data: Vec<f64> = (0..30).map(|i| 100.0 + i as f64).collect();

        let ann_result = hv_ann.calculate(&data);
        let raw_result = hv_raw.calculate(&data);

        // Annualized should be sqrt(252) times larger
        for i in 10..30 {
            if !ann_result[i].is_nan() && !raw_result[i].is_nan() {
                let ratio = ann_result[i] / raw_result[i];
                assert!((ratio - 252.0_f64.sqrt()).abs() < 0.01);
            }
        }
    }
}
