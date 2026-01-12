//! Realized Volatility implementation.
//!
//! Calculates realized volatility from intraday returns.

use crate::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};

/// Realized Volatility (RV).
///
/// Realized volatility is calculated as the square root of the sum of squared
/// returns over a given period. This is the standard ex-post measure of volatility.
///
/// Formula:
/// RV = sqrt(sum(return_i^2)) for returns over the period
///
/// This implementation uses close-to-close returns. For high-frequency data,
/// use sum of squared intraday returns for more accuracy.
#[derive(Debug, Clone)]
pub struct RealizedVolatility {
    /// Lookback period.
    period: usize,
    /// Number of trading days per year for annualization.
    trading_days: f64,
    /// Whether to annualize the volatility.
    annualize: bool,
    /// Whether to use log returns (true) or simple returns (false).
    use_log_returns: bool,
}

impl RealizedVolatility {
    /// Create a new Realized Volatility indicator.
    ///
    /// # Arguments
    /// * `period` - Lookback period
    pub fn new(period: usize) -> Self {
        Self {
            period,
            trading_days: 252.0,
            annualize: true,
            use_log_returns: true,
        }
    }

    /// Create without annualization.
    pub fn without_annualization(period: usize) -> Self {
        Self {
            period,
            trading_days: 252.0,
            annualize: false,
            use_log_returns: true,
        }
    }

    /// Create with simple returns instead of log returns.
    pub fn with_simple_returns(period: usize) -> Self {
        Self {
            period,
            trading_days: 252.0,
            annualize: true,
            use_log_returns: false,
        }
    }

    /// Create with custom trading days.
    pub fn with_trading_days(period: usize, trading_days: f64) -> Self {
        Self {
            period,
            trading_days,
            annualize: true,
            use_log_returns: true,
        }
    }

    /// Calculate returns from price data.
    fn calculate_returns(&self, data: &[f64]) -> Vec<f64> {
        let mut returns = Vec::with_capacity(data.len().saturating_sub(1));

        for i in 1..data.len() {
            if data[i - 1] > 0.0 && data[i] > 0.0 {
                let ret = if self.use_log_returns {
                    (data[i] / data[i - 1]).ln()
                } else {
                    (data[i] - data[i - 1]) / data[i - 1]
                };
                returns.push(ret);
            } else {
                returns.push(f64::NAN);
            }
        }

        returns
    }

    /// Calculate Realized Volatility values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period + 1 || self.period == 0 {
            return vec![f64::NAN; n];
        }

        let returns = self.calculate_returns(data);
        let returns_len = returns.len();

        let mut result = vec![f64::NAN; self.period];

        for i in (self.period - 1)..returns_len {
            let start = i + 1 - self.period;
            let window = &returns[start..=i];

            // Filter out NaN values
            let valid_returns: Vec<f64> = window.iter()
                .filter(|x| !x.is_nan())
                .copied()
                .collect();

            if valid_returns.len() < self.period {
                result.push(f64::NAN);
                continue;
            }

            // Sum of squared returns
            let sum_sq: f64 = valid_returns.iter().map(|x| x.powi(2)).sum();

            // Realized volatility (not dividing by n for realized variance)
            let volatility = sum_sq.sqrt();

            // Annualize if requested
            // For realized volatility, annualization factor is sqrt(trading_days / period)
            let final_vol = if self.annualize {
                volatility * (self.trading_days / self.period as f64).sqrt()
            } else {
                volatility
            };

            result.push(final_vol);
        }

        result
    }
}

impl TechnicalIndicator for RealizedVolatility {
    fn name(&self) -> &str {
        "RealizedVolatility"
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
    fn test_realized_volatility() {
        let rv = RealizedVolatility::new(20);

        // Generate sample price data
        let data: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0 + (i as f64 * 0.05))
            .collect();

        let result = rv.calculate(&data);

        assert_eq!(result.len(), 50);

        // First 20 values should be NaN
        for i in 0..20 {
            assert!(result[i].is_nan());
        }

        // Volatility should be positive
        for i in 20..50 {
            assert!(result[i] > 0.0);
        }
    }

    #[test]
    fn test_simple_vs_log_returns() {
        let rv_log = RealizedVolatility::new(10);
        let rv_simple = RealizedVolatility::with_simple_returns(10);

        let data: Vec<f64> = (0..30).map(|i| 100.0 + i as f64).collect();

        let log_result = rv_log.calculate(&data);
        let simple_result = rv_simple.calculate(&data);

        // Both should produce valid results
        for i in 10..30 {
            assert!(!log_result[i].is_nan());
            assert!(!simple_result[i].is_nan());
        }

        // For small returns, log and simple should be similar
        for i in 10..30 {
            let diff = (log_result[i] - simple_result[i]).abs();
            assert!(diff < 0.1, "Results should be similar for small returns");
        }
    }
}
