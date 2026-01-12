//! Parkinson Volatility implementation.
//!
//! Volatility estimator using high-low range.

use indicator_spi::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};

/// Parkinson Volatility.
///
/// A volatility estimator that uses the high-low range, which is more efficient
/// than close-to-close volatility because it captures intraday volatility.
///
/// Formula:
/// Parkinson = sqrt((1 / (4 * n * ln(2))) * sum(ln(high/low)^2))
///
/// The Parkinson estimator assumes no drift and no opening jumps, making it
/// approximately 5x more efficient than close-to-close volatility.
#[derive(Debug, Clone)]
pub struct ParkinsonVolatility {
    /// Lookback period.
    period: usize,
    /// Number of trading days per year for annualization.
    trading_days: f64,
    /// Whether to annualize the volatility.
    annualize: bool,
}

impl ParkinsonVolatility {
    /// Create a new Parkinson Volatility indicator.
    ///
    /// # Arguments
    /// * `period` - Lookback period
    pub fn new(period: usize) -> Self {
        Self {
            period,
            trading_days: 252.0,
            annualize: true,
        }
    }

    /// Create without annualization.
    pub fn without_annualization(period: usize) -> Self {
        Self {
            period,
            trading_days: 252.0,
            annualize: false,
        }
    }

    /// Create with custom trading days.
    pub fn with_trading_days(period: usize, trading_days: f64) -> Self {
        Self {
            period,
            trading_days,
            annualize: true,
        }
    }

    /// Calculate Parkinson Volatility values.
    pub fn calculate(&self, high: &[f64], low: &[f64]) -> Vec<f64> {
        let n = high.len();
        if n < self.period || self.period == 0 {
            return vec![f64::NAN; n];
        }

        let ln2 = 2.0_f64.ln();
        let constant = 1.0 / (4.0 * ln2);

        // Calculate squared log of high/low ratio for each bar
        let log_hl_sq: Vec<f64> = high.iter()
            .zip(low.iter())
            .map(|(&h, &l)| {
                if h > 0.0 && l > 0.0 && h >= l {
                    (h / l).ln().powi(2)
                } else {
                    f64::NAN
                }
            })
            .collect();

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &log_hl_sq[start..=i];

            // Check for NaN values
            let valid_values: Vec<f64> = window.iter()
                .filter(|x| !x.is_nan())
                .copied()
                .collect();

            if valid_values.len() < self.period {
                result.push(f64::NAN);
                continue;
            }

            // Calculate Parkinson variance
            let sum: f64 = valid_values.iter().sum();
            let variance = constant * sum / self.period as f64;
            let volatility = variance.sqrt();

            // Annualize if requested
            let final_vol = if self.annualize {
                volatility * self.trading_days.sqrt()
            } else {
                volatility
            };

            result.push(final_vol);
        }

        result
    }
}

impl TechnicalIndicator for ParkinsonVolatility {
    fn name(&self) -> &str {
        "ParkinsonVolatility"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.high.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.high.len(),
            });
        }

        let values = self.calculate(&data.high, &data.low);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.period
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parkinson_volatility() {
        let pv = ParkinsonVolatility::new(20);

        // Generate sample data with typical H-L range
        let high: Vec<f64> = (0..50)
            .map(|i| 102.0 + (i as f64 * 0.1).sin() * 2.0)
            .collect();
        let low: Vec<f64> = (0..50)
            .map(|i| 98.0 + (i as f64 * 0.1).sin() * 2.0)
            .collect();

        let result = pv.calculate(&high, &low);

        assert_eq!(result.len(), 50);

        // First 19 values should be NaN
        for i in 0..19 {
            assert!(result[i].is_nan());
        }

        // Volatility should be positive
        for i in 19..50 {
            assert!(result[i] > 0.0);
        }
    }

    #[test]
    fn test_parkinson_no_annualization() {
        let pv_ann = ParkinsonVolatility::new(10);
        let pv_raw = ParkinsonVolatility::without_annualization(10);

        let high: Vec<f64> = (0..30).map(|i| 105.0 + i as f64 * 0.1).collect();
        let low: Vec<f64> = (0..30).map(|i| 95.0 + i as f64 * 0.1).collect();

        let ann = pv_ann.calculate(&high, &low);
        let raw = pv_raw.calculate(&high, &low);

        for i in 10..30 {
            if !ann[i].is_nan() && !raw[i].is_nan() {
                let ratio = ann[i] / raw[i];
                assert!((ratio - 252.0_f64.sqrt()).abs() < 0.01);
            }
        }
    }
}
