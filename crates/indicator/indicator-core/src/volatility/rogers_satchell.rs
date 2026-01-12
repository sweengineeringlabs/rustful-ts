//! Rogers-Satchell Volatility implementation.
//!
//! Volatility estimator that handles drift in prices.

use crate::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};

/// Rogers-Satchell Volatility.
///
/// A volatility estimator that accounts for drift in prices, unlike Parkinson
/// or Garman-Klass. This makes it more suitable for trending markets.
///
/// Formula:
/// RS = sqrt((1/n) * sum(ln(H/C) * ln(H/O) + ln(L/C) * ln(L/O)))
///
/// The Rogers-Satchell estimator provides unbiased estimates even when
/// prices have a non-zero drift (trend).
#[derive(Debug, Clone)]
pub struct RogersSatchellVolatility {
    /// Lookback period.
    period: usize,
    /// Number of trading days per year for annualization.
    trading_days: f64,
    /// Whether to annualize the volatility.
    annualize: bool,
}

impl RogersSatchellVolatility {
    /// Create a new Rogers-Satchell Volatility indicator.
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

    /// Calculate Rogers-Satchell Volatility values.
    pub fn calculate(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = high.len();
        if n < self.period || self.period == 0 {
            return vec![f64::NAN; n];
        }

        // Calculate RS component for each bar
        let rs_component: Vec<f64> = open.iter()
            .zip(high.iter())
            .zip(low.iter())
            .zip(close.iter())
            .map(|(((&o, &h), &l), &c)| {
                if o > 0.0 && h > 0.0 && l > 0.0 && c > 0.0 && h >= l {
                    let ln_hc = (h / c).ln();
                    let ln_ho = (h / o).ln();
                    let ln_lc = (l / c).ln();
                    let ln_lo = (l / o).ln();
                    ln_hc * ln_ho + ln_lc * ln_lo
                } else {
                    f64::NAN
                }
            })
            .collect();

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &rs_component[start..=i];

            // Check for NaN values
            let valid_values: Vec<f64> = window.iter()
                .filter(|x| !x.is_nan())
                .copied()
                .collect();

            if valid_values.len() < self.period {
                result.push(f64::NAN);
                continue;
            }

            // Calculate Rogers-Satchell variance
            let sum: f64 = valid_values.iter().sum();
            let variance = sum / self.period as f64;

            // Handle negative variance
            let volatility = if variance >= 0.0 {
                variance.sqrt()
            } else {
                f64::NAN
            };

            // Annualize if requested
            let final_vol = if self.annualize && !volatility.is_nan() {
                volatility * self.trading_days.sqrt()
            } else {
                volatility
            };

            result.push(final_vol);
        }

        result
    }
}

impl TechnicalIndicator for RogersSatchellVolatility {
    fn name(&self) -> &str {
        "RogersSatchellVolatility"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.high.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.high.len(),
            });
        }

        let values = self.calculate(&data.open, &data.high, &data.low, &data.close);
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
    fn test_rogers_satchell_volatility() {
        let rs = RogersSatchellVolatility::new(20);

        // Generate sample OHLC data with a trend
        let open: Vec<f64> = (0..50)
            .map(|i| 100.0 + i as f64 * 0.5 + (i as f64 * 0.1).sin() * 2.0)
            .collect();
        let high: Vec<f64> = (0..50)
            .map(|i| 102.0 + i as f64 * 0.5 + (i as f64 * 0.1).sin() * 2.0)
            .collect();
        let low: Vec<f64> = (0..50)
            .map(|i| 98.0 + i as f64 * 0.5 + (i as f64 * 0.1).sin() * 2.0)
            .collect();
        let close: Vec<f64> = (0..50)
            .map(|i| 100.5 + i as f64 * 0.5 + (i as f64 * 0.1).sin() * 2.0)
            .collect();

        let result = rs.calculate(&open, &high, &low, &close);

        assert_eq!(result.len(), 50);

        // First 19 values should be NaN
        for i in 0..19 {
            assert!(result[i].is_nan());
        }

        // Volatility should be positive where valid
        for i in 19..50 {
            if !result[i].is_nan() {
                assert!(result[i] > 0.0);
            }
        }
    }

    #[test]
    fn test_rogers_satchell_no_annualization() {
        let rs_ann = RogersSatchellVolatility::new(10);
        let rs_raw = RogersSatchellVolatility::without_annualization(10);

        let open: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 * 0.1).collect();
        let high: Vec<f64> = (0..30).map(|i| 105.0 + i as f64 * 0.1).collect();
        let low: Vec<f64> = (0..30).map(|i| 95.0 + i as f64 * 0.1).collect();
        let close: Vec<f64> = (0..30).map(|i| 101.0 + i as f64 * 0.1).collect();

        let ann = rs_ann.calculate(&open, &high, &low, &close);
        let raw = rs_raw.calculate(&open, &high, &low, &close);

        for i in 10..30 {
            if !ann[i].is_nan() && !raw[i].is_nan() && raw[i] > 0.0 {
                let ratio = ann[i] / raw[i];
                assert!((ratio - 252.0_f64.sqrt()).abs() < 0.01);
            }
        }
    }
}
