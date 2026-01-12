//! Garman-Klass Volatility implementation.
//!
//! Enhanced volatility estimator using OHLC data.

use crate::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};

/// Garman-Klass Volatility.
///
/// An extension of the Parkinson estimator that also incorporates open and close prices.
/// This estimator is approximately 8x more efficient than close-to-close volatility.
///
/// Formula:
/// GK = sqrt((1/n) * sum(0.5 * ln(H/L)^2 - (2*ln(2) - 1) * ln(C/O)^2))
///
/// The Garman-Klass estimator assumes no drift and accounts for overnight jumps
/// through the open-close component.
#[derive(Debug, Clone)]
pub struct GarmanKlassVolatility {
    /// Lookback period.
    period: usize,
    /// Number of trading days per year for annualization.
    trading_days: f64,
    /// Whether to annualize the volatility.
    annualize: bool,
}

impl GarmanKlassVolatility {
    /// Create a new Garman-Klass Volatility indicator.
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

    /// Calculate Garman-Klass Volatility values.
    pub fn calculate(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = high.len();
        if n < self.period || self.period == 0 {
            return vec![f64::NAN; n];
        }

        let ln2 = 2.0_f64.ln();
        let co_coeff = 2.0 * ln2 - 1.0;

        // Calculate GK component for each bar
        let gk_component: Vec<f64> = open.iter()
            .zip(high.iter())
            .zip(low.iter())
            .zip(close.iter())
            .map(|(((&o, &h), &l), &c)| {
                if o > 0.0 && h > 0.0 && l > 0.0 && c > 0.0 && h >= l {
                    let hl_term = 0.5 * (h / l).ln().powi(2);
                    let co_term = co_coeff * (c / o).ln().powi(2);
                    hl_term - co_term
                } else {
                    f64::NAN
                }
            })
            .collect();

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &gk_component[start..=i];

            // Check for NaN values
            let valid_values: Vec<f64> = window.iter()
                .filter(|x| !x.is_nan())
                .copied()
                .collect();

            if valid_values.len() < self.period {
                result.push(f64::NAN);
                continue;
            }

            // Calculate Garman-Klass variance
            let sum: f64 = valid_values.iter().sum();
            let variance = sum / self.period as f64;

            // Handle negative variance (can happen with extreme data)
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

impl TechnicalIndicator for GarmanKlassVolatility {
    fn name(&self) -> &str {
        "GarmanKlassVolatility"
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
    fn test_garman_klass_volatility() {
        let gk = GarmanKlassVolatility::new(20);

        // Generate sample OHLC data
        let open: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64 * 0.1).sin() * 2.0)
            .collect();
        let high: Vec<f64> = (0..50)
            .map(|i| 102.0 + (i as f64 * 0.1).sin() * 2.0)
            .collect();
        let low: Vec<f64> = (0..50)
            .map(|i| 98.0 + (i as f64 * 0.1).sin() * 2.0)
            .collect();
        let close: Vec<f64> = (0..50)
            .map(|i| 100.5 + (i as f64 * 0.1).sin() * 2.0)
            .collect();

        let result = gk.calculate(&open, &high, &low, &close);

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
    fn test_garman_klass_no_annualization() {
        let gk_ann = GarmanKlassVolatility::new(10);
        let gk_raw = GarmanKlassVolatility::without_annualization(10);

        let open: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 * 0.1).collect();
        let high: Vec<f64> = (0..30).map(|i| 105.0 + i as f64 * 0.1).collect();
        let low: Vec<f64> = (0..30).map(|i| 95.0 + i as f64 * 0.1).collect();
        let close: Vec<f64> = (0..30).map(|i| 101.0 + i as f64 * 0.1).collect();

        let ann = gk_ann.calculate(&open, &high, &low, &close);
        let raw = gk_raw.calculate(&open, &high, &low, &close);

        for i in 10..30 {
            if !ann[i].is_nan() && !raw[i].is_nan() && raw[i] > 0.0 {
                let ratio = ann[i] / raw[i];
                assert!((ratio - 252.0_f64.sqrt()).abs() < 0.01);
            }
        }
    }
}
