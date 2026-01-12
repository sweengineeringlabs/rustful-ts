//! Yang-Zhang Volatility implementation.
//!
//! The most comprehensive OHLC volatility estimator.

use indicator_spi::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};

/// Yang-Zhang Volatility.
///
/// The most comprehensive OHLC volatility estimator, combining overnight volatility,
/// open-to-close volatility, and Rogers-Satchell volatility. It handles both drift
/// and opening jumps.
///
/// Formula:
/// YZ = sqrt(Vo + k * Vc + (1-k) * Vrs)
///
/// Where:
/// - Vo = overnight (close-to-open) variance
/// - Vc = open-to-close variance
/// - Vrs = Rogers-Satchell variance
/// - k = 0.34 / (1.34 + (n+1)/(n-1))
#[derive(Debug, Clone)]
pub struct YangZhangVolatility {
    /// Lookback period.
    period: usize,
    /// Number of trading days per year for annualization.
    trading_days: f64,
    /// Whether to annualize the volatility.
    annualize: bool,
}

impl YangZhangVolatility {
    /// Create a new Yang-Zhang Volatility indicator.
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

    /// Calculate the k parameter for Yang-Zhang formula.
    fn k_parameter(n: usize) -> f64 {
        0.34 / (1.34 + (n as f64 + 1.0) / (n as f64 - 1.0))
    }

    /// Calculate Yang-Zhang Volatility values.
    pub fn calculate(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = high.len();
        // Need at least period + 1 for overnight returns
        if n < self.period + 1 || self.period < 2 {
            return vec![f64::NAN; n];
        }

        let k = Self::k_parameter(self.period);

        // Calculate overnight returns (close[i-1] to open[i])
        let mut overnight_returns = vec![f64::NAN];
        for i in 1..n {
            if close[i - 1] > 0.0 && open[i] > 0.0 {
                overnight_returns.push((open[i] / close[i - 1]).ln());
            } else {
                overnight_returns.push(f64::NAN);
            }
        }

        // Calculate open-to-close returns
        let open_close_returns: Vec<f64> = open.iter()
            .zip(close.iter())
            .map(|(&o, &c)| {
                if o > 0.0 && c > 0.0 {
                    (c / o).ln()
                } else {
                    f64::NAN
                }
            })
            .collect();

        // Calculate Rogers-Satchell components
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

        let mut result = vec![f64::NAN; self.period];

        for i in self.period..n {
            let start = i + 1 - self.period;

            // Get windows for each component
            let on_window = &overnight_returns[start..=i];
            let oc_window = &open_close_returns[start..=i];
            let rs_window = &rs_component[start..=i];

            // Filter out NaN values
            let on_valid: Vec<f64> = on_window.iter().filter(|x| !x.is_nan()).copied().collect();
            let oc_valid: Vec<f64> = oc_window.iter().filter(|x| !x.is_nan()).copied().collect();
            let rs_valid: Vec<f64> = rs_window.iter().filter(|x| !x.is_nan()).copied().collect();

            if on_valid.len() < self.period - 1
                || oc_valid.len() < self.period
                || rs_valid.len() < self.period {
                result.push(f64::NAN);
                continue;
            }

            // Calculate overnight variance
            let on_mean: f64 = on_valid.iter().sum::<f64>() / on_valid.len() as f64;
            let overnight_var: f64 = on_valid.iter()
                .map(|x| (x - on_mean).powi(2))
                .sum::<f64>() / (on_valid.len() - 1) as f64;

            // Calculate open-to-close variance
            let oc_mean: f64 = oc_valid.iter().sum::<f64>() / oc_valid.len() as f64;
            let open_close_var: f64 = oc_valid.iter()
                .map(|x| (x - oc_mean).powi(2))
                .sum::<f64>() / (oc_valid.len() - 1) as f64;

            // Calculate Rogers-Satchell variance (mean of components)
            let rs_var: f64 = rs_valid.iter().sum::<f64>() / rs_valid.len() as f64;

            // Yang-Zhang formula
            let yz_variance = overnight_var + k * open_close_var + (1.0 - k) * rs_var;

            let volatility = if yz_variance >= 0.0 {
                yz_variance.sqrt()
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

impl TechnicalIndicator for YangZhangVolatility {
    fn name(&self) -> &str {
        "YangZhangVolatility"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.high.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.high.len(),
            });
        }

        let values = self.calculate(&data.open, &data.high, &data.low, &data.close);
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
    fn test_yang_zhang_volatility() {
        let yz = YangZhangVolatility::new(20);

        // Generate sample OHLC data
        let open: Vec<f64> = (0..60)
            .map(|i| 100.0 + (i as f64 * 0.1).sin() * 2.0)
            .collect();
        let high: Vec<f64> = (0..60)
            .map(|i| 102.0 + (i as f64 * 0.1).sin() * 2.0)
            .collect();
        let low: Vec<f64> = (0..60)
            .map(|i| 98.0 + (i as f64 * 0.1).sin() * 2.0)
            .collect();
        let close: Vec<f64> = (0..60)
            .map(|i| 100.5 + (i as f64 * 0.1).sin() * 2.0)
            .collect();

        let result = yz.calculate(&open, &high, &low, &close);

        assert_eq!(result.len(), 60);

        // First 20 values should be NaN
        for i in 0..20 {
            assert!(result[i].is_nan());
        }

        // Volatility should be positive where valid
        for i in 20..60 {
            if !result[i].is_nan() {
                assert!(result[i] > 0.0);
            }
        }
    }

    #[test]
    fn test_k_parameter() {
        // k should be between 0 and 1
        let k = YangZhangVolatility::k_parameter(20);
        assert!(k > 0.0 && k < 1.0);

        // k increases as n increases (converges toward ~0.34/2.34 ≈ 0.145)
        let k10 = YangZhangVolatility::k_parameter(10);
        let k30 = YangZhangVolatility::k_parameter(30);
        assert!(k10 < k30, "k should increase with n: k10={} k30={}", k10, k30);
    }
}
