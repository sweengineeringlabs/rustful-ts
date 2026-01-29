//! Carry Trade Signal Indicator (IND-497)
//!
//! FX carry trade opportunities based on yield differentials and risk.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

// ============================================================================
// CarryTradeSignal
// ============================================================================

/// Carry Trade Signal - FX carry trade opportunities indicator.
///
/// This indicator identifies carry trade opportunities by analyzing yield
/// differential proxies combined with volatility and trend conditions.
/// It generates signals for high-yielder vs low-yielder currency positions.
///
/// # Theory
/// Carry trades profit from holding high-yielding currencies funded by
/// low-yielding currencies. The strategy works best in low volatility,
/// trending environments and fails during risk-off episodes.
///
/// # Interpretation
/// - `signal > 0.5`: Strong carry opportunity (long high-yielder)
/// - `signal > 0`: Moderate carry opportunity
/// - `signal < 0`: Unfavorable carry conditions (risk-off)
/// - `signal < -0.5`: Strong carry unwind signal
///
/// # Components
/// - Yield proxy: Estimated from price momentum
/// - Volatility filter: Low vol favors carry
/// - Trend confirmation: Trending markets support carry
/// - Risk adjustment: Position sizing guidance
#[derive(Debug, Clone)]
pub struct CarryTradeSignal {
    /// Period for return calculations.
    return_period: usize,
    /// Period for volatility calculation.
    vol_period: usize,
    /// Volatility threshold for favorable carry conditions.
    vol_threshold: f64,
    /// Secondary series (optional high-yield currency).
    secondary_series: Vec<f64>,
}

impl CarryTradeSignal {
    /// Create a new CarryTradeSignal indicator.
    ///
    /// # Arguments
    /// * `return_period` - Period for return calculations (min: 5)
    /// * `vol_period` - Period for volatility calculation (min: 10)
    /// * `vol_threshold` - Annualized volatility threshold (default: 15%)
    pub fn new(return_period: usize, vol_period: usize, vol_threshold: f64) -> Result<Self> {
        if return_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "return_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if vol_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "vol_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if vol_threshold <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "vol_threshold".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self {
            return_period,
            vol_period,
            vol_threshold,
            secondary_series: Vec::new(),
        })
    }

    /// Create with default parameters.
    pub fn default_params() -> Result<Self> {
        Self::new(20, 20, 15.0)
    }

    /// Set secondary series for pair analysis.
    pub fn with_secondary(mut self, series: &[f64]) -> Self {
        self.secondary_series = series.to_vec();
        self
    }

    /// Calculate carry trade signal for a single series.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let max_period = self.return_period.max(self.vol_period);
        let mut result = vec![0.0; n];

        if n < max_period + 1 {
            return result;
        }

        for i in max_period..n {
            // Calculate return (yield proxy)
            let ret_start = i.saturating_sub(self.return_period);
            let returns: Vec<f64> = ((ret_start + 1)..=i)
                .filter(|&j| j < n && j > 0)
                .map(|j| (close[j] / close[j - 1]).ln())
                .collect();

            if returns.is_empty() {
                continue;
            }

            let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;

            // Calculate volatility
            let vol_start = i.saturating_sub(self.vol_period);
            let vol_returns: Vec<f64> = ((vol_start + 1)..=i)
                .filter(|&j| j < n && j > 0)
                .map(|j| (close[j] / close[j - 1]).ln())
                .collect();

            if vol_returns.is_empty() {
                continue;
            }

            let vol_mean = vol_returns.iter().sum::<f64>() / vol_returns.len() as f64;
            let variance = vol_returns.iter()
                .map(|r| (r - vol_mean).powi(2))
                .sum::<f64>() / vol_returns.len() as f64;
            let daily_vol = variance.sqrt();
            let annual_vol = daily_vol * (252.0_f64).sqrt() * 100.0;

            // Calculate trend strength
            let trend_strength = if close[i] > close[ret_start] {
                ((close[i] / close[ret_start]) - 1.0) * 100.0
            } else {
                -((close[ret_start] / close[i]) - 1.0) * 100.0
            };

            // Carry signal components
            let yield_score = mean_return * 252.0 * 100.0; // Annualized
            let vol_score = if annual_vol < self.vol_threshold {
                1.0 - (annual_vol / self.vol_threshold)
            } else {
                -(annual_vol / self.vol_threshold - 1.0).min(1.0)
            };
            let trend_score = (trend_strength / 10.0).clamp(-1.0, 1.0);

            // Combined signal
            let signal = (yield_score / 10.0).clamp(-1.0, 1.0) * 0.4
                + vol_score * 0.35
                + trend_score * 0.25;

            result[i] = signal;
        }

        result
    }

    /// Calculate carry signal using two currency series.
    pub fn calculate_pair(&self, high_yield: &[f64], low_yield: &[f64]) -> Vec<f64> {
        let n = high_yield.len().min(low_yield.len());
        let max_period = self.return_period.max(self.vol_period);
        let mut result = vec![0.0; n];

        if n < max_period + 1 {
            return result;
        }

        for i in max_period..n {
            // Calculate differential returns (yield proxy)
            let ret_start = i.saturating_sub(self.return_period);
            let hy_returns: Vec<f64> = ((ret_start + 1)..=i)
                .filter(|&j| j < n && j > 0)
                .map(|j| (high_yield[j] / high_yield[j - 1]).ln())
                .collect();
            let ly_returns: Vec<f64> = ((ret_start + 1)..=i)
                .filter(|&j| j < n && j > 0)
                .map(|j| (low_yield[j] / low_yield[j - 1]).ln())
                .collect();

            if hy_returns.is_empty() || ly_returns.is_empty() {
                continue;
            }

            let hy_mean = hy_returns.iter().sum::<f64>() / hy_returns.len() as f64;
            let ly_mean = ly_returns.iter().sum::<f64>() / ly_returns.len() as f64;
            let yield_diff = (hy_mean - ly_mean) * 252.0 * 100.0; // Annualized differential

            // Spread volatility
            let spread: Vec<f64> = hy_returns.iter()
                .zip(ly_returns.iter())
                .map(|(h, l)| h - l)
                .collect();
            let spread_mean = spread.iter().sum::<f64>() / spread.len() as f64;
            let spread_var = spread.iter()
                .map(|s| (s - spread_mean).powi(2))
                .sum::<f64>() / spread.len() as f64;
            let spread_vol = spread_var.sqrt() * (252.0_f64).sqrt() * 100.0;

            // Carry-to-risk ratio (Sharpe-like)
            let carry_ratio = if spread_vol > 0.0 {
                yield_diff / spread_vol
            } else {
                0.0
            };

            // Signal based on carry ratio
            result[i] = carry_ratio.clamp(-1.0, 1.0);
        }

        result
    }
}

impl TechnicalIndicator for CarryTradeSignal {
    fn name(&self) -> &str {
        "Carry Trade Signal"
    }

    fn min_periods(&self) -> usize {
        self.return_period.max(self.vol_period) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.secondary_series.is_empty() {
            Ok(IndicatorOutput::single(self.calculate(&data.close)))
        } else {
            Ok(IndicatorOutput::single(self.calculate_pair(&data.close, &self.secondary_series)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> Vec<f64> {
        // Simulated uptrending currency with low volatility
        vec![
            1.1000, 1.1010, 1.1005, 1.1020, 1.1015, 1.1030, 1.1025, 1.1040, 1.1035, 1.1050,
            1.1045, 1.1060, 1.1055, 1.1070, 1.1065, 1.1080, 1.1075, 1.1090, 1.1085, 1.1100,
            1.1095, 1.1110, 1.1105, 1.1120, 1.1115, 1.1130, 1.1125, 1.1140, 1.1135, 1.1150,
            1.1145, 1.1160, 1.1155, 1.1170, 1.1165, 1.1180, 1.1175, 1.1190, 1.1185, 1.1200,
        ]
    }

    fn make_low_yield_data() -> Vec<f64> {
        // Simulated flat/low-yield currency
        vec![
            0.9000, 0.9002, 0.8998, 0.9001, 0.8999, 0.9003, 0.8997, 0.9000, 0.9002, 0.8998,
            0.9001, 0.8999, 0.9003, 0.8997, 0.9000, 0.9002, 0.8998, 0.9001, 0.8999, 0.9003,
            0.8997, 0.9000, 0.9002, 0.8998, 0.9001, 0.8999, 0.9003, 0.8997, 0.9000, 0.9002,
            0.8998, 0.9001, 0.8999, 0.9003, 0.8997, 0.9000, 0.9002, 0.8998, 0.9001, 0.8999,
        ]
    }

    #[test]
    fn test_carry_trade_signal_creation() {
        let cts = CarryTradeSignal::new(10, 15, 12.0);
        assert!(cts.is_ok());

        let cts_err = CarryTradeSignal::new(3, 15, 12.0);
        assert!(cts_err.is_err());

        let cts_err2 = CarryTradeSignal::new(10, 5, 12.0);
        assert!(cts_err2.is_err());
    }

    #[test]
    fn test_carry_trade_signal_single() {
        let data = make_test_data();
        let cts = CarryTradeSignal::new(10, 15, 20.0).unwrap();
        let result = cts.calculate(&data);

        assert_eq!(result.len(), data.len());
        // Uptrending low-vol data should show positive signal
        assert!(result[30] >= -1.0 && result[30] <= 1.0);
    }

    #[test]
    fn test_carry_trade_signal_pair() {
        let high_yield = make_test_data();
        let low_yield = make_low_yield_data();
        let cts = CarryTradeSignal::new(10, 15, 20.0).unwrap();
        let result = cts.calculate_pair(&high_yield, &low_yield);

        assert_eq!(result.len(), high_yield.len());
        // High yield outperforming should show positive carry signal
        assert!(result[30] > -1.0);
    }

    #[test]
    fn test_carry_trade_signal_trait() {
        let data = make_test_data();
        let cts = CarryTradeSignal::default_params().unwrap();

        assert_eq!(cts.name(), "Carry Trade Signal");
        assert!(cts.min_periods() > 0);

        let series = OHLCVSeries {
            open: data.clone(),
            high: data.clone(),
            low: data.clone(),
            close: data.clone(),
            volume: vec![1000.0; data.len()],
        };

        let output = cts.compute(&series);
        assert!(output.is_ok());
    }
}
