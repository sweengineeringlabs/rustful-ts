//! Close-to-Close Volatility implementation.
//!
//! The simplest volatility measure using standard deviation of logarithmic returns
//! calculated from closing prices.

use crate::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result, SignalIndicator,
    TechnicalIndicator,
};

/// Close-to-Close Volatility.
///
/// The most basic historical volatility measure, calculating the standard
/// deviation of logarithmic returns between consecutive closing prices.
///
/// Formula:
/// 1. Log returns: r_i = ln(close_i / close_{i-1})
/// 2. Mean return: mean = sum(r_i) / n
/// 3. Variance: var = sum((r_i - mean)^2) / n
/// 4. Volatility: vol = sqrt(var) * sqrt(trading_days) [if annualized]
///
/// This is also known as:
/// - Historical Volatility (HV)
/// - Realized Volatility (simple version)
/// - Close-Close Volatility
///
/// # Advantages
/// - Simple and widely understood
/// - Only requires closing prices
/// - Foundation for more complex volatility models
///
/// # Limitations
/// - Ignores intraday price movements
/// - May underestimate true volatility in volatile markets
/// - Sensitive to overnight gaps
///
/// # Signal Logic
/// - Volatility above high threshold: Bearish (high risk)
/// - Volatility below low threshold: Bullish (low risk, potential breakout)
/// - Otherwise: Neutral
#[derive(Debug, Clone)]
pub struct CloseToCloseVolatility {
    /// Lookback period for volatility calculation.
    period: usize,
    /// Number of trading days per year for annualization.
    trading_days: f64,
    /// Whether to annualize the volatility.
    annualize: bool,
    /// High volatility threshold (e.g., 0.30 for 30%).
    high_threshold: f64,
    /// Low volatility threshold (e.g., 0.10 for 10%).
    low_threshold: f64,
}

impl CloseToCloseVolatility {
    /// Create a new Close-to-Close Volatility indicator.
    ///
    /// # Arguments
    /// * `period` - Lookback period (commonly 20, 30, or 60)
    pub fn new(period: usize) -> Self {
        Self {
            period,
            trading_days: 252.0,
            annualize: true,
            high_threshold: 0.30,
            low_threshold: 0.10,
        }
    }

    /// Create with custom trading days for annualization.
    pub fn with_trading_days(period: usize, trading_days: f64) -> Self {
        Self {
            period,
            trading_days,
            annualize: true,
            high_threshold: 0.30,
            low_threshold: 0.10,
        }
    }

    /// Create without annualization (returns raw daily standard deviation).
    pub fn without_annualization(period: usize) -> Self {
        Self {
            period,
            trading_days: 252.0,
            annualize: false,
            high_threshold: 0.02, // ~2% daily volatility threshold
            low_threshold: 0.005, // ~0.5% daily volatility threshold
        }
    }

    /// Set volatility thresholds for signal generation.
    pub fn with_thresholds(mut self, high: f64, low: f64) -> Self {
        self.high_threshold = high;
        self.low_threshold = low;
        self
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

    /// Calculate close-to-close volatility values.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < self.period + 1 || self.period == 0 {
            return vec![f64::NAN; n];
        }

        let returns = Self::log_returns(close);
        let returns_len = returns.len();

        // Need `period` returns for first valid value
        let mut result = vec![f64::NAN; self.period];

        for i in (self.period - 1)..returns_len {
            let start = i + 1 - self.period;
            let window = &returns[start..=i];

            // Check for NaN values in window
            let valid_returns: Vec<f64> = window.iter().filter(|x| !x.is_nan()).copied().collect();

            if valid_returns.len() < self.period {
                result.push(f64::NAN);
                continue;
            }

            // Calculate mean
            let mean: f64 = valid_returns.iter().sum::<f64>() / valid_returns.len() as f64;

            // Calculate variance (population variance)
            let variance: f64 = valid_returns
                .iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>()
                / valid_returns.len() as f64;

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

    /// Calculate returns for analysis.
    pub fn returns(&self, close: &[f64]) -> Vec<f64> {
        Self::log_returns(close)
    }
}

impl TechnicalIndicator for CloseToCloseVolatility {
    fn name(&self) -> &str {
        "CloseToCloseVolatility"
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

impl SignalIndicator for CloseToCloseVolatility {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.close);
        let last = values.last().copied().unwrap_or(f64::NAN);

        if last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // High volatility = higher risk, potential mean reversion
        if last > self.high_threshold {
            Ok(IndicatorSignal::Bearish)
        }
        // Low volatility = lower risk, potential breakout setup
        else if last < self.low_threshold {
            Ok(IndicatorSignal::Bullish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.close);

        let signals = values
            .iter()
            .map(|&vol| {
                if vol.is_nan() {
                    IndicatorSignal::Neutral
                } else if vol > self.high_threshold {
                    IndicatorSignal::Bearish
                } else if vol < self.low_threshold {
                    IndicatorSignal::Bullish
                } else {
                    IndicatorSignal::Neutral
                }
            })
            .collect();

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_close_to_close_volatility() {
        let vol = CloseToCloseVolatility::new(20);

        // Generate sample price data with some volatility
        let data: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0 + (i as f64 * 0.05))
            .collect();

        let result = vol.calculate(&data);

        assert_eq!(result.len(), 50);

        // First 20 values should be NaN
        for i in 0..20 {
            assert!(result[i].is_nan(), "Expected NaN at index {}", i);
        }

        // Volatility should be positive
        for i in 20..50 {
            assert!(
                result[i] > 0.0,
                "Volatility should be positive at index {}",
                i
            );
        }
    }

    #[test]
    fn test_log_returns() {
        let data = vec![100.0, 105.0, 103.0, 108.0];
        let returns = CloseToCloseVolatility::log_returns(&data);

        assert_eq!(returns.len(), 3);
        assert!((returns[0] - (105.0_f64 / 100.0).ln()).abs() < 1e-10);
        assert!((returns[1] - (103.0_f64 / 105.0).ln()).abs() < 1e-10);
        assert!((returns[2] - (108.0_f64 / 103.0).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_non_annualized() {
        let vol_ann = CloseToCloseVolatility::new(10);
        let vol_raw = CloseToCloseVolatility::without_annualization(10);

        let data: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 * 0.5).collect();

        let ann_result = vol_ann.calculate(&data);
        let raw_result = vol_raw.calculate(&data);

        // Annualized should be sqrt(252) times larger
        for i in 10..30 {
            if !ann_result[i].is_nan() && !raw_result[i].is_nan() {
                let ratio = ann_result[i] / raw_result[i];
                assert!(
                    (ratio - 252.0_f64.sqrt()).abs() < 0.01,
                    "Ratio should be sqrt(252) at index {}",
                    i
                );
            }
        }
    }

    #[test]
    fn test_constant_prices() {
        let vol = CloseToCloseVolatility::new(10);

        // Constant prices should have zero volatility
        let data: Vec<f64> = vec![100.0; 30];
        let result = vol.calculate(&data);

        for i in 10..30 {
            assert!(
                (result[i] - 0.0).abs() < 1e-10,
                "Constant prices should have zero volatility at index {}",
                i
            );
        }
    }

    #[test]
    fn test_trending_prices() {
        let vol = CloseToCloseVolatility::new(10);

        // Steadily increasing prices (constant percent change)
        let data: Vec<f64> = (0..30).map(|i| 100.0 * 1.01_f64.powi(i as i32)).collect();

        let result = vol.calculate(&data);

        // With constant percent returns, volatility should be very low (near zero std dev)
        for i in 10..30 {
            assert!(
                result[i] < 0.05,
                "Constant percent change should have low vol at index {}",
                i
            );
        }
    }

    #[test]
    fn test_custom_trading_days() {
        let vol_252 = CloseToCloseVolatility::with_trading_days(10, 252.0);
        let vol_365 = CloseToCloseVolatility::with_trading_days(10, 365.0);

        let data: Vec<f64> = (0..30).map(|i| 100.0 + (i as f64).sin() * 5.0).collect();

        let result_252 = vol_252.calculate(&data);
        let result_365 = vol_365.calculate(&data);

        // 365-day annualized should be higher
        for i in 10..30 {
            if !result_252[i].is_nan() && !result_365[i].is_nan() {
                let ratio = result_365[i] / result_252[i];
                let expected_ratio = (365.0_f64 / 252.0).sqrt();
                assert!(
                    (ratio - expected_ratio).abs() < 0.01,
                    "Ratio should be sqrt(365/252) at index {}",
                    i
                );
            }
        }
    }

    #[test]
    fn test_close_to_close_technical_indicator() {
        let vol = CloseToCloseVolatility::new(20);
        assert_eq!(vol.name(), "CloseToCloseVolatility");
        assert_eq!(vol.min_periods(), 21);
    }

    #[test]
    fn test_close_to_close_insufficient_data() {
        let vol = CloseToCloseVolatility::new(20);

        let series = OHLCVSeries {
            open: vec![100.0; 10],
            high: vec![102.0; 10],
            low: vec![98.0; 10],
            close: vec![100.0; 10],
            volume: vec![1000.0; 10],
        };

        let result = vol.compute(&series);
        assert!(result.is_err());
    }

    #[test]
    fn test_close_to_close_signal_high_vol() {
        let vol = CloseToCloseVolatility::new(5).with_thresholds(0.20, 0.05);

        // Generate volatile data
        let close: Vec<f64> = (0..20)
            .map(|i| if i % 2 == 0 { 100.0 } else { 130.0 })
            .collect();

        let series = OHLCVSeries {
            open: close.clone(),
            high: close.iter().map(|&c| c + 5.0).collect(),
            low: close.iter().map(|&c| c - 5.0).collect(),
            close,
            volume: vec![1000.0; 20],
        };

        let signal = vol.signal(&series).unwrap();
        // High volatility should generate bearish signal
        assert_eq!(signal, IndicatorSignal::Bearish);
    }

    #[test]
    fn test_close_to_close_signal_low_vol() {
        let vol = CloseToCloseVolatility::new(5).with_thresholds(0.20, 0.05);

        // Generate low volatility data (constant with tiny movement)
        let close: Vec<f64> = (0..20).map(|i| 100.0 + (i as f64) * 0.001).collect();

        let series = OHLCVSeries {
            open: close.clone(),
            high: close.iter().map(|&c| c + 0.1).collect(),
            low: close.iter().map(|&c| c - 0.1).collect(),
            close,
            volume: vec![1000.0; 20],
        };

        let signal = vol.signal(&series).unwrap();
        // Low volatility should generate bullish signal
        assert_eq!(signal, IndicatorSignal::Bullish);
    }

    #[test]
    fn test_returns_method() {
        let vol = CloseToCloseVolatility::new(10);
        let close = vec![100.0, 101.0, 102.0, 103.0];

        let returns = vol.returns(&close);

        assert_eq!(returns.len(), 3);
        assert!((returns[0] - (101.0_f64 / 100.0).ln()).abs() < 1e-10);
    }
}
