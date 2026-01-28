//! Market Thermometer implementation.
//!
//! Elder's Market Thermometer measures intraday volatility to help with
//! position sizing and market timing.

use crate::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result, SignalIndicator,
    TechnicalIndicator,
};
use indicator_api::ElderThermometerConfig;

/// Market Thermometer (Elder's).
///
/// The Market Thermometer measures intraday volatility by comparing the current
/// bar's range to the previous bar. It helps identify when the market is
/// "hot" (highly volatile) or "cold" (calm).
///
/// Formula (IND-178):
/// 1. Calculate absolute difference: |high - previous_high| and |low - previous_low|
/// 2. Thermometer = max(|H - prev_H|, |L - prev_L|)
/// 3. Calculate EMA of thermometer values for smoothing
/// 4. High readings indicate increased volatility
///
/// The indicator is typically smoothed with an EMA and compared to a threshold
/// (often the 22-period EMA multiplied by a factor like 2).
///
/// # Trading Applications
/// - High readings: Market is volatile, use wider stops
/// - Low readings: Market is calm, tighter stops possible
/// - Spikes: Often occur at trend reversals
///
/// # Signal Logic
/// - Thermometer below EMA: Calm market, potential breakout setup (Bullish)
/// - Thermometer above 2x EMA: Hot market, increased risk (Bearish)
/// - Otherwise: Normal conditions (Neutral)
#[derive(Debug, Clone)]
pub struct MarketThermometer {
    /// EMA period for smoothing (typically 22).
    period: usize,
    /// Multiplier for hot threshold (typically 2.0).
    hot_multiplier: f64,
}

impl MarketThermometer {
    /// Create a new Market Thermometer indicator.
    ///
    /// # Arguments
    /// * `period` - Period for EMA smoothing (commonly 22)
    pub fn new(period: usize) -> Self {
        Self {
            period,
            hot_multiplier: 2.0,
        }
    }

    /// Create from configuration.
    pub fn from_config(config: ElderThermometerConfig) -> Self {
        Self {
            period: config.period,
            hot_multiplier: 2.0,
        }
    }

    /// Create with default parameters (22-period EMA).
    pub fn default_params() -> Self {
        Self::new(22)
    }

    /// Set custom hot multiplier threshold.
    pub fn with_hot_multiplier(mut self, multiplier: f64) -> Self {
        self.hot_multiplier = multiplier;
        self
    }

    /// Calculate EMA of a series, handling leading NaN values.
    fn ema(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        if n < period + 1 || period == 0 {
            return vec![f64::NAN; n];
        }

        let multiplier = 2.0 / (period as f64 + 1.0);

        // Find first valid (non-NaN) index
        let first_valid = data.iter().position(|x| !x.is_nan()).unwrap_or(n);
        if first_valid + period > n {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; first_valid + period - 1];

        // Initial EMA is SMA of first `period` valid values
        let initial_sma: f64 = data[first_valid..first_valid + period]
            .iter()
            .sum::<f64>()
            / period as f64;
        result.push(initial_sma);

        let mut prev_ema = initial_sma;
        for i in (first_valid + period)..n {
            let ema = (data[i] - prev_ema) * multiplier + prev_ema;
            result.push(ema);
            prev_ema = ema;
        }

        result
    }

    /// Calculate raw thermometer values (before smoothing).
    ///
    /// IND-178 Algorithm:
    /// 1. Calculate absolute difference: |high - previous_high| and |low - previous_low|
    /// 2. Thermometer = max(|H - prev_H|, |L - prev_L|)
    fn raw_thermometer(high: &[f64], low: &[f64]) -> Vec<f64> {
        let n = high.len();
        if n < 2 {
            return vec![f64::NAN; n];
        }

        let mut thermo = vec![f64::NAN]; // First bar has no previous bar

        for i in 1..n {
            // Absolute difference from previous high
            let high_diff = (high[i] - high[i - 1]).abs();
            // Absolute difference from previous low
            let low_diff = (low[i] - low[i - 1]).abs();

            // Thermometer is the greater of the two absolute differences
            thermo.push(high_diff.max(low_diff));
        }

        thermo
    }

    /// Calculate Market Thermometer values.
    ///
    /// Returns (raw_thermometer, smoothed_thermometer, hot_threshold).
    pub fn calculate(&self, high: &[f64], low: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = high.len();
        if n < self.period + 1 || self.period == 0 {
            return (vec![f64::NAN; n], vec![f64::NAN; n], vec![f64::NAN; n]);
        }

        // Calculate raw thermometer
        let raw = Self::raw_thermometer(high, low);

        // Calculate EMA of thermometer
        let smoothed = Self::ema(&raw, self.period);

        // Calculate hot threshold (EMA * multiplier)
        let hot: Vec<f64> = smoothed
            .iter()
            .map(|&s| {
                if s.is_nan() {
                    f64::NAN
                } else {
                    s * self.hot_multiplier
                }
            })
            .collect();

        (raw, smoothed, hot)
    }

    /// Calculate average thermometer reading (useful for position sizing).
    pub fn average_reading(&self, high: &[f64], low: &[f64]) -> f64 {
        let (_, smoothed, _) = self.calculate(high, low);
        smoothed.last().copied().unwrap_or(f64::NAN)
    }
}

impl TechnicalIndicator for MarketThermometer {
    fn name(&self) -> &str {
        "MarketThermometer"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.high.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.high.len(),
            });
        }

        let (raw, smoothed, hot) = self.calculate(&data.high, &data.low);
        Ok(IndicatorOutput::triple(raw, smoothed, hot))
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn output_features(&self) -> usize {
        3 // raw, smoothed, hot_threshold
    }
}

impl SignalIndicator for MarketThermometer {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (raw, smoothed, hot) = self.calculate(&data.high, &data.low);
        let n = raw.len();

        if n == 0 {
            return Ok(IndicatorSignal::Neutral);
        }

        let raw_val = raw[n - 1];
        let smoothed_val = smoothed[n - 1];
        let hot_val = hot[n - 1];

        if raw_val.is_nan() || smoothed_val.is_nan() || hot_val.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Hot market: raw thermometer above hot threshold
        if raw_val > hot_val {
            Ok(IndicatorSignal::Bearish)
        }
        // Calm market: raw thermometer below smoothed average
        else if raw_val < smoothed_val {
            Ok(IndicatorSignal::Bullish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (raw, smoothed, hot) = self.calculate(&data.high, &data.low);

        let signals = raw
            .iter()
            .zip(smoothed.iter())
            .zip(hot.iter())
            .map(|((&r, &s), &h)| {
                if r.is_nan() || s.is_nan() || h.is_nan() {
                    IndicatorSignal::Neutral
                } else if r > h {
                    IndicatorSignal::Bearish
                } else if r < s {
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
    fn test_market_thermometer_basic() {
        let thermo = MarketThermometer::new(10);

        // Generate sample OHLC data
        let high: Vec<f64> = (0..50)
            .map(|i| 102.0 + (i as f64 * 0.2).sin() * 3.0 + i as f64 * 0.05)
            .collect();
        let low: Vec<f64> = (0..50)
            .map(|i| 98.0 + (i as f64 * 0.2).sin() * 3.0 + i as f64 * 0.05)
            .collect();

        let (raw, smoothed, hot) = thermo.calculate(&high, &low);

        assert_eq!(raw.len(), 50);
        assert_eq!(smoothed.len(), 50);
        assert_eq!(hot.len(), 50);

        // First bar has no previous, should be NaN
        assert!(raw[0].is_nan());

        // After warmup, should have valid values
        for i in 10..50 {
            assert!(
                !smoothed[i].is_nan(),
                "Smoothed should be valid at index {}",
                i
            );
            assert!(
                !hot[i].is_nan(),
                "Hot threshold should be valid at index {}",
                i
            );
        }
    }

    #[test]
    fn test_raw_thermometer_expanding_bars() {
        // Test case where each bar expands beyond the previous
        let high = vec![100.0, 102.0, 104.0, 106.0];
        let low = vec![98.0, 97.0, 96.0, 95.0];

        let raw = MarketThermometer::raw_thermometer(&high, &low);

        assert!(raw[0].is_nan()); // First bar
        // Bar 1: max(|102-100|, |97-98|) = max(2, 1) = 2
        assert!((raw[1] - 2.0).abs() < 1e-10);
        // Bar 2: max(|104-102|, |96-97|) = max(2, 1) = 2
        assert!((raw[2] - 2.0).abs() < 1e-10);
        // Bar 3: max(|106-104|, |95-96|) = max(2, 1) = 2
        assert!((raw[3] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_raw_thermometer_contracting_bars() {
        // Test case where bars contract (inside bars)
        // With absolute differences, contracting bars still have positive thermometer values
        let high = vec![110.0, 108.0, 106.0, 104.0];
        let low = vec![90.0, 92.0, 94.0, 96.0];

        let raw = MarketThermometer::raw_thermometer(&high, &low);

        assert!(raw[0].is_nan()); // First bar
        // Bar 1: max(|108-110|, |92-90|) = max(2, 2) = 2
        assert!((raw[1] - 2.0).abs() < 1e-10);
        // Bar 2: max(|106-108|, |94-92|) = max(2, 2) = 2
        assert!((raw[2] - 2.0).abs() < 1e-10);
        // Bar 3: max(|104-106|, |96-94|) = max(2, 2) = 2
        assert!((raw[3] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_raw_thermometer_mixed() {
        // Test mixed scenario with absolute differences
        let high = vec![100.0, 103.0, 101.0, 105.0];
        let low = vec![98.0, 96.0, 99.0, 94.0];

        let raw = MarketThermometer::raw_thermometer(&high, &low);

        assert!(raw[0].is_nan());
        // Bar 1: max(|103-100|, |96-98|) = max(3, 2) = 3
        assert!((raw[1] - 3.0).abs() < 1e-10);
        // Bar 2: max(|101-103|, |99-96|) = max(2, 3) = 3
        assert!((raw[2] - 3.0).abs() < 1e-10);
        // Bar 3: max(|105-101|, |94-99|) = max(4, 5) = 5
        assert!((raw[3] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_market_thermometer_default() {
        let thermo = MarketThermometer::default_params();
        assert_eq!(thermo.period, 22);
        assert!((thermo.hot_multiplier - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_from_config() {
        let config = ElderThermometerConfig::new(14);
        let thermo = MarketThermometer::from_config(config);
        assert_eq!(thermo.period, 14);
    }

    #[test]
    fn test_hot_threshold() {
        let thermo = MarketThermometer::new(5).with_hot_multiplier(3.0);

        let high: Vec<f64> = (0..20).map(|i| 102.0 + i as f64 * 0.1).collect();
        let low: Vec<f64> = (0..20).map(|i| 98.0 + i as f64 * 0.1).collect();

        let (_, smoothed, hot) = thermo.calculate(&high, &low);

        // Hot should be 3x smoothed
        for i in 5..20 {
            if !smoothed[i].is_nan() && !hot[i].is_nan() {
                assert!(
                    (hot[i] - smoothed[i] * 3.0).abs() < 1e-10,
                    "Hot should be 3x smoothed at index {}",
                    i
                );
            }
        }
    }

    #[test]
    fn test_market_thermometer_technical_indicator() {
        let thermo = MarketThermometer::new(22);
        assert_eq!(thermo.name(), "MarketThermometer");
        assert_eq!(thermo.min_periods(), 23);
        assert_eq!(thermo.output_features(), 3);
    }

    #[test]
    fn test_market_thermometer_insufficient_data() {
        let thermo = MarketThermometer::new(22);

        let series = OHLCVSeries {
            open: vec![100.0; 10],
            high: vec![102.0; 10],
            low: vec![98.0; 10],
            close: vec![100.0; 10],
            volume: vec![1000.0; 10],
        };

        let result = thermo.compute(&series);
        assert!(result.is_err());
    }

    #[test]
    fn test_market_thermometer_signal_hot() {
        let thermo = MarketThermometer::new(5);

        // Generate data with a big spike at the end
        let mut high: Vec<f64> = vec![102.0; 15];
        let mut low: Vec<f64> = vec![98.0; 15];

        // Last bar has huge extension
        high[14] = 120.0;
        low[14] = 98.0;

        let series = OHLCVSeries {
            open: vec![100.0; 15],
            high,
            low,
            close: vec![100.0; 15],
            volume: vec![1000.0; 15],
        };

        let signal = thermo.signal(&series).unwrap();
        // Big spike should be above hot threshold
        assert_eq!(signal, IndicatorSignal::Bearish);
    }

    #[test]
    fn test_market_thermometer_signal_calm() {
        let thermo = MarketThermometer::new(5);

        // Generate data with large initial volatility, then very small changes
        let mut high = Vec::new();
        let mut low = Vec::new();

        // Volatile period - each bar shifts significantly
        for i in 0..10 {
            high.push(110.0 + i as f64 * 5.0);
            low.push(90.0 - i as f64 * 3.0);
        }

        // Calm period - bars barely move (small absolute differences)
        let last_high = high[9];
        let last_low = low[9];
        for _ in 10..15 {
            // Tiny movements of 0.01
            high.push(last_high + 0.01);
            low.push(last_low + 0.01);
        }

        let series = OHLCVSeries {
            open: vec![100.0; 15],
            high,
            low,
            close: vec![100.0; 15],
            volume: vec![1000.0; 15],
        };

        let signal = thermo.signal(&series).unwrap();
        // Very small movements should be below smoothed average (bullish = calm)
        assert_eq!(signal, IndicatorSignal::Bullish);
    }

    #[test]
    fn test_average_reading() {
        let thermo = MarketThermometer::new(5);

        let high: Vec<f64> = (0..20)
            .map(|i| 102.0 + (i as f64 * 0.3).sin() * 2.0)
            .collect();
        let low: Vec<f64> = (0..20)
            .map(|i| 98.0 + (i as f64 * 0.3).sin() * 2.0)
            .collect();

        let avg = thermo.average_reading(&high, &low);

        // Should return the last smoothed value
        assert!(!avg.is_nan());
        assert!(avg >= 0.0);
    }
}
