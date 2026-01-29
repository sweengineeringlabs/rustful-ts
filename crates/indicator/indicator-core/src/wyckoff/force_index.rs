//! Wyckoff Force Index - Price x Volume force indicator (IND-230)
//!
//! A Wyckoff-style force measurement that combines price change,
//! volume, and spread to quantify the force behind price movements.

use crate::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result,
    SignalIndicator, TechnicalIndicator,
};

/// Wyckoff Force Index configuration.
#[derive(Debug, Clone)]
pub struct WyckoffForceIndexConfig {
    /// Smoothing period
    pub period: usize,
    /// Include spread in force calculation
    pub use_spread: bool,
    /// Weight for volume component
    pub volume_weight: f64,
}

impl Default for WyckoffForceIndexConfig {
    fn default() -> Self {
        Self {
            period: 13,
            use_spread: true,
            volume_weight: 1.0,
        }
    }
}

/// Wyckoff Force Index.
///
/// This indicator measures the "force" behind price movements using
/// Wyckoff principles. It combines:
/// - Price change (direction and magnitude)
/// - Volume (effort)
/// - Spread (range, indicating volatility/conviction)
///
/// Formula:
/// - Basic: Force = (Close - Prev Close) * Volume
/// - With Spread: Force = (Close - Prev Close) * Volume * (Spread / Avg Spread)
///
/// Interpretation:
/// - Large positive force = strong buying
/// - Large negative force = strong selling
/// - Divergence between force and price = potential reversal
#[derive(Debug, Clone)]
pub struct WyckoffForceIndex {
    config: WyckoffForceIndexConfig,
}

impl WyckoffForceIndex {
    pub fn new(period: usize) -> Self {
        Self {
            config: WyckoffForceIndexConfig {
                period,
                ..Default::default()
            },
        }
    }

    pub fn from_config(config: WyckoffForceIndexConfig) -> Self {
        Self { config }
    }

    /// Calculate raw force values.
    pub fn calculate_raw(&self, data: &OHLCVSeries) -> Vec<f64> {
        let n = data.close.len();

        if n < 2 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; n];

        // Calculate average spread for normalization
        let spreads: Vec<f64> = data.high.iter()
            .zip(data.low.iter())
            .map(|(h, l)| h - l)
            .collect();

        let avg_spread: f64 = spreads.iter().sum::<f64>() / n as f64;
        let avg_spread = if avg_spread > 0.0 { avg_spread } else { 1.0 };

        for i in 1..n {
            let price_change = data.close[i] - data.close[i - 1];
            let volume = data.volume[i] * self.config.volume_weight;

            let force = if self.config.use_spread {
                let spread_ratio = spreads[i] / avg_spread;
                price_change * volume * spread_ratio
            } else {
                price_change * volume
            };

            result[i] = force;
        }

        result
    }

    /// Calculate smoothed force index using EMA.
    pub fn calculate(&self, data: &OHLCVSeries) -> Vec<f64> {
        let raw = self.calculate_raw(data);
        let n = raw.len();

        if n < self.config.period + 1 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; n];
        let alpha = 2.0 / (self.config.period as f64 + 1.0);

        // Calculate initial SMA (skip first NaN)
        let mut sum = 0.0;
        let mut count = 0;
        for i in 1..=self.config.period {
            if i < n && !raw[i].is_nan() {
                sum += raw[i];
                count += 1;
            }
        }

        if count > 0 {
            result[self.config.period] = sum / count as f64;
        }

        // Apply EMA
        for i in (self.config.period + 1)..n {
            if !raw[i].is_nan() && !result[i - 1].is_nan() {
                result[i] = alpha * raw[i] + (1.0 - alpha) * result[i - 1];
            } else if !result[i - 1].is_nan() {
                result[i] = result[i - 1];
            }
        }

        result
    }

    /// Calculate force ratio (current force / average force).
    pub fn calculate_force_ratio(&self, data: &OHLCVSeries) -> Vec<f64> {
        let raw = self.calculate_raw(data);
        let n = raw.len();

        if n < self.config.period + 1 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; n];

        for i in self.config.period..n {
            let window_start = i + 1 - self.config.period;
            let mut abs_sum = 0.0;
            let mut count = 0;

            for j in window_start..=i {
                if !raw[j].is_nan() {
                    abs_sum += raw[j].abs();
                    count += 1;
                }
            }

            if count > 0 && abs_sum > 0.0 {
                let avg_force = abs_sum / count as f64;
                result[i] = raw[i] / avg_force;
            }
        }

        result
    }
}

impl Default for WyckoffForceIndex {
    fn default() -> Self {
        Self::from_config(WyckoffForceIndexConfig::default())
    }
}

impl TechnicalIndicator for WyckoffForceIndex {
    fn name(&self) -> &str {
        "WyckoffForceIndex"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.config.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.config.period + 1,
                got: data.close.len(),
            });
        }

        let values = self.calculate(data);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.config.period + 1
    }

    fn output_features(&self) -> usize {
        1
    }
}

impl SignalIndicator for WyckoffForceIndex {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(data);

        if let Some(&last) = values.last() {
            if !last.is_nan() {
                if last > 0.0 {
                    return Ok(IndicatorSignal::Bullish);
                } else if last < 0.0 {
                    return Ok(IndicatorSignal::Bearish);
                }
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(data);

        Ok(values
            .iter()
            .map(|&v| {
                if v.is_nan() {
                    IndicatorSignal::Neutral
                } else if v > 0.0 {
                    IndicatorSignal::Bullish
                } else if v < 0.0 {
                    IndicatorSignal::Bearish
                } else {
                    IndicatorSignal::Neutral
                }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data(n: usize) -> OHLCVSeries {
        let mut open = Vec::with_capacity(n);
        let mut high = Vec::with_capacity(n);
        let mut low = Vec::with_capacity(n);
        let mut close = Vec::with_capacity(n);
        let mut volume = Vec::with_capacity(n);

        for i in 0..n {
            let base = 100.0 + (i as f64) * 1.0;
            open.push(base);
            high.push(base + 2.0);
            low.push(base - 1.0);
            close.push(base + 1.5);
            volume.push(1000.0);
        }

        OHLCVSeries { open, high, low, close, volume }
    }

    fn create_downtrend_data(n: usize) -> OHLCVSeries {
        let mut open = Vec::with_capacity(n);
        let mut high = Vec::with_capacity(n);
        let mut low = Vec::with_capacity(n);
        let mut close = Vec::with_capacity(n);
        let mut volume = Vec::with_capacity(n);

        for i in 0..n {
            let base = 150.0 - (i as f64) * 1.0;
            open.push(base);
            high.push(base + 1.0);
            low.push(base - 2.0);
            close.push(base - 1.5);
            volume.push(1000.0);
        }

        OHLCVSeries { open, high, low, close, volume }
    }

    #[test]
    fn test_wyckoff_force_index_basic() {
        let wfi = WyckoffForceIndex::new(5);
        let data = create_test_data(20);
        let result = wfi.calculate(&data);

        assert_eq!(result.len(), 20);

        // In uptrend, force should be positive
        for &val in result.iter().skip(6) {
            if !val.is_nan() {
                assert!(val > 0.0);
            }
        }
    }

    #[test]
    fn test_wyckoff_force_index_raw() {
        let wfi = WyckoffForceIndex::new(5);
        let data = create_test_data(10);
        let raw = wfi.calculate_raw(&data);

        assert_eq!(raw.len(), 10);
        assert!(raw[0].is_nan()); // First value is NaN

        // Remaining values should be positive in uptrend
        for &val in raw.iter().skip(1) {
            if !val.is_nan() {
                assert!(val > 0.0);
            }
        }
    }

    #[test]
    fn test_wyckoff_force_index_downtrend() {
        let wfi = WyckoffForceIndex::new(5);
        let data = create_downtrend_data(20);
        let result = wfi.calculate(&data);

        // In downtrend, force should be negative
        for &val in result.iter().skip(6) {
            if !val.is_nan() {
                assert!(val < 0.0);
            }
        }
    }

    #[test]
    fn test_force_ratio() {
        let wfi = WyckoffForceIndex::new(5);
        let data = create_test_data(20);
        let ratio = wfi.calculate_force_ratio(&data);

        assert_eq!(ratio.len(), 20);

        // Force ratio should be around 1 in steady trend
        for &val in ratio.iter().skip(5) {
            if !val.is_nan() {
                assert!(val.abs() < 10.0); // Reasonable bounds
            }
        }
    }

    #[test]
    fn test_signal_generation() {
        let wfi = WyckoffForceIndex::new(5);

        let uptrend = create_test_data(15);
        let signal = wfi.signal(&uptrend).unwrap();
        assert!(matches!(signal, IndicatorSignal::Bullish));

        let downtrend = create_downtrend_data(15);
        let signal = wfi.signal(&downtrend).unwrap();
        assert!(matches!(signal, IndicatorSignal::Bearish));
    }
}
