//! Rainbow Oscillator implementation.

use crate::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result, SignalIndicator,
    TechnicalIndicator,
};

/// Rainbow Oscillator - IND-078
///
/// Measures price deviation from Rainbow MA levels (10 recursive SMAs).
///
/// Formula:
/// 1. Calculate Rainbow MA levels (10 SMAs where each is SMA of previous)
/// 2. Calculate highest and lowest Rainbow levels at each bar
/// 3. Rainbow Oscillator = 100 * (Close - Lowest) / (Highest - Lowest)
///
/// Values range from 0 to 100:
/// - Above 80: Overbought
/// - Below 20: Oversold
#[derive(Debug, Clone)]
pub struct RainbowOscillator {
    period: usize,
    levels: usize,
    overbought: f64,
    oversold: f64,
}

impl RainbowOscillator {
    /// Create a new Rainbow Oscillator with the specified SMA period.
    /// Uses 10 recursive SMA levels by default.
    pub fn new(period: usize) -> Self {
        Self {
            period,
            levels: 10,
            overbought: 80.0,
            oversold: 20.0,
        }
    }

    /// Create a Rainbow Oscillator with custom levels and thresholds.
    pub fn with_params(period: usize, levels: usize, overbought: f64, oversold: f64) -> Self {
        Self {
            period,
            levels,
            overbought,
            oversold,
        }
    }

    /// Calculate SMA of the given data.
    fn sma(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        if n < period {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; period - 1];

        // Calculate initial sum, handling NaN values
        let mut sum: f64 = data[..period].iter().filter(|x| !x.is_nan()).sum();
        let count = data[..period].iter().filter(|x| !x.is_nan()).count();

        if count > 0 {
            result.push(sum / count as f64);
        } else {
            result.push(f64::NAN);
        }

        for i in period..n {
            let old = if data[i - period].is_nan() {
                0.0
            } else {
                data[i - period]
            };
            let new = if data[i].is_nan() { 0.0 } else { data[i] };
            sum = sum - old + new;
            result.push(sum / period as f64);
        }

        result
    }

    /// Calculate Rainbow MA levels (recursive SMAs).
    fn calculate_rainbow_levels(&self, data: &[f64]) -> Vec<Vec<f64>> {
        let mut levels = Vec::with_capacity(self.levels);

        // First level is SMA of price
        let mut current = Self::sma(data, self.period);
        levels.push(current.clone());

        // Each subsequent level is SMA of previous level
        for _ in 1..self.levels {
            current = Self::sma(&current, self.period);
            levels.push(current.clone());
        }

        levels
    }

    /// Calculate the Rainbow Oscillator values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.min_data_points() {
            return vec![f64::NAN; n];
        }

        let levels = self.calculate_rainbow_levels(data);

        (0..n)
            .map(|i| {
                // Collect all valid (non-NaN) level values at this index
                let level_values: Vec<f64> = levels
                    .iter()
                    .filter_map(|level| {
                        let val = level[i];
                        if val.is_nan() {
                            None
                        } else {
                            Some(val)
                        }
                    })
                    .collect();

                // Need at least one valid level to calculate
                if level_values.is_empty() {
                    return f64::NAN;
                }

                // Find highest and lowest rainbow levels
                let highest = level_values
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);
                let lowest = level_values.iter().cloned().fold(f64::INFINITY, f64::min);

                let close = data[i];
                if close.is_nan() {
                    return f64::NAN;
                }

                let range = highest - lowest;

                // Avoid division by zero
                if range == 0.0 || range.abs() < f64::EPSILON {
                    // When all levels are equal, check if close is at that level
                    if (close - lowest).abs() < f64::EPSILON {
                        50.0 // Neutral position
                    } else if close > lowest {
                        100.0
                    } else {
                        0.0
                    }
                } else {
                    // Clamp to [0, 100] range
                    (100.0 * (close - lowest) / range).clamp(0.0, 100.0)
                }
            })
            .collect()
    }

    /// Minimum data points required for valid calculation.
    fn min_data_points(&self) -> usize {
        // Need enough data for all recursive SMA levels
        self.period * self.levels
    }
}

impl Default for RainbowOscillator {
    fn default() -> Self {
        Self::new(2)
    }
}

impl TechnicalIndicator for RainbowOscillator {
    fn name(&self) -> &str {
        "RainbowOscillator"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.min_data_points() {
            return Err(IndicatorError::InsufficientData {
                required: self.min_data_points(),
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.min_data_points()
    }
}

impl SignalIndicator for RainbowOscillator {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.close);
        let last = values.last().copied().unwrap_or(f64::NAN);

        if last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Overbought = Bearish (expecting reversal down)
        // Oversold = Bullish (expecting reversal up)
        if last >= self.overbought {
            Ok(IndicatorSignal::Bearish)
        } else if last <= self.oversold {
            Ok(IndicatorSignal::Bullish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.close);
        let signals = values
            .iter()
            .map(|&val| {
                if val.is_nan() {
                    IndicatorSignal::Neutral
                } else if val >= self.overbought {
                    IndicatorSignal::Bearish
                } else if val <= self.oversold {
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
    fn test_rainbow_oscillator_basic() {
        let ro = RainbowOscillator::new(2);
        let data: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64)).collect();
        let result = ro.calculate(&data);

        assert_eq!(result.len(), 50);

        // Check that values are in valid range after warmup
        for (i, &val) in result.iter().enumerate() {
            if i >= ro.min_data_points() && !val.is_nan() {
                assert!(
                    val >= 0.0 && val <= 100.0,
                    "Value {} at index {} out of range",
                    val,
                    i
                );
            }
        }
    }

    #[test]
    fn test_rainbow_oscillator_uptrend() {
        let ro = RainbowOscillator::new(2);
        // Strong uptrend: price should be above rainbow levels
        let data: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 2.0).collect();
        let result = ro.calculate(&data);

        // In uptrend, later values should be high (close above rainbow levels)
        let last = result.last().unwrap();
        assert!(
            *last > 50.0,
            "Expected high oscillator in uptrend, got {}",
            last
        );
    }

    #[test]
    fn test_rainbow_oscillator_downtrend() {
        let ro = RainbowOscillator::new(2);
        // Strong downtrend: price should be below rainbow levels
        let data: Vec<f64> = (0..50).map(|i| 200.0 - (i as f64) * 2.0).collect();
        let result = ro.calculate(&data);

        // In downtrend, later values should be low (close below rainbow levels)
        let last = result.last().unwrap();
        assert!(
            *last < 50.0,
            "Expected low oscillator in downtrend, got {}",
            last
        );
    }

    #[test]
    fn test_rainbow_oscillator_signals() {
        let ro = RainbowOscillator::with_params(2, 10, 80.0, 20.0);
        let data: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 3.0).collect();

        let ohlcv = OHLCVSeries {
            open: data.clone(),
            high: data.clone(),
            low: data.clone(),
            close: data.clone(),
            volume: vec![1000.0; 50],
        };

        let signal = ro.signal(&ohlcv).unwrap();
        // Strong uptrend should give overbought (bearish) signal
        assert!(
            matches!(signal, IndicatorSignal::Bearish)
                || matches!(signal, IndicatorSignal::Neutral),
            "Expected Bearish or Neutral signal in strong uptrend"
        );
    }

    #[test]
    fn test_rainbow_oscillator_insufficient_data() {
        let ro = RainbowOscillator::new(2);
        let data: Vec<f64> = vec![100.0, 101.0, 102.0]; // Too few points

        let ohlcv = OHLCVSeries {
            open: data.clone(),
            high: data.clone(),
            low: data.clone(),
            close: data.clone(),
            volume: vec![1000.0; 3],
        };

        let result = ro.compute(&ohlcv);
        assert!(result.is_err());
    }

    #[test]
    fn test_rainbow_oscillator_custom_params() {
        let ro = RainbowOscillator::with_params(3, 5, 70.0, 30.0);
        let data: Vec<f64> = (0..30).map(|i| 100.0 + (i as f64)).collect();
        let result = ro.calculate(&data);

        assert_eq!(result.len(), 30);
    }

    #[test]
    fn test_rainbow_oscillator_constant_price() {
        let ro = RainbowOscillator::new(2);
        // Constant price: all levels should be equal
        let data: Vec<f64> = vec![100.0; 50];
        let result = ro.calculate(&data);

        // When price equals all rainbow levels, oscillator should be 50 (neutral)
        for &val in result.iter().skip(ro.min_data_points()) {
            if !val.is_nan() {
                assert!(
                    (val - 50.0).abs() < 0.01,
                    "Expected 50.0 for constant price, got {}",
                    val
                );
            }
        }
    }

    #[test]
    fn test_rainbow_oscillator_default() {
        let ro = RainbowOscillator::default();
        assert_eq!(ro.period, 2);
        assert_eq!(ro.levels, 10);
    }

    #[test]
    fn test_technical_indicator_impl() {
        let ro = RainbowOscillator::new(2);
        assert_eq!(ro.name(), "RainbowOscillator");
        assert_eq!(ro.min_periods(), 20); // 2 * 10 levels
    }

    #[test]
    fn test_signals_vector() {
        let ro = RainbowOscillator::new(2);
        let data: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64)).collect();

        let ohlcv = OHLCVSeries {
            open: data.clone(),
            high: data.clone(),
            low: data.clone(),
            close: data.clone(),
            volume: vec![1000.0; 50],
        };

        let signals = ro.signals(&ohlcv).unwrap();
        assert_eq!(signals.len(), 50);
    }
}
