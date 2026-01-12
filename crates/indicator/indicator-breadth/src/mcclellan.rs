//! McClellan Oscillator indicator.

use crate::{BreadthIndicator, BreadthSeries};
use indicator_spi::{IndicatorError, IndicatorOutput, Result};

/// McClellan Oscillator
///
/// A breadth momentum indicator that measures the difference between
/// two exponential moving averages of net advances. It identifies
/// overbought/oversold conditions and momentum shifts in market breadth.
///
/// # Formula
/// 1. Calculate Net Advances = Advances - Declines
/// 2. Or use Ratio-Adjusted: Net Advances = (Advances - Declines) / (Advances + Declines)
/// 3. Fast EMA = 19-period EMA of Net Advances (smoothing factor ~0.10)
/// 4. Slow EMA = 39-period EMA of Net Advances (smoothing factor ~0.05)
/// 5. McClellan Oscillator = Fast EMA - Slow EMA
///
/// # Interpretation
/// - Above +100: Overbought (potential pullback)
/// - Below -100: Oversold (potential bounce)
/// - Zero line crossovers: Momentum shifts
/// - Divergence from price index: Trend reversal signal
#[derive(Debug, Clone)]
pub struct McClellanOscillator {
    /// Fast EMA period (default: 19)
    fast_period: usize,
    /// Slow EMA period (default: 39)
    slow_period: usize,
    /// Use ratio-adjusted calculation (default: true)
    ratio_adjusted: bool,
}

impl Default for McClellanOscillator {
    fn default() -> Self {
        Self::new()
    }
}

impl McClellanOscillator {
    pub fn new() -> Self {
        Self {
            fast_period: 19,
            slow_period: 39,
            ratio_adjusted: true,
        }
    }

    pub fn with_periods(fast_period: usize, slow_period: usize) -> Self {
        Self {
            fast_period,
            slow_period,
            ratio_adjusted: true,
        }
    }

    pub fn with_ratio_adjusted(mut self, ratio_adjusted: bool) -> Self {
        self.ratio_adjusted = ratio_adjusted;
        self
    }

    /// Calculate EMA smoothing factor
    fn ema_multiplier(period: usize) -> f64 {
        2.0 / (period as f64 + 1.0)
    }

    /// Calculate EMA series
    fn calculate_ema(&self, data: &[f64], period: usize) -> Vec<f64> {
        if data.is_empty() {
            return Vec::new();
        }

        let mut result = vec![f64::NAN; data.len()];
        let multiplier = Self::ema_multiplier(period);

        // Find first valid value for initial SMA
        let mut sum = 0.0;
        let mut count = 0;
        let mut start_idx = 0;

        for (i, &value) in data.iter().enumerate() {
            if !value.is_nan() {
                sum += value;
                count += 1;
                if count == period {
                    start_idx = i;
                    break;
                }
            }
        }

        if count < period {
            return result;
        }

        // Initial SMA as first EMA value
        let mut ema = sum / period as f64;
        result[start_idx] = ema;

        // Calculate EMA for remaining values
        for i in (start_idx + 1)..data.len() {
            if !data[i].is_nan() {
                ema = (data[i] - ema) * multiplier + ema;
                result[i] = ema;
            }
        }

        result
    }

    /// Calculate net advances (optionally ratio-adjusted)
    fn calculate_net_advances(&self, data: &BreadthSeries) -> Vec<f64> {
        if self.ratio_adjusted {
            data.advances
                .iter()
                .zip(data.declines.iter())
                .map(|(a, d)| {
                    let total = a + d;
                    if total == 0.0 {
                        0.0
                    } else {
                        (a - d) / total * 1000.0 // Scale to typical McClellan range
                    }
                })
                .collect()
        } else {
            data.net_advances()
        }
    }

    /// Calculate McClellan Oscillator from BreadthSeries
    pub fn calculate(&self, data: &BreadthSeries) -> Vec<f64> {
        let net_advances = self.calculate_net_advances(data);

        let fast_ema = self.calculate_ema(&net_advances, self.fast_period);
        let slow_ema = self.calculate_ema(&net_advances, self.slow_period);

        fast_ema
            .iter()
            .zip(slow_ema.iter())
            .map(|(f, s)| {
                if f.is_nan() || s.is_nan() {
                    f64::NAN
                } else {
                    f - s
                }
            })
            .collect()
    }

    /// Calculate and return all components (oscillator, fast EMA, slow EMA)
    pub fn calculate_with_components(&self, data: &BreadthSeries) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let net_advances = self.calculate_net_advances(data);

        let fast_ema = self.calculate_ema(&net_advances, self.fast_period);
        let slow_ema = self.calculate_ema(&net_advances, self.slow_period);

        let oscillator: Vec<f64> = fast_ema
            .iter()
            .zip(slow_ema.iter())
            .map(|(f, s)| {
                if f.is_nan() || s.is_nan() {
                    f64::NAN
                } else {
                    f - s
                }
            })
            .collect();

        (oscillator, fast_ema, slow_ema)
    }
}

impl BreadthIndicator for McClellanOscillator {
    fn name(&self) -> &str {
        "McClellan Oscillator"
    }

    fn compute_breadth(&self, data: &BreadthSeries) -> Result<IndicatorOutput> {
        if data.len() < self.slow_period {
            return Err(IndicatorError::InsufficientData {
                required: self.slow_period,
                got: data.len(),
            });
        }

        let values = self.calculate(data);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.slow_period
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BreadthData;

    fn create_test_series(len: usize) -> BreadthSeries {
        let mut series = BreadthSeries::new();
        // Create oscillating advance/decline data
        for i in 0..len {
            let advances = 1500.0 + (i as f64 * 10.0).sin() * 200.0;
            let declines = 1400.0 + (i as f64 * 10.0).cos() * 150.0;
            series.push(BreadthData::from_ad(advances, declines));
        }
        series
    }

    #[test]
    fn test_mcclellan_oscillator_basic() {
        let mcclellan = McClellanOscillator::new();
        let series = create_test_series(50);
        let result = mcclellan.calculate(&series);

        assert_eq!(result.len(), 50);

        // First 38 values should be NaN (need slow_period = 39)
        for i in 0..38 {
            assert!(result[i].is_nan(), "Expected NaN at index {}", i);
        }

        // After slow period, values should be valid
        assert!(!result[39].is_nan());
    }

    #[test]
    fn test_mcclellan_oscillator_non_ratio() {
        let mcclellan = McClellanOscillator::new().with_ratio_adjusted(false);
        let series = create_test_series(50);
        let result = mcclellan.calculate(&series);

        assert_eq!(result.len(), 50);
        assert!(!result[39].is_nan());
    }

    #[test]
    fn test_mcclellan_components() {
        let mcclellan = McClellanOscillator::new();
        let series = create_test_series(50);
        let (oscillator, fast, slow) = mcclellan.calculate_with_components(&series);

        assert_eq!(oscillator.len(), 50);
        assert_eq!(fast.len(), 50);
        assert_eq!(slow.len(), 50);

        // Verify oscillator = fast - slow where both are valid
        for i in 39..50 {
            let expected = fast[i] - slow[i];
            assert!(
                (oscillator[i] - expected).abs() < 1e-10,
                "Mismatch at index {}: {} vs {}",
                i,
                oscillator[i],
                expected
            );
        }
    }

    #[test]
    fn test_mcclellan_insufficient_data() {
        let mcclellan = McClellanOscillator::new();
        let series = create_test_series(30); // Less than slow_period (39)
        let result = mcclellan.compute_breadth(&series);

        assert!(result.is_err());
    }

    #[test]
    fn test_mcclellan_custom_periods() {
        let mcclellan = McClellanOscillator::with_periods(10, 20);
        let series = create_test_series(30);
        let result = mcclellan.calculate(&series);

        assert_eq!(result.len(), 30);
        // Should have valid values earlier with shorter periods
        assert!(!result[20].is_nan());
    }
}
