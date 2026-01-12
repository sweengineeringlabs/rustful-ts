//! Deviation Filtered Average implementation.
//!
//! A step-based moving average that only updates when price changes exceed
//! a threshold based on the standard deviation of recent price changes.
//!
//! Based on: filtered_averages.mq5 by mladen (2018)
//!
//! # Algorithm
//! 1. Calculate base MA (EMA by default)
//! 2. Track absolute changes: Δ = |MA_t - MA_{t-1}|
//! 3. Calculate std dev of changes: σ(Δ)
//! 4. Threshold: θ = filter × σ(Δ)
//! 5. If current change < threshold: hold previous value
//! 6. Otherwise: snap to MA
//!
//! # SVHMA Classification
//! - Threshold Mode: 7.9 Change Volatility
//! - Update Mode: 9.2 Snap-to-MA

use indicator_spi::{TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries};
use indicator_api::DeviationFilteredAverageConfig;

/// Deviation Filtered Average.
///
/// A noise-filtering moving average that only updates when price changes
/// are statistically significant relative to recent volatility.
#[derive(Debug, Clone)]
pub struct DeviationFilteredAverage {
    config: DeviationFilteredAverageConfig,
}

impl DeviationFilteredAverage {
    pub fn new(period: usize, filter: f64) -> Self {
        Self {
            config: DeviationFilteredAverageConfig::new(period, filter),
        }
    }

    pub fn from_config(config: DeviationFilteredAverageConfig) -> Self {
        Self { config }
    }

    /// Calculate filtered average values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let period = self.config.period;

        if n < period + 1 {
            return vec![f64::NAN; n];
        }

        // Calculate base MA
        let ma = if self.config.use_ema {
            self.calculate_ema(data)
        } else {
            self.calculate_sma(data)
        };

        // Calculate absolute changes of MA
        let mut changes = vec![0.0; n];
        for i in 1..n {
            if !ma[i].is_nan() && !ma[i - 1].is_nan() {
                changes[i] = (ma[i] - ma[i - 1]).abs();
            }
        }

        // Calculate rolling statistics of changes
        let mut sum_change = vec![0.0; n];
        let mut sum_power = vec![0.0; n];

        for i in 1..n {
            if i <= period {
                // Warmup: accumulate
                sum_change[i] = changes[1..=i].iter().sum();
                let mean = sum_change[i] / i as f64;
                sum_power[i] = changes[1..=i]
                    .iter()
                    .map(|&c| (c - mean).powi(2))
                    .sum();
            } else {
                // Rolling update
                sum_change[i] = sum_change[i - 1] - changes[i - period] + changes[i];
                let mean = sum_change[i] / period as f64;

                // Recalculate power for the window
                let start = i + 1 - period;
                sum_power[i] = changes[start..=i]
                    .iter()
                    .map(|&c| (c - mean).powi(2))
                    .sum();
            }
        }

        // Apply filter
        let mut result = vec![f64::NAN; n];

        // Initialize with first valid MA value
        for i in 0..n {
            if !ma[i].is_nan() {
                result[i] = ma[i];
                break;
            }
        }

        for i in period..n {
            let prev_value = if i > 0 && !result[i - 1].is_nan() {
                result[i - 1]
            } else {
                ma[i]
            };

            // Calculate threshold: filter × std_dev(changes)
            let variance = sum_power[i] / period as f64;
            let threshold = self.config.filter * variance.sqrt();

            // Check if change exceeds threshold
            if changes[i] < threshold {
                // Hold previous value
                result[i] = prev_value;
            } else {
                // Snap to MA
                result[i] = ma[i];
            }
        }

        result
    }

    /// Calculate EMA.
    fn calculate_ema(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];

        if n == 0 {
            return result;
        }

        let alpha = 2.0 / (self.config.period as f64 + 1.0);
        result[0] = data[0];

        for i in 1..n {
            result[i] = alpha * data[i] + (1.0 - alpha) * result[i - 1];
        }

        result
    }

    /// Calculate SMA.
    fn calculate_sma(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let period = self.config.period;
        let mut result = vec![f64::NAN; n];

        if n < period {
            return result;
        }

        let mut sum: f64 = data[..period].iter().sum();
        result[period - 1] = sum / period as f64;

        for i in period..n {
            sum = sum - data[i - period] + data[i];
            result[i] = sum / period as f64;
        }

        result
    }

    /// Calculate with trend coloring info.
    /// Returns (values, colors) where colors: 0=neutral, 1=up, 2=down
    pub fn calculate_with_color(&self, data: &[f64]) -> (Vec<f64>, Vec<i32>) {
        let values = self.calculate(data);
        let n = values.len();
        let mut colors = vec![0i32; n];

        for i in 1..n {
            if values[i].is_nan() || values[i - 1].is_nan() {
                colors[i] = colors[i - 1];
            } else if values[i] > values[i - 1] {
                colors[i] = 1; // Up (green in MQL5)
            } else if values[i] < values[i - 1] {
                colors[i] = 2; // Down (pink in MQL5)
            } else {
                colors[i] = colors[i - 1];
            }
        }

        (values, colors)
    }
}

impl Default for DeviationFilteredAverage {
    fn default() -> Self {
        Self::from_config(DeviationFilteredAverageConfig::default())
    }
}

impl TechnicalIndicator for DeviationFilteredAverage {
    fn name(&self) -> &str {
        "DeviationFilteredAverage"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.config.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.config.period + 1,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.config.period + 1
    }

    fn output_features(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data() -> Vec<f64> {
        vec![
            100.0, 101.0, 102.5, 101.5, 103.0, 104.0, 103.5, 105.0, 106.0, 105.5,
            107.0, 108.0, 107.5, 109.0, 110.0, 109.5, 111.0, 112.0, 111.5, 113.0,
            114.0, 113.5, 115.0, 116.0, 115.5, 117.0, 118.0, 117.5, 119.0, 120.0,
        ]
    }

    fn ranging_data() -> Vec<f64> {
        vec![
            100.0, 100.5, 100.2, 100.8, 100.3, 100.7, 100.4, 100.6, 100.1, 100.9,
            100.2, 100.8, 100.3, 100.7, 100.4, 100.6, 100.1, 100.9, 100.2, 100.8,
            100.3, 100.7, 100.4, 100.6, 100.1, 100.9, 100.2, 100.8, 100.3, 100.5,
        ]
    }

    #[test]
    fn test_default() {
        let dfa = DeviationFilteredAverage::default();
        let data = sample_data();
        let result = dfa.calculate(&data);

        assert!(!result[29].is_nan());
    }

    #[test]
    fn test_filtering_behavior() {
        let dfa = DeviationFilteredAverage::new(5, 2.5);
        let data = ranging_data();
        let result = dfa.calculate(&data);

        // Count value changes
        let mut changes = 0;
        for i in 6..result.len() {
            if !result[i].is_nan() && !result[i - 1].is_nan() && result[i] != result[i - 1] {
                changes += 1;
            }
        }

        // In ranging market, should filter out most noise
        assert!(changes < data.len() / 2);
    }

    #[test]
    fn test_trending_market() {
        let dfa = DeviationFilteredAverage::new(5, 2.5);
        let data = sample_data();
        let result = dfa.calculate(&data);

        // In trending market, should follow the trend
        assert!(result[29] > 110.0);
    }

    #[test]
    fn test_with_color() {
        let dfa = DeviationFilteredAverage::new(5, 2.5);
        let data = sample_data();
        let (values, colors) = dfa.calculate_with_color(&data);

        assert_eq!(values.len(), colors.len());

        // In uptrend, should have some green (1) colors
        let up_count = colors.iter().filter(|&&c| c == 1).count();
        assert!(up_count > 0);
    }

    #[test]
    fn test_sma_variant() {
        let config = DeviationFilteredAverageConfig::new(14, 2.5).with_sma();
        let dfa = DeviationFilteredAverage::from_config(config);
        let data = sample_data();
        let result = dfa.calculate(&data);

        assert!(!result[29].is_nan());
    }

    #[test]
    fn test_technical_indicator_trait() {
        let dfa = DeviationFilteredAverage::default();
        assert_eq!(dfa.name(), "DeviationFilteredAverage");
        assert_eq!(dfa.min_periods(), 15);
        assert_eq!(dfa.output_features(), 1);
    }
}
