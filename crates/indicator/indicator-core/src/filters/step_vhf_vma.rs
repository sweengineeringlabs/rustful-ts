//! Step VHF Adaptive VMA implementation.
//!
//! A step-based variable moving average that uses the Vertical Horizontal Filter
//! to modulate the smoothing factor, with quantized step output.
//!
//! Based on: step_vhf_adaptive_vma.mq5 by mladen (2018)
//!
//! # Algorithm
//! 1. Calculate VHF = (Highest - Lowest) / Σ|changes|
//! 2. Modulate EMA alpha: α_t = α_base × VHF × 2
//! 3. Calculate VMA: VMA_t = VMA_{t-1} + α_t × (price - VMA_{t-1})
//! 4. Apply step filter: if |VMA - prev| < step_size, hold
//! 5. Quantize output to step_size increments
//!
//! # SVHMA Classification
//! - Threshold Mode: 7.1 Fixed (step_size)
//! - Update Mode: 9.4 + 9.5 (Quantized + VHF-Adaptive)

use indicator_spi::{TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries};
use indicator_api::StepVhfAdaptiveVMAConfig;

/// Step VHF Adaptive Variable Moving Average.
///
/// A trend-adaptive moving average that:
/// - Responds faster in trending markets (high VHF)
/// - Smooths more in ranging markets (low VHF)
/// - Outputs quantized step values for cleaner signals
#[derive(Debug, Clone)]
pub struct StepVhfAdaptiveVMA {
    config: StepVhfAdaptiveVMAConfig,
}

impl StepVhfAdaptiveVMA {
    pub fn new(period: usize, step_size: f64) -> Self {
        Self {
            config: StepVhfAdaptiveVMAConfig::new(period, step_size),
        }
    }

    pub fn from_config(config: StepVhfAdaptiveVMAConfig) -> Self {
        Self { config }
    }

    /// Calculate Step VHF Adaptive VMA values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let vhf_period = self.config.vhf_period;

        if n < vhf_period + 1 {
            return vec![f64::NAN; n];
        }

        // Calculate VHF values
        let vhf = self.calculate_vhf(data);

        // Calculate VHF-adaptive VMA
        let alpha_base = 2.0 / (self.config.vma_period as f64 + 1.0);
        let mut vma = vec![f64::NAN; n];
        vma[0] = data[0];

        for i in 1..n {
            let vhf_val = if vhf[i].is_nan() { 0.5 } else { vhf[i] };
            let alpha = alpha_base * vhf_val * 2.0;
            vma[i] = vma[i - 1] + alpha * (data[i] - vma[i - 1]);
        }

        // Apply step filter and quantization
        let mut result = vec![f64::NAN; n];
        let step_size = self.config.step_size;

        if step_size <= 0.0 {
            // No stepping, return raw VMA
            return vma;
        }

        // Initialize with quantized value
        result[0] = (vma[0] / step_size).round() * step_size;

        for i in 1..n {
            let prev = result[i - 1];
            let diff = vma[i] - prev;

            // Check if diff exceeds step threshold
            if diff.abs() < step_size {
                // Hold previous value
                result[i] = prev;
            } else {
                // Move by integer number of steps
                let steps = (diff / step_size).trunc() as i64;
                result[i] = prev + (steps as f64) * step_size;
            }
        }

        result
    }

    /// Calculate Vertical Horizontal Filter.
    fn calculate_vhf(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let period = self.config.vhf_period;
        let mut result = vec![f64::NAN; n];

        if n < period + 1 {
            return result;
        }

        // Pre-calculate absolute differences
        let mut diffs = vec![0.0; n];
        for i in 1..n {
            diffs[i] = (data[i] - data[i - 1]).abs();
        }

        // Calculate VHF for each bar
        for i in period..n {
            // Highest and lowest in window
            let window_start = i - period;
            let window = &data[window_start..=i];
            let highest = window.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let lowest = window.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let numerator = highest - lowest;

            // Sum of absolute changes
            let denominator: f64 = diffs[(window_start + 1)..=i].iter().sum();

            if denominator != 0.0 {
                result[i] = (numerator / denominator).clamp(0.0, 1.0);
            } else {
                result[i] = 0.0;
            }
        }

        result
    }

    /// Calculate with shadow and color info (matches MQL5 output).
    /// Returns (shadow, value, color) where:
    /// - shadow: same as value (displayed thicker in MQL5)
    /// - value: the step VMA value
    /// - color: 0=neutral, 1=up (green), 2=down (pink)
    pub fn calculate_with_color(&self, data: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<i32>) {
        let values = self.calculate(data);
        let n = values.len();
        let shadow = values.clone();
        let mut colors = vec![0i32; n];

        for i in 1..n {
            if values[i].is_nan() || values[i - 1].is_nan() {
                colors[i] = colors[i - 1];
            } else if values[i] > values[i - 1] {
                colors[i] = 1; // Up (green)
            } else if values[i] < values[i - 1] {
                colors[i] = 2; // Down (pink)
            } else {
                colors[i] = colors[i - 1];
            }
        }

        (shadow, values, colors)
    }

    /// Get the underlying VMA without step filtering.
    pub fn calculate_vma(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let vhf = self.calculate_vhf(data);
        let alpha_base = 2.0 / (self.config.vma_period as f64 + 1.0);

        let mut result = vec![f64::NAN; n];
        if n == 0 {
            return result;
        }

        result[0] = data[0];
        for i in 1..n {
            let vhf_val = if vhf[i].is_nan() { 0.5 } else { vhf[i] };
            let alpha = alpha_base * vhf_val * 2.0;
            result[i] = result[i - 1] + alpha * (data[i] - result[i - 1]);
        }

        result
    }

    /// Get VHF values for analysis.
    pub fn get_vhf(&self, data: &[f64]) -> Vec<f64> {
        self.calculate_vhf(data)
    }
}

impl Default for StepVhfAdaptiveVMA {
    fn default() -> Self {
        Self::from_config(StepVhfAdaptiveVMAConfig::default())
    }
}

impl TechnicalIndicator for StepVhfAdaptiveVMA {
    fn name(&self) -> &str {
        "StepVhfAdaptiveVMA"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.config.vhf_period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.config.vhf_period + 1,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.config.vhf_period + 1
    }

    fn output_features(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trending_data() -> Vec<f64> {
        (0..50).map(|i| 100.0 + i as f64 * 2.0).collect()
    }

    fn ranging_data() -> Vec<f64> {
        (0..50)
            .map(|i| 100.0 + ((i as f64) % 4.0) - 2.0)
            .collect()
    }

    fn sample_data() -> Vec<f64> {
        vec![
            100.0, 101.0, 102.5, 101.5, 103.0, 104.0, 103.5, 105.0, 106.0, 105.5,
            107.0, 108.0, 107.5, 109.0, 110.0, 109.5, 111.0, 112.0, 111.5, 113.0,
            114.0, 113.5, 115.0, 116.0, 115.5, 117.0, 118.0, 117.5, 119.0, 120.0,
        ]
    }

    #[test]
    fn test_default() {
        let vma = StepVhfAdaptiveVMA::default();
        let data = sample_data();
        let result = vma.calculate(&data);

        assert!(!result[29].is_nan());
    }

    #[test]
    fn test_quantized_output() {
        let vma = StepVhfAdaptiveVMA::new(14, 1.0);
        let data = sample_data();
        let result = vma.calculate(&data);

        // All non-NaN values should be multiples of step_size
        for &val in result.iter().filter(|v| !v.is_nan()) {
            let remainder = val % 1.0;
            assert!(
                remainder.abs() < 0.001 || (1.0 - remainder).abs() < 0.001,
                "Value {} is not quantized to step 1.0",
                val
            );
        }
    }

    #[test]
    fn test_step_behavior() {
        let vma = StepVhfAdaptiveVMA::new(5, 2.0);
        let data = ranging_data();
        let result = vma.calculate(&data);

        // In ranging market with step=2.0, should have few changes
        let mut changes = 0;
        for i in 1..result.len() {
            if !result[i].is_nan() && !result[i - 1].is_nan() && result[i] != result[i - 1] {
                changes += 1;
            }
        }

        // Should be significantly filtered
        assert!(changes < data.len() / 3);
    }

    #[test]
    fn test_vhf_adaptation() {
        let vma = StepVhfAdaptiveVMA::new(14, 0.0); // No stepping

        let trending = trending_data();
        let ranging = ranging_data();

        let vhf_trending = vma.get_vhf(&trending);
        let vhf_ranging = vma.get_vhf(&ranging);

        // VHF should be higher in trending market
        let avg_trending: f64 = vhf_trending.iter().filter(|v| !v.is_nan()).sum::<f64>()
            / vhf_trending.iter().filter(|v| !v.is_nan()).count() as f64;
        let avg_ranging: f64 = vhf_ranging.iter().filter(|v| !v.is_nan()).sum::<f64>()
            / vhf_ranging.iter().filter(|v| !v.is_nan()).count() as f64;

        assert!(avg_trending > avg_ranging);
    }

    #[test]
    fn test_with_color() {
        let vma = StepVhfAdaptiveVMA::new(5, 1.0);
        let data = trending_data();
        let (shadow, values, colors) = vma.calculate_with_color(&data);

        assert_eq!(shadow.len(), values.len());
        assert_eq!(values.len(), colors.len());

        // In strong uptrend, should have green (1) colors
        let up_count = colors.iter().filter(|&&c| c == 1).count();
        assert!(up_count > colors.len() / 2);
    }

    #[test]
    fn test_different_vhf_period() {
        let config = StepVhfAdaptiveVMAConfig::new(14, 1.0).with_vhf_period(28);
        let vma = StepVhfAdaptiveVMA::from_config(config);
        let data = sample_data();
        let result = vma.calculate(&data);

        assert!(!result[29].is_nan());
    }

    #[test]
    fn test_no_step() {
        let vma = StepVhfAdaptiveVMA::new(14, 0.0);
        let data = sample_data();
        let result = vma.calculate(&data);
        let raw_vma = vma.calculate_vma(&data);

        // Without stepping, should equal raw VMA
        for i in 0..result.len() {
            if !result[i].is_nan() {
                assert!((result[i] - raw_vma[i]).abs() < 0.0001);
            }
        }
    }

    #[test]
    fn test_technical_indicator_trait() {
        let vma = StepVhfAdaptiveVMA::default();
        assert_eq!(vma.name(), "StepVhfAdaptiveVMA");
        assert_eq!(vma.min_periods(), 15);
        assert_eq!(vma.output_features(), 1);
    }
}
