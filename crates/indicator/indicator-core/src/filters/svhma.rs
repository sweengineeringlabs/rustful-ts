//! Step Variable Horizontal Moving Average (SVHMA) implementation.
//!
//! A unified framework for step-based moving averages that only update
//! when price deviations exceed a configurable threshold.
//!
//! Based on the formal specification: "SVHMA: A Unified Framework for Step-Based Moving Averages"
//!
//! # Threshold Modes
//! - Fixed: Constant threshold value
//! - Percentage: θ = p × |x|
//! - ATR: θ = k × ATR
//! - StdDev: θ = k × σ(x)
//! - ChangeVolatility: θ = k × σ(Δx)
//! - Donchian: θ = k × (H_n - L_n)
//! - VHF: θ = k × (1 - VHF) × ATR
//!
//! # Update Modes
//! - SnapToPrice: φ = x_t
//! - SnapToMA: φ = MA_t
//! - Damped: φ = y + α(x - y)
//! - Quantized: φ = y + ⌊δ/s⌋ × s
//! - VhfAdaptive: VHF-modulated EMA

use indicator_spi::{TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries};
use indicator_api::{SVHMAConfig, SVHMAThresholdMode, SVHMAUpdateMode};

/// Step Variable Horizontal Moving Average.
///
/// A non-linear, event-triggered, threshold-adaptive, state-holding filter
/// that updates only when price deviations exceed a configurable dynamic threshold.
#[derive(Debug, Clone)]
pub struct SVHMA {
    config: SVHMAConfig,
}

impl SVHMA {
    pub fn new(period: usize) -> Self {
        Self {
            config: SVHMAConfig::new(period),
        }
    }

    pub fn from_config(config: SVHMAConfig) -> Self {
        Self { config }
    }

    /// Create SVHMA matching filtered_averages.mq5 behavior.
    pub fn filtered_average(period: usize, filter: f64) -> Self {
        Self {
            config: SVHMAConfig::filtered_average(period, filter),
        }
    }

    /// Create SVHMA matching step_vhf_adaptive_vma.mq5 behavior.
    pub fn step_vhf_adaptive(period: usize, step_size: f64) -> Self {
        Self {
            config: SVHMAConfig::step_vhf_adaptive(period, step_size),
        }
    }

    /// Calculate SVHMA values from price data.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let period = self.config.period;

        if n < period + 1 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; n];

        // Pre-calculate required series based on modes
        let ma = self.calculate_ma(data);
        let thresholds = self.calculate_thresholds(data, &ma);
        let vhf = if matches!(self.config.threshold_mode, SVHMAThresholdMode::Vhf)
            || matches!(self.config.update_mode, SVHMAUpdateMode::VhfAdaptive)
        {
            Some(self.calculate_vhf(data))
        } else {
            None
        };

        // Initialize first valid value
        result[period] = match self.config.update_mode {
            SVHMAUpdateMode::Quantized | SVHMAUpdateMode::VhfAdaptive
                if self.config.step_size > 0.0 =>
            {
                (data[period] / self.config.step_size).round() * self.config.step_size
            }
            _ => ma[period],
        };

        // VMA state for VhfAdaptive mode
        let mut vma = result[period];
        let alpha_base = 2.0 / (period as f64 + 1.0);

        // Main calculation loop
        for i in (period + 1)..n {
            let prev_value = result[i - 1];
            let current_price = data[i];
            let threshold = thresholds[i];

            // Calculate deviation based on threshold mode
            let deviation = match self.config.threshold_mode {
                SVHMAThresholdMode::ChangeVolatility => {
                    // Compare change magnitude to threshold
                    (ma[i] - ma[i - 1]).abs()
                }
                _ => {
                    // Compare price deviation from previous output
                    (current_price - prev_value).abs()
                }
            };

            // Update VMA if using VhfAdaptive mode
            if let Some(ref vhf_values) = vhf {
                let vhf_val = vhf_values[i];
                if !vhf_val.is_nan() {
                    let alpha = alpha_base * vhf_val * 2.0;
                    vma = vma + alpha * (current_price - vma);
                }
            }

            // Check if update condition is met
            let should_update = !threshold.is_nan() && deviation > threshold;

            if should_update {
                // Calculate new value based on update mode
                let new_value = self.calculate_update(
                    i,
                    current_price,
                    prev_value,
                    &ma,
                    vma,
                );

                // Apply directional constraint if enabled
                if self.config.directional {
                    let direction = (current_price - prev_value).signum();
                    if direction > 0.0 {
                        result[i] = prev_value.max(new_value);
                    } else {
                        result[i] = prev_value.min(new_value);
                    }
                } else {
                    result[i] = new_value;
                }
            } else {
                // Hold previous value
                result[i] = prev_value;
            }
        }

        result
    }

    /// Calculate SVHMA from OHLCV data (uses close prices by default).
    pub fn calculate_ohlcv(&self, data: &OHLCVSeries) -> Vec<f64> {
        self.calculate(&data.close)
    }

    /// Calculate simple moving average.
    fn calculate_ma(&self, data: &[f64]) -> Vec<f64> {
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

    /// Calculate thresholds based on the configured mode.
    fn calculate_thresholds(&self, data: &[f64], ma: &[f64]) -> Vec<f64> {
        let n = data.len();
        let _period = self.config.period;
        let k = self.config.threshold_multiplier;

        match self.config.threshold_mode {
            SVHMAThresholdMode::Fixed => {
                vec![self.config.fixed_threshold; n]
            }

            SVHMAThresholdMode::Percentage => {
                data.iter().map(|&x| k * x.abs()).collect()
            }

            SVHMAThresholdMode::Atr => {
                self.calculate_atr_threshold(data, k)
            }

            SVHMAThresholdMode::StdDev => {
                self.calculate_stddev_threshold(data, k)
            }

            SVHMAThresholdMode::ChangeVolatility => {
                self.calculate_change_volatility_threshold(ma, k)
            }

            SVHMAThresholdMode::Donchian => {
                self.calculate_donchian_threshold(data, k)
            }

            SVHMAThresholdMode::Vhf => {
                self.calculate_vhf_threshold(data, k)
            }
        }
    }

    /// Calculate ATR-based threshold: θ = k × ATR
    fn calculate_atr_threshold(&self, data: &[f64], k: f64) -> Vec<f64> {
        let n = data.len();
        let period = self.config.period;
        let mut result = vec![f64::NAN; n];

        if n < period + 1 {
            return result;
        }

        // Calculate true ranges (simplified for close-only data)
        let mut tr = vec![0.0; n];
        for i in 1..n {
            tr[i] = (data[i] - data[i - 1]).abs();
        }

        // Calculate ATR as SMA of TR
        for i in period..n {
            let atr: f64 = tr[(i - period + 1)..=i].iter().sum::<f64>() / period as f64;
            result[i] = k * atr;
        }

        result
    }

    /// Calculate standard deviation threshold: θ = k × σ(x)
    fn calculate_stddev_threshold(&self, data: &[f64], k: f64) -> Vec<f64> {
        let n = data.len();
        let period = self.config.period;
        let mut result = vec![f64::NAN; n];

        if n < period {
            return result;
        }

        for i in (period - 1)..n {
            let start = i + 1 - period; // Reorder to avoid overflow
            let window = &data[start..=i];
            let mean: f64 = window.iter().sum::<f64>() / period as f64;
            let variance: f64 = window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / period as f64;
            result[i] = k * variance.sqrt();
        }

        result
    }

    /// Calculate change volatility threshold: θ = k × σ(Δx)
    fn calculate_change_volatility_threshold(&self, ma: &[f64], k: f64) -> Vec<f64> {
        let n = ma.len();
        let period = self.config.period;
        let mut result = vec![f64::NAN; n];

        if n < period + 1 {
            return result;
        }

        // Calculate absolute changes
        let mut changes = vec![0.0; n];
        for i in 1..n {
            if !ma[i].is_nan() && !ma[i - 1].is_nan() {
                changes[i] = (ma[i] - ma[i - 1]).abs();
            }
        }

        // Calculate rolling std dev of changes
        for i in period..n {
            let start = i + 1 - period; // Reorder to avoid overflow
            let window = &changes[start..=i];
            let mean: f64 = window.iter().sum::<f64>() / period as f64;
            let variance: f64 = window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / period as f64;
            result[i] = k * variance.sqrt();
        }

        result
    }

    /// Calculate Donchian range threshold: θ = k × (H_n - L_n)
    fn calculate_donchian_threshold(&self, data: &[f64], k: f64) -> Vec<f64> {
        let n = data.len();
        let period = self.config.period;
        let mut result = vec![f64::NAN; n];

        if n < period {
            return result;
        }

        for i in (period - 1)..n {
            let start = i + 1 - period; // Reorder to avoid overflow
            let window = &data[start..=i];
            let high = window.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let low = window.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            result[i] = k * (high - low);
        }

        result
    }

    /// Calculate VHF-based threshold: θ = k × (1 - VHF) × ATR
    fn calculate_vhf_threshold(&self, data: &[f64], k: f64) -> Vec<f64> {
        let n = data.len();
        let vhf = self.calculate_vhf(data);
        let atr = self.calculate_atr_threshold(data, 1.0);

        let mut result = vec![f64::NAN; n];
        for i in 0..n {
            if !vhf[i].is_nan() && !atr[i].is_nan() {
                result[i] = k * (1.0 - vhf[i]) * atr[i];
            }
        }

        result
    }

    /// Calculate Vertical Horizontal Filter values.
    fn calculate_vhf(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let period = self.config.period;
        let mut result = vec![f64::NAN; n];

        if n < period + 1 {
            return result;
        }

        for i in period..n {
            let window = &data[(i - period)..=i];

            // Numerator: highest - lowest
            let highest = window.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let lowest = window.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let numerator = highest - lowest;

            // Denominator: sum of absolute changes
            let mut denominator = 0.0;
            for j in 1..=period {
                let idx = i - period + j;
                denominator += (data[idx] - data[idx - 1]).abs();
            }

            if denominator != 0.0 {
                result[i] = (numerator / denominator).clamp(0.0, 1.0);
            }
        }

        result
    }

    /// Calculate the new value based on update mode.
    fn calculate_update(
        &self,
        i: usize,
        current_price: f64,
        prev_value: f64,
        ma: &[f64],
        vma: f64,
    ) -> f64 {
        match self.config.update_mode {
            SVHMAUpdateMode::SnapToPrice => current_price,

            SVHMAUpdateMode::SnapToMA => ma[i],

            SVHMAUpdateMode::Damped => {
                let alpha = self.config.damping_factor;
                prev_value + alpha * (current_price - prev_value)
            }

            SVHMAUpdateMode::Quantized => {
                let step = self.config.step_size;
                let delta = ma[i] - prev_value;
                let steps = (delta / step).trunc();
                prev_value + steps * step
            }

            SVHMAUpdateMode::VhfAdaptive => {
                // Use pre-calculated VMA, with quantization if step_size > 0
                if self.config.step_size > 0.0 {
                    let delta = vma - prev_value;
                    let steps = (delta / self.config.step_size).trunc();
                    prev_value + steps * self.config.step_size
                } else {
                    vma
                }
            }
        }
    }
}

impl Default for SVHMA {
    fn default() -> Self {
        Self::from_config(SVHMAConfig::default())
    }
}

impl TechnicalIndicator for SVHMA {
    fn name(&self) -> &str {
        "SVHMA"
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

// ============================================================================
// Convenience type aliases for common configurations
// ============================================================================

/// Deviation Filtered Average - matches filtered_averages.mq5
pub type DeviationFilteredAverage = SVHMA;

/// Step VHF Adaptive VMA - matches step_vhf_adaptive_vma.mq5
pub type StepVhfAdaptiveVMA = SVHMA;

// ============================================================================
// Tests
// ============================================================================

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
    fn test_svhma_default() {
        let svhma = SVHMA::default();
        let data = sample_data();
        let result = svhma.calculate(&data);

        // Should have NaN initially
        assert!(result[0].is_nan());
        // After period, should have values
        assert!(!result[29].is_nan());
    }

    #[test]
    fn test_svhma_step_behavior() {
        let svhma = SVHMA::new(5);
        let data = ranging_data();
        let result = svhma.calculate(&data);

        // In ranging market with small movements, SVHMA should hold values
        // Count how many times value changed
        let mut changes = 0;
        for i in 6..result.len() {
            if !result[i].is_nan() && !result[i - 1].is_nan() && result[i] != result[i - 1] {
                changes += 1;
            }
        }

        // Should have fewer changes than a regular MA would
        assert!(changes < data.len() / 2);
    }

    #[test]
    fn test_filtered_average_preset() {
        let svhma = SVHMA::filtered_average(14, 2.5);
        let data = sample_data();
        let result = svhma.calculate(&data);

        assert!(!result[29].is_nan());
        assert!(result[29] > 100.0); // Should follow the uptrend
    }

    #[test]
    fn test_step_vhf_adaptive_preset() {
        let svhma = SVHMA::step_vhf_adaptive(14, 1.0);
        let data = sample_data();
        let result = svhma.calculate(&data);

        assert!(!result[29].is_nan());

        // Output should be quantized to step size
        for &val in result.iter().filter(|v| !v.is_nan()) {
            let remainder = val % 1.0;
            assert!(remainder.abs() < 0.001 || (1.0 - remainder).abs() < 0.001);
        }
    }

    #[test]
    fn test_directional_mode() {
        let config = SVHMAConfig::new(5).with_directional(true);
        let svhma = SVHMA::from_config(config);

        // Uptrending data
        let data: Vec<f64> = (0..30).map(|i| 100.0 + i as f64).collect();
        let result = svhma.calculate(&data);

        // In directional mode with uptrend, values should only increase
        for i in 6..result.len() {
            if !result[i].is_nan() && !result[i - 1].is_nan() {
                assert!(result[i] >= result[i - 1]);
            }
        }
    }

    #[test]
    fn test_threshold_modes() {
        let data = sample_data();

        // Test each threshold mode produces valid output
        let modes = [
            SVHMAThresholdMode::Fixed,
            SVHMAThresholdMode::Percentage,
            SVHMAThresholdMode::Atr,
            SVHMAThresholdMode::StdDev,
            SVHMAThresholdMode::ChangeVolatility,
            SVHMAThresholdMode::Donchian,
            SVHMAThresholdMode::Vhf,
        ];

        for mode in modes {
            let config = SVHMAConfig::new(5).with_threshold_mode(mode);
            let svhma = SVHMA::from_config(config);
            let result = svhma.calculate(&data);

            assert!(!result[29].is_nan(), "Mode {:?} produced NaN at end", mode);
        }
    }

    #[test]
    fn test_update_modes() {
        let data = sample_data();

        // Test each update mode produces valid output
        let modes = [
            SVHMAUpdateMode::SnapToPrice,
            SVHMAUpdateMode::SnapToMA,
            SVHMAUpdateMode::Damped,
            SVHMAUpdateMode::Quantized,
            SVHMAUpdateMode::VhfAdaptive,
        ];

        for mode in modes {
            let config = SVHMAConfig::new(5).with_update_mode(mode);
            let svhma = SVHMA::from_config(config);
            let result = svhma.calculate(&data);

            assert!(!result[29].is_nan(), "Mode {:?} produced NaN at end", mode);
        }
    }

    #[test]
    fn test_insufficient_data() {
        let svhma = SVHMA::new(20);
        let data = vec![100.0; 10]; // Less than period
        let result = svhma.calculate(&data);

        // All values should be NaN
        assert!(result.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn test_technical_indicator_trait() {
        let svhma = SVHMA::default();
        assert_eq!(svhma.name(), "SVHMA");
        assert_eq!(svhma.min_periods(), 15); // period + 1
        assert_eq!(svhma.output_features(), 1);
    }
}
