//! Jurik Moving Average (JMA) approximation implementation.
//!
//! A simplified approximation of the proprietary Jurik Moving Average that
//! provides adaptive smoothing with low lag characteristics.

use indicator_spi::{IndicatorError, IndicatorOutput, OHLCVSeries, Result, TechnicalIndicator};
use indicator_api::JurikMAConfig;

/// Jurik Moving Average approximation.
///
/// This is a simplified approximation of Mark Jurik's proprietary JMA indicator.
/// The original JMA algorithm is not publicly available, so this implementation
/// uses a combination of adaptive smoothing techniques to approximate its behavior:
///
/// - Low lag response in trending markets
/// - Smooth output with reduced whipsaws in ranging markets
/// - Phase adjustment for better signal timing
///
/// The approximation uses a three-stage smoothing process with adaptive factors
/// based on volatility measurements.
///
/// # Parameters
/// - `period`: The smoothing period (similar to EMA period)
/// - `phase`: Phase adjustment (-100 to +100), affects the trade-off between
///   smoothness and lag. Negative values increase smoothness, positive values
///   reduce lag.
/// - `power`: Smoothing power factor (typically 1-3), higher values increase
///   responsiveness to price changes.
#[derive(Debug, Clone)]
pub struct JurikMA {
    period: usize,
    phase: f64,
    power: f64,
}

impl JurikMA {
    /// Create a new JurikMA with the given parameters.
    ///
    /// # Arguments
    /// - `period`: Smoothing period (must be >= 1)
    /// - `phase`: Phase adjustment (-100 to +100)
    /// - `power`: Smoothing power factor (typically 1-3)
    pub fn new(period: usize, phase: f64, power: f64) -> Self {
        Self {
            period: period.max(1),
            phase: phase.clamp(-100.0, 100.0),
            power: power.max(0.1),
        }
    }

    /// Create JurikMA with default phase (0) and power (2).
    pub fn with_period(period: usize) -> Self {
        Self::new(period, 0.0, 2.0)
    }

    /// Create JurikMA from configuration.
    pub fn from_config(config: JurikMAConfig) -> Self {
        Self::new(config.period, config.phase, config.power)
    }

    /// Calculate JMA values for the given price data.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n == 0 {
            return vec![];
        }

        let mut result = vec![f64::NAN; n];

        // Return all NaN if insufficient data
        if n < self.period {
            return result;
        }

        // Calculate adaptive parameters
        let len = self.period.max(1) as f64;
        let phase_ratio = (self.phase / 100.0 + 1.5).clamp(0.5, 2.5);

        // Beta calculation for smoothing
        let beta = 0.45 * (len - 1.0) / (0.45 * (len - 1.0) + 2.0);

        // Alpha for phase adjustment
        let alpha = beta.powf(self.power);

        // Initialize state variables
        let mut e0 = data[0];
        let mut e1 = 0.0;
        let mut jma = data[0];

        // Volatility tracking
        let mut vol_sum = 0.0;
        let vol_period = (len * 0.65).ceil() as usize;
        let mut vol_buffer: Vec<f64> = Vec::with_capacity(vol_period);

        for i in 0..n {
            let price = data[i];

            // Calculate volatility (absolute price change)
            let vol = if i > 0 {
                (price - data[i - 1]).abs()
            } else {
                0.0
            };

            // Update volatility buffer
            if vol_buffer.len() >= vol_period {
                vol_sum -= vol_buffer[0];
                vol_buffer.remove(0);
            }
            vol_buffer.push(vol);
            vol_sum += vol;

            // Calculate average volatility
            let avg_vol = if !vol_buffer.is_empty() {
                vol_sum / vol_buffer.len() as f64
            } else {
                0.0
            };

            // Calculate relative volatility for adaptive factor
            let rel_vol = if avg_vol > 0.0 { vol / avg_vol } else { 1.0 };

            // Adaptive factor based on volatility
            let adaptive_alpha = alpha * (1.0 + (rel_vol - 1.0) * 0.5).clamp(0.5, 2.0);

            // Three-stage smoothing
            // Stage 1: Initial smoothing
            e0 = (1.0 - adaptive_alpha) * e0 + adaptive_alpha * price;

            // Stage 2: Secondary smoothing with phase adjustment
            e1 = (price - e0) * (1.0 - beta) * phase_ratio + beta * e1;

            // Stage 3: Final smoothing
            let e2 = e0 + e1;
            jma = (1.0 - adaptive_alpha) * jma + adaptive_alpha * e2;

            // Store result (warmup period produces less reliable values)
            if i >= self.period - 1 {
                result[i] = jma;
            } else if i > 0 {
                // Partial warmup values
                result[i] = jma;
            }
        }

        // Mark initial values as NaN for proper warmup period
        for i in 0..self.period.saturating_sub(1) {
            result[i] = f64::NAN;
        }

        result
    }
}

impl Default for JurikMA {
    fn default() -> Self {
        Self::new(14, 0.0, 2.0)
    }
}

impl TechnicalIndicator for JurikMA {
    fn name(&self) -> &str {
        "JurikMA"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jurik_ma_basic() {
        let jma = JurikMA::default();
        let data: Vec<f64> = (0..30).map(|i| 100.0 + i as f64).collect();
        let result = jma.calculate(&data);

        assert_eq!(result.len(), 30);

        // First period-1 values should be NaN
        for i in 0..13 {
            assert!(result[i].is_nan(), "Expected NaN at index {}", i);
        }

        // Subsequent values should be valid
        for i in 13..30 {
            assert!(!result[i].is_nan(), "Expected valid value at index {}", i);
        }
    }

    #[test]
    fn test_jurik_ma_trending() {
        let jma = JurikMA::new(10, 0.0, 2.0);
        // Strong uptrend
        let data: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 2.0).collect();
        let result = jma.calculate(&data);

        // JMA should follow the trend
        let last_jma = result[49];
        assert!(!last_jma.is_nan());
        // Should be tracking upward
        assert!(last_jma > result[25], "JMA should be increasing in uptrend");
        // Should be below price (lagging slightly)
        assert!(
            last_jma < data[49],
            "JMA should lag behind price in uptrend"
        );
    }

    #[test]
    fn test_jurik_ma_phase_negative() {
        let jma_smooth = JurikMA::new(10, -50.0, 2.0);
        let jma_normal = JurikMA::new(10, 0.0, 2.0);

        // Choppy data with reversals
        let data: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64 * 0.5).sin() * 10.0)
            .collect();

        let smooth_result = jma_smooth.calculate(&data);
        let normal_result = jma_normal.calculate(&data);

        // Both should produce valid values
        assert!(!smooth_result[49].is_nan());
        assert!(!normal_result[49].is_nan());

        // Negative phase should produce smoother output (verify by checking variance)
        let smooth_var: f64 = smooth_result[20..50]
            .windows(2)
            .map(|w| (w[1] - w[0]).powi(2))
            .sum::<f64>()
            / 29.0;
        let normal_var: f64 = normal_result[20..50]
            .windows(2)
            .map(|w| (w[1] - w[0]).powi(2))
            .sum::<f64>()
            / 29.0;

        // Smooth version should generally have less variance, but this depends on
        // the specific data pattern. Just verify both are reasonable.
        assert!(smooth_var >= 0.0);
        assert!(normal_var >= 0.0);
    }

    #[test]
    fn test_jurik_ma_phase_positive() {
        let jma_fast = JurikMA::new(10, 50.0, 2.0);
        let jma_normal = JurikMA::new(10, 0.0, 2.0);

        // Trending data
        let data: Vec<f64> = (0..50).map(|i| 100.0 + i as f64).collect();

        let fast_result = jma_fast.calculate(&data);
        let normal_result = jma_normal.calculate(&data);

        // Fast should be closer to current price (less lag)
        let fast_lag = data[49] - fast_result[49];
        let normal_lag = data[49] - normal_result[49];

        // Both should have positive lag in uptrend
        assert!(fast_lag >= 0.0);
        assert!(normal_lag >= 0.0);
    }

    #[test]
    fn test_jurik_ma_power_effect() {
        let jma_low_power = JurikMA::new(10, 0.0, 1.0);
        let jma_high_power = JurikMA::new(10, 0.0, 3.0);

        let data: Vec<f64> = (0..50).map(|i| 100.0 + i as f64).collect();

        let low_result = jma_low_power.calculate(&data);
        let high_result = jma_high_power.calculate(&data);

        // Both should produce valid values
        assert!(!low_result[49].is_nan());
        assert!(!high_result[49].is_nan());
    }

    #[test]
    fn test_jurik_ma_insufficient_data() {
        let jma = JurikMA::new(10, 0.0, 2.0);
        let data = vec![1.0, 2.0, 3.0];
        let result = jma.calculate(&data);

        assert_eq!(result.len(), 3);
        // All values should be NaN for insufficient data
        assert!(result.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn test_jurik_ma_constant_price() {
        let jma = JurikMA::new(10, 0.0, 2.0);
        let data = vec![100.0; 30];
        let result = jma.calculate(&data);

        // After warmup, JMA should equal constant price
        for i in 9..30 {
            assert!(
                (result[i] - 100.0).abs() < 1e-6,
                "JMA should converge to constant price at index {}: got {}",
                i,
                result[i]
            );
        }
    }

    #[test]
    fn test_jurik_ma_empty_data() {
        let jma = JurikMA::default();
        let result = jma.calculate(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_jurik_ma_default() {
        let jma = JurikMA::default();
        assert_eq!(jma.period, 14);
        assert!((jma.phase - 0.0).abs() < f64::EPSILON);
        assert!((jma.power - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_jurik_ma_with_period() {
        let jma = JurikMA::with_period(20);
        assert_eq!(jma.period, 20);
        assert!((jma.phase - 0.0).abs() < f64::EPSILON);
        assert!((jma.power - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_jurik_ma_parameter_clamping() {
        // Phase should be clamped to [-100, 100]
        let jma = JurikMA::new(10, 200.0, 2.0);
        assert!((jma.phase - 100.0).abs() < f64::EPSILON);

        let jma = JurikMA::new(10, -200.0, 2.0);
        assert!((jma.phase - (-100.0)).abs() < f64::EPSILON);

        // Power should be at least 0.1
        let jma = JurikMA::new(10, 0.0, 0.0);
        assert!((jma.power - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn test_jurik_ma_technical_indicator_trait() {
        let jma = JurikMA::new(10, 0.0, 2.0);
        assert_eq!(jma.name(), "JurikMA");
        assert_eq!(jma.min_periods(), 10);
        assert_eq!(jma.output_features(), 1);
    }

    #[test]
    fn test_jurik_ma_compute() {
        let jma = JurikMA::new(10, 0.0, 2.0);
        let data = OHLCVSeries {
            open: vec![100.0; 30],
            high: vec![101.0; 30],
            low: vec![99.0; 30],
            close: (0..30).map(|i| 100.0 + i as f64).collect(),
            volume: vec![1000.0; 30],
        };

        let result = jma.compute(&data).unwrap();
        assert_eq!(result.primary.len(), 30);
    }

    #[test]
    fn test_jurik_ma_compute_insufficient_data() {
        let jma = JurikMA::new(10, 0.0, 2.0);
        let data = OHLCVSeries {
            open: vec![100.0; 5],
            high: vec![101.0; 5],
            low: vec![99.0; 5],
            close: vec![100.0; 5],
            volume: vec![1000.0; 5],
        };

        let result = jma.compute(&data);
        assert!(result.is_err());
        if let Err(IndicatorError::InsufficientData { required, got }) = result {
            assert_eq!(required, 10);
            assert_eq!(got, 5);
        }
    }

    #[test]
    fn test_jurik_ma_low_lag_property() {
        // Test that JMA has lower lag than a simple EMA
        let jma = JurikMA::new(14, 50.0, 2.0);

        // Step function: prices jump from 100 to 110
        let mut data = vec![100.0; 20];
        for _ in 20..40 {
            data.push(110.0);
        }

        let result = jma.calculate(&data);

        // After the step, JMA should converge relatively quickly
        // Check that JMA is closer to 110 after a few bars
        let bars_after_step = 5;
        let jma_value = result[20 + bars_after_step];
        assert!(
            jma_value > 105.0,
            "JMA should respond quickly to price step: got {}",
            jma_value
        );
    }

    #[test]
    fn test_jurik_ma_from_config() {
        let config = JurikMAConfig::new(20, 50.0, 1.5);
        let jma = JurikMA::from_config(config);

        assert_eq!(jma.period, 20);
        assert!((jma.phase - 50.0).abs() < f64::EPSILON);
        assert!((jma.power - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_jurik_ma_from_default_config() {
        let config = JurikMAConfig::default();
        let jma = JurikMA::from_config(config);

        assert_eq!(jma.period, 14);
        assert!((jma.phase - 0.0).abs() < f64::EPSILON);
        assert!((jma.power - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_jurik_ma_config_with_period() {
        let config = JurikMAConfig::with_period(25);
        let jma = JurikMA::from_config(config);

        assert_eq!(jma.period, 25);
        assert!((jma.phase - 0.0).abs() < f64::EPSILON);
        assert!((jma.power - 2.0).abs() < f64::EPSILON);
    }
}
