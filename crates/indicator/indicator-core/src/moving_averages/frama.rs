//! Fractal Adaptive Moving Average (FRAMA) implementation.
//!
//! An adaptive moving average that uses fractal geometry to adjust smoothing.

use indicator_spi::{TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries};
use indicator_api::FRAMAConfig;

/// Fractal Adaptive Moving Average (FRAMA).
///
/// FRAMA uses fractal dimension to adapt the smoothing factor. When the market
/// is trending (low fractal dimension), FRAMA responds quickly. When the market
/// is choppy (high fractal dimension), FRAMA smooths heavily to filter noise.
///
/// The fractal dimension is calculated using the difference between highs and
/// lows across different time frames.
#[derive(Debug, Clone)]
pub struct FRAMA {
    period: usize,
    /// Fast EMA constant (SC = 1 when D is low)
    fast_sc: f64,
    /// Slow EMA constant (SC = 0.01 when D is high)
    slow_sc: f64,
}

impl FRAMA {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            fast_sc: 1.0,
            slow_sc: 0.01,
        }
    }

    pub fn with_smoothing(period: usize, fast_sc: f64, slow_sc: f64) -> Self {
        Self { period, fast_sc, slow_sc }
    }

    pub fn from_config(config: FRAMAConfig) -> Self {
        Self {
            period: config.period,
            fast_sc: config.fast_sc.unwrap_or(1.0),
            slow_sc: config.slow_sc.unwrap_or(0.01),
        }
    }

    /// Calculate FRAMA values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        if data.len() < self.period || self.period < 2 {
            return vec![f64::NAN; data.len()];
        }

        let half = self.period / 2;
        let mut result = vec![f64::NAN; self.period - 1];

        // Initialize with first price at period position
        result.push(data[self.period - 1]);
        let mut frama = data[self.period - 1];

        for i in self.period..data.len() {
            let window = &data[(i + 1 - self.period)..=i];

            // Calculate fractal dimension
            let alpha = self.calculate_alpha(window, half);

            // Apply FRAMA formula
            frama = alpha * data[i] + (1.0 - alpha) * frama;
            result.push(frama);
        }

        result
    }

    /// Calculate the adaptive alpha using fractal dimension.
    fn calculate_alpha(&self, window: &[f64], half: usize) -> f64 {
        let n = window.len();
        if n < 2 || half == 0 {
            return self.slow_sc;
        }

        // Calculate N1: (High - Low) for first half
        let first_half = &window[0..half];
        let n1 = self.range(first_half);

        // Calculate N2: (High - Low) for second half
        let second_half = &window[half..];
        let n2 = self.range(second_half);

        // Calculate N3: (High - Low) for full period
        let n3 = self.range(window);

        if n3 == 0.0 || (n1 + n2) == 0.0 {
            return self.slow_sc;
        }

        // Calculate fractal dimension D
        let d = ((n1 + n2 - n3) / n3).ln() / 2.0_f64.ln();

        // Calculate alpha from D
        // D ranges from 1 (trending) to 2 (choppy)
        let alpha = ((d - 1.0) * (self.slow_sc - self.fast_sc) + self.fast_sc)
            .max(self.slow_sc)
            .min(self.fast_sc);

        alpha
    }

    /// Calculate range (max - min) of a slice.
    fn range(&self, data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        max - min
    }
}

impl Default for FRAMA {
    fn default() -> Self {
        Self::from_config(FRAMAConfig::default())
    }
}

impl TechnicalIndicator for FRAMA {
    fn name(&self) -> &str {
        "FRAMA"
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frama() {
        let frama = FRAMA::new(10);
        let data: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        let result = frama.calculate(&data);

        // First 9 values should be NaN
        for i in 0..9 {
            assert!(result[i].is_nan());
        }
        // Subsequent values should be valid
        for i in 9..20 {
            assert!(!result[i].is_nan());
        }
    }

    #[test]
    fn test_frama_trending() {
        let frama = FRAMA::new(10);
        // Strong uptrend
        let data: Vec<f64> = (0..20).map(|i| 100.0 + i as f64 * 2.0).collect();
        let result = frama.calculate(&data);

        // In trending market, FRAMA should track upward
        let last_frama = result[19];
        // FRAMA should be valid
        assert!(!last_frama.is_nan());
        // Should be moving up with the trend
        assert!(last_frama > result[9]);
    }

    #[test]
    fn test_frama_insufficient_data() {
        let frama = FRAMA::new(10);
        let data = vec![1.0, 2.0, 3.0];
        let result = frama.calculate(&data);

        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn test_frama_default() {
        let frama = FRAMA::default();
        assert_eq!(frama.period, 16);
    }

    #[test]
    fn test_frama_technical_indicator_trait() {
        let frama = FRAMA::new(10);
        assert_eq!(frama.name(), "FRAMA");
        assert_eq!(frama.min_periods(), 10);
    }
}
