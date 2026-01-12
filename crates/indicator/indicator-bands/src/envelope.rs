//! Envelope (Moving Average Envelope) implementation.
//!
//! Price envelope bands based on percentage deviation from a moving average.

use indicator_spi::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};

/// Moving Average Envelope.
///
/// Price envelope indicator consisting of:
/// - Middle Band: SMA or EMA of close
/// - Upper Band: Middle + (percentage x Middle)
/// - Lower Band: Middle - (percentage x Middle)
///
/// Envelopes are used to identify overbought and oversold conditions,
/// as well as trend direction.
#[derive(Debug, Clone)]
pub struct Envelope {
    /// Period for the moving average.
    period: usize,
    /// Percentage deviation for bands (e.g., 0.025 for 2.5%).
    percentage: f64,
    /// Use EMA instead of SMA.
    use_ema: bool,
}

impl Envelope {
    /// Create a new Envelope indicator.
    ///
    /// # Arguments
    /// * `period` - Period for the moving average (typically 20)
    /// * `percentage` - Percentage deviation as decimal (e.g., 0.025 for 2.5%)
    /// * `use_ema` - If true, use EMA; otherwise use SMA
    pub fn new(period: usize, percentage: f64, use_ema: bool) -> Self {
        Self {
            period,
            percentage,
            use_ema,
        }
    }

    /// Create with SMA (default).
    pub fn with_sma(period: usize, percentage: f64) -> Self {
        Self::new(period, percentage, false)
    }

    /// Create with EMA.
    pub fn with_ema(period: usize, percentage: f64) -> Self {
        Self::new(period, percentage, true)
    }

    /// Calculate SMA values.
    fn calculate_sma(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period || self.period == 0 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period - 1];
        let mut sum: f64 = data[0..self.period].iter().sum();
        result.push(sum / self.period as f64);

        for i in self.period..n {
            sum = sum - data[i - self.period] + data[i];
            result.push(sum / self.period as f64);
        }

        result
    }

    /// Calculate EMA values.
    fn calculate_ema(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period || self.period == 0 {
            return vec![f64::NAN; n];
        }

        let alpha = 2.0 / (self.period as f64 + 1.0);
        let mut result = vec![f64::NAN; self.period - 1];

        // Initial SMA as seed
        let initial_sma: f64 = data[0..self.period].iter().sum::<f64>() / self.period as f64;
        result.push(initial_sma);

        // EMA calculation
        let mut ema = initial_sma;
        for i in self.period..n {
            ema = alpha * data[i] + (1.0 - alpha) * ema;
            result.push(ema);
        }

        result
    }

    /// Calculate Envelope bands (middle, upper, lower).
    pub fn calculate(&self, close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();

        // Calculate moving average (middle band)
        let middle = if self.use_ema {
            self.calculate_ema(close)
        } else {
            self.calculate_sma(close)
        };

        // Calculate upper and lower bands
        let mut upper = Vec::with_capacity(n);
        let mut lower = Vec::with_capacity(n);

        for i in 0..n {
            if middle[i].is_nan() {
                upper.push(f64::NAN);
                lower.push(f64::NAN);
            } else {
                upper.push(middle[i] * (1.0 + self.percentage));
                lower.push(middle[i] * (1.0 - self.percentage));
            }
        }

        (middle, upper, lower)
    }
}

impl Default for Envelope {
    fn default() -> Self {
        Self::with_sma(20, 0.025)
    }
}

impl TechnicalIndicator for Envelope {
    fn name(&self) -> &str {
        "Envelope"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let (middle, upper, lower) = self.calculate(&data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_envelope_sma() {
        let env = Envelope::with_sma(10, 0.02);
        let close: Vec<f64> = (0..30).map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0).collect();

        let (middle, upper, lower) = env.calculate(&close);

        assert_eq!(middle.len(), 30);
        assert_eq!(upper.len(), 30);
        assert_eq!(lower.len(), 30);

        // Check bands after warmup
        for i in 10..30 {
            if !middle[i].is_nan() {
                assert!(upper[i] > middle[i], "Upper should be above middle");
                assert!(lower[i] < middle[i], "Lower should be below middle");
                // Check percentage relationship
                let expected_upper = middle[i] * 1.02;
                let expected_lower = middle[i] * 0.98;
                assert!((upper[i] - expected_upper).abs() < 1e-10);
                assert!((lower[i] - expected_lower).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_envelope_ema() {
        let env = Envelope::with_ema(10, 0.03);
        let close: Vec<f64> = (0..30).map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0).collect();

        let (middle, upper, lower) = env.calculate(&close);

        // Check bands after warmup
        for i in 10..30 {
            if !middle[i].is_nan() {
                assert!(upper[i] > middle[i]);
                assert!(lower[i] < middle[i]);
            }
        }
    }

    #[test]
    fn test_envelope_default() {
        let env = Envelope::default();
        assert_eq!(env.period, 20);
        assert!((env.percentage - 0.025).abs() < 1e-10);
        assert!(!env.use_ema);
    }
}
