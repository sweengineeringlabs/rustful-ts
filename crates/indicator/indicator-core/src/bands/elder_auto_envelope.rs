//! Elder's AutoEnvelope implementation.
//!
//! Adaptive envelope bands that automatically adjust width based on recent volatility
//! using standard deviation.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};
use indicator_api::ElderAutoEnvelopeConfig;

/// Elder's AutoEnvelope.
///
/// An adaptive envelope indicator that uses EMA as the middle band and
/// standard deviation to automatically adjust band width based on market volatility.
///
/// **Algorithm:**
/// 1. Calculate EMA of close prices
/// 2. Calculate standard deviation of close prices over period
/// 3. Upper band = EMA + (multiplier x StdDev)
/// 4. Lower band = EMA - (multiplier x StdDev)
///
/// The bands adapt to market volatility automatically, expanding during volatile
/// periods and contracting during quiet markets.
#[derive(Debug, Clone)]
pub struct ElderAutoEnvelope {
    /// Period for the EMA calculation.
    ema_period: usize,
    /// Period for the standard deviation calculation.
    std_period: usize,
    /// Multiplier for standard deviation to determine band width.
    multiplier: f64,
}

impl ElderAutoEnvelope {
    /// Create a new ElderAutoEnvelope indicator.
    ///
    /// # Arguments
    /// * `ema_period` - Period for the EMA (default: 13)
    /// * `std_period` - Period for the standard deviation (default: 13)
    /// * `multiplier` - Standard deviation multiplier (default: 2.7)
    pub fn new(ema_period: usize, std_period: usize, multiplier: f64) -> Self {
        Self {
            ema_period,
            std_period,
            multiplier,
        }
    }

    /// Create from configuration.
    pub fn from_config(config: ElderAutoEnvelopeConfig) -> Self {
        Self {
            ema_period: config.ema_period,
            std_period: config.std_period,
            multiplier: config.multiplier,
        }
    }

    /// Calculate EMA values.
    fn calculate_ema(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.ema_period || self.ema_period == 0 {
            return vec![f64::NAN; n];
        }

        let alpha = 2.0 / (self.ema_period as f64 + 1.0);
        let mut result = vec![f64::NAN; self.ema_period - 1];

        // Initial SMA as seed
        let initial_sma: f64 = data[0..self.ema_period].iter().sum::<f64>() / self.ema_period as f64;
        result.push(initial_sma);

        // EMA calculation
        let mut ema = initial_sma;
        for i in self.ema_period..n {
            ema = alpha * data[i] + (1.0 - alpha) * ema;
            result.push(ema);
        }

        result
    }

    /// Calculate rolling standard deviation.
    fn calculate_std(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.std_period || self.std_period == 0 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.std_period - 1];

        for i in (self.std_period - 1)..n {
            let start = i + 1 - self.std_period;
            let window = &data[start..=i];

            // Calculate mean
            let mean: f64 = window.iter().sum::<f64>() / self.std_period as f64;

            // Calculate variance (population)
            let variance: f64 = window.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / self.std_period as f64;

            result.push(variance.sqrt());
        }

        result
    }

    /// Calculate Elder's AutoEnvelope bands (middle, upper, lower).
    pub fn calculate(&self, close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();

        // Calculate EMA of close (middle band)
        let middle = self.calculate_ema(close);

        // Calculate standard deviation
        let std_values = self.calculate_std(close);

        // Calculate upper and lower bands
        let mut upper = Vec::with_capacity(n);
        let mut lower = Vec::with_capacity(n);

        for i in 0..n {
            if middle[i].is_nan() || std_values[i].is_nan() {
                upper.push(f64::NAN);
                lower.push(f64::NAN);
            } else {
                let band_offset = self.multiplier * std_values[i];
                upper.push(middle[i] + band_offset);
                lower.push(middle[i] - band_offset);
            }
        }

        (middle, upper, lower)
    }

    /// Calculate the band width as a percentage of price.
    pub fn band_width_percent(&self, close: &[f64]) -> Vec<f64> {
        let (middle, upper, lower) = self.calculate(close);
        middle.iter()
            .zip(upper.iter().zip(lower.iter()))
            .map(|(&m, (&u, &l))| {
                if m.is_nan() || u.is_nan() || l.is_nan() || m.abs() < 1e-10 {
                    f64::NAN
                } else {
                    (u - l) / m
                }
            })
            .collect()
    }

    /// Calculate %B (position within bands).
    pub fn percent_b(&self, close: &[f64]) -> Vec<f64> {
        let (_, upper, lower) = self.calculate(close);
        close.iter()
            .zip(upper.iter().zip(lower.iter()))
            .map(|(&price, (&u, &l))| {
                if u.is_nan() || l.is_nan() || (u - l).abs() < 1e-10 {
                    f64::NAN
                } else {
                    (price - l) / (u - l)
                }
            })
            .collect()
    }
}

impl Default for ElderAutoEnvelope {
    fn default() -> Self {
        Self {
            ema_period: 13,
            std_period: 13,
            multiplier: 2.7,
        }
    }
}

impl TechnicalIndicator for ElderAutoEnvelope {
    fn name(&self) -> &str {
        "ElderAutoEnvelope"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.ema_period.max(self.std_period);
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let (middle, upper, lower) = self.calculate(&data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }

    fn min_periods(&self) -> usize {
        self.ema_period.max(self.std_period)
    }

    fn output_features(&self) -> usize {
        3
    }
}

impl SignalIndicator for ElderAutoEnvelope {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (middle, upper, lower) = self.calculate(&data.close);

        let n = data.close.len();
        if n == 0 {
            return Ok(IndicatorSignal::Neutral);
        }

        let price = data.close[n - 1];
        let m = middle[n - 1];
        let u = upper[n - 1];
        let l = lower[n - 1];

        if u.is_nan() || l.is_nan() || m.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Price at/below lower band = potential buy (oversold)
        if price <= l {
            Ok(IndicatorSignal::Bullish)
        }
        // Price at/above upper band = potential sell (overbought)
        else if price >= u {
            Ok(IndicatorSignal::Bearish)
        }
        else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (_, upper, lower) = self.calculate(&data.close);

        let signals: Vec<_> = data.close.iter()
            .zip(upper.iter().zip(lower.iter()))
            .map(|(&price, (&u, &l))| {
                if u.is_nan() || l.is_nan() {
                    IndicatorSignal::Neutral
                } else if price <= l {
                    IndicatorSignal::Bullish
                } else if price >= u {
                    IndicatorSignal::Bearish
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

    fn create_test_data(n: usize) -> Vec<f64> {
        (0..n).map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0).collect()
    }

    #[test]
    fn test_elder_auto_envelope_basic() {
        let eae = ElderAutoEnvelope::new(10, 10, 2.0);
        let close = create_test_data(30);

        let (middle, upper, lower) = eae.calculate(&close);

        assert_eq!(middle.len(), 30);
        assert_eq!(upper.len(), 30);
        assert_eq!(lower.len(), 30);

        // Check warmup period
        for i in 0..9 {
            assert!(middle[i].is_nan());
        }

        // Check bands after warmup
        for i in 9..30 {
            if !middle[i].is_nan() && !upper[i].is_nan() {
                assert!(upper[i] > middle[i], "Upper should be above middle at index {}", i);
                assert!(lower[i] < middle[i], "Lower should be below middle at index {}", i);
            }
        }
    }

    #[test]
    fn test_elder_auto_envelope_default() {
        let eae = ElderAutoEnvelope::default();
        assert_eq!(eae.ema_period, 13);
        assert_eq!(eae.std_period, 13);
        assert!((eae.multiplier - 2.7).abs() < 1e-10);
    }

    #[test]
    fn test_elder_auto_envelope_from_config() {
        let config = ElderAutoEnvelopeConfig::new(20, 15, 3.0);
        let eae = ElderAutoEnvelope::from_config(config);
        assert_eq!(eae.ema_period, 20);
        assert_eq!(eae.std_period, 15);
        assert!((eae.multiplier - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_elder_auto_envelope_adapts_to_volatility() {
        let eae = ElderAutoEnvelope::new(5, 5, 2.0);

        // Low volatility data (constant price)
        let close_low_vol: Vec<f64> = (0..20).map(|_| 100.0).collect();

        // High volatility data (oscillating price)
        let close_high_vol: Vec<f64> = (0..20)
            .map(|i| if i % 2 == 0 { 110.0 } else { 90.0 })
            .collect();

        let (_, upper_low, lower_low) = eae.calculate(&close_low_vol);
        let (_, upper_high, lower_high) = eae.calculate(&close_high_vol);

        // After warmup, high volatility should have wider bands
        let idx = 15;
        let width_low = upper_low[idx] - lower_low[idx];
        let width_high = upper_high[idx] - lower_high[idx];

        // Low volatility constant data should have zero or near-zero band width
        assert!(width_low < 1e-10, "Low volatility should have minimal band width");
        // High volatility should have significant band width
        assert!(width_high > width_low, "High volatility should have wider bands");
    }

    #[test]
    fn test_elder_auto_envelope_band_width_percent() {
        let eae = ElderAutoEnvelope::new(10, 10, 2.0);
        let close = create_test_data(30);

        let bw_percent = eae.band_width_percent(&close);

        assert_eq!(bw_percent.len(), 30);

        // After warmup, bandwidth percent should be non-negative
        for i in 9..30 {
            if !bw_percent[i].is_nan() {
                assert!(bw_percent[i] >= 0.0, "Band width percent should be non-negative");
            }
        }
    }

    #[test]
    fn test_elder_auto_envelope_percent_b() {
        let eae = ElderAutoEnvelope::new(10, 10, 2.0);
        let close = create_test_data(30);

        let percent_b = eae.percent_b(&close);

        assert_eq!(percent_b.len(), 30);

        // After warmup, %B should be finite
        for i in 9..30 {
            if !percent_b[i].is_nan() {
                assert!(percent_b[i].is_finite());
            }
        }
    }

    #[test]
    fn test_elder_auto_envelope_signal_oversold() {
        let eae = ElderAutoEnvelope::new(5, 5, 2.0);

        // Create downtrending data
        let close: Vec<f64> = (0..20).map(|i| 105.0 - i as f64 * 2.0).collect();

        let data = OHLCVSeries {
            open: close.clone(),
            high: close.iter().map(|&c| c + 5.0).collect(),
            low: close.iter().map(|&c| c - 5.0).collect(),
            close,
            volume: vec![1000.0; 20],
        };

        let signals = eae.signals(&data).unwrap();
        assert_eq!(signals.len(), 20);
    }

    #[test]
    fn test_elder_auto_envelope_signal_overbought() {
        let eae = ElderAutoEnvelope::new(5, 5, 2.0);

        // Create uptrending data
        let close: Vec<f64> = (0..20).map(|i| 95.0 + i as f64 * 2.0).collect();

        let data = OHLCVSeries {
            open: close.clone(),
            high: close.iter().map(|&c| c + 5.0).collect(),
            low: close.iter().map(|&c| c - 5.0).collect(),
            close,
            volume: vec![1000.0; 20],
        };

        let signals = eae.signals(&data).unwrap();
        assert_eq!(signals.len(), 20);
    }

    #[test]
    fn test_elder_auto_envelope_compute() {
        let eae = ElderAutoEnvelope::new(10, 10, 2.0);
        let close = create_test_data(30);

        let data = OHLCVSeries {
            open: close.clone(),
            high: close.iter().map(|&c| c + 5.0).collect(),
            low: close.iter().map(|&c| c - 5.0).collect(),
            close,
            volume: vec![1000.0; 30],
        };

        let output = eae.compute(&data).unwrap();
        assert_eq!(output.primary.len(), 30);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    #[test]
    fn test_elder_auto_envelope_insufficient_data() {
        let eae = ElderAutoEnvelope::new(20, 20, 2.0);

        let data = OHLCVSeries {
            open: vec![100.0; 10],
            high: vec![101.0; 10],
            low: vec![99.0; 10],
            close: vec![100.0; 10],
            volume: vec![1000.0; 10],
        };

        let result = eae.compute(&data);
        assert!(result.is_err());

        match result {
            Err(IndicatorError::InsufficientData { required, got }) => {
                assert_eq!(required, 20);
                assert_eq!(got, 10);
            }
            _ => panic!("Expected InsufficientData error"),
        }
    }

    #[test]
    fn test_elder_auto_envelope_min_periods() {
        let eae = ElderAutoEnvelope::new(13, 15, 2.7);
        assert_eq!(eae.min_periods(), 15);

        let eae2 = ElderAutoEnvelope::new(20, 10, 2.0);
        assert_eq!(eae2.min_periods(), 20);
    }

    #[test]
    fn test_elder_auto_envelope_different_periods() {
        // Test with different EMA and StdDev periods
        let eae = ElderAutoEnvelope::new(10, 20, 2.5);
        let close = create_test_data(40);

        let (middle, upper, lower) = eae.calculate(&close);

        assert_eq!(middle.len(), 40);
        assert_eq!(upper.len(), 40);
        assert_eq!(lower.len(), 40);

        // StdDev period is 20, so first valid value at index 19
        for i in 0..19 {
            assert!(upper[i].is_nan() || lower[i].is_nan());
        }

        // After warmup, bands should be valid
        for i in 19..40 {
            if !middle[i].is_nan() && !upper[i].is_nan() {
                assert!(upper[i] >= middle[i], "Upper should be at or above middle");
                assert!(lower[i] <= middle[i], "Lower should be at or below middle");
            }
        }
    }

    #[test]
    fn test_elder_auto_envelope_output_features() {
        let eae = ElderAutoEnvelope::default();
        assert_eq!(eae.output_features(), 3);
    }

    #[test]
    fn test_elder_auto_envelope_name() {
        let eae = ElderAutoEnvelope::default();
        assert_eq!(eae.name(), "ElderAutoEnvelope");
    }
}
