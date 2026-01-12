//! High-Low Bands implementation.
//!
//! Bands based on moving averages of high and low prices.

use crate::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};

/// High-Low Bands.
///
/// Bands constructed from moving averages of high and low prices:
/// - Upper Band: Moving average of high prices
/// - Middle Band: (Upper + Lower) / 2 or moving average of close
/// - Lower Band: Moving average of low prices
///
/// These bands provide a smoothed price channel that adapts to recent
/// price extremes while filtering out noise.
#[derive(Debug, Clone)]
pub struct HighLowBands {
    /// Period for the moving average calculation.
    period: usize,
    /// Use EMA instead of SMA.
    use_ema: bool,
    /// Shift amount for the bands (typically 0).
    shift: i32,
}

impl HighLowBands {
    /// Create a new High-Low Bands indicator.
    ///
    /// # Arguments
    /// * `period` - Period for the moving average (typically 20)
    /// * `use_ema` - If true, use EMA; otherwise use SMA
    /// * `shift` - Number of bars to shift (positive = forward, negative = backward)
    pub fn new(period: usize, use_ema: bool, shift: i32) -> Self {
        Self {
            period,
            use_ema,
            shift,
        }
    }

    /// Create with SMA (default).
    pub fn with_sma(period: usize) -> Self {
        Self::new(period, false, 0)
    }

    /// Create with EMA.
    pub fn with_ema(period: usize) -> Self {
        Self::new(period, true, 0)
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

    /// Apply shift to a series.
    fn apply_shift(&self, data: Vec<f64>) -> Vec<f64> {
        if self.shift == 0 {
            return data;
        }

        let n = data.len();
        let mut result = vec![f64::NAN; n];

        if self.shift > 0 {
            // Shift forward (values move to later indices)
            let shift = self.shift as usize;
            for i in shift..n {
                result[i] = data[i - shift];
            }
        } else {
            // Shift backward (values move to earlier indices)
            let shift = (-self.shift) as usize;
            for i in 0..(n.saturating_sub(shift)) {
                result[i] = data[i + shift];
            }
        }

        result
    }

    /// Calculate High-Low Bands (middle, upper, lower).
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();

        if n < self.period || self.period == 0 {
            return (
                vec![f64::NAN; n],
                vec![f64::NAN; n],
                vec![f64::NAN; n],
            );
        }

        // Calculate moving averages
        let (upper_raw, lower_raw, middle_raw) = if self.use_ema {
            (
                self.calculate_ema(high),
                self.calculate_ema(low),
                self.calculate_ema(close),
            )
        } else {
            (
                self.calculate_sma(high),
                self.calculate_sma(low),
                self.calculate_sma(close),
            )
        };

        // Apply shift
        let upper = self.apply_shift(upper_raw);
        let lower = self.apply_shift(lower_raw);
        let middle = self.apply_shift(middle_raw);

        (middle, upper, lower)
    }

    /// Calculate High-Low Bands with middle as average of upper and lower.
    pub fn calculate_avg_middle(&self, high: &[f64], low: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = high.len();

        if n < self.period || self.period == 0 {
            return (
                vec![f64::NAN; n],
                vec![f64::NAN; n],
                vec![f64::NAN; n],
            );
        }

        // Calculate moving averages
        let (upper_raw, lower_raw) = if self.use_ema {
            (
                self.calculate_ema(high),
                self.calculate_ema(low),
            )
        } else {
            (
                self.calculate_sma(high),
                self.calculate_sma(low),
            )
        };

        // Calculate middle as average of upper and lower
        let middle_raw: Vec<f64> = upper_raw
            .iter()
            .zip(lower_raw.iter())
            .map(|(&u, &l)| {
                if u.is_nan() || l.is_nan() {
                    f64::NAN
                } else {
                    (u + l) / 2.0
                }
            })
            .collect();

        // Apply shift
        let upper = self.apply_shift(upper_raw);
        let lower = self.apply_shift(lower_raw);
        let middle = self.apply_shift(middle_raw);

        (middle, upper, lower)
    }

    /// Calculate band width (upper - lower).
    pub fn band_width(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let (_, upper, lower) = self.calculate(high, low, close);
        upper
            .iter()
            .zip(lower.iter())
            .map(|(&u, &l)| {
                if u.is_nan() || l.is_nan() {
                    f64::NAN
                } else {
                    u - l
                }
            })
            .collect()
    }

    /// Calculate %B (position within bands).
    pub fn percent_b(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let (_, upper, lower) = self.calculate(high, low, close);
        close
            .iter()
            .zip(upper.iter().zip(lower.iter()))
            .map(|(&c, (&u, &l))| {
                if u.is_nan() || l.is_nan() || (u - l).abs() < 1e-10 {
                    f64::NAN
                } else {
                    (c - l) / (u - l)
                }
            })
            .collect()
    }
}

impl Default for HighLowBands {
    fn default() -> Self {
        Self::with_sma(20)
    }
}

impl TechnicalIndicator for HighLowBands {
    fn name(&self) -> &str {
        "HighLowBands"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.high.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.high.len(),
            });
        }

        let (middle, upper, lower) = self.calculate(&data.high, &data.low, &data.close);
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
    fn test_high_low_bands_sma() {
        let hlb = HighLowBands::with_sma(10);
        let n = 30;
        let high: Vec<f64> = (0..n).map(|i| 105.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0).collect();

        let (middle, upper, lower) = hlb.calculate(&high, &low, &close);

        assert_eq!(middle.len(), n);
        assert_eq!(upper.len(), n);
        assert_eq!(lower.len(), n);

        // Check bands after warmup
        for i in 10..n {
            if !middle[i].is_nan() && !upper[i].is_nan() && !lower[i].is_nan() {
                assert!(upper[i] > lower[i], "Upper should be above lower");
                // Middle should be between upper and lower
                assert!(middle[i] >= lower[i] && middle[i] <= upper[i],
                    "Middle should be between upper and lower");
            }
        }
    }

    #[test]
    fn test_high_low_bands_ema() {
        let hlb = HighLowBands::with_ema(10);
        let n = 30;
        let high: Vec<f64> = (0..n).map(|i| 105.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0).collect();

        let (middle, upper, lower) = hlb.calculate(&high, &low, &close);

        // Check bands after warmup
        for i in 10..n {
            if !upper[i].is_nan() && !lower[i].is_nan() {
                assert!(upper[i] > lower[i], "Upper should be above lower");
            }
        }
    }

    #[test]
    fn test_high_low_bands_with_shift() {
        let hlb = HighLowBands::new(10, false, 2);
        let n = 30;
        let high: Vec<f64> = (0..n).map(|i| 105.0 + i as f64).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + i as f64).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();

        let (middle, _, _) = hlb.calculate(&high, &low, &close);

        // With shift 2, values at index 11 should come from calculation at index 9
        // First two values should be NaN due to shift
        assert!(middle[0].is_nan());
        assert!(middle[1].is_nan());
    }

    #[test]
    fn test_avg_middle() {
        let hlb = HighLowBands::with_sma(5);
        let high = vec![110.0, 112.0, 111.0, 113.0, 114.0, 115.0];
        let low = vec![90.0, 92.0, 91.0, 93.0, 94.0, 95.0];

        let (middle, upper, lower) = hlb.calculate_avg_middle(&high, &low);

        // Middle should be exactly (upper + lower) / 2
        for i in 5..6 {
            if !upper[i].is_nan() && !lower[i].is_nan() {
                let expected_middle = (upper[i] + lower[i]) / 2.0;
                assert!((middle[i] - expected_middle).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_band_width() {
        let hlb = HighLowBands::with_sma(5);
        let n = 20;
        let high: Vec<f64> = (0..n).map(|_| 110.0).collect();
        let low: Vec<f64> = (0..n).map(|_| 90.0).collect();
        let close: Vec<f64> = (0..n).map(|_| 100.0).collect();

        let width = hlb.band_width(&high, &low, &close);

        // Width should be constant for constant high-low range
        for i in 5..n {
            if !width[i].is_nan() {
                assert!((width[i] - 20.0).abs() < 1e-10, "Width should be 20");
            }
        }
    }

    #[test]
    fn test_high_low_bands_default() {
        let hlb = HighLowBands::default();
        assert_eq!(hlb.period, 20);
        assert!(!hlb.use_ema);
        assert_eq!(hlb.shift, 0);
    }
}
