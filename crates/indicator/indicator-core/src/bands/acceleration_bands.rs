//! Acceleration Bands implementation.
//!
//! Bands that expand with volatility and contract as the market stabilizes.

use crate::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};

/// Acceleration Bands.
///
/// Developed by Price Headley, Acceleration Bands are envelope lines
/// plotted around a simple moving average. The width of the bands is
/// based on the High-Low range compared to the sum of High and Low:
/// - Middle Band: SMA of close
/// - Upper Band: SMA of (High * (1 + 4 * (High - Low) / (High + Low)))
/// - Lower Band: SMA of (Low * (1 - 4 * (High - Low) / (High + Low)))
///
/// The factor 4 in the formula is the default acceleration factor.
#[derive(Debug, Clone)]
pub struct AccelerationBands {
    /// Period for the SMA calculation.
    period: usize,
    /// Acceleration factor (typically 4.0).
    factor: f64,
}

impl AccelerationBands {
    /// Create a new Acceleration Bands indicator.
    ///
    /// # Arguments
    /// * `period` - Period for SMA calculation (typically 20)
    /// * `factor` - Acceleration factor (typically 4.0)
    pub fn new(period: usize, factor: f64) -> Self {
        Self { period, factor }
    }

    /// Create with default parameters (20-period, 4.0 factor).
    pub fn default_params() -> Self {
        Self::new(20, 4.0)
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

    /// Calculate Acceleration Bands (middle, upper, lower).
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();

        if n < self.period || self.period == 0 {
            return (
                vec![f64::NAN; n],
                vec![f64::NAN; n],
                vec![f64::NAN; n],
            );
        }

        // Calculate the accelerated high and low values
        let mut upper_raw = Vec::with_capacity(n);
        let mut lower_raw = Vec::with_capacity(n);

        for i in 0..n {
            let h = high[i];
            let l = low[i];
            let sum_hl = h + l;

            if sum_hl.abs() < 1e-10 {
                upper_raw.push(h);
                lower_raw.push(l);
            } else {
                let range_factor = self.factor * (h - l) / sum_hl;
                upper_raw.push(h * (1.0 + range_factor));
                lower_raw.push(l * (1.0 - range_factor));
            }
        }

        // Calculate SMAs
        let middle = self.calculate_sma(close);
        let upper = self.calculate_sma(&upper_raw);
        let lower = self.calculate_sma(&lower_raw);

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

impl Default for AccelerationBands {
    fn default() -> Self {
        Self::default_params()
    }
}

impl TechnicalIndicator for AccelerationBands {
    fn name(&self) -> &str {
        "AccelerationBands"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
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
    fn test_acceleration_bands() {
        let ab = AccelerationBands::new(10, 4.0);
        let n = 30;
        let high: Vec<f64> = (0..n).map(|i| 105.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0).collect();

        let (middle, upper, lower) = ab.calculate(&high, &low, &close);

        assert_eq!(middle.len(), n);
        assert_eq!(upper.len(), n);
        assert_eq!(lower.len(), n);

        // Check bands after warmup
        for i in 10..n {
            if !middle[i].is_nan() && !upper[i].is_nan() && !lower[i].is_nan() {
                assert!(upper[i] > middle[i], "Upper should be above middle at {}", i);
                assert!(lower[i] < middle[i], "Lower should be below middle at {}", i);
            }
        }
    }

    #[test]
    fn test_acceleration_bands_percent_b() {
        let ab = AccelerationBands::new(10, 4.0);
        let n = 30;
        let high: Vec<f64> = (0..n).map(|i| 105.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0).collect();

        let percent_b = ab.percent_b(&high, &low, &close);

        // %B should be between 0 and 1 when price is within bands
        for i in 10..n {
            if !percent_b[i].is_nan() {
                assert!(percent_b[i] >= -0.5 && percent_b[i] <= 1.5, "Percent B at {} out of expected range", i);
            }
        }
    }

    #[test]
    fn test_acceleration_bands_default() {
        let ab = AccelerationBands::default();
        assert_eq!(ab.period, 20);
        assert!((ab.factor - 4.0).abs() < 1e-10);
    }
}
