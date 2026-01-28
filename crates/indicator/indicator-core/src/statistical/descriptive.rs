//! Descriptive Statistical Measures
//!
//! Basic descriptive statistics for rolling window analysis.

use indicator_spi::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};
use std::collections::HashMap;

/// Rolling Mode - Most frequent value over a period.
///
/// For continuous data, values are discretized to detect the mode.
/// Returns NaN if no clear mode exists.
#[derive(Debug, Clone)]
pub struct Mode {
    period: usize,
    precision: i32, // Decimal places for discretization
}

impl Mode {
    pub fn new(period: usize) -> Self {
        Self { period, precision: 2 }
    }

    pub fn with_precision(period: usize, precision: i32) -> Self {
        Self { period, precision }
    }

    /// Calculate rolling mode.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period || self.period == 0 {
            return vec![f64::NAN; n];
        }

        let scale = 10f64.powi(self.precision);
        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &data[start..=i];

            // Discretize and count frequencies
            let mut counts: HashMap<i64, usize> = HashMap::new();
            for &val in window {
                let key = (val * scale).round() as i64;
                *counts.entry(key).or_insert(0) += 1;
            }

            // Find mode
            let max_count = *counts.values().max().unwrap_or(&0);
            if max_count > 1 {
                let modes: Vec<i64> = counts.iter()
                    .filter(|(_, &c)| c == max_count)
                    .map(|(&k, _)| k)
                    .collect();

                // If there's a unique mode, return it
                if modes.len() == 1 {
                    result.push(modes[0] as f64 / scale);
                } else {
                    // Multiple modes - return their average
                    let sum: i64 = modes.iter().sum();
                    result.push((sum as f64 / modes.len() as f64) / scale);
                }
            } else {
                // No repeated values
                result.push(f64::NAN);
            }
        }

        result
    }
}

impl TechnicalIndicator for Mode {
    fn name(&self) -> &str {
        "Mode"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.close);
        Ok(IndicatorOutput::single(result))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Rolling Range - Max minus Min over a period.
///
/// Range = Highest High - Lowest Low
#[derive(Debug, Clone)]
pub struct Range {
    period: usize,
}

impl Range {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Calculate rolling range.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period || self.period == 0 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &data[start..=i];

            let max = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min = window.iter().cloned().fold(f64::INFINITY, f64::min);

            result.push(max - min);
        }

        result
    }

    /// Calculate using OHLC data (true range).
    pub fn calculate_ohlc(&self, high: &[f64], low: &[f64]) -> Vec<f64> {
        let n = high.len();
        if n < self.period || self.period == 0 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window_high = &high[start..=i];
            let window_low = &low[start..=i];

            let max = window_high.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min = window_low.iter().cloned().fold(f64::INFINITY, f64::min);

            result.push(max - min);
        }

        result
    }
}

impl TechnicalIndicator for Range {
    fn name(&self) -> &str {
        "Range"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let result = self.calculate_ohlc(&data.high, &data.low);
        Ok(IndicatorOutput::single(result))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Coefficient of Variation - Standard deviation divided by mean.
///
/// CV = StdDev / Mean * 100
///
/// Measures relative variability, useful for comparing volatility
/// across different price levels.
#[derive(Debug, Clone)]
pub struct CoefficientOfVariation {
    period: usize,
}

impl CoefficientOfVariation {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Calculate rolling coefficient of variation.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period || self.period == 0 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &data[start..=i];

            let mean: f64 = window.iter().sum::<f64>() / self.period as f64;

            if mean.abs() < 1e-10 {
                result.push(f64::NAN);
                continue;
            }

            let variance: f64 = window.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / self.period as f64;

            let std_dev = variance.sqrt();
            result.push((std_dev / mean.abs()) * 100.0);
        }

        result
    }
}

impl TechnicalIndicator for CoefficientOfVariation {
    fn name(&self) -> &str {
        "Coefficient of Variation"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.close);
        Ok(IndicatorOutput::single(result))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Rolling Percentile - Nth percentile over a period.
///
/// Returns the value below which N% of the data falls.
#[derive(Debug, Clone)]
pub struct Percentile {
    period: usize,
    percentile: f64, // 0.0 to 100.0
}

impl Percentile {
    pub fn new(period: usize, percentile: f64) -> Self {
        Self {
            period,
            percentile: percentile.clamp(0.0, 100.0),
        }
    }

    /// Calculate rolling percentile.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period || self.period == 0 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let mut window: Vec<f64> = data[start..=i].to_vec();
            window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Calculate percentile position
            let rank = self.percentile / 100.0 * (self.period - 1) as f64;
            let lower_idx = rank.floor() as usize;
            let upper_idx = (lower_idx + 1).min(self.period - 1);
            let fraction = rank - lower_idx as f64;

            // Linear interpolation
            let value = window[lower_idx] * (1.0 - fraction) + window[upper_idx] * fraction;
            result.push(value);
        }

        result
    }
}

impl TechnicalIndicator for Percentile {
    fn name(&self) -> &str {
        "Percentile"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.close);
        Ok(IndicatorOutput::single(result))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Rolling Quartiles - Q1, Q2 (Median), Q3 over a period.
#[derive(Debug, Clone)]
pub struct Quartiles {
    period: usize,
}

/// Output for Quartiles calculation.
#[derive(Debug, Clone)]
pub struct QuartilesOutput {
    /// First quartile (25th percentile)
    pub q1: Vec<f64>,
    /// Second quartile (50th percentile / median)
    pub q2: Vec<f64>,
    /// Third quartile (75th percentile)
    pub q3: Vec<f64>,
}

impl Quartiles {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Calculate rolling quartiles.
    pub fn calculate(&self, data: &[f64]) -> QuartilesOutput {
        let p25 = Percentile::new(self.period, 25.0);
        let p50 = Percentile::new(self.period, 50.0);
        let p75 = Percentile::new(self.period, 75.0);

        QuartilesOutput {
            q1: p25.calculate(data),
            q2: p50.calculate(data),
            q3: p75.calculate(data),
        }
    }
}

impl TechnicalIndicator for Quartiles {
    fn name(&self) -> &str {
        "Quartiles"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let output = self.calculate(&data.close);
        Ok(IndicatorOutput::triple(output.q1, output.q2, output.q3))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        3
    }
}

/// Interquartile Range (IQR) - Q3 minus Q1.
///
/// IQR = Q3 - Q1
///
/// Robust measure of spread, less sensitive to outliers than range.
#[derive(Debug, Clone)]
pub struct IQR {
    period: usize,
}

impl IQR {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Calculate rolling IQR.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let quartiles = Quartiles::new(self.period);
        let output = quartiles.calculate(data);

        output.q3.iter()
            .zip(output.q1.iter())
            .map(|(&q3, &q1)| {
                if q3.is_nan() || q1.is_nan() {
                    f64::NAN
                } else {
                    q3 - q1
                }
            })
            .collect()
    }
}

impl TechnicalIndicator for IQR {
    fn name(&self) -> &str {
        "IQR"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.close);
        Ok(IndicatorOutput::single(result))
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
    fn test_mode_repeated_values() {
        let data = vec![1.0, 2.0, 2.0, 3.0, 2.0, 4.0, 5.0, 2.0, 6.0, 2.0];
        let mode = Mode::new(5);
        let result = mode.calculate(&data);

        assert_eq!(result.len(), 10);
        // Window [1,2,2,3,2] has mode 2
        assert!((result[4] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_range() {
        let data = vec![10.0, 15.0, 8.0, 12.0, 20.0];
        let range = Range::new(3);
        let result = range.calculate(&data);

        assert_eq!(result.len(), 5);
        assert!(result[1].is_nan());
        assert!((result[2] - 7.0).abs() < 0.001); // 15 - 8 = 7
        assert!((result[3] - 7.0).abs() < 0.001); // 15 - 8 = 7
        assert!((result[4] - 12.0).abs() < 0.001); // 20 - 8 = 12
    }

    #[test]
    fn test_coefficient_of_variation() {
        let data = vec![100.0, 102.0, 98.0, 101.0, 99.0];
        let cov = CoefficientOfVariation::new(5);
        let result = cov.calculate(&data);

        assert_eq!(result.len(), 5);
        // CV should be relatively small for this stable data
        if !result[4].is_nan() {
            assert!(result[4] > 0.0 && result[4] < 10.0);
        }
    }

    #[test]
    fn test_percentile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        // 50th percentile should be median
        let p50 = Percentile::new(5, 50.0);
        let result = p50.calculate(&data);

        assert!((result[4] - 3.0).abs() < 0.001); // median of [1,2,3,4,5] = 3
    }

    #[test]
    fn test_quartiles() {
        let data: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let quartiles = Quartiles::new(20);
        let output = quartiles.calculate(&data);

        assert_eq!(output.q1.len(), 20);
        assert_eq!(output.q2.len(), 20);
        assert_eq!(output.q3.len(), 20);

        // For data 1-20, Q1 should be around 5.75, Q2 around 10.5, Q3 around 15.25
        let q1 = output.q1[19];
        let q2 = output.q2[19];
        let q3 = output.q3[19];

        assert!(!q1.is_nan());
        assert!(!q2.is_nan());
        assert!(!q3.is_nan());
        assert!(q1 < q2 && q2 < q3);
    }

    #[test]
    fn test_iqr() {
        let data: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let iqr = IQR::new(20);
        let result = iqr.calculate(&data);

        assert_eq!(result.len(), 20);

        // IQR = Q3 - Q1, should be approximately 9.5 for uniform 1-20
        let iqr_val = result[19];
        assert!(!iqr_val.is_nan());
        assert!(iqr_val > 8.0 && iqr_val < 11.0);
    }
}
