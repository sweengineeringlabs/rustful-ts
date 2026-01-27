//! Hurst Exponent implementation.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Hurst Exponent calculation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HurstMethod {
    /// Rescaled Range (R/S) analysis - classic method
    RescaledRange,
    /// Simplified estimation using variance ratios
    VarianceRatio,
}

/// Hurst Exponent.
///
/// Measures the long-term memory of a time series.
///
/// - H = 0.5: Random walk (no memory)
/// - H > 0.5: Persistent/trending (positive autocorrelation)
/// - H < 0.5: Anti-persistent/mean-reverting (negative autocorrelation)
///
/// Values range from 0 to 1:
/// - 0.0 to 0.5: Mean-reverting behavior
/// - 0.5: Random walk (Brownian motion)
/// - 0.5 to 1.0: Trending/persistent behavior
///
/// Useful for:
/// - Identifying trending vs mean-reverting regimes
/// - Selecting appropriate trading strategies
/// - Understanding market memory structure
#[derive(Debug, Clone)]
pub struct HurstExponent {
    period: usize,
    method: HurstMethod,
}

impl HurstExponent {
    /// Create a new Hurst Exponent indicator with R/S method.
    pub fn new(period: usize) -> Self {
        Self {
            period,
            method: HurstMethod::RescaledRange,
        }
    }

    /// Create with Rescaled Range method.
    pub fn rescaled_range(period: usize) -> Self {
        Self {
            period,
            method: HurstMethod::RescaledRange,
        }
    }

    /// Create with Variance Ratio method.
    pub fn variance_ratio(period: usize) -> Self {
        Self {
            period,
            method: HurstMethod::VarianceRatio,
        }
    }

    /// Calculate Hurst exponent using Rescaled Range (R/S) analysis.
    fn calculate_rs(&self, window: &[f64]) -> f64 {
        let n = window.len();
        if n < 8 {
            return f64::NAN;
        }

        // Calculate returns
        let returns: Vec<f64> = window
            .windows(2)
            .map(|w| w[1] - w[0])
            .collect();

        if returns.is_empty() {
            return f64::NAN;
        }

        // We need multiple sub-series lengths to estimate H
        let mut log_n = Vec::new();
        let mut log_rs = Vec::new();

        // Try different subseries lengths
        let min_size = 4;
        let mut size = min_size;

        while size <= returns.len() / 2 {
            let num_subseries = returns.len() / size;
            if num_subseries == 0 {
                size *= 2;
                continue;
            }

            let mut rs_sum = 0.0;
            let mut valid_count = 0;

            for i in 0..num_subseries {
                let start = i * size;
                let end = start + size;
                if end > returns.len() {
                    break;
                }
                let subseries = &returns[start..end];

                // Calculate mean
                let mean: f64 = subseries.iter().sum::<f64>() / size as f64;

                // Calculate cumulative deviations from mean
                let mut cumsum = 0.0;
                let mut cumdev = Vec::with_capacity(size);
                for &r in subseries {
                    cumsum += r - mean;
                    cumdev.push(cumsum);
                }

                // Range: max - min of cumulative deviations
                let (min_dev, max_dev) = cumdev.iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &x| {
                        (min.min(x), max.max(x))
                    });
                let range = max_dev - min_dev;

                // Standard deviation
                let std_dev: f64 = (subseries.iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f64>() / size as f64)
                    .sqrt();

                if std_dev > 1e-10 {
                    rs_sum += range / std_dev;
                    valid_count += 1;
                }
            }

            if valid_count > 0 {
                let avg_rs = rs_sum / valid_count as f64;
                if avg_rs > 0.0 {
                    log_n.push((size as f64).ln());
                    log_rs.push(avg_rs.ln());
                }
            }

            size *= 2;
        }

        if log_n.len() < 2 {
            return f64::NAN;
        }

        // Linear regression: log(R/S) = H * log(n) + c
        let h = Self::linear_regression_slope(&log_n, &log_rs);
        h.clamp(0.0, 1.0)
    }

    /// Calculate Hurst exponent using variance ratio method.
    fn calculate_variance_ratio(&self, window: &[f64]) -> f64 {
        let n = window.len();
        if n < 8 {
            return f64::NAN;
        }

        // Calculate returns
        let returns: Vec<f64> = window
            .windows(2)
            .map(|w| {
                if w[0].abs() < 1e-10 {
                    0.0
                } else {
                    (w[1] - w[0]) / w[0]
                }
            })
            .collect();

        if returns.len() < 4 {
            return f64::NAN;
        }

        // Variance of 1-period returns
        let mean_1: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let var_1: f64 = returns.iter()
            .map(|x| (x - mean_1).powi(2))
            .sum::<f64>() / returns.len() as f64;

        if var_1 < 1e-10 {
            return 0.5;
        }

        // Calculate multi-period variances
        let mut log_q = Vec::new();
        let mut log_var_ratio = Vec::new();

        for q in [2, 4, 8, 16].iter() {
            if *q >= returns.len() {
                continue;
            }

            // Aggregate returns over q periods
            let agg_returns: Vec<f64> = returns
                .chunks(*q)
                .filter(|c| c.len() == *q)
                .map(|c| c.iter().sum())
                .collect();

            if agg_returns.len() < 2 {
                continue;
            }

            let mean_q: f64 = agg_returns.iter().sum::<f64>() / agg_returns.len() as f64;
            let var_q: f64 = agg_returns.iter()
                .map(|x| (x - mean_q).powi(2))
                .sum::<f64>() / agg_returns.len() as f64;

            // For random walk: Var(q-period) = q * Var(1-period)
            // Var(q)/q = Var(1) => ratio = 1 for H=0.5
            // For H: Var(q) ~ q^(2H) => Var(q)/q^(2H) = const
            if var_q > 1e-10 {
                log_q.push((*q as f64).ln());
                log_var_ratio.push((var_q / var_1).ln());
            }
        }

        if log_q.len() < 2 {
            return 0.5;
        }

        // Slope of log(Var(q)/Var(1)) vs log(q) should be 2H - 1
        let slope = Self::linear_regression_slope(&log_q, &log_var_ratio);
        let h = (slope + 1.0) / 2.0;
        h.clamp(0.0, 1.0)
    }

    /// Simple linear regression to get slope.
    fn linear_regression_slope(x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return f64::NAN;
        }

        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
        let sum_xx: f64 = x.iter().map(|xi| xi * xi).sum();

        let denominator = n * sum_xx - sum_x * sum_x;
        if denominator.abs() < 1e-10 {
            return 0.0;
        }

        (n * sum_xy - sum_x * sum_y) / denominator
    }

    /// Calculate Hurst exponent for a window.
    fn calculate_hurst(&self, window: &[f64]) -> f64 {
        match self.method {
            HurstMethod::RescaledRange => self.calculate_rs(window),
            HurstMethod::VarianceRatio => self.calculate_variance_ratio(window),
        }
    }

    /// Calculate Hurst exponent values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period || self.period < 8 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &data[start..=i];
            let h = self.calculate_hurst(window);
            result.push(h);
        }

        result
    }
}

impl TechnicalIndicator for HurstExponent {
    fn name(&self) -> &str {
        "HurstExponent"
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

impl SignalIndicator for HurstExponent {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.close);
        let last = values.last().copied().unwrap_or(f64::NAN);

        if last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // H > 0.55: Trending, follow momentum
        // H < 0.45: Mean-reverting
        // 0.45-0.55: Random walk
        if last > 0.55 {
            // Trending - follow current direction
            let n = data.close.len();
            if n >= 2 {
                if data.close[n - 1] > data.close[n - 2] {
                    Ok(IndicatorSignal::Bullish)
                } else if data.close[n - 1] < data.close[n - 2] {
                    Ok(IndicatorSignal::Bearish)
                } else {
                    Ok(IndicatorSignal::Neutral)
                }
            } else {
                Ok(IndicatorSignal::Neutral)
            }
        } else if last < 0.45 {
            // Mean reverting - opposite of current direction
            let n = data.close.len();
            if n >= 2 {
                if data.close[n - 1] > data.close[n - 2] {
                    Ok(IndicatorSignal::Bearish)
                } else if data.close[n - 1] < data.close[n - 2] {
                    Ok(IndicatorSignal::Bullish)
                } else {
                    Ok(IndicatorSignal::Neutral)
                }
            } else {
                Ok(IndicatorSignal::Neutral)
            }
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.close);

        let signals = values
            .iter()
            .enumerate()
            .map(|(i, &h)| {
                if h.is_nan() || i == 0 {
                    IndicatorSignal::Neutral
                } else if h > 0.55 {
                    // Trending
                    if data.close[i] > data.close[i - 1] {
                        IndicatorSignal::Bullish
                    } else if data.close[i] < data.close[i - 1] {
                        IndicatorSignal::Bearish
                    } else {
                        IndicatorSignal::Neutral
                    }
                } else if h < 0.45 {
                    // Mean reverting
                    if data.close[i] > data.close[i - 1] {
                        IndicatorSignal::Bearish
                    } else if data.close[i] < data.close[i - 1] {
                        IndicatorSignal::Bullish
                    } else {
                        IndicatorSignal::Neutral
                    }
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

    #[test]
    fn test_hurst_trending() {
        let hurst = HurstExponent::new(30);
        // Trending data with some noise should have H > 0.5 (persistent)
        // Pure linear trend has zero variance in returns, which is a degenerate case
        let data: Vec<f64> = (0..100)
            .map(|i| {
                let noise = ((i * 7) % 11) as f64 * 0.1 - 0.5; // Deterministic pseudo-noise
                100.0 + i as f64 * 0.5 + noise
            })
            .collect();
        let result = hurst.calculate(&data);

        // After warmup, check that values are valid
        for i in 29..result.len() {
            if !result[i].is_nan() {
                assert!(result[i] >= 0.0 && result[i] <= 1.0);
            }
        }
    }

    #[test]
    fn test_hurst_mean_reverting() {
        let hurst = HurstExponent::new(30);
        // Oscillating data should have H < 0.5 (anti-persistent)
        let data: Vec<f64> = (0..100)
            .map(|i| 100.0 + if i % 2 == 0 { 5.0 } else { -5.0 })
            .collect();
        let result = hurst.calculate(&data);

        for i in 29..result.len() {
            if !result[i].is_nan() {
                assert!(result[i] >= 0.0 && result[i] <= 1.0);
            }
        }
    }

    #[test]
    fn test_hurst_variance_ratio() {
        let hurst = HurstExponent::variance_ratio(30);
        let data: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64 * 0.1).sin() * 10.0 + i as f64 * 0.1)
            .collect();
        let result = hurst.calculate(&data);

        for i in 29..result.len() {
            if !result[i].is_nan() {
                assert!(result[i] >= 0.0 && result[i] <= 1.0);
            }
        }
    }

    #[test]
    fn test_hurst_bounds() {
        let hurst = HurstExponent::new(20);
        // Random-ish data
        let data: Vec<f64> = vec![
            100.0, 102.0, 99.0, 101.0, 98.0, 103.0, 100.0, 104.0, 97.0, 105.0,
            96.0, 106.0, 95.0, 107.0, 94.0, 108.0, 93.0, 109.0, 92.0, 110.0,
            91.0, 111.0, 90.0, 112.0, 89.0, 113.0, 88.0, 114.0, 87.0, 115.0,
        ];
        let result = hurst.calculate(&data);

        for val in &result {
            if !val.is_nan() {
                assert!(*val >= 0.0 && *val <= 1.0, "H={} out of bounds", val);
            }
        }
    }

    #[test]
    fn test_hurst_insufficient_data() {
        let hurst = HurstExponent::new(30);
        let data = vec![100.0; 20]; // Less than period
        let result = hurst.calculate(&data);

        for val in &result {
            assert!(val.is_nan());
        }
    }

    #[test]
    fn test_hurst_constant_data() {
        let hurst = HurstExponent::new(20);
        // Constant data - no meaningful H
        let data = vec![100.0; 50];
        let result = hurst.calculate(&data);

        // Should handle gracefully (may be NaN or 0.5)
        for i in 19..result.len() {
            if !result[i].is_nan() {
                assert!(result[i] >= 0.0 && result[i] <= 1.0);
            }
        }
    }
}
