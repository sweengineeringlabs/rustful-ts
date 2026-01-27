//! Fractal Dimension implementation.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Fractal Dimension method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FractalDimensionMethod {
    /// Box-counting method (grid-based)
    BoxCounting,
    /// Variation method (using price range ratios)
    Variation,
    /// Higuchi method (length-based)
    Higuchi,
}

/// Fractal Dimension.
///
/// Measures the complexity/roughness of price movements.
///
/// - D close to 1.0: trending market (smooth, one-dimensional)
/// - D close to 1.5: random walk (Brownian motion)
/// - D close to 2.0: choppy/mean-reverting market (fills 2D space)
///
/// Values typically range from 1.0 to 2.0 for time series.
///
/// Useful for:
/// - Identifying market regimes (trending vs ranging)
/// - Adaptive strategy selection
/// - Risk assessment based on market complexity
#[derive(Debug, Clone)]
pub struct FractalDimension {
    period: usize,
    method: FractalDimensionMethod,
}

impl FractalDimension {
    /// Create a new Fractal Dimension indicator with the variation method.
    pub fn new(period: usize) -> Self {
        Self {
            period,
            method: FractalDimensionMethod::Variation,
        }
    }

    /// Create with box-counting method.
    pub fn box_counting(period: usize) -> Self {
        Self {
            period,
            method: FractalDimensionMethod::BoxCounting,
        }
    }

    /// Create with variation method.
    pub fn variation(period: usize) -> Self {
        Self {
            period,
            method: FractalDimensionMethod::Variation,
        }
    }

    /// Create with Higuchi method.
    pub fn higuchi(period: usize) -> Self {
        Self {
            period,
            method: FractalDimensionMethod::Higuchi,
        }
    }

    /// Calculate fractal dimension using the variation method.
    ///
    /// Uses the path length vs straight-line distance approach.
    /// D = log(L/d) / log(N) where L is path length, d is straight distance, N is number of points.
    fn calculate_variation(&self, window: &[f64]) -> f64 {
        let n = window.len();
        if n < 4 {
            return f64::NAN;
        }

        // Calculate path length (sum of absolute differences)
        let path_length: f64 = window
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .sum();

        // Straight line distance from first to last
        let straight_distance = (window[n - 1] - window[0]).abs();

        // If path equals straight distance, it's a straight line (D = 1)
        // If path is much longer, it's more complex (D closer to 2)
        if path_length < 1e-10 {
            return 1.0; // No movement
        }

        // Normalized complexity using log ratio
        // For a straight line: path_length ≈ straight_distance, ratio ≈ 1
        // For choppy data: path_length >> straight_distance, ratio >> 1
        let complexity_ratio = if straight_distance > 1e-10 {
            path_length / straight_distance
        } else {
            // If no net movement but there is path, it's very complex
            path_length * (n as f64) / window.iter().map(|&x| x.abs()).sum::<f64>().max(1.0) * 10.0
        };

        // Map complexity ratio to fractal dimension [1, 2]
        // ratio = 1 -> D = 1 (straight line)
        // ratio = n -> D = 2 (fills space)
        let max_ratio = (n - 1) as f64; // Maximum theoretical ratio

        if complexity_ratio <= 1.0 {
            return 1.0;
        }

        // D = 1 + log(ratio) / log(max_ratio)
        let d = 1.0 + (complexity_ratio.ln() / max_ratio.ln());
        d.clamp(1.0, 2.0)
    }

    /// Calculate fractal dimension using box-counting method.
    fn calculate_box_counting(&self, window: &[f64]) -> f64 {
        let n = window.len();
        if n < 4 {
            return f64::NAN;
        }

        // Normalize prices to [0, 1] range
        let (min_price, max_price) = window.iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &x| {
                (min.min(x), max.max(x))
            });

        let price_range = max_price - min_price;
        if price_range < 1e-10 {
            return 1.0; // Flat line
        }

        // Count boxes at different scales
        let scales = [2, 4, 8, 16].iter()
            .filter(|&&s| s < n)
            .cloned()
            .collect::<Vec<_>>();

        if scales.len() < 2 {
            return f64::NAN;
        }

        let mut log_n = Vec::new();
        let mut log_1_r = Vec::new();

        for scale in scales {
            let box_size = (n as f64) / (scale as f64);
            let value_box_size = price_range / (scale as f64);

            let mut boxes_covered = 0usize;

            // Count boxes that contain the curve
            for i in 0..scale {
                let start_idx = ((i as f64) * box_size) as usize;
                let end_idx = ((((i + 1) as f64) * box_size) as usize).min(n);

                if start_idx >= end_idx {
                    continue;
                }

                let segment = &window[start_idx..end_idx];
                let (seg_min, seg_max) = segment.iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &x| {
                        (min.min(x), max.max(x))
                    });

                // Count vertical boxes this segment spans
                if value_box_size > 1e-10 {
                    let boxes_in_column = ((seg_max - seg_min) / value_box_size).ceil() as usize;
                    boxes_covered += boxes_in_column.max(1);
                } else {
                    boxes_covered += 1;
                }
            }

            if boxes_covered > 0 {
                log_n.push((boxes_covered as f64).ln());
                log_1_r.push((scale as f64).ln());
            }
        }

        if log_n.len() < 2 {
            return f64::NAN;
        }

        // Linear regression to find slope
        let d = Self::linear_regression_slope(&log_1_r, &log_n);
        d.clamp(1.0, 2.0)
    }

    /// Calculate fractal dimension using Higuchi method.
    fn calculate_higuchi(&self, window: &[f64]) -> f64 {
        let n = window.len();
        if n < 4 {
            return f64::NAN;
        }

        let k_max = (n / 4).max(2);
        let mut log_l = Vec::new();
        let mut log_k = Vec::new();

        for k in 1..=k_max {
            let mut l_k = 0.0;
            let mut count = 0;

            for m in 0..k {
                let mut length = 0.0;
                let mut points = 0;

                let mut idx = m;
                while idx + k < n {
                    length += (window[idx + k] - window[idx]).abs();
                    points += 1;
                    idx += k;
                }

                if points > 0 {
                    // Normalize by interval and scale
                    let n_m = ((n - 1 - m) / k) as f64;
                    if n_m > 0.0 {
                        l_k += (length * (n - 1) as f64) / (k as f64 * n_m * k as f64);
                        count += 1;
                    }
                }
            }

            if count > 0 {
                l_k /= count as f64;
                if l_k > 0.0 {
                    log_l.push(l_k.ln());
                    log_k.push((k as f64).ln());
                }
            }
        }

        if log_l.len() < 2 {
            return f64::NAN;
        }

        // Slope of log(L(k)) vs log(1/k) gives D
        // Which is -slope of log(L(k)) vs log(k)
        let slope = Self::linear_regression_slope(&log_k, &log_l);
        let d = -slope;
        d.clamp(1.0, 2.0)
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

    /// Calculate fractal dimension for a window.
    fn calculate_fd(&self, window: &[f64]) -> f64 {
        match self.method {
            FractalDimensionMethod::Variation => self.calculate_variation(window),
            FractalDimensionMethod::BoxCounting => self.calculate_box_counting(window),
            FractalDimensionMethod::Higuchi => self.calculate_higuchi(window),
        }
    }

    /// Calculate fractal dimension values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period || self.period < 4 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &data[start..=i];
            let fd = self.calculate_fd(window);
            result.push(fd);
        }

        result
    }
}

impl TechnicalIndicator for FractalDimension {
    fn name(&self) -> &str {
        "FractalDimension"
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

impl SignalIndicator for FractalDimension {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.close);
        let last = values.last().copied().unwrap_or(f64::NAN);

        if last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // D < 1.4: Strong trend, follow momentum
        // D > 1.6: Choppy/ranging, mean reversion
        // 1.4-1.6: Random walk, neutral
        if last < 1.4 {
            // Trending - determine direction
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
        } else if last > 1.6 {
            // Mean reverting - opposite of current move
            let n = data.close.len();
            if n >= 2 {
                if data.close[n - 1] > data.close[n - 2] {
                    Ok(IndicatorSignal::Bearish) // Expect reversal
                } else if data.close[n - 1] < data.close[n - 2] {
                    Ok(IndicatorSignal::Bullish) // Expect reversal
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
            .map(|(i, &fd)| {
                if fd.is_nan() || i == 0 {
                    IndicatorSignal::Neutral
                } else if fd < 1.4 {
                    // Trending
                    if data.close[i] > data.close[i - 1] {
                        IndicatorSignal::Bullish
                    } else if data.close[i] < data.close[i - 1] {
                        IndicatorSignal::Bearish
                    } else {
                        IndicatorSignal::Neutral
                    }
                } else if fd > 1.6 {
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
    fn test_fractal_dimension_trending() {
        let fd = FractalDimension::new(20);
        // Strong linear trend should have low fractal dimension (close to 1)
        let data: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 2.0).collect();
        let result = fd.calculate(&data);

        // After warmup, values should be valid
        for i in 19..result.len() {
            assert!(!result[i].is_nan());
            assert!(result[i] >= 1.0 && result[i] <= 2.0);
            // Trending data should have lower FD (closer to 1)
            // A perfect line has ratio = 1, giving FD = 1
            assert!(result[i] < 1.5, "FD={} expected < 1.5 for trending", result[i]);
        }
    }

    #[test]
    fn test_fractal_dimension_choppy() {
        let fd = FractalDimension::new(20);
        // Oscillating data should have higher fractal dimension
        let data: Vec<f64> = (0..50)
            .map(|i| 100.0 + if i % 2 == 0 { 5.0 } else { -5.0 })
            .collect();
        let result = fd.calculate(&data);

        for i in 19..result.len() {
            assert!(!result[i].is_nan());
            assert!(result[i] >= 1.0 && result[i] <= 2.0);
        }
    }

    #[test]
    fn test_fractal_dimension_box_counting() {
        let fd = FractalDimension::box_counting(20);
        let data: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64).sin() * 10.0).collect();
        let result = fd.calculate(&data);

        for i in 19..result.len() {
            assert!(!result[i].is_nan());
            assert!(result[i] >= 1.0 && result[i] <= 2.0);
        }
    }

    #[test]
    fn test_fractal_dimension_higuchi() {
        let fd = FractalDimension::higuchi(20);
        let data: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.5).sin() * 10.0).collect();
        let result = fd.calculate(&data);

        for i in 19..result.len() {
            assert!(!result[i].is_nan());
            assert!(result[i] >= 1.0 && result[i] <= 2.0);
        }
    }

    #[test]
    fn test_fractal_dimension_constant() {
        let fd = FractalDimension::new(10);
        // Constant data should have FD close to 1
        let data = vec![100.0; 30];
        let result = fd.calculate(&data);

        for i in 9..result.len() {
            assert!(!result[i].is_nan());
            assert!((result[i] - 1.0).abs() < 0.1, "FD={} expected close to 1.0", result[i]);
        }
    }

    #[test]
    fn test_fractal_dimension_insufficient_data() {
        let fd = FractalDimension::new(20);
        let data = vec![100.0; 10]; // Less than period
        let result = fd.calculate(&data);

        // All should be NaN
        for val in &result {
            assert!(val.is_nan());
        }
    }
}
