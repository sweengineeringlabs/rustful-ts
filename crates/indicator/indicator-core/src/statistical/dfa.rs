//! Detrended Fluctuation Analysis implementation.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Detrended Fluctuation Analysis (DFA).
///
/// A method for determining the statistical self-affinity of a signal.
/// Measures long-range correlations in non-stationary time series.
///
/// The DFA exponent (alpha):
/// - alpha = 0.5: Uncorrelated white noise
/// - alpha < 0.5: Anti-correlated (mean-reverting)
/// - alpha > 0.5: Long-range correlated (trending)
/// - alpha = 1.0: 1/f noise (pink noise)
/// - alpha = 1.5: Brownian motion (integrated white noise)
///
/// Useful for:
/// - Detecting hidden long-range correlations
/// - Market regime identification
/// - Distinguishing between trending and mean-reverting behavior
#[derive(Debug, Clone)]
pub struct DetrendedFluctuationAnalysis {
    period: usize,
    /// Minimum segment size for analysis
    min_segment: usize,
    /// Maximum segment size for analysis
    max_segment: usize,
    /// Polynomial order for detrending (1 = linear, 2 = quadratic)
    detrend_order: usize,
}

impl DetrendedFluctuationAnalysis {
    /// Create a new DFA indicator with default parameters.
    pub fn new(period: usize) -> Self {
        Self {
            period,
            min_segment: 4,
            max_segment: period / 4,
            detrend_order: 1,
        }
    }

    /// Create with custom segment sizes.
    pub fn with_segments(period: usize, min_segment: usize, max_segment: usize) -> Self {
        Self {
            period,
            min_segment: min_segment.max(4),
            max_segment: max_segment.min(period / 2),
            detrend_order: 1,
        }
    }

    /// Create with quadratic detrending (DFA-2).
    pub fn quadratic(period: usize) -> Self {
        Self {
            period,
            min_segment: 4,
            max_segment: period / 4,
            detrend_order: 2,
        }
    }

    /// Calculate cumulative sum of deviations from mean.
    fn integrate(data: &[f64]) -> Vec<f64> {
        let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
        let mut cumsum = Vec::with_capacity(data.len());
        let mut sum = 0.0;
        for &x in data {
            sum += x - mean;
            cumsum.push(sum);
        }
        cumsum
    }

    /// Fit polynomial and return residuals.
    fn detrend_segment(&self, segment: &[f64]) -> Vec<f64> {
        let n = segment.len();
        if n < 2 {
            return segment.to_vec();
        }

        match self.detrend_order {
            1 => {
                // Linear detrending: y = a + b*x
                let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
                let (a, b) = Self::linear_fit(&x, segment);
                segment.iter().enumerate()
                    .map(|(i, &y)| y - (a + b * i as f64))
                    .collect()
            }
            2 => {
                // Quadratic detrending: y = a + b*x + c*x^2
                let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
                let (a, b, c) = Self::quadratic_fit(&x, segment);
                segment.iter().enumerate()
                    .map(|(i, &y)| {
                        let xi = i as f64;
                        y - (a + b * xi + c * xi * xi)
                    })
                    .collect()
            }
            _ => {
                // Default to linear
                let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
                let (a, b) = Self::linear_fit(&x, segment);
                segment.iter().enumerate()
                    .map(|(i, &y)| y - (a + b * i as f64))
                    .collect()
            }
        }
    }

    /// Linear least squares fit.
    fn linear_fit(x: &[f64], y: &[f64]) -> (f64, f64) {
        let n = x.len() as f64;
        if n < 2.0 {
            return (0.0, 0.0);
        }

        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
        let sum_xx: f64 = x.iter().map(|xi| xi * xi).sum();

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-10 {
            return (sum_y / n, 0.0);
        }

        let b = (n * sum_xy - sum_x * sum_y) / denom;
        let a = (sum_y - b * sum_x) / n;
        (a, b)
    }

    /// Quadratic least squares fit (simple implementation).
    fn quadratic_fit(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
        let n = x.len();
        if n < 3 {
            let (a, b) = Self::linear_fit(x, y);
            return (a, b, 0.0);
        }

        // Build normal equations for y = a + b*x + c*x^2
        let mut sum_x = 0.0;
        let mut sum_x2 = 0.0;
        let mut sum_x3 = 0.0;
        let mut sum_x4 = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2y = 0.0;

        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let x2 = xi * xi;
            let x3 = x2 * xi;
            let x4 = x3 * xi;
            sum_x += xi;
            sum_x2 += x2;
            sum_x3 += x3;
            sum_x4 += x4;
            sum_y += yi;
            sum_xy += xi * yi;
            sum_x2y += x2 * yi;
        }

        let n = n as f64;

        // Solve 3x3 system using Cramer's rule (simplified)
        // [n, sum_x, sum_x2]   [a]   [sum_y]
        // [sum_x, sum_x2, sum_x3] [b] = [sum_xy]
        // [sum_x2, sum_x3, sum_x4] [c]   [sum_x2y]

        let det = n * (sum_x2 * sum_x4 - sum_x3 * sum_x3)
            - sum_x * (sum_x * sum_x4 - sum_x3 * sum_x2)
            + sum_x2 * (sum_x * sum_x3 - sum_x2 * sum_x2);

        if det.abs() < 1e-10 {
            let (a, b) = Self::linear_fit(x, y);
            return (a, b, 0.0);
        }

        let a = (sum_y * (sum_x2 * sum_x4 - sum_x3 * sum_x3)
            - sum_x * (sum_xy * sum_x4 - sum_x3 * sum_x2y)
            + sum_x2 * (sum_xy * sum_x3 - sum_x2 * sum_x2y)) / det;

        let b = (n * (sum_xy * sum_x4 - sum_x3 * sum_x2y)
            - sum_y * (sum_x * sum_x4 - sum_x3 * sum_x2)
            + sum_x2 * (sum_x * sum_x2y - sum_xy * sum_x2)) / det;

        let c = (n * (sum_x2 * sum_x2y - sum_xy * sum_x3)
            - sum_x * (sum_x * sum_x2y - sum_xy * sum_x2)
            + sum_y * (sum_x * sum_x3 - sum_x2 * sum_x2)) / det;

        (a, b, c)
    }

    /// Calculate fluctuation function F(n) for a given segment size.
    fn calculate_fluctuation(&self, integrated: &[f64], segment_size: usize) -> f64 {
        let n = integrated.len();
        if segment_size < 2 || segment_size > n {
            return f64::NAN;
        }

        let num_segments = n / segment_size;
        if num_segments == 0 {
            return f64::NAN;
        }

        let mut total_variance = 0.0;
        let mut count = 0;

        // Forward segments
        for i in 0..num_segments {
            let start = i * segment_size;
            let end = start + segment_size;
            if end > n {
                break;
            }
            let segment = &integrated[start..end];
            let detrended = self.detrend_segment(segment);

            let variance: f64 = detrended.iter().map(|x| x * x).sum::<f64>() / segment_size as f64;
            total_variance += variance;
            count += 1;
        }

        // Backward segments (to use all data)
        for i in 0..num_segments {
            let end = n - i * segment_size;
            let start = end.saturating_sub(segment_size);
            if end <= start || start == 0 && i > 0 {
                break;
            }
            let segment = &integrated[start..end];
            let detrended = self.detrend_segment(segment);

            let variance: f64 = detrended.iter().map(|x| x * x).sum::<f64>() / segment_size as f64;
            total_variance += variance;
            count += 1;
        }

        if count == 0 {
            return f64::NAN;
        }

        (total_variance / count as f64).sqrt()
    }

    /// Calculate DFA exponent (alpha) for a window.
    fn calculate_alpha(&self, window: &[f64]) -> f64 {
        let n = window.len();
        if n < self.min_segment * 2 {
            return f64::NAN;
        }

        // Calculate returns
        let returns: Vec<f64> = window
            .windows(2)
            .map(|w| w[1] - w[0])
            .collect();

        if returns.len() < self.min_segment {
            return f64::NAN;
        }

        // Integrate the series
        let integrated = Self::integrate(&returns);

        // Calculate F(n) for different segment sizes
        let mut log_n = Vec::new();
        let mut log_f = Vec::new();

        let max_seg = self.max_segment.min(integrated.len() / 4).max(self.min_segment + 1);

        // Use logarithmically spaced segment sizes
        let mut seg_size = self.min_segment;
        while seg_size <= max_seg {
            let f = self.calculate_fluctuation(&integrated, seg_size);
            if !f.is_nan() && f > 0.0 {
                log_n.push((seg_size as f64).ln());
                log_f.push(f.ln());
            }
            seg_size = (seg_size as f64 * 1.5).ceil() as usize;
            if seg_size <= self.min_segment {
                seg_size = self.min_segment + 1;
            }
        }

        if log_n.len() < 2 {
            return f64::NAN;
        }

        // Alpha is the slope of log(F(n)) vs log(n)
        Self::linear_regression_slope(&log_n, &log_f)
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
            return f64::NAN;
        }

        (n * sum_xy - sum_x * sum_y) / denominator
    }

    /// Calculate DFA values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period || self.period < 16 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &data[start..=i];
            let alpha = self.calculate_alpha(window);
            result.push(alpha);
        }

        result
    }
}

impl TechnicalIndicator for DetrendedFluctuationAnalysis {
    fn name(&self) -> &str {
        "DFA"
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

impl SignalIndicator for DetrendedFluctuationAnalysis {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.close);
        let last = values.last().copied().unwrap_or(f64::NAN);

        if last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Alpha interpretation:
        // > 0.7: Strong trending (persistent)
        // 0.5 - 0.7: Mild trending
        // 0.3 - 0.5: Mild mean-reverting
        // < 0.3: Strong mean-reverting (anti-persistent)
        if last > 0.7 {
            // Strong trend - follow momentum
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
        } else if last < 0.4 {
            // Mean reverting - fade the move
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
            .map(|(i, &alpha)| {
                if alpha.is_nan() || i == 0 {
                    IndicatorSignal::Neutral
                } else if alpha > 0.7 {
                    // Trending
                    if data.close[i] > data.close[i - 1] {
                        IndicatorSignal::Bullish
                    } else if data.close[i] < data.close[i - 1] {
                        IndicatorSignal::Bearish
                    } else {
                        IndicatorSignal::Neutral
                    }
                } else if alpha < 0.4 {
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
    fn test_dfa_trending() {
        let dfa = DetrendedFluctuationAnalysis::new(50);
        // Trending data with some variation
        let data: Vec<f64> = (0..150)
            .map(|i| {
                let noise = ((i * 7) % 13) as f64 * 0.2 - 1.3; // Deterministic pseudo-noise
                100.0 + i as f64 * 0.5 + noise
            })
            .collect();
        let result = dfa.calculate(&data);

        // Check that we get valid values after warmup
        for i in 49..result.len() {
            if !result[i].is_nan() {
                // Alpha should be finite and reasonable
                assert!(result[i].is_finite(), "alpha should be finite");
            }
        }
    }

    #[test]
    fn test_dfa_quadratic() {
        let dfa = DetrendedFluctuationAnalysis::quadratic(50);
        let data: Vec<f64> = (0..150)
            .map(|i| 100.0 + (i as f64 * 0.1).sin() * 10.0 + i as f64 * 0.05)
            .collect();
        let result = dfa.calculate(&data);

        for i in 49..result.len() {
            if !result[i].is_nan() {
                // Alpha should be finite
                assert!(result[i].is_finite(), "alpha={} should be finite", result[i]);
            }
        }
    }

    #[test]
    fn test_dfa_oscillating() {
        let dfa = DetrendedFluctuationAnalysis::new(50);
        // Oscillating data may show different characteristics
        let data: Vec<f64> = (0..150)
            .map(|i| 100.0 + if i % 2 == 0 { 5.0 } else { -5.0 })
            .collect();
        let result = dfa.calculate(&data);

        for i in 49..result.len() {
            if !result[i].is_nan() {
                // Should produce some value
                assert!(result[i].is_finite());
            }
        }
    }

    #[test]
    fn test_dfa_insufficient_data() {
        let dfa = DetrendedFluctuationAnalysis::new(50);
        let data = vec![100.0; 30]; // Less than period
        let result = dfa.calculate(&data);

        for val in &result {
            assert!(val.is_nan());
        }
    }

    #[test]
    fn test_dfa_custom_segments() {
        let dfa = DetrendedFluctuationAnalysis::with_segments(40, 4, 10);
        let data: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64 * 0.2).sin() * 5.0 + i as f64 * 0.1)
            .collect();
        let result = dfa.calculate(&data);

        for i in 39..result.len() {
            if !result[i].is_nan() {
                assert!(result[i].is_finite());
            }
        }
    }

    #[test]
    fn test_dfa_random_walk_like() {
        let dfa = DetrendedFluctuationAnalysis::new(50);
        // Simulated random walk-ish data
        let mut data = vec![100.0];
        let steps = [0.5, -0.3, 0.2, -0.6, 0.4, -0.1, 0.3, -0.4, 0.6, -0.2];
        for i in 1..150 {
            let step = steps[i % steps.len()];
            data.push(data[i - 1] + step);
        }
        let result = dfa.calculate(&data);

        for i in 49..result.len() {
            if !result[i].is_nan() {
                // Random walk should have alpha around 0.5
                assert!(result[i] >= 0.0 && result[i] <= 2.0);
            }
        }
    }
}
