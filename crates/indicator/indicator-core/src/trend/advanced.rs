//! Advanced Trend Indicators
//!
//! Sophisticated trend analysis indicators for detecting trend acceleration,
//! consistency, adaptive trend lines, and multi-scale trend analysis.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Trend Acceleration - Measures trend acceleration/deceleration
///
/// Calculates the rate of change of the trend's slope, indicating
/// whether the trend is gaining or losing momentum.
#[derive(Debug, Clone)]
pub struct TrendAcceleration {
    period: usize,
    smoothing: usize,
}

impl TrendAcceleration {
    pub fn new(period: usize, smoothing: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if smoothing < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period, smoothing })
    }

    /// Calculate trend acceleration values
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.period + self.smoothing {
            return result;
        }

        // First calculate slopes (first derivative)
        let mut slopes = vec![0.0; n];
        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            // Linear regression slope
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            let mut sum_xy = 0.0;
            let mut sum_xx = 0.0;

            for (j, idx) in (start..=i).enumerate() {
                let x = j as f64;
                let y = close[idx];
                sum_x += x;
                sum_y += y;
                sum_xy += x * y;
                sum_xx += x * x;
            }

            let count = (i - start + 1) as f64;
            let denom = count * sum_xx - sum_x * sum_x;
            if denom.abs() > 1e-10 {
                slopes[i] = (count * sum_xy - sum_x * sum_y) / denom;
            }
        }

        // Smooth the slopes
        let mut smoothed_slopes = vec![0.0; n];
        for i in (self.period + self.smoothing)..n {
            let start = i.saturating_sub(self.smoothing);
            let sum: f64 = slopes[start..=i].iter().sum();
            smoothed_slopes[i] = sum / self.smoothing as f64;
        }

        // Calculate acceleration (second derivative - change in slope)
        for i in (self.period + self.smoothing + 1)..n {
            result[i] = smoothed_slopes[i] - smoothed_slopes[i - 1];
        }

        result
    }
}

impl TechnicalIndicator for TrendAcceleration {
    fn name(&self) -> &str {
        "Trend Acceleration"
    }

    fn min_periods(&self) -> usize {
        self.period + self.smoothing + 2
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Trend Consistency - Measures how consistent the trend is
///
/// Evaluates the uniformity of price movement in the trend direction,
/// returning a value from 0 (choppy) to 100 (perfectly consistent).
#[derive(Debug, Clone)]
pub struct TrendConsistency {
    period: usize,
}

impl TrendConsistency {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate trend consistency (0-100)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate directional consistency
            let mut up_count = 0;
            let mut down_count = 0;

            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    up_count += 1;
                } else if close[j] < close[j - 1] {
                    down_count += 1;
                }
            }

            let total_moves = up_count + down_count;
            if total_moves > 0 {
                // Consistency is how one-sided the moves are
                let dominant = up_count.max(down_count);
                let consistency = dominant as f64 / total_moves as f64;

                // Calculate path efficiency (direct path vs actual path)
                let direct_move = (close[i] - close[start]).abs();
                let mut actual_path = 0.0;
                for j in (start + 1)..=i {
                    actual_path += (close[j] - close[j - 1]).abs();
                }

                let efficiency = if actual_path > 1e-10 {
                    direct_move / actual_path
                } else {
                    0.0
                };

                // Combine consistency and efficiency
                result[i] = (consistency * 0.5 + efficiency * 0.5) * 100.0;
            }
        }
        result
    }
}

impl TechnicalIndicator for TrendConsistency {
    fn name(&self) -> &str {
        "Trend Consistency"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Adaptive Trend Line - Self-adjusting trend line
///
/// A trend line that adapts its sensitivity based on market volatility,
/// becoming more responsive in trending markets and more stable in ranging markets.
#[derive(Debug, Clone)]
pub struct AdaptiveTrendLine {
    period: usize,
    fast_alpha: f64,
    slow_alpha: f64,
}

impl AdaptiveTrendLine {
    pub fn new(period: usize, fast_alpha: f64, slow_alpha: f64) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if fast_alpha <= 0.0 || fast_alpha > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "fast_alpha".to_string(),
                reason: "must be between 0 and 1".to_string(),
            });
        }
        if slow_alpha <= 0.0 || slow_alpha > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "slow_alpha".to_string(),
                reason: "must be between 0 and 1".to_string(),
            });
        }
        if slow_alpha >= fast_alpha {
            return Err(IndicatorError::InvalidParameter {
                name: "slow_alpha".to_string(),
                reason: "must be less than fast_alpha".to_string(),
            });
        }
        Ok(Self { period, fast_alpha, slow_alpha })
    }

    /// Calculate efficiency ratio for adaptiveness
    fn efficiency_ratio(&self, close: &[f64], start: usize, end: usize) -> f64 {
        let change = (close[end] - close[start]).abs();
        let mut volatility = 0.0;
        for i in (start + 1)..=end {
            volatility += (close[i] - close[i - 1]).abs();
        }
        if volatility > 1e-10 {
            change / volatility
        } else {
            0.0
        }
    }

    /// Calculate adaptive trend line values
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.period {
            return result;
        }

        // Initialize with simple MA
        let initial_sum: f64 = close[..self.period].iter().sum();
        result[self.period - 1] = initial_sum / self.period as f64;

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let er = self.efficiency_ratio(close, start, i);

            // Scaled smoothing constant: more efficient = faster response
            let sc = er * (self.fast_alpha - self.slow_alpha) + self.slow_alpha;
            let alpha = sc * sc; // Square for faster adaptation

            result[i] = result[i - 1] + alpha * (close[i] - result[i - 1]);
        }

        result
    }
}

impl TechnicalIndicator for AdaptiveTrendLine {
    fn name(&self) -> &str {
        "Adaptive Trend Line"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Trend Strength Meter - Composite trend strength measurement
///
/// Combines multiple factors to provide a comprehensive trend strength reading:
/// - Price vs moving average
/// - Slope of the trend
/// - Consistency of movement
/// - Volatility-adjusted momentum
#[derive(Debug, Clone)]
pub struct TrendStrengthMeter {
    short_period: usize,
    long_period: usize,
}

impl TrendStrengthMeter {
    pub fn new(short_period: usize, long_period: usize) -> Result<Self> {
        if short_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if long_period <= short_period {
            return Err(IndicatorError::InvalidParameter {
                name: "long_period".to_string(),
                reason: "must be greater than short_period".to_string(),
            });
        }
        Ok(Self { short_period, long_period })
    }

    /// Calculate trend strength (-100 to +100)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len());
        let mut result = vec![0.0; n];

        for i in self.long_period..n {
            let mut strength = 0.0;

            // Factor 1: Price position relative to MAs (25 points)
            let short_start = i.saturating_sub(self.short_period);
            let short_ma: f64 = close[short_start..=i].iter().sum::<f64>() / self.short_period as f64;

            let long_start = i.saturating_sub(self.long_period);
            let long_ma: f64 = close[long_start..=i].iter().sum::<f64>() / self.long_period as f64;

            if close[i] > short_ma && close[i] > long_ma {
                strength += 25.0;
            } else if close[i] < short_ma && close[i] < long_ma {
                strength -= 25.0;
            }

            // Factor 2: MA alignment (25 points)
            if short_ma > long_ma {
                strength += 25.0;
            } else if short_ma < long_ma {
                strength -= 25.0;
            }

            // Factor 3: Slope strength (25 points)
            let slope = (close[i] - close[long_start]) / self.long_period as f64;
            let avg_range = (0..self.long_period)
                .map(|j| high[long_start + j] - low[long_start + j])
                .sum::<f64>() / self.long_period as f64;

            if avg_range > 1e-10 {
                let normalized_slope = slope / avg_range;
                let slope_score = (normalized_slope * 100.0).clamp(-25.0, 25.0);
                strength += slope_score;
            }

            // Factor 4: Directional consistency (25 points)
            let mut up_days = 0;
            let mut down_days = 0;
            for j in (long_start + 1)..=i {
                if close[j] > close[j - 1] {
                    up_days += 1;
                } else if close[j] < close[j - 1] {
                    down_days += 1;
                }
            }

            let total_days = up_days + down_days;
            if total_days > 0 {
                let consistency = (up_days as f64 - down_days as f64) / total_days as f64;
                strength += consistency * 25.0;
            }

            result[i] = strength;
        }
        result
    }
}

impl TechnicalIndicator for TrendStrengthMeter {
    fn name(&self) -> &str {
        "Trend Strength Meter"
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

/// Trend Change Detector - Detects trend changes early
///
/// Uses multiple detection methods to identify potential trend reversals:
/// - Price-MA crossovers
/// - Momentum divergence
/// - Volatility expansion
/// Returns a signal strength from -100 (strong bearish change) to +100 (strong bullish change)
#[derive(Debug, Clone)]
pub struct TrendChangeDetector {
    period: usize,
    sensitivity: f64,
}

impl TrendChangeDetector {
    pub fn new(period: usize, sensitivity: f64) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if sensitivity <= 0.0 || sensitivity > 3.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "sensitivity".to_string(),
                reason: "must be between 0 and 3".to_string(),
            });
        }
        Ok(Self { period, sensitivity })
    }

    /// Calculate trend change signal (-100 to +100)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.period * 2 {
            return result;
        }

        // Calculate moving averages
        let mut ma = vec![0.0; n];
        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            ma[i] = close[start..=i].iter().sum::<f64>() / self.period as f64;
        }

        for i in (self.period * 2)..n {
            let mut signal = 0.0;

            // Signal 1: MA crossover detection (33 points)
            let prev_diff = close[i - 1] - ma[i - 1];
            let curr_diff = close[i] - ma[i];

            if prev_diff <= 0.0 && curr_diff > 0.0 {
                signal += 33.0 * self.sensitivity;
            } else if prev_diff >= 0.0 && curr_diff < 0.0 {
                signal -= 33.0 * self.sensitivity;
            }

            // Signal 2: Momentum shift (33 points)
            let start = i.saturating_sub(self.period);
            let mid = (start + i) / 2;

            let recent_momentum = close[i] - close[mid];
            let older_momentum = close[mid] - close[start];

            // Detect momentum reversal
            if older_momentum < 0.0 && recent_momentum > 0.0 {
                let strength = (recent_momentum / older_momentum.abs()).min(1.0);
                signal += 33.0 * strength * self.sensitivity;
            } else if older_momentum > 0.0 && recent_momentum < 0.0 {
                let strength = (recent_momentum.abs() / older_momentum).min(1.0);
                signal -= 33.0 * strength * self.sensitivity;
            }

            // Signal 3: Volatility breakout (34 points)
            let volatility: f64 = (start + 1..=i)
                .map(|j| (close[j] - close[j - 1]).abs())
                .sum::<f64>() / self.period as f64;

            let recent_move = close[i] - close[i - 1];
            if volatility > 1e-10 {
                let z_score = recent_move / volatility;
                if z_score > 1.5 {
                    signal += (34.0 * (z_score - 1.5) / 1.5).min(34.0) * self.sensitivity;
                } else if z_score < -1.5 {
                    signal -= (34.0 * (-z_score - 1.5) / 1.5).min(34.0) * self.sensitivity;
                }
            }

            result[i] = signal.clamp(-100.0, 100.0);
        }
        result
    }
}

impl TechnicalIndicator for TrendChangeDetector {
    fn name(&self) -> &str {
        "Trend Change Detector"
    }

    fn min_periods(&self) -> usize {
        self.period * 2 + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Multi-Scale Trend - Analyzes trend across multiple timeframes
///
/// Evaluates trend strength at multiple lookback periods and combines
/// them into a unified signal. Strong readings indicate trend alignment
/// across all timeframes.
#[derive(Debug, Clone)]
pub struct MultiScaleTrend {
    scales: Vec<usize>,
    weights: Vec<f64>,
}

impl MultiScaleTrend {
    pub fn new(scales: Vec<usize>, weights: Option<Vec<f64>>) -> Result<Self> {
        if scales.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "scales".to_string(),
                reason: "must have at least one scale".to_string(),
            });
        }
        if scales.iter().any(|&s| s < 2) {
            return Err(IndicatorError::InvalidParameter {
                name: "scales".to_string(),
                reason: "all scales must be at least 2".to_string(),
            });
        }

        let weights = match weights {
            Some(w) => {
                if w.len() != scales.len() {
                    return Err(IndicatorError::InvalidParameter {
                        name: "weights".to_string(),
                        reason: "must have same length as scales".to_string(),
                    });
                }
                w
            }
            None => {
                // Default: equal weights
                vec![1.0 / scales.len() as f64; scales.len()]
            }
        };

        Ok(Self { scales, weights })
    }

    /// Calculate trend at a single scale
    fn calculate_scale_trend(&self, close: &[f64], scale: usize, idx: usize) -> f64 {
        let start = idx.saturating_sub(scale);
        if start >= idx {
            return 0.0;
        }

        // Calculate MA
        let ma: f64 = close[start..=idx].iter().sum::<f64>() / scale as f64;

        // Calculate slope
        let slope = (close[idx] - close[start]) / scale as f64;

        // Calculate volatility for normalization
        let mut volatility = 0.0;
        for j in (start + 1)..=idx {
            volatility += (close[j] - close[j - 1]).abs();
        }
        volatility /= scale as f64;

        if volatility < 1e-10 {
            return 0.0;
        }

        // Combine factors
        let position_score = if close[idx] > ma { 1.0 } else { -1.0 };
        let slope_score = (slope / volatility).clamp(-1.0, 1.0);

        // Weighted combination
        (position_score * 0.5 + slope_score * 0.5) * 100.0
    }

    /// Calculate multi-scale trend values (-100 to +100)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];
        let max_scale = *self.scales.iter().max().unwrap_or(&0);

        for i in max_scale..n {
            let mut weighted_sum = 0.0;
            let mut weight_total = 0.0;

            for (scale_idx, &scale) in self.scales.iter().enumerate() {
                let scale_trend = self.calculate_scale_trend(close, scale, i);
                weighted_sum += scale_trend * self.weights[scale_idx];
                weight_total += self.weights[scale_idx];
            }

            if weight_total > 1e-10 {
                result[i] = weighted_sum / weight_total;
            }
        }
        result
    }
}

impl TechnicalIndicator for MultiScaleTrend {
    fn name(&self) -> &str {
        "Multi-Scale Trend"
    }

    fn min_periods(&self) -> usize {
        *self.scales.iter().max().unwrap_or(&0) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        // Uptrending data
        let high = vec![
            102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
            112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0,
            122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0, 130.0, 131.0,
            132.0, 133.0, 134.0, 135.0, 136.0, 137.0, 138.0, 139.0, 140.0, 141.0,
        ];
        let low = vec![
            98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0,
            108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0,
            118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0,
            128.0, 129.0, 130.0, 131.0, 132.0, 133.0, 134.0, 135.0, 136.0, 137.0,
        ];
        let close = vec![
            100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0,
            110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0,
            120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0,
            130.0, 131.0, 132.0, 133.0, 134.0, 135.0, 136.0, 137.0, 138.0, 139.0,
        ];
        (high, low, close)
    }

    #[test]
    fn test_trend_acceleration() {
        let (_, _, close) = make_test_data();
        let ta = TrendAcceleration::new(10, 5).unwrap();
        let result = ta.calculate(&close);

        assert_eq!(result.len(), close.len());
        // In a linear uptrend, acceleration should be near zero
        // Check that values are computed for later indices
        let last_idx = close.len() - 1;
        assert!(result[last_idx].abs() < 0.5 || result[last_idx - 1].abs() < 0.5);
    }

    #[test]
    fn test_trend_acceleration_invalid_params() {
        assert!(TrendAcceleration::new(2, 5).is_err());
        assert!(TrendAcceleration::new(10, 1).is_err());
    }

    #[test]
    fn test_trend_consistency() {
        let (_, _, close) = make_test_data();
        let tc = TrendConsistency::new(10).unwrap();
        let result = tc.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Consistent uptrend should have high consistency
        assert!(result[30] > 50.0);
    }

    #[test]
    fn test_trend_consistency_invalid_params() {
        assert!(TrendConsistency::new(2).is_err());
    }

    #[test]
    fn test_adaptive_trend_line() {
        let (_, _, close) = make_test_data();
        let atl = AdaptiveTrendLine::new(10, 0.5, 0.1).unwrap();
        let result = atl.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Adaptive line should follow the trend
        assert!(result[30] > 100.0);
        assert!(result[30] < close[30]);
    }

    #[test]
    fn test_adaptive_trend_line_invalid_params() {
        assert!(AdaptiveTrendLine::new(2, 0.5, 0.1).is_err());
        assert!(AdaptiveTrendLine::new(10, 1.5, 0.1).is_err());
        assert!(AdaptiveTrendLine::new(10, 0.5, 0.6).is_err());
    }

    #[test]
    fn test_trend_strength_meter() {
        let (high, low, close) = make_test_data();
        let tsm = TrendStrengthMeter::new(5, 15).unwrap();
        let result = tsm.calculate(&high, &low, &close);

        assert_eq!(result.len(), close.len());
        // Strong uptrend should show positive strength
        assert!(result[30] > 0.0);
        assert!(result[30] >= -100.0 && result[30] <= 100.0);
    }

    #[test]
    fn test_trend_strength_meter_invalid_params() {
        assert!(TrendStrengthMeter::new(2, 15).is_err());
        assert!(TrendStrengthMeter::new(15, 10).is_err());
    }

    #[test]
    fn test_trend_change_detector() {
        let (_, _, close) = make_test_data();
        let tcd = TrendChangeDetector::new(10, 1.0).unwrap();
        let result = tcd.calculate(&close);

        assert_eq!(result.len(), close.len());
        // All values should be in range
        for val in &result {
            assert!(*val >= -100.0 && *val <= 100.0);
        }
    }

    #[test]
    fn test_trend_change_detector_invalid_params() {
        assert!(TrendChangeDetector::new(2, 1.0).is_err());
        assert!(TrendChangeDetector::new(10, 0.0).is_err());
        assert!(TrendChangeDetector::new(10, 4.0).is_err());
    }

    #[test]
    fn test_multi_scale_trend() {
        let (_, _, close) = make_test_data();
        let mst = MultiScaleTrend::new(vec![5, 10, 20], None).unwrap();
        let result = mst.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Consistent uptrend should show positive multi-scale trend
        assert!(result[35] > 0.0);
    }

    #[test]
    fn test_multi_scale_trend_with_weights() {
        let (_, _, close) = make_test_data();
        let mst = MultiScaleTrend::new(
            vec![5, 10, 20],
            Some(vec![0.2, 0.3, 0.5]),
        ).unwrap();
        let result = mst.calculate(&close);

        assert_eq!(result.len(), close.len());
        assert!(result[35] > 0.0);
    }

    #[test]
    fn test_multi_scale_trend_invalid_params() {
        assert!(MultiScaleTrend::new(vec![], None).is_err());
        assert!(MultiScaleTrend::new(vec![1], None).is_err());
        assert!(MultiScaleTrend::new(vec![5, 10], Some(vec![0.5])).is_err());
    }

    #[test]
    fn test_technical_indicator_trait() {
        let ta = TrendAcceleration::new(10, 5).unwrap();
        assert_eq!(ta.name(), "Trend Acceleration");
        assert_eq!(ta.min_periods(), 17);

        let tc = TrendConsistency::new(10).unwrap();
        assert_eq!(tc.name(), "Trend Consistency");
        assert_eq!(tc.min_periods(), 11);

        let atl = AdaptiveTrendLine::new(10, 0.5, 0.1).unwrap();
        assert_eq!(atl.name(), "Adaptive Trend Line");
        assert_eq!(atl.min_periods(), 10);

        let tsm = TrendStrengthMeter::new(5, 15).unwrap();
        assert_eq!(tsm.name(), "Trend Strength Meter");
        assert_eq!(tsm.min_periods(), 16);

        let tcd = TrendChangeDetector::new(10, 1.0).unwrap();
        assert_eq!(tcd.name(), "Trend Change Detector");
        assert_eq!(tcd.min_periods(), 21);

        let mst = MultiScaleTrend::new(vec![5, 10, 20], None).unwrap();
        assert_eq!(mst.name(), "Multi-Scale Trend");
        assert_eq!(mst.min_periods(), 21);
    }
}
