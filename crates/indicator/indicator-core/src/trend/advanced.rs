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

/// Adaptive Trend Follower - Uses multiple timeframes to adaptively follow trends
///
/// Combines short, medium, and long-term trend signals with adaptive weighting
/// based on trend consistency at each timeframe. Returns values from -100 to +100.
#[derive(Debug, Clone)]
pub struct AdaptiveTrendFollower {
    short_period: usize,
    medium_period: usize,
    long_period: usize,
}

impl AdaptiveTrendFollower {
    pub fn new(short_period: usize, medium_period: usize, long_period: usize) -> Result<Self> {
        if short_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if medium_period <= short_period {
            return Err(IndicatorError::InvalidParameter {
                name: "medium_period".to_string(),
                reason: "must be greater than short_period".to_string(),
            });
        }
        if long_period <= medium_period {
            return Err(IndicatorError::InvalidParameter {
                name: "long_period".to_string(),
                reason: "must be greater than medium_period".to_string(),
            });
        }
        Ok(Self { short_period, medium_period, long_period })
    }

    /// Calculate trend signal for a single timeframe
    fn timeframe_signal(&self, close: &[f64], period: usize, idx: usize) -> (f64, f64) {
        let start = idx.saturating_sub(period);
        if start >= idx || idx - start < period {
            return (0.0, 0.0);
        }

        // Calculate MA
        let ma: f64 = close[start..=idx].iter().sum::<f64>() / period as f64;

        // Calculate slope
        let slope = (close[idx] - close[start]) / period as f64;

        // Calculate volatility for normalization
        let mut volatility = 0.0;
        for j in (start + 1)..=idx {
            volatility += (close[j] - close[j - 1]).abs();
        }
        volatility /= period as f64;

        // Calculate consistency (how many bars move in trend direction)
        let trend_direction = if slope > 0.0 { 1 } else { -1 };
        let mut consistent_moves = 0;
        for j in (start + 1)..=idx {
            let move_direction = if close[j] > close[j - 1] { 1 } else { -1 };
            if move_direction == trend_direction {
                consistent_moves += 1;
            }
        }
        let consistency = consistent_moves as f64 / (idx - start) as f64;

        // Signal combines position relative to MA and normalized slope
        let position_signal = if close[idx] > ma { 50.0 } else { -50.0 };
        let slope_signal = if volatility > 1e-10 {
            (slope / volatility * 50.0).clamp(-50.0, 50.0)
        } else {
            0.0
        };

        let signal = position_signal + slope_signal;
        (signal, consistency)
    }

    /// Calculate adaptive trend follower values (-100 to +100)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.long_period..n {
            let (short_signal, short_consistency) = self.timeframe_signal(close, self.short_period, i);
            let (medium_signal, medium_consistency) = self.timeframe_signal(close, self.medium_period, i);
            let (long_signal, long_consistency) = self.timeframe_signal(close, self.long_period, i);

            // Adaptive weighting based on consistency
            let total_consistency = short_consistency + medium_consistency + long_consistency;
            if total_consistency > 1e-10 {
                let short_weight = short_consistency / total_consistency;
                let medium_weight = medium_consistency / total_consistency;
                let long_weight = long_consistency / total_consistency;

                result[i] = short_signal * short_weight
                    + medium_signal * medium_weight
                    + long_signal * long_weight;
            } else {
                // Equal weights if no consistency
                result[i] = (short_signal + medium_signal + long_signal) / 3.0;
            }

            result[i] = result[i].clamp(-100.0, 100.0);
        }
        result
    }
}

impl TechnicalIndicator for AdaptiveTrendFollower {
    fn name(&self) -> &str {
        "Adaptive Trend Follower"
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Trend Quality Index - Measures trend quality using directional movement and consistency
///
/// Evaluates the overall quality of a trend by combining:
/// - Directional movement strength
/// - Price consistency (efficiency ratio)
/// - Volatility-adjusted momentum
/// Returns values from 0 to 100 (higher = better quality trend).
#[derive(Debug, Clone)]
pub struct TrendQualityIndex {
    period: usize,
}

impl TrendQualityIndex {
    pub fn new(period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate trend quality index (0-100)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len());
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Factor 1: Directional Movement Strength (0-33)
            let mut plus_dm = 0.0;
            let mut minus_dm = 0.0;
            let mut tr_sum = 0.0;

            for j in (start + 1)..=i {
                let up_move = high[j] - high[j - 1];
                let down_move = low[j - 1] - low[j];

                if up_move > down_move && up_move > 0.0 {
                    plus_dm += up_move;
                }
                if down_move > up_move && down_move > 0.0 {
                    minus_dm += down_move;
                }

                let tr = (high[j] - low[j])
                    .max((high[j] - close[j - 1]).abs())
                    .max((low[j] - close[j - 1]).abs());
                tr_sum += tr;
            }

            let dm_strength = if tr_sum > 1e-10 {
                let di_plus = plus_dm / tr_sum;
                let di_minus = minus_dm / tr_sum;
                let di_diff = (di_plus - di_minus).abs();
                let di_sum = di_plus + di_minus;
                if di_sum > 1e-10 {
                    (di_diff / di_sum * 33.0).min(33.0)
                } else {
                    0.0
                }
            } else {
                0.0
            };

            // Factor 2: Price Efficiency (0-33)
            let direct_move = (close[i] - close[start]).abs();
            let mut path_length = 0.0;
            for j in (start + 1)..=i {
                path_length += (close[j] - close[j - 1]).abs();
            }
            let efficiency = if path_length > 1e-10 {
                (direct_move / path_length * 33.0).min(33.0)
            } else {
                0.0
            };

            // Factor 3: Consistency of Direction (0-34)
            let trend_direction = if close[i] > close[start] { 1 } else { -1 };
            let mut consistent_bars = 0;
            for j in (start + 1)..=i {
                let bar_direction = if close[j] > close[j - 1] { 1 } else { -1 };
                if bar_direction == trend_direction {
                    consistent_bars += 1;
                }
            }
            let consistency = consistent_bars as f64 / (i - start) as f64 * 34.0;

            result[i] = dm_strength + efficiency + consistency;
        }
        result
    }
}

impl TechnicalIndicator for TrendQualityIndex {
    fn name(&self) -> &str {
        "Trend Quality Index"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

/// Trend Breakout Strength - Measures strength of trend breakouts
///
/// Quantifies breakout strength by analyzing:
/// - Price movement beyond recent range
/// - Volume confirmation (if available)
/// - Momentum at breakout
/// Returns values from 0 to 100 (higher = stronger breakout).
#[derive(Debug, Clone)]
pub struct TrendBreakoutStrength {
    lookback_period: usize,
    breakout_threshold: f64,
}

impl TrendBreakoutStrength {
    pub fn new(lookback_period: usize, breakout_threshold: f64) -> Result<Self> {
        if lookback_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if breakout_threshold <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "breakout_threshold".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { lookback_period, breakout_threshold })
    }

    /// Calculate breakout strength (0-100)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len()).min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.lookback_period..n {
            let start = i.saturating_sub(self.lookback_period);

            // Calculate recent range (excluding current bar)
            let range_high = high[start..i].iter().cloned().fold(f64::MIN, f64::max);
            let range_low = low[start..i].iter().cloned().fold(f64::MAX, f64::min);
            let range = range_high - range_low;

            if range < 1e-10 {
                continue;
            }

            // Factor 1: Breakout Magnitude (0-40)
            let upside_breakout = if close[i] > range_high {
                (close[i] - range_high) / range
            } else {
                0.0
            };
            let downside_breakout = if close[i] < range_low {
                (range_low - close[i]) / range
            } else {
                0.0
            };
            let breakout_magnitude = upside_breakout.max(downside_breakout);

            let magnitude_score = if breakout_magnitude > self.breakout_threshold {
                ((breakout_magnitude / self.breakout_threshold) * 40.0).min(40.0)
            } else {
                breakout_magnitude / self.breakout_threshold * 20.0
            };

            // Factor 2: Volume Confirmation (0-30)
            let avg_volume: f64 = volume[start..i].iter().sum::<f64>() / (i - start) as f64;
            let volume_ratio = if avg_volume > 1e-10 {
                volume[i] / avg_volume
            } else {
                1.0
            };
            let volume_score = ((volume_ratio - 1.0) * 15.0).clamp(0.0, 30.0);

            // Factor 3: Momentum at Breakout (0-30)
            let momentum = (close[i] - close[i - 1]).abs();
            let avg_momentum: f64 = (start + 1..i)
                .map(|j| (close[j] - close[j - 1]).abs())
                .sum::<f64>() / (i - start - 1).max(1) as f64;
            let momentum_ratio = if avg_momentum > 1e-10 {
                momentum / avg_momentum
            } else {
                1.0
            };
            let momentum_score = ((momentum_ratio - 1.0) * 15.0).clamp(0.0, 30.0);

            result[i] = magnitude_score + volume_score + momentum_score;
        }
        result
    }
}

impl TechnicalIndicator for TrendBreakoutStrength {
    fn name(&self) -> &str {
        "Trend Breakout Strength"
    }

    fn min_periods(&self) -> usize {
        self.lookback_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close, &data.volume)))
    }
}

/// Trend Persistence Metric - Quantifies how persistent trends are
///
/// Measures trend persistence through:
/// - Duration of trend (consecutive bars in direction)
/// - Depth of retracements
/// - Recovery speed from pullbacks
/// Returns values from 0 to 100 (higher = more persistent trend).
#[derive(Debug, Clone)]
pub struct TrendPersistenceMetric {
    period: usize,
}

impl TrendPersistenceMetric {
    pub fn new(period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate trend persistence (0-100)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Determine overall trend direction
            let trend_direction = if close[i] >= close[start] { 1.0 } else { -1.0 };

            // Factor 1: Consecutive bars in trend direction (0-35)
            let mut max_consecutive = 0;
            let mut current_consecutive = 0;
            for j in (start + 1)..=i {
                let bar_direction = if close[j] >= close[j - 1] { 1.0 } else { -1.0 };
                if bar_direction == trend_direction {
                    current_consecutive += 1;
                    max_consecutive = max_consecutive.max(current_consecutive);
                } else {
                    current_consecutive = 0;
                }
            }
            let duration_score = (max_consecutive as f64 / (i - start) as f64 * 35.0).min(35.0);

            // Factor 2: Shallow retracements (0-35)
            let total_move = (close[i] - close[start]).abs();
            let mut max_retracement: f64 = 0.0;

            if trend_direction > 0.0 {
                // Uptrend: track pullbacks from highs
                let mut peak = close[start];
                for j in start..=i {
                    if close[j] > peak {
                        peak = close[j];
                    }
                    let retracement = (peak - close[j]).abs();
                    max_retracement = max_retracement.max(retracement);
                }
            } else {
                // Downtrend: track rallies from lows
                let mut trough = close[start];
                for j in start..=i {
                    if close[j] < trough {
                        trough = close[j];
                    }
                    let retracement = (close[j] - trough).abs();
                    max_retracement = max_retracement.max(retracement);
                }
            }

            let retracement_ratio = if total_move > 1e-10 {
                max_retracement / total_move
            } else {
                1.0
            };
            let retracement_score = ((1.0 - retracement_ratio.min(1.0)) * 35.0).max(0.0);

            // Factor 3: Trend continuity (0-30)
            let mut trend_aligned_moves = 0;
            let total_moves = i - start;
            for j in (start + 1)..=i {
                let bar_direction = if close[j] >= close[j - 1] { 1.0 } else { -1.0 };
                if bar_direction == trend_direction {
                    trend_aligned_moves += 1;
                }
            }
            let continuity_score = if total_moves > 0 {
                trend_aligned_moves as f64 / total_moves as f64 * 30.0
            } else {
                0.0
            };

            result[i] = duration_score + retracement_score + continuity_score;
        }
        result
    }
}

impl TechnicalIndicator for TrendPersistenceMetric {
    fn name(&self) -> &str {
        "Trend Persistence Metric"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Trend Cycle Finder - Identifies trend cycles and their phases
///
/// Detects market cycles and identifies the current phase:
/// - Accumulation (0-25): Base building after downtrend
/// - Markup (25-50): Uptrend phase
/// - Distribution (50-75): Top formation
/// - Markdown (75-100): Downtrend phase
#[derive(Debug, Clone)]
pub struct TrendCycleFinder {
    short_period: usize,
    long_period: usize,
}

impl TrendCycleFinder {
    pub fn new(short_period: usize, long_period: usize) -> Result<Self> {
        if short_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be at least 2".to_string(),
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

    /// Calculate cycle phase (0-100)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.long_period..n {
            let short_start = i.saturating_sub(self.short_period);
            let long_start = i.saturating_sub(self.long_period);

            // Calculate short and long MAs
            let short_ma: f64 = close[short_start..=i].iter().sum::<f64>() / self.short_period as f64;
            let long_ma: f64 = close[long_start..=i].iter().sum::<f64>() / self.long_period as f64;

            // Calculate slopes
            let short_slope = (close[i] - close[short_start]) / self.short_period as f64;
            let long_slope = (close[i] - close[long_start]) / self.long_period as f64;

            // Determine phase based on MA relationship and slopes
            let price_above_short = close[i] > short_ma;
            let price_above_long = close[i] > long_ma;
            let short_above_long = short_ma > long_ma;
            let short_slope_positive = short_slope > 0.0;
            let long_slope_positive = long_slope > 0.0;

            // Phase detection logic
            let phase = if !price_above_long && !short_above_long && !long_slope_positive && short_slope_positive {
                // Accumulation: price below long MA, short starting to turn up
                // Range: 0-25
                let recovery_strength = if short_slope.abs() > 1e-10 {
                    (short_slope.abs() / (long_slope.abs() + 1e-10)).min(1.0)
                } else {
                    0.0
                };
                recovery_strength * 25.0
            } else if price_above_long && short_above_long && short_slope_positive && long_slope_positive {
                // Markup: uptrend in full swing
                // Range: 25-50
                let trend_strength = if price_above_short { 1.0 } else { 0.5 };
                25.0 + trend_strength * 25.0
            } else if price_above_long && short_above_long && !short_slope_positive {
                // Distribution: price high but momentum waning
                // Range: 50-75
                let distribution_progress = if long_slope_positive {
                    0.5 // Early distribution
                } else {
                    1.0 // Late distribution
                };
                50.0 + distribution_progress * 25.0
            } else if !price_above_long || !short_above_long {
                // Markdown: downtrend
                // Range: 75-100
                let markdown_depth = if !price_above_short && !short_slope_positive {
                    1.0 // Deep markdown
                } else {
                    0.5 // Early markdown
                };
                75.0 + markdown_depth * 25.0
            } else {
                // Transition state
                50.0
            };

            result[i] = phase;
        }
        result
    }
}

impl TechnicalIndicator for TrendCycleFinder {
    fn name(&self) -> &str {
        "Trend Cycle Finder"
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Trend Volatility Ratio - Ratio of trend strength to volatility
///
/// Compares directional movement to overall volatility:
/// - High values indicate strong trend with low noise
/// - Low values indicate weak trend or high noise
/// Returns values typically ranging from 0 to 2+ (higher = better trend/volatility ratio).
#[derive(Debug, Clone)]
pub struct TrendVolatilityRatio {
    period: usize,
}

impl TrendVolatilityRatio {
    pub fn new(period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate trend/volatility ratio
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len());
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Trend strength: absolute price change
            let trend_move = (close[i] - close[start]).abs();

            // Volatility: sum of true ranges
            let mut total_volatility = 0.0;
            for j in (start + 1)..=i {
                let tr = (high[j] - low[j])
                    .max((high[j] - close[j - 1]).abs())
                    .max((low[j] - close[j - 1]).abs());
                total_volatility += tr;
            }

            // Ratio: trend strength divided by volatility
            if total_volatility > 1e-10 {
                result[i] = trend_move / total_volatility * (i - start) as f64;
            }
        }
        result
    }
}

impl TechnicalIndicator for TrendVolatilityRatio {
    fn name(&self) -> &str {
        "Trend Volatility Ratio"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

/// Trend Continuity - Measures trend continuity/consistency over time
///
/// Evaluates how continuous a trend is by analyzing:
/// - Consecutive bars moving in the same direction
/// - Gap-free price progression
/// - Smooth vs choppy movement
/// Returns values from 0 to 100 (higher = more continuous trend).
#[derive(Debug, Clone)]
pub struct TrendContinuity {
    period: usize,
    smoothing: usize,
}

impl TrendContinuity {
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

    /// Calculate trend continuity (0-100)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.period + self.smoothing {
            return result;
        }

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Determine overall trend direction
            let trend_direction = if close[i] >= close[start] { 1.0 } else { -1.0 };

            // Factor 1: Consecutive moves in trend direction (0-40)
            let mut max_consecutive = 0;
            let mut current_consecutive = 0;
            let mut total_aligned = 0;

            for j in (start + 1)..=i {
                let bar_direction = if close[j] >= close[j - 1] { 1.0 } else { -1.0 };
                if bar_direction == trend_direction {
                    current_consecutive += 1;
                    max_consecutive = max_consecutive.max(current_consecutive);
                    total_aligned += 1;
                } else {
                    current_consecutive = 0;
                }
            }

            let bars = i - start;
            let consecutive_score = (max_consecutive as f64 / bars as f64 * 40.0).min(40.0);

            // Factor 2: Overall alignment ratio (0-30)
            let alignment_score = total_aligned as f64 / bars as f64 * 30.0;

            // Factor 3: Smoothness - penalize large individual moves (0-30)
            let mut moves: Vec<f64> = Vec::with_capacity(bars);
            for j in (start + 1)..=i {
                moves.push((close[j] - close[j - 1]).abs());
            }

            let avg_move = moves.iter().sum::<f64>() / moves.len() as f64;
            let variance = if avg_move > 1e-10 {
                moves.iter().map(|m| ((m - avg_move) / avg_move).powi(2)).sum::<f64>() / moves.len() as f64
            } else {
                0.0
            };
            let smoothness = (1.0 - variance.sqrt().min(1.0)) * 30.0;

            result[i] = consecutive_score + alignment_score + smoothness;
        }

        // Apply smoothing
        if self.smoothing > 1 {
            let mut smoothed = vec![0.0; n];
            for i in (self.period + self.smoothing - 1)..n {
                let start = i.saturating_sub(self.smoothing - 1);
                let sum: f64 = result[start..=i].iter().sum();
                smoothed[i] = sum / self.smoothing as f64;
            }
            return smoothed;
        }

        result
    }
}

impl TechnicalIndicator for TrendContinuity {
    fn name(&self) -> &str {
        "Trend Continuity"
    }

    fn min_periods(&self) -> usize {
        self.period + self.smoothing
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Trend Momentum Convergence - Convergence of trend and momentum
///
/// Measures the degree to which price trend and momentum are aligned:
/// - When trend and momentum agree, signal is strong
/// - When they diverge, signal weakens
/// Returns values from -100 to +100 (positive = bullish convergence, negative = bearish).
#[derive(Debug, Clone)]
pub struct TrendMomentumConvergence {
    trend_period: usize,
    momentum_period: usize,
}

impl TrendMomentumConvergence {
    pub fn new(trend_period: usize, momentum_period: usize) -> Result<Self> {
        if trend_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "trend_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if momentum_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { trend_period, momentum_period })
    }

    /// Calculate trend-momentum convergence (-100 to +100)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];
        let min_period = self.trend_period.max(self.momentum_period);

        if n <= min_period {
            return result;
        }

        for i in min_period..n {
            let trend_start = i.saturating_sub(self.trend_period);
            let momentum_start = i.saturating_sub(self.momentum_period);

            // Calculate trend component: normalized slope
            let trend_ma: f64 = close[trend_start..=i].iter().sum::<f64>() / self.trend_period as f64;
            let trend_slope = (close[i] - close[trend_start]) / self.trend_period as f64;

            // Calculate volatility for normalization
            let mut volatility = 0.0;
            for j in (trend_start + 1)..=i {
                volatility += (close[j] - close[j - 1]).abs();
            }
            volatility /= self.trend_period as f64;

            let normalized_trend = if volatility > 1e-10 {
                (trend_slope / volatility).clamp(-1.0, 1.0)
            } else {
                0.0
            };

            // Calculate momentum component: rate of change
            let momentum = if close[momentum_start].abs() > 1e-10 {
                (close[i] - close[momentum_start]) / close[momentum_start]
            } else {
                0.0
            };
            let normalized_momentum = (momentum * 10.0).clamp(-1.0, 1.0);

            // Calculate position relative to MA
            let position = if close[i] > trend_ma { 1.0 } else { -1.0 };

            // Convergence: all three components agree
            let trend_sign = if normalized_trend > 0.0 { 1.0 } else { -1.0 };
            let momentum_sign = if normalized_momentum > 0.0 { 1.0 } else { -1.0 };

            // Base signal from trend
            let mut signal = normalized_trend * 50.0;

            // Momentum confirmation (adds up to 30 points)
            if trend_sign == momentum_sign {
                signal += normalized_momentum.abs() * 30.0 * trend_sign;
            } else {
                // Divergence penalty
                signal -= normalized_momentum.abs() * 15.0 * trend_sign;
            }

            // Position confirmation (adds up to 20 points)
            if position == trend_sign {
                signal += 20.0 * trend_sign;
            } else {
                signal -= 10.0 * trend_sign;
            }

            result[i] = signal.clamp(-100.0, 100.0);
        }

        result
    }
}

impl TechnicalIndicator for TrendMomentumConvergence {
    fn name(&self) -> &str {
        "Trend Momentum Convergence"
    }

    fn min_periods(&self) -> usize {
        self.trend_period.max(self.momentum_period) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Adaptive Trend Strength - Volatility-adaptive trend strength measurement
///
/// Measures trend strength with automatic adaptation to volatility:
/// - High volatility: requires stronger trend for positive signal
/// - Low volatility: more sensitive to trend changes
/// Returns values from 0 to 100 (higher = stronger trend relative to volatility).
#[derive(Debug, Clone)]
pub struct AdaptiveTrendStrength {
    period: usize,
    volatility_period: usize,
    sensitivity: f64,
}

impl AdaptiveTrendStrength {
    pub fn new(period: usize, volatility_period: usize, sensitivity: f64) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if volatility_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if sensitivity <= 0.0 || sensitivity > 5.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "sensitivity".to_string(),
                reason: "must be between 0 and 5".to_string(),
            });
        }
        Ok(Self { period, volatility_period, sensitivity })
    }

    /// Calculate adaptive trend strength (0-100)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len());
        let mut result = vec![0.0; n];
        let min_period = self.period.max(self.volatility_period);

        if n <= min_period {
            return result;
        }

        for i in min_period..n {
            let trend_start = i.saturating_sub(self.period);
            let vol_start = i.saturating_sub(self.volatility_period);

            // Calculate ATR-based volatility
            let mut atr_sum = 0.0;
            for j in (vol_start + 1)..=i {
                let tr = (high[j] - low[j])
                    .max((high[j] - close[j - 1]).abs())
                    .max((low[j] - close[j - 1]).abs());
                atr_sum += tr;
            }
            let atr = atr_sum / self.volatility_period as f64;

            // Calculate directional movement
            let trend_move = (close[i] - close[trend_start]).abs();

            // Calculate price efficiency
            let mut total_path = 0.0;
            for j in (trend_start + 1)..=i {
                total_path += (close[j] - close[j - 1]).abs();
            }
            let efficiency = if total_path > 1e-10 {
                trend_move / total_path
            } else {
                0.0
            };

            // Adaptive threshold based on volatility
            let expected_move = atr * self.period as f64;
            let adaptive_threshold = expected_move * self.sensitivity;

            // Factor 1: Trend magnitude vs adaptive threshold (0-50)
            let magnitude_score = if adaptive_threshold > 1e-10 {
                (trend_move / adaptive_threshold * 50.0).min(50.0)
            } else {
                0.0
            };

            // Factor 2: Price efficiency (0-30)
            let efficiency_score = efficiency * 30.0;

            // Factor 3: Directional consistency (0-20)
            let trend_direction = if close[i] >= close[trend_start] { 1.0 } else { -1.0 };
            let mut aligned_moves = 0;
            for j in (trend_start + 1)..=i {
                let move_direction = if close[j] >= close[j - 1] { 1.0 } else { -1.0 };
                if move_direction == trend_direction {
                    aligned_moves += 1;
                }
            }
            let consistency_score = aligned_moves as f64 / (i - trend_start) as f64 * 20.0;

            result[i] = magnitude_score + efficiency_score + consistency_score;
        }

        result
    }
}

impl TechnicalIndicator for AdaptiveTrendStrength {
    fn name(&self) -> &str {
        "Adaptive Trend Strength"
    }

    fn min_periods(&self) -> usize {
        self.period.max(self.volatility_period) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

/// Trend Direction Index - Clear directional index with smoothing
///
/// Provides a smoothed directional trend reading:
/// - Positive values indicate uptrend
/// - Negative values indicate downtrend
/// - Magnitude indicates strength
/// Returns values from -100 to +100.
#[derive(Debug, Clone)]
pub struct TrendDirectionIndex {
    period: usize,
    smoothing: usize,
}

impl TrendDirectionIndex {
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

    /// Calculate trend direction index (-100 to +100)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len());
        let mut result = vec![0.0; n];

        if n < self.period + self.smoothing {
            return result;
        }

        // First pass: calculate raw directional index
        let mut raw_index = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate +DM and -DM
            let mut plus_dm = 0.0;
            let mut minus_dm = 0.0;
            let mut tr_sum = 0.0;

            for j in (start + 1)..=i {
                let up_move = high[j] - high[j - 1];
                let down_move = low[j - 1] - low[j];

                if up_move > down_move && up_move > 0.0 {
                    plus_dm += up_move;
                }
                if down_move > up_move && down_move > 0.0 {
                    minus_dm += down_move;
                }

                let tr = (high[j] - low[j])
                    .max((high[j] - close[j - 1]).abs())
                    .max((low[j] - close[j - 1]).abs());
                tr_sum += tr;
            }

            // Calculate DI+ and DI-
            let di_plus = if tr_sum > 1e-10 { plus_dm / tr_sum * 100.0 } else { 0.0 };
            let di_minus = if tr_sum > 1e-10 { minus_dm / tr_sum * 100.0 } else { 0.0 };

            // Calculate DX-like index but directional
            let di_sum = di_plus + di_minus;
            if di_sum > 1e-10 {
                // Positive when DI+ > DI-, negative when DI- > DI+
                raw_index[i] = (di_plus - di_minus) / di_sum * 100.0;
            }
        }

        // Second pass: apply smoothing
        for i in (self.period + self.smoothing - 1)..n {
            let start = i.saturating_sub(self.smoothing - 1);
            let sum: f64 = raw_index[start..=i].iter().sum();
            result[i] = sum / self.smoothing as f64;
        }

        result
    }
}

impl TechnicalIndicator for TrendDirectionIndex {
    fn name(&self) -> &str {
        "Trend Direction Index"
    }

    fn min_periods(&self) -> usize {
        self.period + self.smoothing
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

/// Trend Maturity - Measures how mature/exhausted a trend is
///
/// Evaluates trend maturity through:
/// - Duration of the current trend
/// - Momentum decay
/// - Volatility changes
/// Returns values from 0 to 100 (higher = more mature/exhausted trend).
#[derive(Debug, Clone)]
pub struct TrendMaturity {
    short_period: usize,
    long_period: usize,
}

impl TrendMaturity {
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

    /// Calculate trend maturity (0-100)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n <= self.long_period {
            return result;
        }

        for i in self.long_period..n {
            let short_start = i.saturating_sub(self.short_period);
            let long_start = i.saturating_sub(self.long_period);

            // Determine if we're in an uptrend or downtrend
            let long_trend = close[i] - close[long_start];
            let is_uptrend = long_trend >= 0.0;

            // Factor 1: Duration analysis - how long has trend been going? (0-35)
            // Count consecutive bars in trend direction from the end
            let mut trend_bars = 0;
            let trend_direction = if is_uptrend { 1.0 } else { -1.0 };

            for j in (long_start + 1..=i).rev() {
                let bar_direction = if close[j] >= close[j - 1] { 1.0 } else { -1.0 };
                if bar_direction == trend_direction {
                    trend_bars += 1;
                } else {
                    break;
                }
            }

            // More bars = more mature
            let duration_ratio = trend_bars as f64 / self.long_period as f64;
            let duration_score = (duration_ratio * 35.0).min(35.0);

            // Factor 2: Momentum decay - compare recent vs older momentum (0-35)
            let recent_momentum = (close[i] - close[short_start]).abs();
            let mid_point = (short_start + long_start) / 2;
            let older_momentum = (close[short_start] - close[mid_point]).abs();

            let momentum_decay = if older_momentum > 1e-10 {
                // If recent momentum is less than older, trend is maturing
                1.0 - (recent_momentum / older_momentum).min(2.0) / 2.0
            } else if recent_momentum < 1e-10 {
                1.0 // No momentum at all = fully mature
            } else {
                0.0 // Recent momentum but no older = fresh trend
            };
            let momentum_score = (momentum_decay.max(0.0) * 35.0).min(35.0);

            // Factor 3: Volatility exhaustion - compare recent vs older volatility (0-30)
            let recent_start = i.saturating_sub(self.short_period / 2);
            let mut recent_vol = 0.0;
            for j in (recent_start + 1)..=i {
                recent_vol += (close[j] - close[j - 1]).abs();
            }
            recent_vol /= (i - recent_start).max(1) as f64;

            let older_end = short_start;
            let older_start = older_end.saturating_sub(self.short_period / 2);
            let mut older_vol = 0.0;
            for j in (older_start + 1)..=older_end {
                older_vol += (close[j] - close[j - 1]).abs();
            }
            older_vol /= (older_end - older_start).max(1) as f64;

            let vol_change = if older_vol > 1e-10 {
                recent_vol / older_vol
            } else {
                1.0
            };

            // Decreasing volatility suggests exhaustion, increasing suggests fresh move
            let vol_score = if vol_change < 1.0 {
                (1.0 - vol_change) * 30.0
            } else {
                0.0
            };

            result[i] = duration_score + momentum_score + vol_score;
        }

        result
    }
}

impl TechnicalIndicator for TrendMaturity {
    fn name(&self) -> &str {
        "Trend Maturity"
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Multi-Period Trend Alignment - Alignment across multiple periods
///
/// Measures trend alignment across short, medium, and long timeframes:
/// - All aligned bullish: strong positive signal
/// - All aligned bearish: strong negative signal
/// - Mixed signals: weak or no signal
/// Returns values from -100 to +100.
#[derive(Debug, Clone)]
pub struct MultiPeriodTrendAlignment {
    short_period: usize,
    medium_period: usize,
    long_period: usize,
}

impl MultiPeriodTrendAlignment {
    pub fn new(short_period: usize, medium_period: usize, long_period: usize) -> Result<Self> {
        if short_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if medium_period <= short_period {
            return Err(IndicatorError::InvalidParameter {
                name: "medium_period".to_string(),
                reason: "must be greater than short_period".to_string(),
            });
        }
        if long_period <= medium_period {
            return Err(IndicatorError::InvalidParameter {
                name: "long_period".to_string(),
                reason: "must be greater than medium_period".to_string(),
            });
        }
        Ok(Self { short_period, medium_period, long_period })
    }

    /// Calculate single timeframe signal
    fn timeframe_signal(&self, close: &[f64], period: usize, idx: usize) -> f64 {
        let start = idx.saturating_sub(period);
        if start >= idx {
            return 0.0;
        }

        // Calculate MA
        let ma: f64 = close[start..=idx].iter().sum::<f64>() / period as f64;

        // Calculate slope
        let slope = close[idx] - close[start];

        // Position relative to MA
        let above_ma = close[idx] > ma;

        // Slope direction
        let slope_positive = slope > 0.0;

        // Combined signal
        match (above_ma, slope_positive) {
            (true, true) => 1.0,   // Bullish
            (false, false) => -1.0, // Bearish
            _ => 0.0,              // Mixed
        }
    }

    /// Calculate strength of a timeframe signal
    fn timeframe_strength(&self, close: &[f64], period: usize, idx: usize) -> f64 {
        let start = idx.saturating_sub(period);
        if start >= idx {
            return 0.0;
        }

        // Price efficiency as strength measure
        let direct_move = (close[idx] - close[start]).abs();
        let mut total_path = 0.0;
        for j in (start + 1)..=idx {
            total_path += (close[j] - close[j - 1]).abs();
        }

        if total_path > 1e-10 {
            direct_move / total_path
        } else {
            0.0
        }
    }

    /// Calculate multi-period trend alignment (-100 to +100)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n <= self.long_period {
            return result;
        }

        for i in self.long_period..n {
            // Get signals for each timeframe
            let short_signal = self.timeframe_signal(close, self.short_period, i);
            let medium_signal = self.timeframe_signal(close, self.medium_period, i);
            let long_signal = self.timeframe_signal(close, self.long_period, i);

            // Get strengths for each timeframe
            let short_strength = self.timeframe_strength(close, self.short_period, i);
            let medium_strength = self.timeframe_strength(close, self.medium_period, i);
            let long_strength = self.timeframe_strength(close, self.long_period, i);

            // Check alignment
            let all_bullish = short_signal > 0.0 && medium_signal > 0.0 && long_signal > 0.0;
            let all_bearish = short_signal < 0.0 && medium_signal < 0.0 && long_signal < 0.0;

            if all_bullish {
                // All timeframes aligned bullish
                // Weight: short=20%, medium=30%, long=50%
                let weighted_strength = short_strength * 0.2 + medium_strength * 0.3 + long_strength * 0.5;
                result[i] = weighted_strength * 100.0;
            } else if all_bearish {
                // All timeframes aligned bearish
                let weighted_strength = short_strength * 0.2 + medium_strength * 0.3 + long_strength * 0.5;
                result[i] = -weighted_strength * 100.0;
            } else {
                // Partial alignment or no alignment
                let bullish_count = [short_signal, medium_signal, long_signal].iter().filter(|&&s| s > 0.0).count();
                let bearish_count = [short_signal, medium_signal, long_signal].iter().filter(|&&s| s < 0.0).count();

                if bullish_count > bearish_count {
                    // Partial bullish alignment
                    let partial_strength = (short_strength + medium_strength + long_strength) / 3.0;
                    result[i] = partial_strength * 100.0 * (bullish_count as f64 - bearish_count as f64) / 3.0;
                } else if bearish_count > bullish_count {
                    // Partial bearish alignment
                    let partial_strength = (short_strength + medium_strength + long_strength) / 3.0;
                    result[i] = -partial_strength * 100.0 * (bearish_count as f64 - bullish_count as f64) / 3.0;
                }
                // If equal, result stays 0 (no alignment)
            }

            result[i] = result[i].clamp(-100.0, 100.0);
        }

        result
    }
}

impl TechnicalIndicator for MultiPeriodTrendAlignment {
    fn name(&self) -> &str {
        "Multi-Period Trend Alignment"
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
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
        let volume = vec![
            1000.0, 1100.0, 1200.0, 1050.0, 1150.0, 1000.0, 1300.0, 1100.0, 1250.0, 1200.0,
            1100.0, 1150.0, 1300.0, 1400.0, 1200.0, 1100.0, 1000.0, 1250.0, 1350.0, 1400.0,
            1500.0, 1450.0, 1300.0, 1200.0, 1350.0, 1400.0, 1550.0, 1600.0, 1500.0, 1450.0,
            1400.0, 1350.0, 1500.0, 1600.0, 1700.0, 1650.0, 1550.0, 1500.0, 1600.0, 1700.0,
        ];
        (high, low, close, volume)
    }

    #[test]
    fn test_trend_acceleration() {
        let (_, _, close, _) = make_test_data();
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
        let (_, _, close, _) = make_test_data();
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
        let (_, _, close, _) = make_test_data();
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
        let (high, low, close, _) = make_test_data();
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
        let (_, _, close, _) = make_test_data();
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
        let (_, _, close, _) = make_test_data();
        let mst = MultiScaleTrend::new(vec![5, 10, 20], None).unwrap();
        let result = mst.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Consistent uptrend should show positive multi-scale trend
        assert!(result[35] > 0.0);
    }

    #[test]
    fn test_multi_scale_trend_with_weights() {
        let (_, _, close, _) = make_test_data();
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

    // ============= Tests for NEW indicators =============

    #[test]
    fn test_adaptive_trend_follower() {
        let (_, _, close, _) = make_test_data();
        let atf = AdaptiveTrendFollower::new(5, 10, 20).unwrap();
        let result = atf.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Uptrend should produce positive values
        assert!(result[30] > 0.0);
        // Values should be in range
        for val in &result[20..] {
            assert!(*val >= -100.0 && *val <= 100.0);
        }
    }

    #[test]
    fn test_adaptive_trend_follower_invalid_params() {
        // short_period must be at least 2
        assert!(AdaptiveTrendFollower::new(1, 10, 20).is_err());
        // medium_period must be > short_period
        assert!(AdaptiveTrendFollower::new(10, 10, 20).is_err());
        assert!(AdaptiveTrendFollower::new(10, 5, 20).is_err());
        // long_period must be > medium_period
        assert!(AdaptiveTrendFollower::new(5, 10, 10).is_err());
        assert!(AdaptiveTrendFollower::new(5, 10, 8).is_err());
    }

    #[test]
    fn test_adaptive_trend_follower_trait() {
        let atf = AdaptiveTrendFollower::new(5, 10, 20).unwrap();
        assert_eq!(atf.name(), "Adaptive Trend Follower");
        assert_eq!(atf.min_periods(), 21);
    }

    #[test]
    fn test_trend_quality_index() {
        let (high, low, close, _) = make_test_data();
        let tqi = TrendQualityIndex::new(10).unwrap();
        let result = tqi.calculate(&high, &low, &close);

        assert_eq!(result.len(), close.len());
        // Consistent uptrend should have good quality
        assert!(result[30] > 0.0);
        // Values should be in 0-100 range
        for val in &result[10..] {
            assert!(*val >= 0.0 && *val <= 100.0);
        }
    }

    #[test]
    fn test_trend_quality_index_invalid_params() {
        assert!(TrendQualityIndex::new(1).is_err());
    }

    #[test]
    fn test_trend_quality_index_trait() {
        let tqi = TrendQualityIndex::new(10).unwrap();
        assert_eq!(tqi.name(), "Trend Quality Index");
        assert_eq!(tqi.min_periods(), 11);
    }

    #[test]
    fn test_trend_breakout_strength() {
        let (high, low, close, volume) = make_test_data();
        let tbs = TrendBreakoutStrength::new(10, 0.5).unwrap();
        let result = tbs.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        // Values should be non-negative
        for val in &result {
            assert!(*val >= 0.0);
        }
    }

    #[test]
    fn test_trend_breakout_strength_invalid_params() {
        // lookback_period must be at least 2
        assert!(TrendBreakoutStrength::new(1, 0.5).is_err());
        // breakout_threshold must be positive
        assert!(TrendBreakoutStrength::new(10, 0.0).is_err());
        assert!(TrendBreakoutStrength::new(10, -0.5).is_err());
    }

    #[test]
    fn test_trend_breakout_strength_trait() {
        let tbs = TrendBreakoutStrength::new(10, 0.5).unwrap();
        assert_eq!(tbs.name(), "Trend Breakout Strength");
        assert_eq!(tbs.min_periods(), 11);
    }

    #[test]
    fn test_trend_persistence_metric() {
        let (_, _, close, _) = make_test_data();
        let tpm = TrendPersistenceMetric::new(10).unwrap();
        let result = tpm.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Consistent uptrend should show high persistence
        assert!(result[30] > 50.0);
        // Values should be in 0-100 range
        for val in &result[10..] {
            assert!(*val >= 0.0 && *val <= 100.0);
        }
    }

    #[test]
    fn test_trend_persistence_metric_invalid_params() {
        assert!(TrendPersistenceMetric::new(1).is_err());
    }

    #[test]
    fn test_trend_persistence_metric_trait() {
        let tpm = TrendPersistenceMetric::new(10).unwrap();
        assert_eq!(tpm.name(), "Trend Persistence Metric");
        assert_eq!(tpm.min_periods(), 11);
    }

    #[test]
    fn test_trend_cycle_finder() {
        let (_, _, close, _) = make_test_data();
        let tcf = TrendCycleFinder::new(5, 20).unwrap();
        let result = tcf.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Values should be in 0-100 range
        for val in &result[20..] {
            assert!(*val >= 0.0 && *val <= 100.0);
        }
        // In consistent uptrend, should be in markup phase (25-50)
        assert!(result[30] >= 25.0 && result[30] <= 50.0);
    }

    #[test]
    fn test_trend_cycle_finder_invalid_params() {
        // short_period must be at least 2
        assert!(TrendCycleFinder::new(1, 20).is_err());
        // long_period must be > short_period
        assert!(TrendCycleFinder::new(10, 10).is_err());
        assert!(TrendCycleFinder::new(10, 5).is_err());
    }

    #[test]
    fn test_trend_cycle_finder_trait() {
        let tcf = TrendCycleFinder::new(5, 20).unwrap();
        assert_eq!(tcf.name(), "Trend Cycle Finder");
        assert_eq!(tcf.min_periods(), 21);
    }

    #[test]
    fn test_trend_volatility_ratio() {
        let (high, low, close, _) = make_test_data();
        let tvr = TrendVolatilityRatio::new(10).unwrap();
        let result = tvr.calculate(&high, &low, &close);

        assert_eq!(result.len(), close.len());
        // Values should be non-negative
        for val in &result {
            assert!(*val >= 0.0);
        }
        // Strong trend should have positive ratio
        assert!(result[30] > 0.0);
    }

    #[test]
    fn test_trend_volatility_ratio_invalid_params() {
        assert!(TrendVolatilityRatio::new(1).is_err());
    }

    #[test]
    fn test_trend_volatility_ratio_trait() {
        let tvr = TrendVolatilityRatio::new(10).unwrap();
        assert_eq!(tvr.name(), "Trend Volatility Ratio");
        assert_eq!(tvr.min_periods(), 11);
    }

    #[test]
    fn test_new_indicators_technical_indicator_trait() {
        // Test all 6 new indicators implement TechnicalIndicator correctly
        let atf = AdaptiveTrendFollower::new(5, 10, 20).unwrap();
        assert_eq!(atf.name(), "Adaptive Trend Follower");
        assert_eq!(atf.min_periods(), 21);

        let tqi = TrendQualityIndex::new(10).unwrap();
        assert_eq!(tqi.name(), "Trend Quality Index");
        assert_eq!(tqi.min_periods(), 11);

        let tbs = TrendBreakoutStrength::new(10, 0.5).unwrap();
        assert_eq!(tbs.name(), "Trend Breakout Strength");
        assert_eq!(tbs.min_periods(), 11);

        let tpm = TrendPersistenceMetric::new(10).unwrap();
        assert_eq!(tpm.name(), "Trend Persistence Metric");
        assert_eq!(tpm.min_periods(), 11);

        let tcf = TrendCycleFinder::new(5, 20).unwrap();
        assert_eq!(tcf.name(), "Trend Cycle Finder");
        assert_eq!(tcf.min_periods(), 21);

        let tvr = TrendVolatilityRatio::new(10).unwrap();
        assert_eq!(tvr.name(), "Trend Volatility Ratio");
        assert_eq!(tvr.min_periods(), 11);
    }

    #[test]
    fn test_new_indicators_with_short_data() {
        // Test behavior with data shorter than min_periods
        let short_close = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let short_high = vec![102.0, 103.0, 104.0, 105.0, 106.0];
        let short_low = vec![98.0, 99.0, 100.0, 101.0, 102.0];
        let short_volume = vec![1000.0, 1100.0, 1200.0, 1050.0, 1150.0];

        let atf = AdaptiveTrendFollower::new(5, 10, 20).unwrap();
        let result = atf.calculate(&short_close);
        assert_eq!(result.len(), short_close.len());
        // All values should be 0.0 since not enough data
        for val in &result {
            assert_eq!(*val, 0.0);
        }

        let tqi = TrendQualityIndex::new(10).unwrap();
        let result = tqi.calculate(&short_high, &short_low, &short_close);
        assert_eq!(result.len(), short_close.len());

        let tbs = TrendBreakoutStrength::new(10, 0.5).unwrap();
        let result = tbs.calculate(&short_high, &short_low, &short_close, &short_volume);
        assert_eq!(result.len(), short_close.len());

        let tpm = TrendPersistenceMetric::new(10).unwrap();
        let result = tpm.calculate(&short_close);
        assert_eq!(result.len(), short_close.len());

        let tcf = TrendCycleFinder::new(5, 10).unwrap();
        let result = tcf.calculate(&short_close);
        assert_eq!(result.len(), short_close.len());

        let tvr = TrendVolatilityRatio::new(10).unwrap();
        let result = tvr.calculate(&short_high, &short_low, &short_close);
        assert_eq!(result.len(), short_close.len());
    }

    #[test]
    fn test_downtrend_data() {
        // Test with downtrending data
        let close: Vec<f64> = (0..40).map(|i| 140.0 - i as f64).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 2.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 2.0).collect();
        let volume: Vec<f64> = vec![1000.0; 40];

        let atf = AdaptiveTrendFollower::new(5, 10, 20).unwrap();
        let result = atf.calculate(&close);
        // Downtrend should produce negative values
        assert!(result[30] < 0.0);

        let tpm = TrendPersistenceMetric::new(10).unwrap();
        let result = tpm.calculate(&close);
        // Downtrend should still show persistence
        assert!(result[30] > 0.0);

        let tvr = TrendVolatilityRatio::new(10).unwrap();
        let result = tvr.calculate(&high, &low, &close);
        // Strong downtrend should have good ratio
        assert!(result[30] > 0.0);
    }

    // ============= Tests for 6 NEW indicators (TrendContinuity, TrendMomentumConvergence, etc.) =============

    #[test]
    fn test_trend_continuity() {
        let (_, _, close, _) = make_test_data();
        let tc = TrendContinuity::new(10, 3).unwrap();
        let result = tc.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Consistent uptrend should show high continuity
        assert!(result[30] > 30.0);
        // Values should be in 0-100 range
        for val in &result[13..] {
            assert!(*val >= 0.0 && *val <= 100.0);
        }
    }

    #[test]
    fn test_trend_continuity_invalid_params() {
        // period must be at least 5
        assert!(TrendContinuity::new(3, 3).is_err());
        // smoothing must be at least 2
        assert!(TrendContinuity::new(10, 1).is_err());
    }

    #[test]
    fn test_trend_continuity_trait() {
        let tc = TrendContinuity::new(10, 3).unwrap();
        assert_eq!(tc.name(), "Trend Continuity");
        assert_eq!(tc.min_periods(), 13);
    }

    #[test]
    fn test_trend_continuity_short_data() {
        let short_close = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let tc = TrendContinuity::new(10, 3).unwrap();
        let result = tc.calculate(&short_close);
        assert_eq!(result.len(), short_close.len());
        // All values should be 0.0 since not enough data
        for val in &result {
            assert_eq!(*val, 0.0);
        }
    }

    #[test]
    fn test_trend_momentum_convergence() {
        let (_, _, close, _) = make_test_data();
        let tmc = TrendMomentumConvergence::new(10, 5).unwrap();
        let result = tmc.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Uptrend with momentum should show positive convergence
        assert!(result[30] > 0.0);
        // Values should be in -100 to +100 range
        for val in &result[10..] {
            assert!(*val >= -100.0 && *val <= 100.0);
        }
    }

    #[test]
    fn test_trend_momentum_convergence_invalid_params() {
        // trend_period must be at least 5
        assert!(TrendMomentumConvergence::new(3, 5).is_err());
        // momentum_period must be at least 2
        assert!(TrendMomentumConvergence::new(10, 1).is_err());
    }

    #[test]
    fn test_trend_momentum_convergence_trait() {
        let tmc = TrendMomentumConvergence::new(10, 5).unwrap();
        assert_eq!(tmc.name(), "Trend Momentum Convergence");
        assert_eq!(tmc.min_periods(), 11);
    }

    #[test]
    fn test_trend_momentum_convergence_downtrend() {
        let close: Vec<f64> = (0..40).map(|i| 140.0 - i as f64).collect();
        let tmc = TrendMomentumConvergence::new(10, 5).unwrap();
        let result = tmc.calculate(&close);
        // Downtrend should show negative convergence
        assert!(result[30] < 0.0);
    }

    #[test]
    fn test_adaptive_trend_strength() {
        let (high, low, close, _) = make_test_data();
        let ats = AdaptiveTrendStrength::new(10, 5, 1.0).unwrap();
        let result = ats.calculate(&high, &low, &close);

        assert_eq!(result.len(), close.len());
        // Consistent uptrend should show good strength
        assert!(result[30] > 0.0);
        // Values should be in 0-100 range
        for val in &result[10..] {
            assert!(*val >= 0.0 && *val <= 100.0);
        }
    }

    #[test]
    fn test_adaptive_trend_strength_invalid_params() {
        // period must be at least 5
        assert!(AdaptiveTrendStrength::new(3, 5, 1.0).is_err());
        // volatility_period must be at least 2
        assert!(AdaptiveTrendStrength::new(10, 1, 1.0).is_err());
        // sensitivity must be between 0 and 5
        assert!(AdaptiveTrendStrength::new(10, 5, 0.0).is_err());
        assert!(AdaptiveTrendStrength::new(10, 5, 6.0).is_err());
    }

    #[test]
    fn test_adaptive_trend_strength_trait() {
        let ats = AdaptiveTrendStrength::new(10, 5, 1.0).unwrap();
        assert_eq!(ats.name(), "Adaptive Trend Strength");
        assert_eq!(ats.min_periods(), 11);
    }

    #[test]
    fn test_adaptive_trend_strength_sensitivity() {
        let (high, low, close, _) = make_test_data();

        // Lower sensitivity should give higher scores for same trend
        let ats_low = AdaptiveTrendStrength::new(10, 5, 0.5).unwrap();
        let ats_high = AdaptiveTrendStrength::new(10, 5, 2.0).unwrap();

        let result_low = ats_low.calculate(&high, &low, &close);
        let result_high = ats_high.calculate(&high, &low, &close);

        // Lower sensitivity threshold means same trend appears stronger
        assert!(result_low[30] >= result_high[30]);
    }

    #[test]
    fn test_trend_direction_index() {
        let (high, low, close, _) = make_test_data();
        let tdi = TrendDirectionIndex::new(10, 3).unwrap();
        let result = tdi.calculate(&high, &low, &close);

        assert_eq!(result.len(), close.len());
        // Uptrend should show positive direction
        assert!(result[30] > 0.0);
        // Values should be in -100 to +100 range
        for val in &result[13..] {
            assert!(*val >= -100.0 && *val <= 100.0);
        }
    }

    #[test]
    fn test_trend_direction_index_invalid_params() {
        // period must be at least 5
        assert!(TrendDirectionIndex::new(3, 3).is_err());
        // smoothing must be at least 2
        assert!(TrendDirectionIndex::new(10, 1).is_err());
    }

    #[test]
    fn test_trend_direction_index_trait() {
        let tdi = TrendDirectionIndex::new(10, 3).unwrap();
        assert_eq!(tdi.name(), "Trend Direction Index");
        assert_eq!(tdi.min_periods(), 13);
    }

    #[test]
    fn test_trend_direction_index_downtrend() {
        let close: Vec<f64> = (0..40).map(|i| 140.0 - i as f64).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 2.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 2.0).collect();

        let tdi = TrendDirectionIndex::new(10, 3).unwrap();
        let result = tdi.calculate(&high, &low, &close);
        // Downtrend should show negative direction
        assert!(result[30] < 0.0);
    }

    #[test]
    fn test_trend_maturity() {
        let (_, _, close, _) = make_test_data();
        let tm = TrendMaturity::new(5, 20).unwrap();
        let result = tm.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Values should be in 0-100 range
        for val in &result[20..] {
            assert!(*val >= 0.0 && *val <= 100.0);
        }
    }

    #[test]
    fn test_trend_maturity_invalid_params() {
        // short_period must be at least 5
        assert!(TrendMaturity::new(3, 20).is_err());
        // long_period must be > short_period
        assert!(TrendMaturity::new(10, 10).is_err());
        assert!(TrendMaturity::new(10, 5).is_err());
    }

    #[test]
    fn test_trend_maturity_trait() {
        let tm = TrendMaturity::new(5, 20).unwrap();
        assert_eq!(tm.name(), "Trend Maturity");
        assert_eq!(tm.min_periods(), 21);
    }

    #[test]
    fn test_trend_maturity_extended_trend() {
        // Create a longer trend to test maturity detection
        let close: Vec<f64> = (0..60).map(|i| 100.0 + i as f64).collect();
        let tm = TrendMaturity::new(10, 30).unwrap();
        let result = tm.calculate(&close);

        // At the end of a long consistent trend, maturity should be elevated
        // because duration is long even if momentum hasn't decayed
        assert!(result[55] > 0.0);
    }

    #[test]
    fn test_multi_period_trend_alignment() {
        let (_, _, close, _) = make_test_data();
        let mpta = MultiPeriodTrendAlignment::new(5, 10, 20).unwrap();
        let result = mpta.calculate(&close);

        assert_eq!(result.len(), close.len());
        // All timeframes aligned in uptrend should show positive alignment
        assert!(result[30] > 0.0);
        // Values should be in -100 to +100 range
        for val in &result[20..] {
            assert!(*val >= -100.0 && *val <= 100.0);
        }
    }

    #[test]
    fn test_multi_period_trend_alignment_invalid_params() {
        // short_period must be at least 2
        assert!(MultiPeriodTrendAlignment::new(1, 10, 20).is_err());
        // medium_period must be > short_period
        assert!(MultiPeriodTrendAlignment::new(10, 10, 20).is_err());
        assert!(MultiPeriodTrendAlignment::new(10, 5, 20).is_err());
        // long_period must be > medium_period
        assert!(MultiPeriodTrendAlignment::new(5, 10, 10).is_err());
        assert!(MultiPeriodTrendAlignment::new(5, 10, 8).is_err());
    }

    #[test]
    fn test_multi_period_trend_alignment_trait() {
        let mpta = MultiPeriodTrendAlignment::new(5, 10, 20).unwrap();
        assert_eq!(mpta.name(), "Multi-Period Trend Alignment");
        assert_eq!(mpta.min_periods(), 21);
    }

    #[test]
    fn test_multi_period_trend_alignment_downtrend() {
        let close: Vec<f64> = (0..40).map(|i| 140.0 - i as f64).collect();
        let mpta = MultiPeriodTrendAlignment::new(5, 10, 20).unwrap();
        let result = mpta.calculate(&close);
        // All timeframes aligned in downtrend should show negative alignment
        assert!(result[30] < 0.0);
    }

    #[test]
    fn test_multi_period_trend_alignment_mixed() {
        // Create data with mixed trends across timeframes
        let mut close = vec![100.0; 40];
        // Short term up, but long term flat/down
        for i in 0..20 {
            close[i] = 100.0 - i as f64 * 0.1; // Slight downtrend early
        }
        for i in 20..40 {
            close[i] = 98.0 + (i - 20) as f64 * 0.3; // Stronger uptrend late
        }

        let mpta = MultiPeriodTrendAlignment::new(5, 10, 20).unwrap();
        let result = mpta.calculate(&close);

        // With mixed trends, alignment should be weaker (closer to 0)
        assert!(result[35].abs() < 50.0);
    }

    #[test]
    fn test_new_six_indicators_technical_indicator_trait() {
        // Test all 6 new indicators implement TechnicalIndicator correctly
        let tc = TrendContinuity::new(10, 3).unwrap();
        assert_eq!(tc.name(), "Trend Continuity");
        assert_eq!(tc.min_periods(), 13);

        let tmc = TrendMomentumConvergence::new(10, 5).unwrap();
        assert_eq!(tmc.name(), "Trend Momentum Convergence");
        assert_eq!(tmc.min_periods(), 11);

        let ats = AdaptiveTrendStrength::new(10, 5, 1.0).unwrap();
        assert_eq!(ats.name(), "Adaptive Trend Strength");
        assert_eq!(ats.min_periods(), 11);

        let tdi = TrendDirectionIndex::new(10, 3).unwrap();
        assert_eq!(tdi.name(), "Trend Direction Index");
        assert_eq!(tdi.min_periods(), 13);

        let tm = TrendMaturity::new(5, 20).unwrap();
        assert_eq!(tm.name(), "Trend Maturity");
        assert_eq!(tm.min_periods(), 21);

        let mpta = MultiPeriodTrendAlignment::new(5, 10, 20).unwrap();
        assert_eq!(mpta.name(), "Multi-Period Trend Alignment");
        assert_eq!(mpta.min_periods(), 21);
    }

    #[test]
    fn test_new_six_indicators_with_short_data() {
        // Test behavior with data shorter than min_periods
        let short_close = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let short_high = vec![102.0, 103.0, 104.0, 105.0, 106.0];
        let short_low = vec![98.0, 99.0, 100.0, 101.0, 102.0];

        let tc = TrendContinuity::new(10, 3).unwrap();
        let result = tc.calculate(&short_close);
        assert_eq!(result.len(), short_close.len());
        for val in &result {
            assert_eq!(*val, 0.0);
        }

        let tmc = TrendMomentumConvergence::new(10, 5).unwrap();
        let result = tmc.calculate(&short_close);
        assert_eq!(result.len(), short_close.len());

        let ats = AdaptiveTrendStrength::new(10, 5, 1.0).unwrap();
        let result = ats.calculate(&short_high, &short_low, &short_close);
        assert_eq!(result.len(), short_close.len());

        let tdi = TrendDirectionIndex::new(10, 3).unwrap();
        let result = tdi.calculate(&short_high, &short_low, &short_close);
        assert_eq!(result.len(), short_close.len());

        let tm = TrendMaturity::new(5, 10).unwrap();
        let result = tm.calculate(&short_close);
        assert_eq!(result.len(), short_close.len());

        let mpta = MultiPeriodTrendAlignment::new(2, 3, 4).unwrap();
        let result = mpta.calculate(&short_close);
        assert_eq!(result.len(), short_close.len());
    }

    #[test]
    fn test_new_six_indicators_downtrend() {
        // Test with downtrending data
        let close: Vec<f64> = (0..40).map(|i| 140.0 - i as f64).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 2.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 2.0).collect();

        let tc = TrendContinuity::new(10, 3).unwrap();
        let result = tc.calculate(&close);
        // Downtrend should still show continuity (direction doesn't matter for continuity)
        assert!(result[30] > 0.0);

        let tmc = TrendMomentumConvergence::new(10, 5).unwrap();
        let result = tmc.calculate(&close);
        // Downtrend should produce negative convergence
        assert!(result[30] < 0.0);

        let ats = AdaptiveTrendStrength::new(10, 5, 1.0).unwrap();
        let result = ats.calculate(&high, &low, &close);
        // Strong downtrend should show good strength
        assert!(result[30] > 0.0);

        let tdi = TrendDirectionIndex::new(10, 3).unwrap();
        let result = tdi.calculate(&high, &low, &close);
        // Downtrend should show negative direction
        assert!(result[30] < 0.0);

        let tm = TrendMaturity::new(5, 20).unwrap();
        let result = tm.calculate(&close);
        // Maturity is measured regardless of direction
        assert!(result[30] >= 0.0);

        let mpta = MultiPeriodTrendAlignment::new(5, 10, 20).unwrap();
        let result = mpta.calculate(&close);
        // Downtrend alignment should be negative
        assert!(result[30] < 0.0);
    }

    #[test]
    fn test_new_six_indicators_choppy_market() {
        // Test with choppy/ranging data
        let close: Vec<f64> = (0..40)
            .map(|i| 100.0 + if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();

        let tc = TrendContinuity::new(10, 3).unwrap();
        let result = tc.calculate(&close);
        // Choppy market should show lower continuity
        assert!(result[30] < 50.0);

        let tmc = TrendMomentumConvergence::new(10, 5).unwrap();
        let result = tmc.calculate(&close);
        // Choppy market should show weak convergence (close to 0)
        assert!(result[30].abs() < 50.0);

        let ats = AdaptiveTrendStrength::new(10, 5, 1.0).unwrap();
        let result = ats.calculate(&high, &low, &close);
        // Choppy market should show low strength
        assert!(result[30] < 50.0);

        let mpta = MultiPeriodTrendAlignment::new(5, 10, 20).unwrap();
        let result = mpta.calculate(&close);
        // Choppy market should show weak alignment (close to 0)
        assert!(result[30].abs() < 50.0);
    }
}
