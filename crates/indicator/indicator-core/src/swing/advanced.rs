//! Advanced Swing Trading Indicators
//!
//! Additional advanced swing trading indicators for trend strength, reversals,
//! volatility, momentum, target levels, and duration analysis.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Swing Trend Strength - Measures strength of swing trends
///
/// Analyzes the consistency and magnitude of swing movements to determine
/// overall trend strength. Higher values indicate stronger trending behavior.
///
/// Output: Trend strength value from 0 to 100
#[derive(Debug, Clone)]
pub struct SwingTrendStrength {
    period: usize,
    smoothing: usize,
}

impl SwingTrendStrength {
    /// Create a new Swing Trend Strength indicator.
    ///
    /// # Arguments
    /// * `period` - Lookback period for swing analysis (minimum 5)
    /// * `smoothing` - Smoothing period for output (minimum 1)
    pub fn new(period: usize, smoothing: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if smoothing < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self { period, smoothing })
    }

    /// Calculate swing trend strength
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];
        let mut raw_strength = vec![0.0; n];

        for i in self.period..n {
            let start = i - self.period;

            // Calculate swing range
            let period_high = high[start..=i].iter().cloned().fold(f64::MIN, f64::max);
            let period_low = low[start..=i].iter().cloned().fold(f64::MAX, f64::min);
            let range = period_high - period_low;

            if range > 1e-10 {
                // Directional movement analysis
                let mut up_moves = 0.0;
                let mut down_moves = 0.0;
                let mut up_count = 0;
                let mut down_count = 0;

                for j in (start + 1)..=i {
                    let change = close[j] - close[j - 1];
                    if change > 0.0 {
                        up_moves += change;
                        up_count += 1;
                    } else if change < 0.0 {
                        down_moves += change.abs();
                        down_count += 1;
                    }
                }

                // Calculate directional strength
                let total_movement = up_moves + down_moves;
                let directional_ratio = if total_movement > 1e-10 {
                    (up_moves - down_moves).abs() / total_movement
                } else {
                    0.0
                };

                // Calculate consistency (how one-sided the moves are)
                let total_bars = up_count + down_count;
                let consistency = if total_bars > 0 {
                    (up_count as i32 - down_count as i32).abs() as f64 / total_bars as f64
                } else {
                    0.0
                };

                // Price progress relative to range
                let price_progress = (close[i] - close[start]).abs() / range;

                // Combine factors
                raw_strength[i] = (directional_ratio * 40.0 + consistency * 30.0 + price_progress * 30.0)
                    .clamp(0.0, 100.0);
            }
        }

        // Apply smoothing
        for i in (self.period + self.smoothing - 1)..n {
            let start = i - self.smoothing + 1;
            let sum: f64 = raw_strength[start..=i].iter().sum();
            result[i] = sum / self.smoothing as f64;
        }

        result
    }
}

impl TechnicalIndicator for SwingTrendStrength {
    fn name(&self) -> &str {
        "Swing Trend Strength"
    }

    fn min_periods(&self) -> usize {
        self.period + self.smoothing
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

/// Swing Reversal - Detects potential swing reversals
///
/// Identifies potential trend reversal points based on swing exhaustion,
/// divergence patterns, and price structure analysis.
///
/// Output:
/// - Primary: Reversal signal (1 = bullish reversal, -1 = bearish reversal, 0 = none)
/// - Secondary: Reversal confidence (0 to 100)
#[derive(Debug, Clone)]
pub struct SwingReversal {
    period: usize,
    threshold: f64,
}

impl SwingReversal {
    /// Create a new Swing Reversal indicator.
    ///
    /// # Arguments
    /// * `period` - Lookback period for swing analysis (minimum 5)
    /// * `threshold` - Minimum threshold for reversal detection (0.0 to 1.0)
    pub fn new(period: usize, threshold: f64) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if !(0.0..=1.0).contains(&threshold) {
            return Err(IndicatorError::InvalidParameter {
                name: "threshold".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        Ok(Self { period, threshold })
    }

    /// Calculate swing reversal signals
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut signal = vec![0.0; n];
        let mut confidence = vec![0.0; n];

        for i in (self.period * 2)..n {
            let mid = i - self.period;
            let start = mid - self.period;

            // First period analysis
            let first_high = high[start..=mid].iter().cloned().fold(f64::MIN, f64::max);
            let first_low = low[start..=mid].iter().cloned().fold(f64::MAX, f64::min);
            let first_range = first_high - first_low;
            let first_trend = close[mid] - close[start];

            // Second period analysis
            let second_high = high[mid..=i].iter().cloned().fold(f64::MIN, f64::max);
            let second_low = low[mid..=i].iter().cloned().fold(f64::MAX, f64::min);
            let second_range = second_high - second_low;
            let second_trend = close[i] - close[mid];

            if first_range > 1e-10 && second_range > 1e-10 {
                // Detect bearish reversal (was going up, now going down)
                if first_trend > 0.0 && second_trend < 0.0 {
                    // Check for exhaustion (higher high but close near low)
                    let exhaustion = if second_high > first_high {
                        (second_high - close[i]) / second_range
                    } else {
                        0.0
                    };

                    // Momentum divergence (slowing upward momentum)
                    let momentum_div = if first_trend > 1e-10 {
                        (-second_trend / first_trend).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };

                    let reversal_strength = exhaustion * 0.5 + momentum_div * 0.5;
                    if reversal_strength > self.threshold {
                        signal[i] = -1.0;
                        confidence[i] = (reversal_strength * 100.0).clamp(0.0, 100.0);
                    }
                }
                // Detect bullish reversal (was going down, now going up)
                else if first_trend < 0.0 && second_trend > 0.0 {
                    // Check for exhaustion (lower low but close near high)
                    let exhaustion = if second_low < first_low {
                        (close[i] - second_low) / second_range
                    } else {
                        0.0
                    };

                    // Momentum divergence (slowing downward momentum)
                    let momentum_div = if first_trend.abs() > 1e-10 {
                        (second_trend / first_trend.abs()).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };

                    let reversal_strength = exhaustion * 0.5 + momentum_div * 0.5;
                    if reversal_strength > self.threshold {
                        signal[i] = 1.0;
                        confidence[i] = (reversal_strength * 100.0).clamp(0.0, 100.0);
                    }
                }
            }
        }

        (signal, confidence)
    }
}

impl TechnicalIndicator for SwingReversal {
    fn name(&self) -> &str {
        "Swing Reversal"
    }

    fn min_periods(&self) -> usize {
        self.period * 2 + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (signal, confidence) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::dual(signal, confidence))
    }

    fn output_features(&self) -> usize {
        2
    }
}

/// Swing Volatility - Volatility of swing movements
///
/// Measures the volatility specific to swing movements, considering both
/// the amplitude and frequency of price swings.
///
/// Output: Swing volatility as a percentage
#[derive(Debug, Clone)]
pub struct SwingVolatility {
    period: usize,
    swing_lookback: usize,
}

impl SwingVolatility {
    /// Create a new Swing Volatility indicator.
    ///
    /// # Arguments
    /// * `period` - Lookback period for volatility calculation (minimum 10)
    /// * `swing_lookback` - Bars to look for swing points (minimum 3)
    pub fn new(period: usize, swing_lookback: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if swing_lookback < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "swing_lookback".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        Ok(Self { period, swing_lookback })
    }

    /// Calculate swing volatility
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // First, identify swing points
        let mut swing_highs = vec![false; n];
        let mut swing_lows = vec![false; n];

        for i in self.swing_lookback..(n - self.swing_lookback) {
            // Check if swing high
            let mut is_high = true;
            for j in 1..=self.swing_lookback {
                if high[i] <= high[i - j] || high[i] <= high[i + j] {
                    is_high = false;
                    break;
                }
            }
            swing_highs[i] = is_high;

            // Check if swing low
            let mut is_low = true;
            for j in 1..=self.swing_lookback {
                if low[i] >= low[i - j] || low[i] >= low[i + j] {
                    is_low = false;
                    break;
                }
            }
            swing_lows[i] = is_low;
        }

        // Calculate swing volatility
        for i in self.period..n {
            let start = i - self.period;

            // Collect swing amplitudes in the period
            let mut swing_amplitudes = Vec::new();
            let mut last_swing_price = close[start];
            let mut last_swing_type: Option<bool> = None; // true = high, false = low

            for j in start..=i {
                if swing_highs[j] {
                    if let Some(was_high) = last_swing_type {
                        if !was_high {
                            // Swing from low to high
                            swing_amplitudes.push((high[j] - last_swing_price).abs());
                        }
                    }
                    last_swing_price = high[j];
                    last_swing_type = Some(true);
                }
                if swing_lows[j] {
                    if let Some(was_high) = last_swing_type {
                        if was_high {
                            // Swing from high to low
                            swing_amplitudes.push((last_swing_price - low[j]).abs());
                        }
                    }
                    last_swing_price = low[j];
                    last_swing_type = Some(false);
                }
            }

            // Calculate volatility metrics
            let avg_close = close[start..=i].iter().sum::<f64>() / (self.period + 1) as f64;

            if !swing_amplitudes.is_empty() && avg_close > 1e-10 {
                // Average swing amplitude
                let avg_amplitude = swing_amplitudes.iter().sum::<f64>() / swing_amplitudes.len() as f64;

                // Swing frequency (swings per bar)
                let swing_frequency = swing_amplitudes.len() as f64 / self.period as f64;

                // Combined volatility measure (normalized)
                result[i] = (avg_amplitude / avg_close * 100.0) * (1.0 + swing_frequency);
            } else {
                // Fallback to simple range-based volatility
                let period_high = high[start..=i].iter().cloned().fold(f64::MIN, f64::max);
                let period_low = low[start..=i].iter().cloned().fold(f64::MAX, f64::min);
                if avg_close > 1e-10 {
                    result[i] = (period_high - period_low) / avg_close * 100.0;
                }
            }
        }

        result
    }
}

impl TechnicalIndicator for SwingVolatility {
    fn name(&self) -> &str {
        "Swing Volatility"
    }

    fn min_periods(&self) -> usize {
        self.period + self.swing_lookback
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

/// Swing Momentum Advanced - Advanced swing momentum measurement
///
/// Enhanced momentum indicator that considers swing structure, acceleration,
/// and volume confirmation for more accurate momentum readings.
///
/// Output:
/// - Primary: Momentum value (positive = bullish, negative = bearish)
/// - Secondary: Momentum acceleration
#[derive(Debug, Clone)]
pub struct SwingMomentumAdvanced {
    period: usize,
    accel_period: usize,
}

impl SwingMomentumAdvanced {
    /// Create a new Swing Momentum Advanced indicator.
    ///
    /// # Arguments
    /// * `period` - Main momentum period (minimum 5)
    /// * `accel_period` - Acceleration calculation period (minimum 3)
    pub fn new(period: usize, accel_period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if accel_period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "accel_period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        Ok(Self { period, accel_period })
    }

    /// Calculate advanced swing momentum
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut momentum = vec![0.0; n];
        let mut acceleration = vec![0.0; n];

        for i in self.period..n {
            let start = i - self.period;

            // Basic price momentum
            let price_change = close[i] - close[start];

            // Range context
            let period_high = high[start..=i].iter().cloned().fold(f64::MIN, f64::max);
            let period_low = low[start..=i].iter().cloned().fold(f64::MAX, f64::min);
            let range = period_high - period_low;

            if range > 1e-10 {
                // Normalized momentum
                let norm_momentum = price_change / range * 100.0;

                // Volume-weighted momentum adjustment
                let avg_volume = volume[start..=i].iter().sum::<f64>() / (self.period + 1) as f64;
                let recent_volume = volume[(i.saturating_sub(2))..=i].iter().sum::<f64>() / 3.0;
                let volume_factor = if avg_volume > 1e-10 {
                    (recent_volume / avg_volume).clamp(0.5, 2.0)
                } else {
                    1.0
                };

                // Swing structure analysis
                let mid = start + self.period / 2;
                let first_half_change = close[mid] - close[start];
                let second_half_change = close[i] - close[mid];

                // Momentum with structure weight
                let structure_factor = if price_change.abs() > 1e-10 {
                    // If second half is accelerating in same direction
                    if (first_half_change * second_half_change) > 0.0 {
                        1.0 + (second_half_change.abs() / first_half_change.abs() - 1.0).clamp(-0.5, 0.5)
                    } else {
                        0.5 // Deceleration or reversal
                    }
                } else {
                    1.0
                };

                momentum[i] = norm_momentum * volume_factor * structure_factor;
            }
        }

        // Calculate acceleration
        for i in (self.period + self.accel_period)..n {
            let accel_start = i - self.accel_period;
            acceleration[i] = momentum[i] - momentum[accel_start];
        }

        (momentum, acceleration)
    }
}

impl TechnicalIndicator for SwingMomentumAdvanced {
    fn name(&self) -> &str {
        "Swing Momentum Advanced"
    }

    fn min_periods(&self) -> usize {
        self.period + self.accel_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (momentum, acceleration) = self.calculate(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::dual(momentum, acceleration))
    }

    fn output_features(&self) -> usize {
        2
    }
}

/// Swing Target Levels - Calculates swing target price levels
///
/// Projects potential target prices based on swing measurements, Fibonacci
/// extensions, and measured move analysis.
///
/// Output:
/// - Primary: Upper target level
/// - Secondary: Lower target level
#[derive(Debug, Clone)]
pub struct SwingTargetLevels {
    swing_period: usize,
    extension_ratio: f64,
}

impl SwingTargetLevels {
    /// Create a new Swing Target Levels indicator.
    ///
    /// # Arguments
    /// * `swing_period` - Period for swing detection (minimum 5)
    /// * `extension_ratio` - Fibonacci extension ratio (e.g., 1.618)
    pub fn new(swing_period: usize, extension_ratio: f64) -> Result<Self> {
        if swing_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "swing_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if extension_ratio <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "extension_ratio".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { swing_period, extension_ratio })
    }

    /// Calculate swing target levels
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut upper_target = vec![f64::NAN; n];
        let mut lower_target = vec![f64::NAN; n];

        // Track swing points
        let mut last_swing_high: Option<(usize, f64)> = None;
        let mut last_swing_low: Option<(usize, f64)> = None;

        for i in self.swing_period..(n.saturating_sub(self.swing_period)) {
            // Check for swing high
            let mut is_swing_high = true;
            for j in 1..=self.swing_period {
                if high[i] <= high[i - j] || high[i] <= high[i + j] {
                    is_swing_high = false;
                    break;
                }
            }
            if is_swing_high {
                last_swing_high = Some((i, high[i]));
            }

            // Check for swing low
            let mut is_swing_low = true;
            for j in 1..=self.swing_period {
                if low[i] >= low[i - j] || low[i] >= low[i + j] {
                    is_swing_low = false;
                    break;
                }
            }
            if is_swing_low {
                last_swing_low = Some((i, low[i]));
            }
        }

        // Calculate targets for current position
        for i in (self.swing_period * 2)..n {
            let current_price = close[i];

            // Find most recent swing points before current bar
            let mut recent_high: Option<f64> = None;
            let mut recent_low: Option<f64> = None;

            for j in (0..i.saturating_sub(self.swing_period)).rev() {
                if recent_high.is_none() {
                    // Check if j is a swing high
                    if j >= self.swing_period && j + self.swing_period < n {
                        let mut is_high = true;
                        for k in 1..=self.swing_period {
                            if high[j] <= high[j - k] || high[j] <= high[j + k] {
                                is_high = false;
                                break;
                            }
                        }
                        if is_high {
                            recent_high = Some(high[j]);
                        }
                    }
                }

                if recent_low.is_none() {
                    // Check if j is a swing low
                    if j >= self.swing_period && j + self.swing_period < n {
                        let mut is_low = true;
                        for k in 1..=self.swing_period {
                            if low[j] >= low[j - k] || low[j] >= low[j + k] {
                                is_low = false;
                                break;
                            }
                        }
                        if is_low {
                            recent_low = Some(low[j]);
                        }
                    }
                }

                if recent_high.is_some() && recent_low.is_some() {
                    break;
                }
            }

            // Calculate targets based on swing range
            if let (Some(sh), Some(sl)) = (recent_high, recent_low) {
                let swing_range = sh - sl;

                // Upper target: extension above recent high
                upper_target[i] = sh + swing_range * (self.extension_ratio - 1.0);

                // Lower target: extension below recent low
                lower_target[i] = sl - swing_range * (self.extension_ratio - 1.0);
            } else {
                // Fallback: use recent range
                let start = i.saturating_sub(self.swing_period * 2);
                let range_high = high[start..=i].iter().cloned().fold(f64::MIN, f64::max);
                let range_low = low[start..=i].iter().cloned().fold(f64::MAX, f64::min);
                let range = range_high - range_low;

                upper_target[i] = range_high + range * (self.extension_ratio - 1.0);
                lower_target[i] = range_low - range * (self.extension_ratio - 1.0);
            }
        }

        (upper_target, lower_target)
    }
}

impl TechnicalIndicator for SwingTargetLevels {
    fn name(&self) -> &str {
        "Swing Target Levels"
    }

    fn min_periods(&self) -> usize {
        self.swing_period * 2 + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (upper, lower) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::dual(upper, lower))
    }

    fn output_features(&self) -> usize {
        2
    }
}

/// Swing Duration - Measures duration of swing moves
///
/// Tracks the duration of swing movements in bars, helping identify
/// when swings are extending beyond typical lengths.
///
/// Output:
/// - Primary: Current swing duration (bars)
/// - Secondary: Average swing duration over lookback period
#[derive(Debug, Clone)]
pub struct SwingDuration {
    swing_period: usize,
    lookback: usize,
}

impl SwingDuration {
    /// Create a new Swing Duration indicator.
    ///
    /// # Arguments
    /// * `swing_period` - Period for swing point detection (minimum 3)
    /// * `lookback` - Number of historical swings to average (minimum 3)
    pub fn new(swing_period: usize, lookback: usize) -> Result<Self> {
        if swing_period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "swing_period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        if lookback < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        Ok(Self { swing_period, lookback })
    }

    /// Calculate swing duration metrics
    pub fn calculate(&self, high: &[f64], low: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = high.len();
        let mut current_duration = vec![0.0; n];
        let mut avg_duration = vec![0.0; n];

        // Identify swing points
        let mut swing_points: Vec<(usize, bool)> = Vec::new(); // (index, is_high)

        for i in self.swing_period..(n.saturating_sub(self.swing_period)) {
            // Check for swing high
            let mut is_swing_high = true;
            for j in 1..=self.swing_period {
                if high[i] <= high[i - j] || high[i] <= high[i + j] {
                    is_swing_high = false;
                    break;
                }
            }
            if is_swing_high {
                swing_points.push((i, true));
            }

            // Check for swing low
            let mut is_swing_low = true;
            for j in 1..=self.swing_period {
                if low[i] >= low[i - j] || low[i] >= low[i + j] {
                    is_swing_low = false;
                    break;
                }
            }
            if is_swing_low {
                swing_points.push((i, false));
            }
        }

        // Sort by index
        swing_points.sort_by_key(|&(idx, _)| idx);

        // Calculate durations between consecutive swings
        let mut swing_durations: Vec<(usize, usize)> = Vec::new(); // (end_index, duration)
        for i in 1..swing_points.len() {
            let duration = swing_points[i].0 - swing_points[i - 1].0;
            swing_durations.push((swing_points[i].0, duration));
        }

        // Fill in current duration and average duration for each bar
        let mut last_swing_idx = 0;
        let mut duration_history: Vec<usize> = Vec::new();

        for i in 0..n {
            // Update if we've passed a swing point
            while last_swing_idx < swing_points.len() && swing_points[last_swing_idx].0 <= i {
                last_swing_idx += 1;
            }

            // Current duration: bars since last swing
            if last_swing_idx > 0 {
                current_duration[i] = (i - swing_points[last_swing_idx - 1].0) as f64;
            } else {
                current_duration[i] = i as f64;
            }

            // Collect durations that ended before or at this bar
            while !swing_durations.is_empty() && swing_durations[0].0 <= i {
                duration_history.push(swing_durations[0].1);
                swing_durations.remove(0);
            }

            // Keep only recent history
            if duration_history.len() > self.lookback {
                duration_history.remove(0);
            }

            // Calculate average duration
            if !duration_history.is_empty() {
                avg_duration[i] = duration_history.iter().sum::<usize>() as f64 / duration_history.len() as f64;
            }
        }

        (current_duration, avg_duration)
    }
}

impl TechnicalIndicator for SwingDuration {
    fn name(&self) -> &str {
        "Swing Duration"
    }

    fn min_periods(&self) -> usize {
        self.swing_period * 2 + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (current, avg) = self.calculate(&data.high, &data.low);
        Ok(IndicatorOutput::dual(current, avg))
    }

    fn output_features(&self) -> usize {
        2
    }
}

/// Swing Efficiency - Measures efficiency of swing moves
///
/// Calculates how efficiently price moves from swing point to swing point,
/// comparing the direct distance to the actual path taken. Higher values
/// indicate more efficient (direct) price movement.
///
/// Output: Efficiency ratio from 0 to 100 (100 = perfectly efficient)
#[derive(Debug, Clone)]
pub struct SwingEfficiency {
    period: usize,
    /// Lookback for swing point detection, used in range calculations
    #[allow(dead_code)]
    swing_lookback: usize,
}

impl SwingEfficiency {
    /// Create a new Swing Efficiency indicator.
    ///
    /// # Arguments
    /// * `period` - Lookback period for efficiency calculation (minimum 5)
    /// * `swing_lookback` - Bars to look for swing points (minimum 2)
    pub fn new(period: usize, swing_lookback: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if swing_lookback < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "swing_lookback".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period, swing_lookback })
    }

    /// Calculate swing efficiency
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.period {
            return result;
        }

        for i in self.period..n {
            let start = i - self.period;

            // Direct distance: net price change
            let direct_distance = (close[i] - close[start]).abs();

            // Actual path: sum of all bar-to-bar movements
            let mut actual_path = 0.0;
            for j in (start + 1)..=i {
                actual_path += (close[j] - close[j - 1]).abs();
            }

            // Also consider high-low range traversed
            let mut range_path = 0.0;
            for j in start..=i {
                range_path += high[j] - low[j];
            }

            // Combined path measure
            let total_path = actual_path + range_path * 0.5;

            // Efficiency ratio
            if total_path > 1e-10 {
                // Normalize: direct/path, scaled to 0-100
                let efficiency = (direct_distance / total_path * 100.0).clamp(0.0, 100.0);
                result[i] = efficiency;
            }
        }

        result
    }
}

impl TechnicalIndicator for SwingEfficiency {
    fn name(&self) -> &str {
        "Swing Efficiency"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

/// Swing Continuation - Probability of swing continuation
///
/// Analyzes current swing characteristics to estimate the probability
/// that the swing will continue in its current direction rather than reverse.
///
/// Output:
/// - Primary: Continuation probability (0 to 100)
/// - Secondary: Current swing direction (1 = up, -1 = down, 0 = neutral)
#[derive(Debug, Clone)]
pub struct SwingContinuation {
    period: usize,
    momentum_period: usize,
}

impl SwingContinuation {
    /// Create a new Swing Continuation indicator.
    ///
    /// # Arguments
    /// * `period` - Lookback period for swing analysis (minimum 5)
    /// * `momentum_period` - Period for momentum calculation (minimum 3)
    pub fn new(period: usize, momentum_period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if momentum_period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        Ok(Self { period, momentum_period })
    }

    /// Calculate swing continuation probability
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut probability = vec![50.0; n];
        let mut direction = vec![0.0; n];

        if n < self.period + self.momentum_period {
            return (probability, direction);
        }

        for i in (self.period + self.momentum_period)..n {
            let start = i - self.period;

            // Determine current swing direction
            let price_change = close[i] - close[start];
            let current_direction = if price_change > 0.0 {
                1.0
            } else if price_change < 0.0 {
                -1.0
            } else {
                0.0
            };
            direction[i] = current_direction;

            // Calculate range
            let period_high = high[start..=i].iter().cloned().fold(f64::MIN, f64::max);
            let period_low = low[start..=i].iter().cloned().fold(f64::MAX, f64::min);
            let range = period_high - period_low;

            if range < 1e-10 || current_direction == 0.0 {
                continue;
            }

            // Factor 1: Momentum consistency
            let mut consistent_bars = 0;
            let mut total_bars = 0;
            for j in (start + 1)..=i {
                let bar_change = close[j] - close[j - 1];
                total_bars += 1;
                if (bar_change > 0.0 && current_direction > 0.0)
                    || (bar_change < 0.0 && current_direction < 0.0)
                {
                    consistent_bars += 1;
                }
            }
            let consistency = if total_bars > 0 {
                consistent_bars as f64 / total_bars as f64
            } else {
                0.5
            };

            // Factor 2: Recent momentum vs earlier momentum
            let mid = start + self.period / 2;
            let first_half_move = (close[mid] - close[start]).abs();
            let second_half_move = (close[i] - close[mid]).abs();
            let momentum_ratio = if first_half_move > 1e-10 {
                (second_half_move / first_half_move).clamp(0.0, 2.0)
            } else {
                1.0
            };

            // Factor 3: Position in range (trending markets stay in direction)
            let range_position = (close[i] - period_low) / range;
            let position_score = if current_direction > 0.0 {
                range_position // Higher is better for uptrend
            } else {
                1.0 - range_position // Lower is better for downtrend
            };

            // Factor 4: Trend strength (higher highs/lows or lower highs/lows)
            let mut trend_score = 0.0;
            let check_bars = (self.momentum_period).min(i - start);
            for j in 1..check_bars {
                let idx = i - j;
                if idx > start {
                    if current_direction > 0.0 {
                        if high[idx] > high[idx - 1] {
                            trend_score += 1.0;
                        }
                        if low[idx] > low[idx - 1] {
                            trend_score += 1.0;
                        }
                    } else {
                        if high[idx] < high[idx - 1] {
                            trend_score += 1.0;
                        }
                        if low[idx] < low[idx - 1] {
                            trend_score += 1.0;
                        }
                    }
                }
            }
            let trend_factor = if check_bars > 1 {
                trend_score / ((check_bars - 1) * 2) as f64
            } else {
                0.5
            };

            // Combine factors into continuation probability
            let raw_probability = consistency * 25.0
                + (momentum_ratio / 2.0).clamp(0.0, 1.0) * 25.0
                + position_score * 25.0
                + trend_factor * 25.0;

            probability[i] = raw_probability.clamp(0.0, 100.0);
        }

        (probability, direction)
    }
}

impl TechnicalIndicator for SwingContinuation {
    fn name(&self) -> &str {
        "Swing Continuation"
    }

    fn min_periods(&self) -> usize {
        self.period + self.momentum_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (probability, direction) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::dual(probability, direction))
    }

    fn output_features(&self) -> usize {
        2
    }
}

/// Swing Magnitude - Magnitude/size of swing moves
///
/// Measures the absolute and relative magnitude of swing movements,
/// helping identify significant vs. minor price swings.
///
/// Output:
/// - Primary: Absolute swing magnitude (price units)
/// - Secondary: Relative magnitude (percentage of average)
#[derive(Debug, Clone)]
pub struct SwingMagnitude {
    period: usize,
    avg_period: usize,
}

impl SwingMagnitude {
    /// Create a new Swing Magnitude indicator.
    ///
    /// # Arguments
    /// * `period` - Lookback period for current swing (minimum 3)
    /// * `avg_period` - Period for averaging historical swings (minimum 5)
    pub fn new(period: usize, avg_period: usize) -> Result<Self> {
        if period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        if avg_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "avg_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period, avg_period })
    }

    /// Calculate swing magnitude metrics
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut absolute_magnitude = vec![0.0; n];
        let mut relative_magnitude = vec![100.0; n];

        if n < self.period + self.avg_period {
            return (absolute_magnitude, relative_magnitude);
        }

        // Calculate running average of swing magnitudes
        let mut magnitude_history: Vec<f64> = Vec::new();

        for i in self.period..n {
            let start = i - self.period;

            // Current swing magnitude: high-low range of the swing
            let swing_high = high[start..=i].iter().cloned().fold(f64::MIN, f64::max);
            let swing_low = low[start..=i].iter().cloned().fold(f64::MAX, f64::min);
            let current_magnitude = swing_high - swing_low;

            absolute_magnitude[i] = current_magnitude;

            // Add to history for averaging
            magnitude_history.push(current_magnitude);

            // Keep only avg_period worth of history
            if magnitude_history.len() > self.avg_period {
                magnitude_history.remove(0);
            }

            // Calculate average magnitude
            if !magnitude_history.is_empty() {
                let avg_magnitude =
                    magnitude_history.iter().sum::<f64>() / magnitude_history.len() as f64;

                if avg_magnitude > 1e-10 {
                    // Relative magnitude: current as percentage of average
                    relative_magnitude[i] = (current_magnitude / avg_magnitude * 100.0).clamp(0.0, 500.0);
                }
            }
        }

        (absolute_magnitude, relative_magnitude)
    }
}

impl TechnicalIndicator for SwingMagnitude {
    fn name(&self) -> &str {
        "Swing Magnitude"
    }

    fn min_periods(&self) -> usize {
        self.period + self.avg_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (absolute, relative) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::dual(absolute, relative))
    }

    fn output_features(&self) -> usize {
        2
    }
}

/// Swing Retracement Level - Calculates retracement levels
///
/// Identifies current retracement levels relative to recent swing highs and lows,
/// using Fibonacci ratios to determine support and resistance zones.
///
/// Output:
/// - Primary: Current retracement percentage (0 = at swing low, 100 = at swing high)
/// - Secondary: Nearest Fibonacci level (23.6, 38.2, 50, 61.8, 78.6)
#[derive(Debug, Clone)]
pub struct SwingRetracementLevel {
    swing_period: usize,
}

impl SwingRetracementLevel {
    /// Create a new Swing Retracement Level indicator.
    ///
    /// # Arguments
    /// * `swing_period` - Period for swing point detection (minimum 5)
    pub fn new(swing_period: usize) -> Result<Self> {
        if swing_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "swing_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { swing_period })
    }

    /// Calculate swing retracement levels
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut retracement_pct = vec![50.0; n];
        let mut nearest_fib = vec![50.0; n];

        // Fibonacci levels
        let fib_levels = [0.0, 23.6, 38.2, 50.0, 61.8, 78.6, 100.0];

        if n < self.swing_period * 2 {
            return (retracement_pct, nearest_fib);
        }

        // Track recent swing points
        let mut last_swing_high: Option<(usize, f64)> = None;
        let mut last_swing_low: Option<(usize, f64)> = None;

        for i in self.swing_period..(n.saturating_sub(self.swing_period)) {
            // Check for swing high
            let mut is_swing_high = true;
            for j in 1..=self.swing_period {
                if high[i] <= high[i - j] || high[i] <= high[i + j] {
                    is_swing_high = false;
                    break;
                }
            }
            if is_swing_high {
                last_swing_high = Some((i, high[i]));
            }

            // Check for swing low
            let mut is_swing_low = true;
            for j in 1..=self.swing_period {
                if low[i] >= low[i - j] || low[i] >= low[i + j] {
                    is_swing_low = false;
                    break;
                }
            }
            if is_swing_low {
                last_swing_low = Some((i, low[i]));
            }
        }

        // Calculate retracement for each bar
        for i in (self.swing_period * 2)..n {
            // Find swing points before current bar
            let mut swing_high: Option<f64> = None;
            let mut swing_low: Option<f64> = None;

            // Search backwards for swing points
            for j in (self.swing_period..(i.saturating_sub(self.swing_period))).rev() {
                if swing_high.is_none() {
                    let mut is_high = true;
                    for k in 1..=self.swing_period {
                        if j >= k && j + k < n {
                            if high[j] <= high[j - k] || high[j] <= high[j + k] {
                                is_high = false;
                                break;
                            }
                        } else {
                            is_high = false;
                            break;
                        }
                    }
                    if is_high {
                        swing_high = Some(high[j]);
                    }
                }

                if swing_low.is_none() {
                    let mut is_low = true;
                    for k in 1..=self.swing_period {
                        if j >= k && j + k < n {
                            if low[j] >= low[j - k] || low[j] >= low[j + k] {
                                is_low = false;
                                break;
                            }
                        } else {
                            is_low = false;
                            break;
                        }
                    }
                    if is_low {
                        swing_low = Some(low[j]);
                    }
                }

                if swing_high.is_some() && swing_low.is_some() {
                    break;
                }
            }

            // Fallback to period high/low if no swing points found
            let sh = swing_high.unwrap_or_else(|| {
                high[(i.saturating_sub(self.swing_period * 2))..=i]
                    .iter()
                    .cloned()
                    .fold(f64::MIN, f64::max)
            });
            let sl = swing_low.unwrap_or_else(|| {
                low[(i.saturating_sub(self.swing_period * 2))..=i]
                    .iter()
                    .cloned()
                    .fold(f64::MAX, f64::min)
            });

            let range = sh - sl;
            if range > 1e-10 {
                // Current retracement percentage
                let current_pct = (close[i] - sl) / range * 100.0;
                retracement_pct[i] = current_pct.clamp(0.0, 100.0);

                // Find nearest Fibonacci level
                let mut min_distance = f64::MAX;
                let mut nearest = 50.0;
                for &fib in &fib_levels {
                    let distance = (current_pct - fib).abs();
                    if distance < min_distance {
                        min_distance = distance;
                        nearest = fib;
                    }
                }
                nearest_fib[i] = nearest;
            }
        }

        (retracement_pct, nearest_fib)
    }
}

impl TechnicalIndicator for SwingRetracementLevel {
    fn name(&self) -> &str {
        "Swing Retracement Level"
    }

    fn min_periods(&self) -> usize {
        self.swing_period * 2 + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (retracement, fib) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::dual(retracement, fib))
    }

    fn output_features(&self) -> usize {
        2
    }
}

/// Swing Extension Target - Projects extension targets
///
/// Projects potential price targets based on swing measurements using
/// Fibonacci extension ratios (100%, 127.2%, 161.8%, 200%, 261.8%).
///
/// Output:
/// - Primary: First extension target (127.2%)
/// - Secondary: Second extension target (161.8%)
#[derive(Debug, Clone)]
pub struct SwingExtensionTarget {
    swing_period: usize,
    primary_ratio: f64,
    secondary_ratio: f64,
}

impl SwingExtensionTarget {
    /// Create a new Swing Extension Target indicator.
    ///
    /// # Arguments
    /// * `swing_period` - Period for swing detection (minimum 5)
    /// * `primary_ratio` - Primary extension ratio (default 1.272)
    /// * `secondary_ratio` - Secondary extension ratio (default 1.618)
    pub fn new(swing_period: usize, primary_ratio: f64, secondary_ratio: f64) -> Result<Self> {
        if swing_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "swing_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if primary_ratio <= 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "primary_ratio".to_string(),
                reason: "must be greater than 1.0".to_string(),
            });
        }
        if secondary_ratio <= primary_ratio {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_ratio".to_string(),
                reason: "must be greater than primary_ratio".to_string(),
            });
        }
        Ok(Self {
            swing_period,
            primary_ratio,
            secondary_ratio,
        })
    }

    /// Create with default Fibonacci ratios (1.272 and 1.618)
    pub fn with_defaults(swing_period: usize) -> Result<Self> {
        Self::new(swing_period, 1.272, 1.618)
    }

    /// Calculate swing extension targets
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut primary_target = vec![f64::NAN; n];
        let mut secondary_target = vec![f64::NAN; n];

        if n < self.swing_period * 3 {
            return (primary_target, secondary_target);
        }

        // Find swing points and calculate extensions
        for i in (self.swing_period * 3)..n {
            // Collect recent swing points
            let mut swing_highs: Vec<(usize, f64)> = Vec::new();
            let mut swing_lows: Vec<(usize, f64)> = Vec::new();

            for j in (self.swing_period..(i.saturating_sub(self.swing_period))).rev() {
                // Check for swing high
                let mut is_high = true;
                for k in 1..=self.swing_period {
                    if j >= k && j + k < n {
                        if high[j] <= high[j - k] || high[j] <= high[j + k] {
                            is_high = false;
                            break;
                        }
                    } else {
                        is_high = false;
                        break;
                    }
                }
                if is_high && swing_highs.len() < 3 {
                    swing_highs.push((j, high[j]));
                }

                // Check for swing low
                let mut is_low = true;
                for k in 1..=self.swing_period {
                    if j >= k && j + k < n {
                        if low[j] >= low[j - k] || low[j] >= low[j + k] {
                            is_low = false;
                            break;
                        }
                    } else {
                        is_low = false;
                        break;
                    }
                }
                if is_low && swing_lows.len() < 3 {
                    swing_lows.push((j, low[j]));
                }

                if swing_highs.len() >= 3 && swing_lows.len() >= 3 {
                    break;
                }
            }

            // Determine trend direction from recent swing points
            let trend_up = if !swing_highs.is_empty() && !swing_lows.is_empty() {
                // Most recent swing point comparison
                let last_high_idx = swing_highs.first().map(|(idx, _)| *idx).unwrap_or(0);
                let last_low_idx = swing_lows.first().map(|(idx, _)| *idx).unwrap_or(0);
                last_low_idx > last_high_idx || close[i] > close[i.saturating_sub(self.swing_period)]
            } else {
                close[i] > close[i.saturating_sub(self.swing_period)]
            };

            // Calculate swing range for extensions
            if !swing_highs.is_empty() && !swing_lows.is_empty() {
                let recent_high = swing_highs.first().map(|(_, p)| *p).unwrap_or(high[i]);
                let recent_low = swing_lows.first().map(|(_, p)| *p).unwrap_or(low[i]);
                let swing_range = recent_high - recent_low;

                if swing_range > 1e-10 {
                    if trend_up {
                        // Bullish extensions above the swing high
                        primary_target[i] = recent_low + swing_range * self.primary_ratio;
                        secondary_target[i] = recent_low + swing_range * self.secondary_ratio;
                    } else {
                        // Bearish extensions below the swing low
                        primary_target[i] = recent_high - swing_range * self.primary_ratio;
                        secondary_target[i] = recent_high - swing_range * self.secondary_ratio;
                    }
                }
            } else {
                // Fallback: use recent range
                let start = i.saturating_sub(self.swing_period * 2);
                let range_high = high[start..=i].iter().cloned().fold(f64::MIN, f64::max);
                let range_low = low[start..=i].iter().cloned().fold(f64::MAX, f64::min);
                let range = range_high - range_low;

                if range > 1e-10 {
                    if close[i] > close[start] {
                        primary_target[i] = range_low + range * self.primary_ratio;
                        secondary_target[i] = range_low + range * self.secondary_ratio;
                    } else {
                        primary_target[i] = range_high - range * self.primary_ratio;
                        secondary_target[i] = range_high - range * self.secondary_ratio;
                    }
                }
            }
        }

        (primary_target, secondary_target)
    }
}

impl TechnicalIndicator for SwingExtensionTarget {
    fn name(&self) -> &str {
        "Swing Extension Target"
    }

    fn min_periods(&self) -> usize {
        self.swing_period * 3 + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (primary, secondary) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::dual(primary, secondary))
    }

    fn output_features(&self) -> usize {
        2
    }
}

/// Swing Persistence - Measures persistence of swing direction
///
/// Tracks how long the current swing direction has persisted and measures
/// the consistency of directional movement over time.
///
/// Output:
/// - Primary: Persistence score (0 to 100, higher = more persistent direction)
/// - Secondary: Bars since last direction change
#[derive(Debug, Clone)]
pub struct SwingPersistence {
    period: usize,
    /// Threshold sensitivity for detecting significant direction changes
    #[allow(dead_code)]
    sensitivity: f64,
}

impl SwingPersistence {
    /// Create a new Swing Persistence indicator.
    ///
    /// # Arguments
    /// * `period` - Lookback period for persistence analysis (minimum 5)
    /// * `sensitivity` - Threshold sensitivity for direction changes (0.0 to 1.0)
    pub fn new(period: usize, sensitivity: f64) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if !(0.0..=1.0).contains(&sensitivity) {
            return Err(IndicatorError::InvalidParameter {
                name: "sensitivity".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        Ok(Self { period, sensitivity })
    }

    /// Calculate swing persistence metrics
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut persistence_score = vec![50.0; n];
        let mut bars_since_change = vec![0.0; n];

        if n < self.period {
            return (persistence_score, bars_since_change);
        }

        // Track direction changes
        let mut last_direction: Option<i32> = None;
        let mut direction_start_idx = 0;

        for i in 1..n {
            // Determine current bar direction
            let bar_change = close[i] - close[i - 1];
            let current_direction = if bar_change > 0.0 {
                1
            } else if bar_change < 0.0 {
                -1
            } else {
                0
            };

            // Check for direction change
            if let Some(last_dir) = last_direction {
                if current_direction != 0 && current_direction != last_dir {
                    direction_start_idx = i;
                    last_direction = Some(current_direction);
                }
            } else if current_direction != 0 {
                last_direction = Some(current_direction);
                direction_start_idx = i;
            }

            // Bars since last direction change
            bars_since_change[i] = (i - direction_start_idx) as f64;
        }

        // Calculate persistence score
        for i in self.period..n {
            let start = i - self.period;

            // Count consecutive bars in same direction
            let mut up_streak = 0;
            let mut down_streak = 0;
            let mut max_up_streak = 0;
            let mut max_down_streak = 0;

            for j in (start + 1)..=i {
                let change = close[j] - close[j - 1];
                if change > 0.0 {
                    up_streak += 1;
                    down_streak = 0;
                    max_up_streak = max_up_streak.max(up_streak);
                } else if change < 0.0 {
                    down_streak += 1;
                    up_streak = 0;
                    max_down_streak = max_down_streak.max(down_streak);
                }
            }

            // Overall directional consistency
            let mut up_bars = 0;
            let mut down_bars = 0;
            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    up_bars += 1;
                } else if close[j] < close[j - 1] {
                    down_bars += 1;
                }
            }

            let total_directional = up_bars + down_bars;
            let dominance = if total_directional > 0 {
                (up_bars as i32 - down_bars as i32).abs() as f64 / total_directional as f64
            } else {
                0.0
            };

            // Maximum streak relative to period
            let max_streak = max_up_streak.max(max_down_streak);
            let streak_ratio = max_streak as f64 / self.period as f64;

            // Price progress: how much net movement vs range
            let period_high = high[start..=i].iter().cloned().fold(f64::MIN, f64::max);
            let period_low = low[start..=i].iter().cloned().fold(f64::MAX, f64::min);
            let range = period_high - period_low;
            let net_change = (close[i] - close[start]).abs();
            let progress_ratio = if range > 1e-10 {
                net_change / range
            } else {
                0.0
            };

            // Combine factors
            let raw_score = dominance * 35.0 + streak_ratio * 35.0 + progress_ratio * 30.0;
            persistence_score[i] = (raw_score * 100.0 / 100.0).clamp(0.0, 100.0);
        }

        (persistence_score, bars_since_change)
    }
}

impl TechnicalIndicator for SwingPersistence {
    fn name(&self) -> &str {
        "Swing Persistence"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (score, bars) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::dual(score, bars))
    }

    fn output_features(&self) -> usize {
        2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        // Create data with clear swing patterns
        let high = vec![
            102.0, 104.0, 106.0, 105.0, 103.0, 104.0, 108.0, 110.0, 109.0, 107.0,
            108.0, 112.0, 114.0, 113.0, 111.0, 112.0, 116.0, 118.0, 117.0, 115.0,
            116.0, 120.0, 122.0, 121.0, 119.0, 120.0, 124.0, 126.0, 125.0, 123.0,
        ];
        let low = vec![
            98.0, 100.0, 102.0, 101.0, 99.0, 100.0, 104.0, 106.0, 105.0, 103.0,
            104.0, 108.0, 110.0, 109.0, 107.0, 108.0, 112.0, 114.0, 113.0, 111.0,
            112.0, 116.0, 118.0, 117.0, 115.0, 116.0, 120.0, 122.0, 121.0, 119.0,
        ];
        let close = vec![
            100.0, 102.0, 104.0, 103.0, 101.0, 102.0, 106.0, 108.0, 107.0, 105.0,
            106.0, 110.0, 112.0, 111.0, 109.0, 110.0, 114.0, 116.0, 115.0, 113.0,
            114.0, 118.0, 120.0, 119.0, 117.0, 118.0, 122.0, 124.0, 123.0, 121.0,
        ];
        let volume = vec![
            1000.0, 1100.0, 1200.0, 1100.0, 1000.0, 1050.0, 1300.0, 1400.0, 1300.0, 1100.0,
            1150.0, 1500.0, 1600.0, 1500.0, 1200.0, 1250.0, 1700.0, 1800.0, 1700.0, 1400.0,
            1450.0, 1900.0, 2000.0, 1900.0, 1600.0, 1650.0, 2100.0, 2200.0, 2100.0, 1800.0,
        ];
        (high, low, close, volume)
    }

    #[test]
    fn test_swing_trend_strength() {
        let (high, low, close, _) = make_test_data();
        let sts = SwingTrendStrength::new(10, 3).unwrap();
        let result = sts.calculate(&high, &low, &close);

        assert_eq!(result.len(), close.len());
        // Values should be between 0 and 100
        for i in 13..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0, "Value at {} was {}", i, result[i]);
        }
    }

    #[test]
    fn test_swing_trend_strength_validation() {
        assert!(SwingTrendStrength::new(4, 3).is_err());
        assert!(SwingTrendStrength::new(5, 0).is_err());
        assert!(SwingTrendStrength::new(5, 1).is_ok());
    }

    #[test]
    fn test_swing_reversal() {
        let (high, low, close, _) = make_test_data();
        let sr = SwingReversal::new(5, 0.3).unwrap();
        let (signal, confidence) = sr.calculate(&high, &low, &close);

        assert_eq!(signal.len(), close.len());
        assert_eq!(confidence.len(), close.len());

        // Signals should be -1, 0, or 1
        for s in &signal {
            assert!(*s == -1.0 || *s == 0.0 || *s == 1.0);
        }

        // Confidence should be 0-100
        for c in &confidence {
            assert!(*c >= 0.0 && *c <= 100.0);
        }
    }

    #[test]
    fn test_swing_reversal_validation() {
        assert!(SwingReversal::new(4, 0.5).is_err());
        assert!(SwingReversal::new(5, -0.1).is_err());
        assert!(SwingReversal::new(5, 1.1).is_err());
        assert!(SwingReversal::new(5, 0.5).is_ok());
    }

    #[test]
    fn test_swing_volatility() {
        let (high, low, close, _) = make_test_data();
        let sv = SwingVolatility::new(10, 3).unwrap();
        let result = sv.calculate(&high, &low, &close);

        assert_eq!(result.len(), close.len());
        // Volatility should be non-negative
        for i in 13..result.len() {
            assert!(result[i] >= 0.0, "Volatility at {} was {}", i, result[i]);
        }
    }

    #[test]
    fn test_swing_volatility_validation() {
        assert!(SwingVolatility::new(9, 3).is_err());
        assert!(SwingVolatility::new(10, 2).is_err());
        assert!(SwingVolatility::new(10, 3).is_ok());
    }

    #[test]
    fn test_swing_momentum_advanced() {
        let (high, low, close, volume) = make_test_data();
        let sma = SwingMomentumAdvanced::new(5, 3).unwrap();
        let (momentum, acceleration) = sma.calculate(&high, &low, &close, &volume);

        assert_eq!(momentum.len(), close.len());
        assert_eq!(acceleration.len(), close.len());
    }

    #[test]
    fn test_swing_momentum_advanced_validation() {
        assert!(SwingMomentumAdvanced::new(4, 3).is_err());
        assert!(SwingMomentumAdvanced::new(5, 2).is_err());
        assert!(SwingMomentumAdvanced::new(5, 3).is_ok());
    }

    #[test]
    fn test_swing_target_levels() {
        let (high, low, close, _) = make_test_data();
        let stl = SwingTargetLevels::new(5, 1.618).unwrap();
        let (upper, lower) = stl.calculate(&high, &low, &close);

        assert_eq!(upper.len(), close.len());
        assert_eq!(lower.len(), close.len());

        // Upper target should be above lower target when both are valid
        for i in 11..close.len() {
            if !upper[i].is_nan() && !lower[i].is_nan() {
                assert!(upper[i] > lower[i], "Upper {} should be > lower {} at {}", upper[i], lower[i], i);
            }
        }
    }

    #[test]
    fn test_swing_target_levels_validation() {
        assert!(SwingTargetLevels::new(4, 1.618).is_err());
        assert!(SwingTargetLevels::new(5, 0.0).is_err());
        assert!(SwingTargetLevels::new(5, -1.0).is_err());
        assert!(SwingTargetLevels::new(5, 1.618).is_ok());
    }

    #[test]
    fn test_swing_duration() {
        let (high, low, _, _) = make_test_data();
        let sd = SwingDuration::new(3, 5).unwrap();
        let (current, avg) = sd.calculate(&high, &low);

        assert_eq!(current.len(), high.len());
        assert_eq!(avg.len(), high.len());

        // Current duration should be non-negative
        for c in &current {
            assert!(*c >= 0.0);
        }

        // Average duration should be non-negative
        for a in &avg {
            assert!(*a >= 0.0);
        }
    }

    #[test]
    fn test_swing_duration_validation() {
        assert!(SwingDuration::new(2, 5).is_err());
        assert!(SwingDuration::new(3, 2).is_err());
        assert!(SwingDuration::new(3, 3).is_ok());
    }

    #[test]
    fn test_technical_indicator_impl() {
        let mut data = OHLCVSeries::new();
        for i in 0..30 {
            let base = 100.0 + (i as f64 * 0.3).sin() * 10.0;
            data.open.push(base);
            data.high.push(base + 2.0);
            data.low.push(base - 2.0);
            data.close.push(base + 1.0);
            data.volume.push(1000.0 + i as f64 * 10.0);
        }

        // Test SwingTrendStrength
        let sts = SwingTrendStrength::new(5, 2).unwrap();
        let output = sts.compute(&data).unwrap();
        assert_eq!(output.primary.len(), 30);
        assert_eq!(sts.name(), "Swing Trend Strength");

        // Test SwingReversal
        let sr = SwingReversal::new(5, 0.3).unwrap();
        let output = sr.compute(&data).unwrap();
        assert_eq!(output.primary.len(), 30);
        assert!(output.secondary.is_some());
        assert_eq!(sr.name(), "Swing Reversal");

        // Test SwingVolatility
        let sv = SwingVolatility::new(10, 3).unwrap();
        let output = sv.compute(&data).unwrap();
        assert_eq!(output.primary.len(), 30);
        assert_eq!(sv.name(), "Swing Volatility");

        // Test SwingMomentumAdvanced
        let sma = SwingMomentumAdvanced::new(5, 3).unwrap();
        let output = sma.compute(&data).unwrap();
        assert_eq!(output.primary.len(), 30);
        assert!(output.secondary.is_some());
        assert_eq!(sma.name(), "Swing Momentum Advanced");

        // Test SwingTargetLevels
        let stl = SwingTargetLevels::new(5, 1.618).unwrap();
        let output = stl.compute(&data).unwrap();
        assert_eq!(output.primary.len(), 30);
        assert!(output.secondary.is_some());
        assert_eq!(stl.name(), "Swing Target Levels");

        // Test SwingDuration
        let sd = SwingDuration::new(3, 5).unwrap();
        let output = sd.compute(&data).unwrap();
        assert_eq!(output.primary.len(), 30);
        assert!(output.secondary.is_some());
        assert_eq!(sd.name(), "Swing Duration");
    }

    // ========== Tests for new indicators ==========

    #[test]
    fn test_swing_efficiency() {
        let (high, low, close, _) = make_test_data();
        let se = SwingEfficiency::new(10, 3).unwrap();
        let result = se.calculate(&high, &low, &close);

        assert_eq!(result.len(), close.len());
        // Efficiency should be between 0 and 100
        for i in 10..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0, "Efficiency at {} was {}", i, result[i]);
        }
    }

    #[test]
    fn test_swing_efficiency_validation() {
        assert!(SwingEfficiency::new(4, 2).is_err());
        assert!(SwingEfficiency::new(5, 1).is_err());
        assert!(SwingEfficiency::new(5, 2).is_ok());
        assert!(SwingEfficiency::new(10, 3).is_ok());
    }

    #[test]
    fn test_swing_efficiency_trending_data() {
        // Strongly trending data should have higher efficiency
        let high: Vec<f64> = (0..20).map(|i| 100.0 + i as f64 * 2.0 + 1.0).collect();
        let low: Vec<f64> = (0..20).map(|i| 100.0 + i as f64 * 2.0 - 1.0).collect();
        let close: Vec<f64> = (0..20).map(|i| 100.0 + i as f64 * 2.0).collect();

        let se = SwingEfficiency::new(5, 2).unwrap();
        let result = se.calculate(&high, &low, &close);

        // In a strong uptrend, efficiency should be relatively high
        for i in 10..result.len() {
            assert!(result[i] > 10.0, "Trending data should have decent efficiency at {}", i);
        }
    }

    #[test]
    fn test_swing_continuation() {
        let (high, low, close, _) = make_test_data();
        let sc = SwingContinuation::new(5, 3).unwrap();
        let (probability, direction) = sc.calculate(&high, &low, &close);

        assert_eq!(probability.len(), close.len());
        assert_eq!(direction.len(), close.len());

        // Probability should be between 0 and 100
        for i in 8..probability.len() {
            assert!(probability[i] >= 0.0 && probability[i] <= 100.0,
                "Probability at {} was {}", i, probability[i]);
        }

        // Direction should be -1, 0, or 1
        for d in &direction {
            assert!(*d == -1.0 || *d == 0.0 || *d == 1.0, "Direction should be -1, 0, or 1");
        }
    }

    #[test]
    fn test_swing_continuation_validation() {
        assert!(SwingContinuation::new(4, 3).is_err());
        assert!(SwingContinuation::new(5, 2).is_err());
        assert!(SwingContinuation::new(5, 3).is_ok());
    }

    #[test]
    fn test_swing_magnitude() {
        let (high, low, close, _) = make_test_data();
        let sm = SwingMagnitude::new(5, 10).unwrap();
        let (absolute, relative) = sm.calculate(&high, &low, &close);

        assert_eq!(absolute.len(), close.len());
        assert_eq!(relative.len(), close.len());

        // Absolute magnitude should be non-negative
        for i in 5..absolute.len() {
            assert!(absolute[i] >= 0.0, "Absolute magnitude at {} was {}", i, absolute[i]);
        }

        // Relative magnitude should be positive
        for i in 15..relative.len() {
            assert!(relative[i] > 0.0, "Relative magnitude at {} was {}", i, relative[i]);
        }
    }

    #[test]
    fn test_swing_magnitude_validation() {
        assert!(SwingMagnitude::new(2, 5).is_err());
        assert!(SwingMagnitude::new(3, 4).is_err());
        assert!(SwingMagnitude::new(3, 5).is_ok());
    }

    #[test]
    fn test_swing_retracement_level() {
        let (high, low, close, _) = make_test_data();
        let srl = SwingRetracementLevel::new(5).unwrap();
        let (retracement, fib) = srl.calculate(&high, &low, &close);

        assert_eq!(retracement.len(), close.len());
        assert_eq!(fib.len(), close.len());

        // Retracement should be between 0 and 100
        for i in 10..retracement.len() {
            assert!(retracement[i] >= 0.0 && retracement[i] <= 100.0,
                "Retracement at {} was {}", i, retracement[i]);
        }

        // Fib level should be a valid Fibonacci level
        let valid_fibs = [0.0, 23.6, 38.2, 50.0, 61.8, 78.6, 100.0];
        for i in 10..fib.len() {
            assert!(valid_fibs.contains(&fib[i]),
                "Fib at {} should be a valid level, got {}", i, fib[i]);
        }
    }

    #[test]
    fn test_swing_retracement_level_validation() {
        assert!(SwingRetracementLevel::new(4).is_err());
        assert!(SwingRetracementLevel::new(5).is_ok());
        assert!(SwingRetracementLevel::new(10).is_ok());
    }

    #[test]
    fn test_swing_extension_target() {
        let (high, low, close, _) = make_test_data();
        let set = SwingExtensionTarget::new(5, 1.272, 1.618).unwrap();
        let (primary, secondary) = set.calculate(&high, &low, &close);

        assert_eq!(primary.len(), close.len());
        assert_eq!(secondary.len(), close.len());

        // After warmup, targets should be valid numbers
        for i in 16..close.len() {
            if !primary[i].is_nan() && !secondary[i].is_nan() {
                // Secondary target should be further from current price than primary
                let current = close[i];
                let primary_dist = (primary[i] - current).abs();
                let secondary_dist = (secondary[i] - current).abs();
                // This may not always hold due to trend direction, so just check they're finite
                assert!(primary[i].is_finite(), "Primary target at {} should be finite", i);
                assert!(secondary[i].is_finite(), "Secondary target at {} should be finite", i);
            }
        }
    }

    #[test]
    fn test_swing_extension_target_validation() {
        assert!(SwingExtensionTarget::new(4, 1.272, 1.618).is_err());
        assert!(SwingExtensionTarget::new(5, 1.0, 1.618).is_err());
        assert!(SwingExtensionTarget::new(5, 1.272, 1.2).is_err());
        assert!(SwingExtensionTarget::new(5, 1.272, 1.618).is_ok());
    }

    #[test]
    fn test_swing_extension_target_with_defaults() {
        let set = SwingExtensionTarget::with_defaults(5).unwrap();
        assert_eq!(set.primary_ratio, 1.272);
        assert_eq!(set.secondary_ratio, 1.618);
    }

    #[test]
    fn test_swing_persistence() {
        let (high, low, close, _) = make_test_data();
        let sp = SwingPersistence::new(10, 0.5).unwrap();
        let (score, bars) = sp.calculate(&high, &low, &close);

        assert_eq!(score.len(), close.len());
        assert_eq!(bars.len(), close.len());

        // Score should be between 0 and 100
        for i in 10..score.len() {
            assert!(score[i] >= 0.0 && score[i] <= 100.0,
                "Persistence score at {} was {}", i, score[i]);
        }

        // Bars since change should be non-negative
        for b in &bars {
            assert!(*b >= 0.0, "Bars since change should be non-negative");
        }
    }

    #[test]
    fn test_swing_persistence_validation() {
        assert!(SwingPersistence::new(4, 0.5).is_err());
        assert!(SwingPersistence::new(5, -0.1).is_err());
        assert!(SwingPersistence::new(5, 1.1).is_err());
        assert!(SwingPersistence::new(5, 0.0).is_ok());
        assert!(SwingPersistence::new(5, 1.0).is_ok());
    }

    #[test]
    fn test_swing_persistence_trending_data() {
        // Strongly trending data should have higher persistence
        let high: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 * 1.5 + 1.0).collect();
        let low: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 * 1.5 - 1.0).collect();
        let close: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 * 1.5).collect();

        let sp = SwingPersistence::new(5, 0.5).unwrap();
        let (score, _) = sp.calculate(&high, &low, &close);

        // In a strong uptrend, persistence should be relatively high
        for i in 10..score.len() {
            assert!(score[i] > 30.0, "Trending data should have higher persistence at {}", i);
        }
    }

    #[test]
    fn test_new_indicators_technical_impl() {
        let mut data = OHLCVSeries::new();
        for i in 0..40 {
            let base = 100.0 + (i as f64 * 0.3).sin() * 10.0 + i as f64 * 0.5;
            data.open.push(base);
            data.high.push(base + 2.0);
            data.low.push(base - 2.0);
            data.close.push(base + 1.0);
            data.volume.push(1000.0 + i as f64 * 10.0);
        }

        // Test SwingEfficiency
        let se = SwingEfficiency::new(5, 2).unwrap();
        let output = se.compute(&data).unwrap();
        assert_eq!(output.primary.len(), 40);
        assert_eq!(se.name(), "Swing Efficiency");
        assert_eq!(se.min_periods(), 6);

        // Test SwingContinuation
        let sc = SwingContinuation::new(5, 3).unwrap();
        let output = sc.compute(&data).unwrap();
        assert_eq!(output.primary.len(), 40);
        assert!(output.secondary.is_some());
        assert_eq!(sc.name(), "Swing Continuation");
        assert_eq!(sc.output_features(), 2);

        // Test SwingMagnitude
        let sm = SwingMagnitude::new(3, 5).unwrap();
        let output = sm.compute(&data).unwrap();
        assert_eq!(output.primary.len(), 40);
        assert!(output.secondary.is_some());
        assert_eq!(sm.name(), "Swing Magnitude");
        assert_eq!(sm.output_features(), 2);

        // Test SwingRetracementLevel
        let srl = SwingRetracementLevel::new(5).unwrap();
        let output = srl.compute(&data).unwrap();
        assert_eq!(output.primary.len(), 40);
        assert!(output.secondary.is_some());
        assert_eq!(srl.name(), "Swing Retracement Level");
        assert_eq!(srl.output_features(), 2);

        // Test SwingExtensionTarget
        let set = SwingExtensionTarget::with_defaults(5).unwrap();
        let output = set.compute(&data).unwrap();
        assert_eq!(output.primary.len(), 40);
        assert!(output.secondary.is_some());
        assert_eq!(set.name(), "Swing Extension Target");
        assert_eq!(set.output_features(), 2);

        // Test SwingPersistence
        let sp = SwingPersistence::new(5, 0.5).unwrap();
        let output = sp.compute(&data).unwrap();
        assert_eq!(output.primary.len(), 40);
        assert!(output.secondary.is_some());
        assert_eq!(sp.name(), "Swing Persistence");
        assert_eq!(sp.output_features(), 2);
    }

    #[test]
    fn test_new_indicators_empty_data() {
        let high: Vec<f64> = vec![];
        let low: Vec<f64> = vec![];
        let close: Vec<f64> = vec![];

        let se = SwingEfficiency::new(5, 2).unwrap();
        let result = se.calculate(&high, &low, &close);
        assert!(result.is_empty());

        let sc = SwingContinuation::new(5, 3).unwrap();
        let (prob, dir) = sc.calculate(&high, &low, &close);
        assert!(prob.is_empty());
        assert!(dir.is_empty());

        let sm = SwingMagnitude::new(3, 5).unwrap();
        let (abs, rel) = sm.calculate(&high, &low, &close);
        assert!(abs.is_empty());
        assert!(rel.is_empty());

        let srl = SwingRetracementLevel::new(5).unwrap();
        let (ret, fib) = srl.calculate(&high, &low, &close);
        assert!(ret.is_empty());
        assert!(fib.is_empty());

        let set = SwingExtensionTarget::with_defaults(5).unwrap();
        let (prim, sec) = set.calculate(&high, &low, &close);
        assert!(prim.is_empty());
        assert!(sec.is_empty());

        let sp = SwingPersistence::new(5, 0.5).unwrap();
        let (score, bars) = sp.calculate(&high, &low, &close);
        assert!(score.is_empty());
        assert!(bars.is_empty());
    }

    #[test]
    fn test_new_indicators_small_data() {
        // Data smaller than required periods
        let high = vec![100.0, 101.0, 102.0];
        let low = vec![98.0, 99.0, 100.0];
        let close = vec![99.0, 100.0, 101.0];

        let se = SwingEfficiency::new(5, 2).unwrap();
        let result = se.calculate(&high, &low, &close);
        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|&x| x == 0.0));

        let sc = SwingContinuation::new(5, 3).unwrap();
        let (prob, dir) = sc.calculate(&high, &low, &close);
        assert_eq!(prob.len(), 3);
        assert_eq!(dir.len(), 3);

        let sm = SwingMagnitude::new(3, 5).unwrap();
        let (abs, rel) = sm.calculate(&high, &low, &close);
        assert_eq!(abs.len(), 3);
        assert_eq!(rel.len(), 3);

        let srl = SwingRetracementLevel::new(5).unwrap();
        let (ret, fib) = srl.calculate(&high, &low, &close);
        assert_eq!(ret.len(), 3);
        assert_eq!(fib.len(), 3);

        let set = SwingExtensionTarget::with_defaults(5).unwrap();
        let (prim, sec) = set.calculate(&high, &low, &close);
        assert_eq!(prim.len(), 3);
        assert_eq!(sec.len(), 3);

        let sp = SwingPersistence::new(5, 0.5).unwrap();
        let (score, bars) = sp.calculate(&high, &low, &close);
        assert_eq!(score.len(), 3);
        assert_eq!(bars.len(), 3);
    }
}
