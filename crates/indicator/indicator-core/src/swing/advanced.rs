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
}
