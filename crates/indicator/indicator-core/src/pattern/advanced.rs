//! Advanced Pattern Recognition Indicators
//!
//! Sophisticated pattern detection algorithms for trend continuation,
//! reversals, breakouts, and consolidation analysis.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

// ============================================================================
// TrendContinuationPattern
// ============================================================================

/// Trend Continuation Pattern - Detects patterns indicating trend continuation
///
/// Analyzes price action to identify patterns suggesting the current trend
/// will continue, including higher highs/lows in uptrends and lower highs/lows
/// in downtrends.
#[derive(Debug, Clone)]
pub struct TrendContinuationPattern {
    /// Lookback period for trend analysis
    lookback: usize,
    /// Minimum consecutive confirmations required
    min_confirmations: usize,
}

impl TrendContinuationPattern {
    /// Create a new TrendContinuationPattern indicator.
    ///
    /// # Arguments
    /// * `lookback` - Lookback period for trend analysis (5-50)
    /// * `min_confirmations` - Minimum confirmations required (2-10)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(lookback: usize, min_confirmations: usize) -> Result<Self> {
        if lookback < 5 || lookback > 50 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be between 5 and 50".to_string(),
            });
        }
        if min_confirmations < 2 || min_confirmations > 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "min_confirmations".to_string(),
                reason: "must be between 2 and 10".to_string(),
            });
        }
        if min_confirmations > lookback {
            return Err(IndicatorError::InvalidParameter {
                name: "min_confirmations".to_string(),
                reason: "cannot exceed lookback period".to_string(),
            });
        }
        Ok(Self { lookback, min_confirmations })
    }

    /// Calculate trend continuation signals.
    ///
    /// Returns:
    /// * +1: Bullish continuation (uptrend continuation)
    /// * -1: Bearish continuation (downtrend continuation)
    /// * 0: No clear continuation pattern
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.lookback..n {
            let start = i - self.lookback;

            // Count higher highs and higher lows (bullish)
            let mut bullish_count = 0;
            for j in (start + 1)..=i {
                if high[j] > high[j - 1] && low[j] > low[j - 1] {
                    bullish_count += 1;
                }
            }

            // Count lower highs and lower lows (bearish)
            let mut bearish_count = 0;
            for j in (start + 1)..=i {
                if high[j] < high[j - 1] && low[j] < low[j - 1] {
                    bearish_count += 1;
                }
            }

            // Check for trend continuation with recent confirmation
            let recent_bullish = close[i] > close[i - 1] && high[i] > high[i - 1];
            let recent_bearish = close[i] < close[i - 1] && low[i] < low[i - 1];

            if bullish_count >= self.min_confirmations && recent_bullish {
                result[i] = 1.0;
            } else if bearish_count >= self.min_confirmations && recent_bearish {
                result[i] = -1.0;
            }
        }

        result
    }
}

impl TechnicalIndicator for TrendContinuationPattern {
    fn name(&self) -> &str {
        "Trend Continuation Pattern"
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

// ============================================================================
// ReversalCandlePattern
// ============================================================================

/// Reversal Candle Pattern - Detects candlestick reversal patterns
///
/// Identifies key candlestick reversal patterns including pin bars,
/// engulfing patterns, and doji at extremes.
#[derive(Debug, Clone)]
pub struct ReversalCandlePattern {
    /// Minimum body-to-range ratio for significant candles
    min_body_ratio: f64,
    /// Minimum wick-to-body ratio for pin bars
    min_wick_ratio: f64,
}

impl ReversalCandlePattern {
    /// Create a new ReversalCandlePattern indicator.
    ///
    /// # Arguments
    /// * `min_body_ratio` - Minimum body/range ratio for engulfing (0.3-0.9)
    /// * `min_wick_ratio` - Minimum wick/body ratio for pin bars (1.5-5.0)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(min_body_ratio: f64, min_wick_ratio: f64) -> Result<Self> {
        if min_body_ratio < 0.3 || min_body_ratio > 0.9 {
            return Err(IndicatorError::InvalidParameter {
                name: "min_body_ratio".to_string(),
                reason: "must be between 0.3 and 0.9".to_string(),
            });
        }
        if min_wick_ratio < 1.5 || min_wick_ratio > 5.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "min_wick_ratio".to_string(),
                reason: "must be between 1.5 and 5.0".to_string(),
            });
        }
        Ok(Self { min_body_ratio, min_wick_ratio })
    }

    /// Calculate reversal candle pattern signals.
    ///
    /// Returns:
    /// * +1: Bullish reversal pattern
    /// * -1: Bearish reversal pattern
    /// * 0: No reversal pattern
    pub fn calculate(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in 1..n {
            let range = high[i] - low[i];
            if range < 1e-10 {
                continue;
            }

            let body = (close[i] - open[i]).abs();
            let body_ratio = body / range;
            let is_bullish = close[i] > open[i];

            // Upper and lower wicks
            let upper_wick = if is_bullish {
                high[i] - close[i]
            } else {
                high[i] - open[i]
            };
            let lower_wick = if is_bullish {
                open[i] - low[i]
            } else {
                close[i] - low[i]
            };

            // Check for pin bar (hammer/shooting star)
            if body > 1e-10 {
                let lower_wick_ratio = lower_wick / body;
                let upper_wick_ratio = upper_wick / body;

                // Bullish pin bar (hammer): long lower wick, small upper wick
                if lower_wick_ratio >= self.min_wick_ratio && upper_wick < lower_wick * 0.3 {
                    result[i] = 1.0;
                    continue;
                }

                // Bearish pin bar (shooting star): long upper wick, small lower wick
                if upper_wick_ratio >= self.min_wick_ratio && lower_wick < upper_wick * 0.3 {
                    result[i] = -1.0;
                    continue;
                }
            }

            // Check for engulfing pattern
            if i >= 1 {
                let prev_body = (close[i - 1] - open[i - 1]).abs();
                let prev_range = high[i - 1] - low[i - 1];
                let prev_is_bullish = close[i - 1] > open[i - 1];

                if prev_range > 1e-10 && body_ratio >= self.min_body_ratio {
                    // Bullish engulfing
                    if !prev_is_bullish && is_bullish && body > prev_body {
                        if close[i] > open[i - 1] && open[i] < close[i - 1] {
                            result[i] = 1.0;
                            continue;
                        }
                    }

                    // Bearish engulfing
                    if prev_is_bullish && !is_bullish && body > prev_body {
                        if close[i] < open[i - 1] && open[i] > close[i - 1] {
                            result[i] = -1.0;
                            continue;
                        }
                    }
                }
            }
        }

        result
    }
}

impl TechnicalIndicator for ReversalCandlePattern {
    fn name(&self) -> &str {
        "Reversal Candle Pattern"
    }

    fn min_periods(&self) -> usize {
        2
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.open, &data.high, &data.low, &data.close)))
    }
}

// ============================================================================
// VolumePricePattern
// ============================================================================

/// Volume Price Pattern - Pattern detection using volume and price
///
/// Identifies patterns where volume confirms or diverges from price action,
/// useful for detecting accumulation, distribution, and climax events.
#[derive(Debug, Clone)]
pub struct VolumePricePattern {
    /// Period for average volume calculation
    volume_period: usize,
    /// Volume spike multiplier
    volume_multiplier: f64,
}

impl VolumePricePattern {
    /// Create a new VolumePricePattern indicator.
    ///
    /// # Arguments
    /// * `volume_period` - Period for average volume (5-50)
    /// * `volume_multiplier` - Multiplier for volume spike detection (1.5-5.0)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(volume_period: usize, volume_multiplier: f64) -> Result<Self> {
        if volume_period < 5 || volume_period > 50 {
            return Err(IndicatorError::InvalidParameter {
                name: "volume_period".to_string(),
                reason: "must be between 5 and 50".to_string(),
            });
        }
        if volume_multiplier < 1.5 || volume_multiplier > 5.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "volume_multiplier".to_string(),
                reason: "must be between 1.5 and 5.0".to_string(),
            });
        }
        Ok(Self { volume_period, volume_multiplier })
    }

    /// Calculate volume-price pattern signals.
    ///
    /// Returns:
    /// * +1: Bullish volume-price pattern (accumulation/buying climax)
    /// * -1: Bearish volume-price pattern (distribution/selling climax)
    /// * 0: No significant pattern
    pub fn calculate(&self, open: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.volume_period..n {
            // Calculate average volume
            let avg_volume: f64 = volume[(i - self.volume_period)..i].iter().sum::<f64>()
                / self.volume_period as f64;

            if avg_volume < 1e-10 {
                continue;
            }

            let volume_ratio = volume[i] / avg_volume;
            let is_volume_spike = volume_ratio >= self.volume_multiplier;

            if !is_volume_spike {
                continue;
            }

            let price_change = close[i] - open[i];
            let is_bullish = price_change > 0.0;
            let is_bearish = price_change < 0.0;

            // High volume with price direction = confirmation
            if is_bullish && is_volume_spike {
                result[i] = 1.0;
            } else if is_bearish && is_volume_spike {
                result[i] = -1.0;
            }
        }

        result
    }
}

impl TechnicalIndicator for VolumePricePattern {
    fn name(&self) -> &str {
        "Volume Price Pattern"
    }

    fn min_periods(&self) -> usize {
        self.volume_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.open, &data.close, &data.volume)))
    }
}

// ============================================================================
// MomentumPattern
// ============================================================================

/// Momentum Pattern - Momentum-based pattern recognition
///
/// Detects momentum patterns including acceleration, deceleration,
/// and momentum divergences that signal potential trend changes.
#[derive(Debug, Clone)]
pub struct MomentumPattern {
    /// Fast momentum period
    fast_period: usize,
    /// Slow momentum period
    slow_period: usize,
    /// Threshold for significant momentum change (percentage)
    threshold: f64,
}

impl MomentumPattern {
    /// Create a new MomentumPattern indicator.
    ///
    /// # Arguments
    /// * `fast_period` - Fast momentum period (3-20)
    /// * `slow_period` - Slow momentum period (10-50)
    /// * `threshold` - Momentum change threshold in percent (0.5-5.0)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(fast_period: usize, slow_period: usize, threshold: f64) -> Result<Self> {
        if fast_period < 3 || fast_period > 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "fast_period".to_string(),
                reason: "must be between 3 and 20".to_string(),
            });
        }
        if slow_period < 10 || slow_period > 50 {
            return Err(IndicatorError::InvalidParameter {
                name: "slow_period".to_string(),
                reason: "must be between 10 and 50".to_string(),
            });
        }
        if fast_period >= slow_period {
            return Err(IndicatorError::InvalidParameter {
                name: "fast_period".to_string(),
                reason: "must be less than slow_period".to_string(),
            });
        }
        if threshold < 0.5 || threshold > 5.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "threshold".to_string(),
                reason: "must be between 0.5 and 5.0".to_string(),
            });
        }
        Ok(Self { fast_period, slow_period, threshold })
    }

    /// Calculate momentum pattern signals.
    ///
    /// Returns:
    /// * +1: Bullish momentum pattern (acceleration/positive divergence)
    /// * -1: Bearish momentum pattern (deceleration/negative divergence)
    /// * 0: No significant pattern
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.slow_period..n {
            // Fast momentum (ROC)
            let fast_base = close[i - self.fast_period];
            let fast_mom = if fast_base > 1e-10 {
                (close[i] / fast_base - 1.0) * 100.0
            } else {
                0.0
            };

            // Slow momentum (ROC)
            let slow_base = close[i - self.slow_period];
            let slow_mom = if slow_base > 1e-10 {
                (close[i] / slow_base - 1.0) * 100.0
            } else {
                0.0
            };

            // Previous fast momentum for acceleration detection
            let prev_fast_base = close[i - self.fast_period - 1];
            let prev_fast_mom = if prev_fast_base > 1e-10 {
                (close[i - 1] / prev_fast_base - 1.0) * 100.0
            } else {
                0.0
            };

            // Momentum acceleration
            let mom_accel = fast_mom - prev_fast_mom;

            // Bullish pattern: positive fast momentum accelerating or crossing above slow
            if fast_mom > self.threshold && (mom_accel > 0.0 || fast_mom > slow_mom) {
                result[i] = 1.0;
            }
            // Bearish pattern: negative fast momentum decelerating or crossing below slow
            else if fast_mom < -self.threshold && (mom_accel < 0.0 || fast_mom < slow_mom) {
                result[i] = -1.0;
            }
        }

        result
    }
}

impl TechnicalIndicator for MomentumPattern {
    fn name(&self) -> &str {
        "Momentum Pattern"
    }

    fn min_periods(&self) -> usize {
        self.slow_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

// ============================================================================
// BreakoutPattern
// ============================================================================

/// Breakout Pattern - Detects breakout patterns
///
/// Identifies price breakouts from consolidation ranges, channels,
/// and key levels with volume confirmation.
#[derive(Debug, Clone)]
pub struct BreakoutPattern {
    /// Lookback period for range detection
    lookback: usize,
    /// Breakout threshold as percentage of range
    breakout_threshold: f64,
    /// Whether to require volume confirmation
    require_volume: bool,
}

impl BreakoutPattern {
    /// Create a new BreakoutPattern indicator.
    ///
    /// # Arguments
    /// * `lookback` - Lookback period for range detection (10-100)
    /// * `breakout_threshold` - Breakout threshold percentage (0.1-1.0)
    /// * `require_volume` - Whether to require volume confirmation
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(lookback: usize, breakout_threshold: f64, require_volume: bool) -> Result<Self> {
        if lookback < 10 || lookback > 100 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be between 10 and 100".to_string(),
            });
        }
        if breakout_threshold < 0.1 || breakout_threshold > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "breakout_threshold".to_string(),
                reason: "must be between 0.1 and 1.0".to_string(),
            });
        }
        Ok(Self { lookback, breakout_threshold, require_volume })
    }

    /// Calculate breakout pattern signals.
    ///
    /// Returns:
    /// * +1: Bullish breakout (above resistance)
    /// * -1: Bearish breakout (below support)
    /// * 0: No breakout
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.lookback..n {
            let start = i - self.lookback;

            // Find highest high and lowest low in lookback (excluding current bar)
            let range_high = high[start..i].iter().cloned().fold(f64::MIN, f64::max);
            let range_low = low[start..i].iter().cloned().fold(f64::MAX, f64::min);
            let range_size = range_high - range_low;

            if range_size < 1e-10 {
                continue;
            }

            // Calculate breakout threshold
            let breakout_distance = range_size * self.breakout_threshold;

            // Volume check
            let volume_ok = if self.require_volume {
                let avg_vol: f64 = volume[start..i].iter().sum::<f64>() / self.lookback as f64;
                avg_vol > 1e-10 && volume[i] > avg_vol
            } else {
                true
            };

            // Bullish breakout: close above range high + threshold
            if close[i] > range_high + breakout_distance && volume_ok {
                result[i] = 1.0;
            }
            // Bearish breakout: close below range low - threshold
            else if close[i] < range_low - breakout_distance && volume_ok {
                result[i] = -1.0;
            }
        }

        result
    }
}

impl TechnicalIndicator for BreakoutPattern {
    fn name(&self) -> &str {
        "Breakout Pattern"
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close, &data.volume)))
    }
}

// ============================================================================
// ConsolidationBreak
// ============================================================================

/// Consolidation Break - Detects breaks from consolidation
///
/// Identifies when price breaks out of a consolidation zone (low volatility
/// period), often signaling the start of a new trend.
#[derive(Debug, Clone)]
pub struct ConsolidationBreak {
    /// Period for consolidation detection
    consolidation_period: usize,
    /// Maximum range as percentage for consolidation
    max_range_pct: f64,
    /// Minimum bars in consolidation before valid break
    min_consolidation_bars: usize,
}

impl ConsolidationBreak {
    /// Create a new ConsolidationBreak indicator.
    ///
    /// # Arguments
    /// * `consolidation_period` - Period for consolidation detection (5-50)
    /// * `max_range_pct` - Max range percentage for consolidation (1.0-10.0)
    /// * `min_consolidation_bars` - Minimum bars in consolidation (3-20)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(consolidation_period: usize, max_range_pct: f64, min_consolidation_bars: usize) -> Result<Self> {
        if consolidation_period < 5 || consolidation_period > 50 {
            return Err(IndicatorError::InvalidParameter {
                name: "consolidation_period".to_string(),
                reason: "must be between 5 and 50".to_string(),
            });
        }
        if max_range_pct < 1.0 || max_range_pct > 10.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "max_range_pct".to_string(),
                reason: "must be between 1.0 and 10.0".to_string(),
            });
        }
        if min_consolidation_bars < 3 || min_consolidation_bars > 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "min_consolidation_bars".to_string(),
                reason: "must be between 3 and 20".to_string(),
            });
        }
        if min_consolidation_bars > consolidation_period {
            return Err(IndicatorError::InvalidParameter {
                name: "min_consolidation_bars".to_string(),
                reason: "cannot exceed consolidation_period".to_string(),
            });
        }
        Ok(Self { consolidation_period, max_range_pct, min_consolidation_bars })
    }

    /// Calculate consolidation break signals.
    ///
    /// Returns:
    /// * +1: Bullish break from consolidation
    /// * -1: Bearish break from consolidation
    /// * 0: No break or not in consolidation
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];
        let mut consolidation_count = vec![0usize; n];

        // First pass: identify consolidation zones
        for i in self.consolidation_period..n {
            let start = i - self.consolidation_period;

            // Find range in consolidation period
            let period_high = high[start..i].iter().cloned().fold(f64::MIN, f64::max);
            let period_low = low[start..i].iter().cloned().fold(f64::MAX, f64::min);
            let range = period_high - period_low;

            // Calculate average price for percentage
            let avg_price: f64 = close[start..i].iter().sum::<f64>() / self.consolidation_period as f64;

            if avg_price > 1e-10 {
                let range_pct = (range / avg_price) * 100.0;

                // Check if in consolidation
                if range_pct <= self.max_range_pct {
                    consolidation_count[i] = consolidation_count[i - 1] + 1;
                } else {
                    consolidation_count[i] = 0;
                }
            }
        }

        // Second pass: detect breaks from consolidation
        for i in self.consolidation_period..n {
            // Only check if we were in consolidation
            if i >= 1 && consolidation_count[i - 1] >= self.min_consolidation_bars {
                let start = i - self.consolidation_period;
                let period_high = high[start..i].iter().cloned().fold(f64::MIN, f64::max);
                let period_low = low[start..i].iter().cloned().fold(f64::MAX, f64::min);

                // Bullish break: close above consolidation high
                if close[i] > period_high {
                    result[i] = 1.0;
                }
                // Bearish break: close below consolidation low
                else if close[i] < period_low {
                    result[i] = -1.0;
                }
            }
        }

        result
    }
}

impl TechnicalIndicator for ConsolidationBreak {
    fn name(&self) -> &str {
        "Consolidation Break"
    }

    fn min_periods(&self) -> usize {
        self.consolidation_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

// ============================================================================
// PatternStrength
// ============================================================================

/// Pattern Strength - Measures strength of detected patterns
///
/// Calculates a strength score for detected patterns based on multiple factors
/// including price momentum, volume confirmation, and trend alignment.
#[derive(Debug, Clone)]
pub struct PatternStrength {
    /// Lookback period for strength calculation
    lookback: usize,
    /// Threshold for significant pattern strength
    strength_threshold: f64,
}

impl PatternStrength {
    /// Create a new PatternStrength indicator.
    ///
    /// # Arguments
    /// * `lookback` - Lookback period for strength calculation (5-100)
    /// * `strength_threshold` - Minimum strength threshold (0.1-1.0)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(lookback: usize, strength_threshold: f64) -> Result<Self> {
        if lookback < 5 || lookback > 100 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be between 5 and 100".to_string(),
            });
        }
        if strength_threshold < 0.1 || strength_threshold > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "strength_threshold".to_string(),
                reason: "must be between 0.1 and 1.0".to_string(),
            });
        }
        Ok(Self { lookback, strength_threshold })
    }

    /// Calculate pattern strength values.
    ///
    /// Returns strength score between 0 and 1, where:
    /// * 0-0.3: Weak pattern
    /// * 0.3-0.6: Moderate pattern
    /// * 0.6-1.0: Strong pattern
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.lookback..n {
            let start = i - self.lookback;

            // Price momentum component
            let price_change = if close[start] > 1e-10 {
                ((close[i] - close[start]) / close[start]).abs()
            } else {
                0.0
            };

            // Volatility component (normalized ATR)
            let mut atr_sum = 0.0;
            for j in (start + 1)..=i {
                let tr = (high[j] - low[j])
                    .max((high[j] - close[j - 1]).abs())
                    .max((low[j] - close[j - 1]).abs());
                atr_sum += tr;
            }
            let avg_atr = atr_sum / self.lookback as f64;
            let avg_price = close[start..=i].iter().sum::<f64>() / (self.lookback + 1) as f64;
            let norm_atr = if avg_price > 1e-10 { avg_atr / avg_price } else { 0.0 };

            // Volume component
            let avg_vol = volume[start..i].iter().sum::<f64>() / self.lookback as f64;
            let vol_ratio = if avg_vol > 1e-10 { volume[i] / avg_vol } else { 1.0 };
            let vol_component = (vol_ratio - 1.0).max(0.0).min(1.0);

            // Trend consistency component
            let mut up_moves = 0;
            let mut down_moves = 0;
            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    up_moves += 1;
                } else if close[j] < close[j - 1] {
                    down_moves += 1;
                }
            }
            let consistency = (up_moves as f64 - down_moves as f64).abs() / self.lookback as f64;

            // Combine components into strength score
            let momentum_score = (price_change * 10.0).min(1.0);
            let volatility_score = (norm_atr * 20.0).min(1.0);

            let strength = (momentum_score * 0.35 + volatility_score * 0.25 +
                           vol_component * 0.2 + consistency * 0.2).min(1.0);

            result[i] = strength;
        }

        result
    }
}

impl TechnicalIndicator for PatternStrength {
    fn name(&self) -> &str {
        "Pattern Strength"
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close, &data.volume)))
    }
}

// ============================================================================
// PatternProbability
// ============================================================================

/// Pattern Probability - Probability of pattern completion
///
/// Calculates the probability that a detected pattern will complete based on
/// historical pattern success rates and current market conditions.
#[derive(Debug, Clone)]
pub struct PatternProbability {
    /// Lookback period for probability calculation
    lookback: usize,
    /// Minimum pattern occurrences for reliable probability
    min_occurrences: usize,
}

impl PatternProbability {
    /// Create a new PatternProbability indicator.
    ///
    /// # Arguments
    /// * `lookback` - Lookback period for probability calculation (5-200)
    /// * `min_occurrences` - Minimum pattern occurrences required (1-20)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(lookback: usize, min_occurrences: usize) -> Result<Self> {
        if lookback < 5 || lookback > 200 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be between 5 and 200".to_string(),
            });
        }
        if min_occurrences < 1 || min_occurrences > 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "min_occurrences".to_string(),
                reason: "must be between 1 and 20".to_string(),
            });
        }
        Ok(Self { lookback, min_occurrences })
    }

    /// Calculate pattern completion probability.
    ///
    /// Returns probability between 0 and 1 based on:
    /// * Historical pattern completion rate
    /// * Current trend alignment
    /// * Volume confirmation
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.lookback..n {
            let start = i - self.lookback;

            // Detect higher highs/higher lows (bullish) or lower highs/lower lows (bearish)
            let mut bullish_patterns = 0;
            let mut bullish_completions = 0;
            let mut bearish_patterns = 0;
            let mut bearish_completions = 0;

            for j in (start + 2)..i {
                // Bullish pattern: higher high and higher low
                if high[j] > high[j - 1] && low[j] > low[j - 1] {
                    bullish_patterns += 1;
                    // Check if pattern completed (price continued higher)
                    if j + 1 < i && close[j + 1] > close[j] {
                        bullish_completions += 1;
                    }
                }

                // Bearish pattern: lower high and lower low
                if high[j] < high[j - 1] && low[j] < low[j - 1] {
                    bearish_patterns += 1;
                    // Check if pattern completed (price continued lower)
                    if j + 1 < i && close[j + 1] < close[j] {
                        bearish_completions += 1;
                    }
                }
            }

            // Calculate probability based on current price action
            let is_bullish_setup = high[i] > high[i - 1] && low[i] > low[i - 1];
            let is_bearish_setup = high[i] < high[i - 1] && low[i] < low[i - 1];

            if is_bullish_setup && bullish_patterns >= self.min_occurrences {
                result[i] = bullish_completions as f64 / bullish_patterns as f64;
            } else if is_bearish_setup && bearish_patterns >= self.min_occurrences {
                result[i] = bearish_completions as f64 / bearish_patterns as f64;
            } else {
                // No clear pattern, use neutral probability
                let total_patterns = bullish_patterns + bearish_patterns;
                let total_completions = bullish_completions + bearish_completions;
                if total_patterns >= self.min_occurrences {
                    result[i] = total_completions as f64 / total_patterns as f64;
                } else {
                    result[i] = 0.5; // Neutral when insufficient data
                }
            }
        }

        result
    }
}

impl TechnicalIndicator for PatternProbability {
    fn name(&self) -> &str {
        "Pattern Probability"
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

// ============================================================================
// MultiTimeframePattern
// ============================================================================

/// Multi-Timeframe Pattern - Detects patterns across timeframes
///
/// Analyzes patterns across multiple simulated timeframes by using different
/// lookback periods to identify confluence and stronger signals.
#[derive(Debug, Clone)]
pub struct MultiTimeframePattern {
    /// Short timeframe period
    short_period: usize,
    /// Long timeframe period
    long_period: usize,
    /// Minimum timeframe agreement for signal
    min_agreement: usize,
}

impl MultiTimeframePattern {
    /// Create a new MultiTimeframePattern indicator.
    ///
    /// # Arguments
    /// * `short_period` - Short timeframe period (3-20)
    /// * `long_period` - Long timeframe period (10-200)
    /// * `min_agreement` - Minimum timeframes that must agree (1-5)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(short_period: usize, long_period: usize, min_agreement: usize) -> Result<Self> {
        if short_period < 3 || short_period > 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be between 3 and 20".to_string(),
            });
        }
        if long_period < 10 || long_period > 200 {
            return Err(IndicatorError::InvalidParameter {
                name: "long_period".to_string(),
                reason: "must be between 10 and 200".to_string(),
            });
        }
        if short_period >= long_period {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be less than long_period".to_string(),
            });
        }
        if min_agreement < 1 || min_agreement > 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "min_agreement".to_string(),
                reason: "must be between 1 and 5".to_string(),
            });
        }
        Ok(Self { short_period, long_period, min_agreement })
    }

    /// Calculate multi-timeframe pattern signals.
    ///
    /// Returns:
    /// * +1: Bullish pattern confirmed across timeframes
    /// * -1: Bearish pattern confirmed across timeframes
    /// * 0: No confirmed pattern or conflicting signals
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // Create multiple timeframe periods
        let mid_period = (self.short_period + self.long_period) / 2;
        let periods = [self.short_period, mid_period, self.long_period];

        for i in self.long_period..n {
            let mut bullish_count = 0;
            let mut bearish_count = 0;

            for &period in &periods {
                let start = i - period;

                // Calculate trend direction for this timeframe
                let period_high = high[start..=i].iter().cloned().fold(f64::MIN, f64::max);
                let period_low = low[start..=i].iter().cloned().fold(f64::MAX, f64::min);
                let mid_price = (period_high + period_low) / 2.0;

                // Check if price is in upper or lower half of range
                let in_upper_half = close[i] > mid_price;

                // Check trend direction (higher highs/lows vs lower highs/lows)
                let recent_start = i.saturating_sub(period / 2);
                let recent_high = high[recent_start..=i].iter().cloned().fold(f64::MIN, f64::max);
                let recent_low = low[recent_start..=i].iter().cloned().fold(f64::MAX, f64::min);

                let earlier_end = recent_start;
                let earlier_start = start;
                if earlier_end > earlier_start {
                    let earlier_high = high[earlier_start..earlier_end].iter().cloned().fold(f64::MIN, f64::max);
                    let earlier_low = low[earlier_start..earlier_end].iter().cloned().fold(f64::MAX, f64::min);

                    // Bullish: higher highs and higher lows, price in upper half
                    if recent_high > earlier_high && recent_low > earlier_low && in_upper_half {
                        bullish_count += 1;
                    }
                    // Bearish: lower highs and lower lows, price in lower half
                    else if recent_high < earlier_high && recent_low < earlier_low && !in_upper_half {
                        bearish_count += 1;
                    }
                }
            }

            // Require minimum agreement across timeframes
            if bullish_count >= self.min_agreement {
                result[i] = 1.0;
            } else if bearish_count >= self.min_agreement {
                result[i] = -1.0;
            }
        }

        result
    }
}

impl TechnicalIndicator for MultiTimeframePattern {
    fn name(&self) -> &str {
        "Multi-Timeframe Pattern"
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

// ============================================================================
// PatternCluster
// ============================================================================

/// Pattern Cluster - Identifies clusters of related patterns
///
/// Detects when multiple pattern signals occur in close proximity, indicating
/// stronger support/resistance zones or trend confirmation.
#[derive(Debug, Clone)]
pub struct PatternCluster {
    /// Lookback period for cluster detection
    lookback: usize,
    /// Maximum distance (% of price) for patterns to cluster
    cluster_distance: f64,
    /// Minimum patterns required in cluster
    min_patterns: usize,
}

impl PatternCluster {
    /// Create a new PatternCluster indicator.
    ///
    /// # Arguments
    /// * `lookback` - Lookback period for cluster detection (5-100)
    /// * `cluster_distance` - Max distance for clustering (0.1-10.0%)
    /// * `min_patterns` - Minimum patterns in cluster (2-10)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(lookback: usize, cluster_distance: f64, min_patterns: usize) -> Result<Self> {
        if lookback < 5 || lookback > 100 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be between 5 and 100".to_string(),
            });
        }
        if cluster_distance < 0.1 || cluster_distance > 10.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "cluster_distance".to_string(),
                reason: "must be between 0.1 and 10.0".to_string(),
            });
        }
        if min_patterns < 2 || min_patterns > 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "min_patterns".to_string(),
                reason: "must be between 2 and 10".to_string(),
            });
        }
        Ok(Self { lookback, cluster_distance, min_patterns })
    }

    /// Calculate pattern cluster signals.
    ///
    /// Returns:
    /// * +1: Bullish pattern cluster detected
    /// * -1: Bearish pattern cluster detected
    /// * 0: No significant cluster
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // First pass: detect individual pattern signals
        let mut bullish_levels: Vec<Vec<f64>> = vec![Vec::new(); n];
        let mut bearish_levels: Vec<Vec<f64>> = vec![Vec::new(); n];

        for i in 2..n {
            // Detect swing lows (potential support / bullish reversal)
            if low[i - 1] < low[i - 2] && low[i - 1] < low[i] {
                bullish_levels[i].push(low[i - 1]);
            }

            // Detect swing highs (potential resistance / bearish reversal)
            if high[i - 1] > high[i - 2] && high[i - 1] > high[i] {
                bearish_levels[i].push(high[i - 1]);
            }

            // Volume spike with bullish close
            if i >= 5 {
                let avg_vol: f64 = volume[(i - 5)..i].iter().sum::<f64>() / 5.0;
                if avg_vol > 1e-10 && volume[i] > avg_vol * 1.5 {
                    if close[i] > close[i - 1] {
                        bullish_levels[i].push(close[i]);
                    } else if close[i] < close[i - 1] {
                        bearish_levels[i].push(close[i]);
                    }
                }
            }
        }

        // Second pass: detect clusters
        for i in self.lookback..n {
            let start = i - self.lookback;
            let current_price = close[i];
            let distance_threshold = current_price * self.cluster_distance / 100.0;

            // Collect all bullish levels in lookback
            let mut all_bullish: Vec<f64> = Vec::new();
            let mut all_bearish: Vec<f64> = Vec::new();

            for j in start..=i {
                all_bullish.extend(&bullish_levels[j]);
                all_bearish.extend(&bearish_levels[j]);
            }

            // Count levels near current price
            let bullish_cluster: usize = all_bullish
                .iter()
                .filter(|&&level| (level - current_price).abs() <= distance_threshold)
                .count();

            let bearish_cluster: usize = all_bearish
                .iter()
                .filter(|&&level| (level - current_price).abs() <= distance_threshold)
                .count();

            // Signal if cluster threshold is met
            if bullish_cluster >= self.min_patterns && bullish_cluster > bearish_cluster {
                result[i] = 1.0;
            } else if bearish_cluster >= self.min_patterns && bearish_cluster > bullish_cluster {
                result[i] = -1.0;
            }
        }

        result
    }
}

impl TechnicalIndicator for PatternCluster {
    fn name(&self) -> &str {
        "Pattern Cluster"
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close, &data.volume)))
    }
}

// ============================================================================
// SequentialPattern
// ============================================================================

/// Sequential Pattern - Detects sequential pattern formations
///
/// Identifies TD Sequential-style patterns where consecutive closes above/below
/// prior closes indicate exhaustion or continuation.
#[derive(Debug, Clone)]
pub struct SequentialPattern {
    /// Number of consecutive closes required for setup
    setup_count: usize,
    /// Bars back to compare for sequential close
    comparison_bars: usize,
}

impl SequentialPattern {
    /// Create a new SequentialPattern indicator.
    ///
    /// # Arguments
    /// * `setup_count` - Number of consecutive closes for setup (5-20)
    /// * `comparison_bars` - Bars back to compare (1-10)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(setup_count: usize, comparison_bars: usize) -> Result<Self> {
        if setup_count < 5 || setup_count > 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "setup_count".to_string(),
                reason: "must be between 5 and 20".to_string(),
            });
        }
        if comparison_bars < 1 || comparison_bars > 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "comparison_bars".to_string(),
                reason: "must be between 1 and 10".to_string(),
            });
        }
        Ok(Self { setup_count, comparison_bars })
    }

    /// Calculate sequential pattern signals.
    ///
    /// Returns normalized count:
    /// * Positive values (0 to 1): Bullish setup count / setup_count
    /// * Negative values (-1 to 0): Bearish setup count / setup_count
    /// * Values at +/-1 indicate completed setup
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        let min_idx = self.comparison_bars;
        if n <= min_idx {
            return result;
        }

        let mut bullish_count = 0;
        let mut bearish_count = 0;

        for i in min_idx..n {
            let compare_idx = i - self.comparison_bars;

            // Check for bullish sequential (close > close N bars ago)
            if close[i] > close[compare_idx] {
                bullish_count += 1;
                bearish_count = 0;
            }
            // Check for bearish sequential (close < close N bars ago)
            else if close[i] < close[compare_idx] {
                bearish_count += 1;
                bullish_count = 0;
            }
            // Reset on no change
            else {
                bullish_count = 0;
                bearish_count = 0;
            }

            // Cap counts at setup_count
            bullish_count = bullish_count.min(self.setup_count);
            bearish_count = bearish_count.min(self.setup_count);

            // Output normalized count
            if bullish_count > 0 {
                result[i] = bullish_count as f64 / self.setup_count as f64;
            } else if bearish_count > 0 {
                result[i] = -(bearish_count as f64 / self.setup_count as f64);
            }
        }

        result
    }
}

impl TechnicalIndicator for SequentialPattern {
    fn name(&self) -> &str {
        "Sequential Pattern"
    }

    fn min_periods(&self) -> usize {
        self.setup_count + self.comparison_bars
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

// ============================================================================
// PatternBreakoutStrength
// ============================================================================

/// Pattern Breakout Strength - Strength of pattern breakout signals
///
/// Measures the strength of breakouts from pattern formations using price
/// momentum, volume surge, and ATR expansion.
#[derive(Debug, Clone)]
pub struct PatternBreakoutStrength {
    /// Lookback period for breakout analysis
    lookback: usize,
    /// ATR multiplier for breakout confirmation
    atr_multiplier: f64,
    /// Volume multiplier for confirmation
    volume_multiplier: f64,
}

impl PatternBreakoutStrength {
    /// Create a new PatternBreakoutStrength indicator.
    ///
    /// # Arguments
    /// * `lookback` - Lookback period for analysis (5-100)
    /// * `atr_multiplier` - ATR multiplier for breakout (0.5-5.0)
    /// * `volume_multiplier` - Volume multiplier for confirmation (1.0-5.0)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(lookback: usize, atr_multiplier: f64, volume_multiplier: f64) -> Result<Self> {
        if lookback < 5 || lookback > 100 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be between 5 and 100".to_string(),
            });
        }
        if atr_multiplier < 0.5 || atr_multiplier > 5.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "atr_multiplier".to_string(),
                reason: "must be between 0.5 and 5.0".to_string(),
            });
        }
        if volume_multiplier < 1.0 || volume_multiplier > 5.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "volume_multiplier".to_string(),
                reason: "must be between 1.0 and 5.0".to_string(),
            });
        }
        Ok(Self { lookback, atr_multiplier, volume_multiplier })
    }

    /// Calculate pattern breakout strength.
    ///
    /// Returns strength between -1 and 1:
    /// * Positive values: Bullish breakout strength
    /// * Negative values: Bearish breakout strength
    /// * Values closer to +/-1 indicate stronger breakouts
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.lookback..n {
            let start = i - self.lookback;

            // Calculate ATR
            let mut atr_sum = 0.0;
            for j in (start + 1)..i {
                let tr = (high[j] - low[j])
                    .max((high[j] - close[j - 1]).abs())
                    .max((low[j] - close[j - 1]).abs());
                atr_sum += tr;
            }
            let atr = atr_sum / (self.lookback - 1) as f64;

            // Find consolidation range
            let range_high = high[start..i].iter().cloned().fold(f64::MIN, f64::max);
            let range_low = low[start..i].iter().cloned().fold(f64::MAX, f64::min);

            // Calculate average volume
            let avg_vol = volume[start..i].iter().sum::<f64>() / self.lookback as f64;

            // Calculate current bar's range
            let current_range = high[i] - low[i];
            let atr_expansion = if atr > 1e-10 { current_range / atr } else { 1.0 };

            // Volume ratio
            let vol_ratio = if avg_vol > 1e-10 { volume[i] / avg_vol } else { 1.0 };

            // Determine breakout direction and strength
            let breakout_threshold = atr * self.atr_multiplier;

            // Bullish breakout: close above range high with ATR expansion
            if close[i] > range_high + breakout_threshold {
                let price_strength = ((close[i] - range_high) / atr).min(2.0) / 2.0;
                let atr_strength = (atr_expansion / self.atr_multiplier).min(1.0);
                let vol_strength = if vol_ratio >= self.volume_multiplier { 1.0 } else { vol_ratio / self.volume_multiplier };

                let strength = (price_strength * 0.4 + atr_strength * 0.3 + vol_strength * 0.3).min(1.0);
                result[i] = strength;
            }
            // Bearish breakout: close below range low with ATR expansion
            else if close[i] < range_low - breakout_threshold {
                let price_strength = ((range_low - close[i]) / atr).min(2.0) / 2.0;
                let atr_strength = (atr_expansion / self.atr_multiplier).min(1.0);
                let vol_strength = if vol_ratio >= self.volume_multiplier { 1.0 } else { vol_ratio / self.volume_multiplier };

                let strength = (price_strength * 0.4 + atr_strength * 0.3 + vol_strength * 0.3).min(1.0);
                result[i] = -strength;
            }
        }

        result
    }
}

impl TechnicalIndicator for PatternBreakoutStrength {
    fn name(&self) -> &str {
        "Pattern Breakout Strength"
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close, &data.volume)))
    }
}

// ============================================================================
// PricePatternRecognizer
// ============================================================================

/// Price Pattern Recognizer - Recognizes common price patterns
///
/// Identifies classic chart patterns including flags, wedges, triangles,
/// and rectangles by analyzing price structure and trend characteristics.
#[derive(Debug, Clone)]
pub struct PricePatternRecognizer {
    /// Lookback period for pattern detection
    lookback: usize,
    /// Minimum points required to confirm pattern
    min_points: usize,
    /// Tolerance for pattern matching (percentage)
    tolerance: f64,
}

impl PricePatternRecognizer {
    /// Create a new PricePatternRecognizer indicator.
    ///
    /// # Arguments
    /// * `lookback` - Lookback period for pattern detection (10-100)
    /// * `min_points` - Minimum swing points for pattern (3-10)
    /// * `tolerance` - Pattern matching tolerance percentage (0.5-10.0)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(lookback: usize, min_points: usize, tolerance: f64) -> Result<Self> {
        if lookback < 10 || lookback > 100 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be between 10 and 100".to_string(),
            });
        }
        if min_points < 3 || min_points > 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "min_points".to_string(),
                reason: "must be between 3 and 10".to_string(),
            });
        }
        if tolerance < 0.5 || tolerance > 10.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "tolerance".to_string(),
                reason: "must be between 0.5 and 10.0".to_string(),
            });
        }
        Ok(Self { lookback, min_points, tolerance })
    }

    /// Calculate pattern recognition signals.
    ///
    /// Returns pattern type encoded as:
    /// * 1.0: Bullish flag pattern
    /// * 2.0: Bearish flag pattern
    /// * 3.0: Ascending wedge
    /// * 4.0: Descending wedge
    /// * 5.0: Symmetrical triangle
    /// * 0.0: No recognized pattern
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.lookback..n {
            let start = i - self.lookback;

            // Find swing highs and lows
            let mut swing_highs: Vec<(usize, f64)> = Vec::new();
            let mut swing_lows: Vec<(usize, f64)> = Vec::new();

            for j in (start + 1)..(i - 1) {
                // Swing high: higher than neighbors
                if high[j] > high[j - 1] && high[j] > high[j + 1] {
                    swing_highs.push((j, high[j]));
                }
                // Swing low: lower than neighbors
                if low[j] < low[j - 1] && low[j] < low[j + 1] {
                    swing_lows.push((j, low[j]));
                }
            }

            // Need minimum points for pattern detection
            if swing_highs.len() < self.min_points || swing_lows.len() < self.min_points {
                continue;
            }

            // Calculate trend lines for highs and lows
            let avg_price = close[start..=i].iter().sum::<f64>() / (self.lookback + 1) as f64;
            let tolerance_abs = avg_price * self.tolerance / 100.0;

            // Analyze high trend (ascending, descending, flat)
            let high_slope = if swing_highs.len() >= 2 {
                let first = swing_highs.first().unwrap();
                let last = swing_highs.last().unwrap();
                if last.0 > first.0 {
                    (last.1 - first.1) / (last.0 - first.0) as f64
                } else {
                    0.0
                }
            } else {
                0.0
            };

            // Analyze low trend (ascending, descending, flat)
            let low_slope = if swing_lows.len() >= 2 {
                let first = swing_lows.first().unwrap();
                let last = swing_lows.last().unwrap();
                if last.0 > first.0 {
                    (last.1 - first.1) / (last.0 - first.0) as f64
                } else {
                    0.0
                }
            } else {
                0.0
            };

            // Normalize slopes relative to price
            let norm_high_slope = high_slope / avg_price * 100.0;
            let norm_low_slope = low_slope / avg_price * 100.0;

            // Pattern detection based on slope characteristics
            // Bullish flag: both lines descending (consolidation in uptrend)
            if norm_high_slope < -0.1 && norm_low_slope < -0.1 {
                // Check for prior uptrend
                let prior_change = if start >= 5 {
                    (close[start] - close[start - 5]) / close[start - 5] * 100.0
                } else {
                    0.0
                };
                if prior_change > 2.0 {
                    result[i] = 1.0;
                    continue;
                }
            }

            // Bearish flag: both lines ascending (consolidation in downtrend)
            if norm_high_slope > 0.1 && norm_low_slope > 0.1 {
                let prior_change = if start >= 5 {
                    (close[start] - close[start - 5]) / close[start - 5] * 100.0
                } else {
                    0.0
                };
                if prior_change < -2.0 {
                    result[i] = 2.0;
                    continue;
                }
            }

            // Ascending wedge: highs converging, lows rising
            if norm_high_slope > 0.0 && norm_low_slope > 0.0 && norm_low_slope > norm_high_slope {
                let range_start = swing_highs.first().map(|h| h.1).unwrap_or(0.0)
                    - swing_lows.first().map(|l| l.1).unwrap_or(0.0);
                let range_end = swing_highs.last().map(|h| h.1).unwrap_or(0.0)
                    - swing_lows.last().map(|l| l.1).unwrap_or(0.0);
                if range_end < range_start * 0.7 {
                    result[i] = 3.0;
                    continue;
                }
            }

            // Descending wedge: lows converging, highs falling
            if norm_high_slope < 0.0 && norm_low_slope < 0.0 && norm_high_slope > norm_low_slope {
                let range_start = swing_highs.first().map(|h| h.1).unwrap_or(0.0)
                    - swing_lows.first().map(|l| l.1).unwrap_or(0.0);
                let range_end = swing_highs.last().map(|h| h.1).unwrap_or(0.0)
                    - swing_lows.last().map(|l| l.1).unwrap_or(0.0);
                if range_end < range_start * 0.7 {
                    result[i] = 4.0;
                    continue;
                }
            }

            // Symmetrical triangle: converging highs and lows from opposite directions
            if (norm_high_slope < -0.05 && norm_low_slope > 0.05)
                || (norm_high_slope.abs() < tolerance_abs / avg_price * 10.0
                    && norm_low_slope.abs() < tolerance_abs / avg_price * 10.0)
            {
                let range_start = swing_highs.first().map(|h| h.1).unwrap_or(0.0)
                    - swing_lows.first().map(|l| l.1).unwrap_or(0.0);
                let range_end = swing_highs.last().map(|h| h.1).unwrap_or(0.0)
                    - swing_lows.last().map(|l| l.1).unwrap_or(0.0);
                if range_end < range_start * 0.6 {
                    result[i] = 5.0;
                }
            }
        }

        result
    }
}

impl TechnicalIndicator for PricePatternRecognizer {
    fn name(&self) -> &str {
        "Price Pattern Recognizer"
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

// ============================================================================
// ConsolidationDetector
// ============================================================================

/// Consolidation Detector - Detects price consolidation periods
///
/// Identifies periods where price is consolidating within a range,
/// characterized by decreasing volatility and range contraction.
#[derive(Debug, Clone)]
pub struct ConsolidationDetector {
    /// Lookback period for consolidation detection
    lookback: usize,
    /// Maximum range as percentage for consolidation
    max_range_pct: f64,
    /// ATR period for volatility measurement
    atr_period: usize,
}

impl ConsolidationDetector {
    /// Create a new ConsolidationDetector indicator.
    ///
    /// # Arguments
    /// * `lookback` - Lookback period for consolidation detection (5-50)
    /// * `max_range_pct` - Maximum range percentage for consolidation (1.0-15.0)
    /// * `atr_period` - ATR period for volatility measurement (5-30)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(lookback: usize, max_range_pct: f64, atr_period: usize) -> Result<Self> {
        if lookback < 5 || lookback > 50 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be between 5 and 50".to_string(),
            });
        }
        if max_range_pct < 1.0 || max_range_pct > 15.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "max_range_pct".to_string(),
                reason: "must be between 1.0 and 15.0".to_string(),
            });
        }
        if atr_period < 5 || atr_period > 30 {
            return Err(IndicatorError::InvalidParameter {
                name: "atr_period".to_string(),
                reason: "must be between 5 and 30".to_string(),
            });
        }
        Ok(Self { lookback, max_range_pct, atr_period })
    }

    /// Calculate consolidation detection signals.
    ///
    /// Returns consolidation strength between 0 and 1:
    /// * 0.0: No consolidation
    /// * 0.0-0.5: Weak consolidation
    /// * 0.5-0.8: Moderate consolidation
    /// * 0.8-1.0: Strong consolidation (tight range)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];
        let min_period = self.lookback.max(self.atr_period);

        for i in min_period..n {
            let start = i - self.lookback;

            // Calculate range in lookback period
            let period_high = high[start..=i].iter().cloned().fold(f64::MIN, f64::max);
            let period_low = low[start..=i].iter().cloned().fold(f64::MAX, f64::min);
            let range = period_high - period_low;

            // Calculate average price for percentage
            let avg_price: f64 = close[start..=i].iter().sum::<f64>() / (self.lookback + 1) as f64;

            if avg_price < 1e-10 {
                continue;
            }

            let range_pct = (range / avg_price) * 100.0;

            // Calculate ATR for volatility comparison
            let atr_start = i.saturating_sub(self.atr_period);
            let mut atr_sum = 0.0;
            for j in (atr_start + 1)..=i {
                let tr = (high[j] - low[j])
                    .max((high[j] - close[j - 1]).abs())
                    .max((low[j] - close[j - 1]).abs());
                atr_sum += tr;
            }
            let atr = atr_sum / self.atr_period as f64;
            let atr_pct = (atr / avg_price) * 100.0;

            // Calculate historical ATR for comparison
            let hist_atr_start = i.saturating_sub(self.lookback);
            let hist_atr_end = i.saturating_sub(self.atr_period);
            let mut hist_atr_sum = 0.0;
            let mut hist_count = 0;
            for j in (hist_atr_start + 1)..=hist_atr_end {
                if j > 0 {
                    let tr = (high[j] - low[j])
                        .max((high[j] - close[j - 1]).abs())
                        .max((low[j] - close[j - 1]).abs());
                    hist_atr_sum += tr;
                    hist_count += 1;
                }
            }
            let hist_atr = if hist_count > 0 { hist_atr_sum / hist_count as f64 } else { atr };
            let hist_atr_pct = (hist_atr / avg_price) * 100.0;

            // Check if in consolidation
            if range_pct <= self.max_range_pct {
                // Calculate consolidation strength based on:
                // 1. Range tightness (how close to min range)
                let range_score = 1.0 - (range_pct / self.max_range_pct);

                // 2. Volatility contraction (current ATR vs historical)
                let vol_score = if hist_atr_pct > 1e-10 {
                    (1.0 - atr_pct / hist_atr_pct).max(0.0).min(1.0)
                } else {
                    0.5
                };

                // 3. Price clustering around mean
                let close_deviation = (close[i] - avg_price).abs() / avg_price;
                let cluster_score = (1.0 - close_deviation * 10.0).max(0.0).min(1.0);

                // Combine scores
                let strength = (range_score * 0.4 + vol_score * 0.35 + cluster_score * 0.25).min(1.0);
                result[i] = strength;
            }
        }

        result
    }
}

impl TechnicalIndicator for ConsolidationDetector {
    fn name(&self) -> &str {
        "Consolidation Detector"
    }

    fn min_periods(&self) -> usize {
        self.lookback.max(self.atr_period) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

// ============================================================================
// BreakoutPatternStrength
// ============================================================================

/// Breakout Pattern Strength - Measures strength of breakout patterns
///
/// Evaluates the quality and strength of breakout patterns based on
/// price momentum, volume confirmation, and technical factors.
#[derive(Debug, Clone)]
pub struct BreakoutPatternStrength {
    /// Lookback period for breakout analysis
    lookback: usize,
    /// Volume confirmation multiplier
    volume_multiplier: f64,
    /// ATR multiplier for breakout threshold
    atr_multiplier: f64,
}

impl BreakoutPatternStrength {
    /// Create a new BreakoutPatternStrength indicator.
    ///
    /// # Arguments
    /// * `lookback` - Lookback period for breakout analysis (5-50)
    /// * `volume_multiplier` - Volume multiplier for confirmation (1.0-5.0)
    /// * `atr_multiplier` - ATR multiplier for breakout threshold (0.5-3.0)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(lookback: usize, volume_multiplier: f64, atr_multiplier: f64) -> Result<Self> {
        if lookback < 5 || lookback > 50 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be between 5 and 50".to_string(),
            });
        }
        if volume_multiplier < 1.0 || volume_multiplier > 5.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "volume_multiplier".to_string(),
                reason: "must be between 1.0 and 5.0".to_string(),
            });
        }
        if atr_multiplier < 0.5 || atr_multiplier > 3.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "atr_multiplier".to_string(),
                reason: "must be between 0.5 and 3.0".to_string(),
            });
        }
        Ok(Self { lookback, volume_multiplier, atr_multiplier })
    }

    /// Calculate breakout pattern strength.
    ///
    /// Returns strength between -1 and 1:
    /// * Positive values (0 to 1): Bullish breakout strength
    /// * Negative values (-1 to 0): Bearish breakout strength
    /// * 0: No breakout detected
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.lookback..n {
            let start = i - self.lookback;

            // Calculate range boundaries
            let range_high = high[start..i].iter().cloned().fold(f64::MIN, f64::max);
            let range_low = low[start..i].iter().cloned().fold(f64::MAX, f64::min);

            // Calculate ATR
            let mut atr_sum = 0.0;
            for j in (start + 1)..i {
                let tr = (high[j] - low[j])
                    .max((high[j] - close[j - 1]).abs())
                    .max((low[j] - close[j - 1]).abs());
                atr_sum += tr;
            }
            let atr = atr_sum / (self.lookback - 1) as f64;
            let breakout_threshold = atr * self.atr_multiplier;

            // Calculate average volume
            let avg_vol = volume[start..i].iter().sum::<f64>() / self.lookback as f64;

            // Volume ratio
            let vol_ratio = if avg_vol > 1e-10 { volume[i] / avg_vol } else { 1.0 };

            // Detect breakout direction
            let is_bullish_breakout = close[i] > range_high + breakout_threshold;
            let is_bearish_breakout = close[i] < range_low - breakout_threshold;

            if !is_bullish_breakout && !is_bearish_breakout {
                continue;
            }

            // Calculate strength components
            // 1. Price momentum (how far beyond breakout level)
            let price_momentum = if is_bullish_breakout {
                ((close[i] - range_high) / atr).min(2.0) / 2.0
            } else {
                ((range_low - close[i]) / atr).min(2.0) / 2.0
            };

            // 2. Volume confirmation
            let volume_score = if vol_ratio >= self.volume_multiplier {
                1.0
            } else {
                (vol_ratio / self.volume_multiplier).min(1.0)
            };

            // 3. Close position in bar (close near high for bullish, near low for bearish)
            let bar_range = high[i] - low[i];
            let close_position = if bar_range > 1e-10 {
                if is_bullish_breakout {
                    (close[i] - low[i]) / bar_range
                } else {
                    (high[i] - close[i]) / bar_range
                }
            } else {
                0.5
            };

            // 4. Consolidation quality (tighter consolidation = stronger breakout)
            let avg_price = close[start..i].iter().sum::<f64>() / self.lookback as f64;
            let range_pct = if avg_price > 1e-10 { (range_high - range_low) / avg_price } else { 0.1 };
            let consolidation_score = (1.0 - range_pct * 5.0).max(0.0).min(1.0);

            // Combine scores
            let strength = (price_momentum * 0.3 + volume_score * 0.25 +
                           close_position * 0.25 + consolidation_score * 0.2).min(1.0);

            result[i] = if is_bullish_breakout { strength } else { -strength };
        }

        result
    }
}

impl TechnicalIndicator for BreakoutPatternStrength {
    fn name(&self) -> &str {
        "Breakout Pattern Strength"
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close, &data.volume)))
    }
}

// ============================================================================
// ReversalPatternScore
// ============================================================================

/// Reversal Pattern Score - Scores reversal pattern quality
///
/// Calculates a quality score for potential reversal patterns based on
/// multiple technical factors including divergence, exhaustion, and structure.
#[derive(Debug, Clone)]
pub struct ReversalPatternScore {
    /// Lookback period for reversal analysis
    lookback: usize,
    /// Momentum period for divergence detection
    momentum_period: usize,
    /// Minimum score threshold for valid reversal
    min_score: f64,
}

impl ReversalPatternScore {
    /// Create a new ReversalPatternScore indicator.
    ///
    /// # Arguments
    /// * `lookback` - Lookback period for reversal analysis (10-50)
    /// * `momentum_period` - Momentum period for divergence (5-20)
    /// * `min_score` - Minimum score for valid reversal (0.3-0.9)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(lookback: usize, momentum_period: usize, min_score: f64) -> Result<Self> {
        if lookback < 10 || lookback > 50 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be between 10 and 50".to_string(),
            });
        }
        if momentum_period < 5 || momentum_period > 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be between 5 and 20".to_string(),
            });
        }
        if momentum_period >= lookback {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be less than lookback".to_string(),
            });
        }
        if min_score < 0.3 || min_score > 0.9 {
            return Err(IndicatorError::InvalidParameter {
                name: "min_score".to_string(),
                reason: "must be between 0.3 and 0.9".to_string(),
            });
        }
        Ok(Self { lookback, momentum_period, min_score })
    }

    /// Calculate reversal pattern score.
    ///
    /// Returns score between -1 and 1:
    /// * Positive values: Bullish reversal score (potential bottom)
    /// * Negative values: Bearish reversal score (potential top)
    /// * Values closer to +/-1 indicate stronger reversal signals
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.lookback..n {
            let start = i - self.lookback;

            // Find extremes in lookback period
            let period_high = high[start..=i].iter().cloned().fold(f64::MIN, f64::max);
            let period_low = low[start..=i].iter().cloned().fold(f64::MAX, f64::min);
            let range = period_high - period_low;

            if range < 1e-10 {
                continue;
            }

            // Calculate price position in range
            let price_position = (close[i] - period_low) / range;

            // Calculate momentum (rate of change)
            let mom_base = close[i - self.momentum_period];
            let momentum = if mom_base > 1e-10 {
                (close[i] / mom_base - 1.0) * 100.0
            } else {
                0.0
            };

            // Check for potential reversals
            let near_low = price_position < 0.25;
            let near_high = price_position > 0.75;

            if !near_low && !near_high {
                continue;
            }

            // Calculate reversal factors
            // 1. Exhaustion: strong prior move followed by stalling
            let prior_momentum = if i >= self.momentum_period * 2 {
                let prior_base = close[i - self.momentum_period * 2];
                if prior_base > 1e-10 {
                    (close[i - self.momentum_period] / prior_base - 1.0) * 100.0
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let exhaustion_score = if near_low {
                // For bullish reversal: prior was bearish, current momentum improving
                if prior_momentum < -1.0 && momentum > prior_momentum {
                    ((momentum - prior_momentum).abs() / 10.0).min(1.0)
                } else {
                    0.0
                }
            } else {
                // For bearish reversal: prior was bullish, current momentum declining
                if prior_momentum > 1.0 && momentum < prior_momentum {
                    ((prior_momentum - momentum).abs() / 10.0).min(1.0)
                } else {
                    0.0
                }
            };

            // 2. Volume climax (high volume at extreme)
            let avg_vol = volume[start..i].iter().sum::<f64>() / self.lookback as f64;
            let vol_ratio = if avg_vol > 1e-10 { volume[i] / avg_vol } else { 1.0 };
            let volume_score = (vol_ratio / 2.0).min(1.0);

            // 3. Candle structure (reversal candle patterns)
            let body = (close[i] - close[i - 1]).abs();
            let bar_range = high[i] - low[i];
            let candle_score = if bar_range > 1e-10 {
                if near_low && close[i] > close[i - 1] {
                    // Bullish candle at low
                    (body / bar_range).min(1.0)
                } else if near_high && close[i] < close[i - 1] {
                    // Bearish candle at high
                    (body / bar_range).min(1.0)
                } else {
                    0.0
                }
            } else {
                0.0
            };

            // 4. Position score (closer to extreme = higher score)
            let position_score = if near_low {
                (0.25 - price_position) * 4.0
            } else {
                (price_position - 0.75) * 4.0
            };

            // Combine scores
            let total_score = (exhaustion_score * 0.3 + volume_score * 0.25 +
                              candle_score * 0.25 + position_score * 0.2).min(1.0);

            if total_score >= self.min_score {
                result[i] = if near_low { total_score } else { -total_score };
            }
        }

        result
    }
}

impl TechnicalIndicator for ReversalPatternScore {
    fn name(&self) -> &str {
        "Reversal Pattern Score"
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close, &data.volume)))
    }
}

// ============================================================================
// PatternSymmetry
// ============================================================================

/// Pattern Symmetry - Measures symmetry in price patterns
///
/// Analyzes the symmetry of price movements, useful for identifying
/// balanced patterns like head and shoulders, double tops/bottoms, and
/// symmetric triangles.
#[derive(Debug, Clone)]
pub struct PatternSymmetry {
    /// Lookback period for symmetry analysis
    lookback: usize,
    /// Tolerance for symmetry matching (percentage)
    tolerance: f64,
}

impl PatternSymmetry {
    /// Create a new PatternSymmetry indicator.
    ///
    /// # Arguments
    /// * `lookback` - Lookback period for symmetry analysis (10-100)
    /// * `tolerance` - Tolerance for symmetry matching (1.0-20.0%)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(lookback: usize, tolerance: f64) -> Result<Self> {
        if lookback < 10 || lookback > 100 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be between 10 and 100".to_string(),
            });
        }
        if tolerance < 1.0 || tolerance > 20.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "tolerance".to_string(),
                reason: "must be between 1.0 and 20.0".to_string(),
            });
        }
        Ok(Self { lookback, tolerance })
    }

    /// Calculate pattern symmetry score.
    ///
    /// Returns symmetry score between 0 and 1:
    /// * 0.0-0.3: Low symmetry (asymmetric pattern)
    /// * 0.3-0.6: Moderate symmetry
    /// * 0.6-1.0: High symmetry (nearly mirror-image pattern)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.lookback..n {
            let start = i - self.lookback;
            let mid = start + self.lookback / 2;

            // Calculate average price for normalization
            let avg_price: f64 = close[start..=i].iter().sum::<f64>() / (self.lookback + 1) as f64;

            if avg_price < 1e-10 {
                continue;
            }

            // Calculate price deviation from mean for each side
            let left_half = &close[start..mid];
            let right_half = &close[mid..=i];

            // Reverse right half for comparison
            let right_reversed: Vec<f64> = right_half.iter().rev().cloned().collect();

            // Calculate symmetry score based on price correlation
            let left_len = left_half.len();
            let right_len = right_reversed.len();
            let compare_len = left_len.min(right_len);

            if compare_len < 2 {
                continue;
            }

            // Calculate price difference for symmetry
            let mut diff_sum = 0.0;

            for j in 0..compare_len {
                diff_sum += (left_half[j] - right_reversed[j]).abs();
            }

            // Normalize difference
            let avg_diff = diff_sum / compare_len as f64;
            let tolerance_abs = avg_price * self.tolerance / 100.0;
            let price_symmetry = (1.0 - avg_diff / tolerance_abs).max(0.0).min(1.0);

            // Calculate shape symmetry (high-low patterns)
            let mut high_diff_sum = 0.0;
            let mut low_diff_sum = 0.0;
            let left_high = &high[start..mid];
            let left_low = &low[start..mid];
            let right_high: Vec<f64> = high[mid..=i].iter().rev().cloned().collect();
            let right_low: Vec<f64> = low[mid..=i].iter().rev().cloned().collect();

            let high_len = left_high.len().min(right_high.len());
            let low_len = left_low.len().min(right_low.len());

            for j in 0..high_len {
                high_diff_sum += (left_high[j] - right_high[j]).abs();
            }
            for j in 0..low_len {
                low_diff_sum += (left_low[j] - right_low[j]).abs();
            }

            let avg_high_diff = if high_len > 0 { high_diff_sum / high_len as f64 } else { 0.0 };
            let avg_low_diff = if low_len > 0 { low_diff_sum / low_len as f64 } else { 0.0 };

            let shape_symmetry = (1.0 - (avg_high_diff + avg_low_diff) / (2.0 * tolerance_abs))
                .max(0.0)
                .min(1.0);

            // Calculate time symmetry (distance from pivot to swing points)
            let mut swing_count_left = 0;
            let mut swing_count_right = 0;

            for j in (start + 1)..(mid - 1) {
                if high[j] > high[j - 1] && high[j] > high[j + 1] {
                    swing_count_left += 1;
                }
                if low[j] < low[j - 1] && low[j] < low[j + 1] {
                    swing_count_left += 1;
                }
            }

            for j in (mid + 1)..(i - 1) {
                if j + 1 <= i {
                    if high[j] > high[j - 1] && high[j] > high[j + 1] {
                        swing_count_right += 1;
                    }
                    if low[j] < low[j - 1] && low[j] < low[j + 1] {
                        swing_count_right += 1;
                    }
                }
            }

            let swing_symmetry = if swing_count_left > 0 || swing_count_right > 0 {
                let max_swings = swing_count_left.max(swing_count_right) as f64;
                let min_swings = swing_count_left.min(swing_count_right) as f64;
                if max_swings > 0.0 { min_swings / max_swings } else { 0.5 }
            } else {
                0.5
            };

            // Combine symmetry scores
            let symmetry = (price_symmetry * 0.4 + shape_symmetry * 0.4 + swing_symmetry * 0.2)
                .min(1.0);

            result[i] = symmetry;
        }

        result
    }
}

impl TechnicalIndicator for PatternSymmetry {
    fn name(&self) -> &str {
        "Pattern Symmetry"
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

// ============================================================================
// TrendContinuationStrength
// ============================================================================

/// Trend Continuation Strength - Measures strength of trend continuation signals
///
/// Evaluates the quality and strength of trend continuation patterns based on
/// momentum alignment, pullback depth, and trend structure.
#[derive(Debug, Clone)]
pub struct TrendContinuationStrength {
    /// Lookback period for trend analysis
    lookback: usize,
    /// Short-term momentum period
    short_period: usize,
    /// Maximum pullback depth as percentage
    max_pullback: f64,
}

impl TrendContinuationStrength {
    /// Create a new TrendContinuationStrength indicator.
    ///
    /// # Arguments
    /// * `lookback` - Lookback period for trend analysis (10-50)
    /// * `short_period` - Short-term momentum period (3-15)
    /// * `max_pullback` - Maximum pullback depth percentage (10.0-50.0)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(lookback: usize, short_period: usize, max_pullback: f64) -> Result<Self> {
        if lookback < 10 || lookback > 50 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be between 10 and 50".to_string(),
            });
        }
        if short_period < 3 || short_period > 15 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be between 3 and 15".to_string(),
            });
        }
        if short_period >= lookback {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be less than lookback".to_string(),
            });
        }
        if max_pullback < 10.0 || max_pullback > 50.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "max_pullback".to_string(),
                reason: "must be between 10.0 and 50.0".to_string(),
            });
        }
        Ok(Self { lookback, short_period, max_pullback })
    }

    /// Calculate trend continuation strength.
    ///
    /// Returns strength between -1 and 1:
    /// * Positive values: Bullish continuation strength
    /// * Negative values: Bearish continuation strength
    /// * Values closer to +/-1 indicate stronger continuation signals
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.lookback..n {
            let start = i - self.lookback;

            // Determine long-term trend direction
            let trend_change = close[i] - close[start];
            let is_uptrend = trend_change > 0.0;
            let is_downtrend = trend_change < 0.0;

            if !is_uptrend && !is_downtrend {
                continue;
            }

            // Calculate trend strength (total move)
            let trend_move = trend_change.abs();

            // Calculate pullback depth
            let pullback = if is_uptrend {
                let swing_high = high[start..=i].iter().cloned().fold(f64::MIN, f64::max);
                swing_high - close[i]
            } else {
                let swing_low = low[start..=i].iter().cloned().fold(f64::MAX, f64::min);
                close[i] - swing_low
            };

            let pullback_pct = if trend_move > 1e-10 {
                pullback / trend_move * 100.0
            } else {
                100.0
            };

            // Check if pullback is within acceptable range
            if pullback_pct > self.max_pullback {
                continue;
            }

            // Calculate continuation factors
            // 1. Trend momentum (short-term aligned with long-term)
            let short_change = close[i] - close[i.saturating_sub(self.short_period)];
            let momentum_aligned = (is_uptrend && short_change > 0.0) ||
                                   (is_downtrend && short_change < 0.0);
            let momentum_score = if momentum_aligned {
                (short_change.abs() / trend_move).min(1.0)
            } else {
                0.0
            };

            // 2. Pullback quality (shallow pullback = stronger)
            let pullback_score = (1.0 - pullback_pct / self.max_pullback).max(0.0);

            // 3. Higher highs/lows (or lower highs/lows for downtrend)
            let mid_point = start + self.lookback / 2;
            let structure_score = if is_uptrend {
                let first_half_high = high[start..mid_point].iter().cloned().fold(f64::MIN, f64::max);
                let second_half_high = high[mid_point..=i].iter().cloned().fold(f64::MIN, f64::max);
                let first_half_low = low[start..mid_point].iter().cloned().fold(f64::MAX, f64::min);
                let second_half_low = low[mid_point..=i].iter().cloned().fold(f64::MAX, f64::min);

                let higher_high = if second_half_high > first_half_high { 0.5 } else { 0.0 };
                let higher_low = if second_half_low > first_half_low { 0.5 } else { 0.0 };
                higher_high + higher_low
            } else {
                let first_half_high = high[start..mid_point].iter().cloned().fold(f64::MIN, f64::max);
                let second_half_high = high[mid_point..=i].iter().cloned().fold(f64::MIN, f64::max);
                let first_half_low = low[start..mid_point].iter().cloned().fold(f64::MAX, f64::min);
                let second_half_low = low[mid_point..=i].iter().cloned().fold(f64::MAX, f64::min);

                let lower_high = if second_half_high < first_half_high { 0.5 } else { 0.0 };
                let lower_low = if second_half_low < first_half_low { 0.5 } else { 0.0 };
                lower_high + lower_low
            };

            // 4. Volume pattern (increasing on trend moves)
            let avg_vol = volume[start..i].iter().sum::<f64>() / self.lookback as f64;
            let vol_ratio = if avg_vol > 1e-10 { volume[i] / avg_vol } else { 1.0 };
            let volume_score = (vol_ratio / 1.5).min(1.0);

            // Combine scores
            let strength = (momentum_score * 0.3 + pullback_score * 0.25 +
                           structure_score * 0.25 + volume_score * 0.2).min(1.0);

            result[i] = if is_uptrend { strength } else { -strength };
        }

        result
    }
}

impl TechnicalIndicator for TrendContinuationStrength {
    fn name(&self) -> &str {
        "Trend Continuation Strength"
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close, &data.volume)))
    }
}

// ============================================================================
// SwingPointDetector
// ============================================================================

/// Swing Point Detector - Detects swing highs and lows in price data
///
/// Identifies significant price pivot points (swing highs and swing lows)
/// that can be used for trend analysis, support/resistance identification,
/// and pattern recognition. A swing high is a high surrounded by lower highs,
/// and a swing low is a low surrounded by higher lows.
#[derive(Debug, Clone)]
pub struct SwingPointDetector {
    /// Number of bars on each side to confirm swing point
    strength: usize,
    /// Minimum percentage move from swing point to confirm
    min_move_pct: f64,
}

impl SwingPointDetector {
    /// Create a new SwingPointDetector indicator.
    ///
    /// # Arguments
    /// * `strength` - Number of bars on each side for swing confirmation (2-20)
    /// * `min_move_pct` - Minimum percentage move to confirm swing (0.1-5.0)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(strength: usize, min_move_pct: f64) -> Result<Self> {
        if strength < 2 || strength > 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "strength".to_string(),
                reason: "must be between 2 and 20".to_string(),
            });
        }
        if min_move_pct < 0.1 || min_move_pct > 5.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "min_move_pct".to_string(),
                reason: "must be between 0.1 and 5.0".to_string(),
            });
        }
        Ok(Self { strength, min_move_pct })
    }

    /// Calculate swing point signals.
    ///
    /// Returns:
    /// * +1.0: Swing low detected (potential support)
    /// * -1.0: Swing high detected (potential resistance)
    /// * 0.0: No swing point
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.strength * 2 + 1 {
            return result;
        }

        for i in self.strength..(n - self.strength) {
            let avg_price = close[i];
            if avg_price < 1e-10 {
                continue;
            }

            let min_move = avg_price * self.min_move_pct / 100.0;

            // Check for swing high
            let mut is_swing_high = true;
            let current_high = high[i];

            for j in 1..=self.strength {
                // Left side: current high must be higher
                if high[i - j] >= current_high {
                    is_swing_high = false;
                    break;
                }
                // Right side: current high must be higher
                if high[i + j] >= current_high {
                    is_swing_high = false;
                    break;
                }
            }

            // Verify minimum move from swing high
            if is_swing_high {
                let left_low = low[(i - self.strength)..i].iter().cloned().fold(f64::MAX, f64::min);
                let right_low = low[(i + 1)..=(i + self.strength)].iter().cloned().fold(f64::MAX, f64::min);
                let max_drop = (current_high - left_low).max(current_high - right_low);
                if max_drop >= min_move {
                    result[i] = -1.0; // Swing high (resistance)
                    continue;
                }
            }

            // Check for swing low
            let mut is_swing_low = true;
            let current_low = low[i];

            for j in 1..=self.strength {
                // Left side: current low must be lower
                if low[i - j] <= current_low {
                    is_swing_low = false;
                    break;
                }
                // Right side: current low must be lower
                if low[i + j] <= current_low {
                    is_swing_low = false;
                    break;
                }
            }

            // Verify minimum move from swing low
            if is_swing_low {
                let left_high = high[(i - self.strength)..i].iter().cloned().fold(f64::MIN, f64::max);
                let right_high = high[(i + 1)..=(i + self.strength)].iter().cloned().fold(f64::MIN, f64::max);
                let max_rise = (left_high - current_low).max(right_high - current_low);
                if max_rise >= min_move {
                    result[i] = 1.0; // Swing low (support)
                }
            }
        }

        result
    }
}

impl TechnicalIndicator for SwingPointDetector {
    fn name(&self) -> &str {
        "Swing Point Detector"
    }

    fn min_periods(&self) -> usize {
        self.strength * 2 + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

// ============================================================================
// SupportResistanceStrength
// ============================================================================

/// Support Resistance Strength - Measures the strength of S/R levels
///
/// Evaluates the quality and strength of support and resistance levels based on
/// multiple price touches, volume at level, time at level, and rejection strength.
#[derive(Debug, Clone)]
pub struct SupportResistanceStrength {
    /// Lookback period for S/R analysis
    lookback: usize,
    /// Tolerance zone for level touches (percentage)
    tolerance_pct: f64,
    /// Minimum touches required to confirm level
    min_touches: usize,
}

impl SupportResistanceStrength {
    /// Create a new SupportResistanceStrength indicator.
    ///
    /// # Arguments
    /// * `lookback` - Lookback period for S/R analysis (10-200)
    /// * `tolerance_pct` - Tolerance zone for level touches (0.1-3.0%)
    /// * `min_touches` - Minimum touches to confirm level (2-10)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(lookback: usize, tolerance_pct: f64, min_touches: usize) -> Result<Self> {
        if lookback < 10 || lookback > 200 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be between 10 and 200".to_string(),
            });
        }
        if tolerance_pct < 0.1 || tolerance_pct > 3.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "tolerance_pct".to_string(),
                reason: "must be between 0.1 and 3.0".to_string(),
            });
        }
        if min_touches < 2 || min_touches > 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "min_touches".to_string(),
                reason: "must be between 2 and 10".to_string(),
            });
        }
        Ok(Self { lookback, tolerance_pct, min_touches })
    }

    /// Calculate S/R strength values.
    ///
    /// Returns strength between -1 and 1:
    /// * Positive values (0 to 1): Support strength (current price near support)
    /// * Negative values (-1 to 0): Resistance strength (current price near resistance)
    /// * 0: No significant S/R level nearby
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.lookback..n {
            let start = i - self.lookback;
            let current_price = close[i];

            if current_price < 1e-10 {
                continue;
            }

            let tolerance = current_price * self.tolerance_pct / 100.0;

            // Find potential support levels (swing lows)
            let mut support_levels: Vec<(f64, f64)> = Vec::new(); // (level, volume)
            // Find potential resistance levels (swing highs)
            let mut resistance_levels: Vec<(f64, f64)> = Vec::new();

            for j in (start + 1)..(i - 1) {
                // Swing low (support)
                if low[j] < low[j - 1] && low[j] < low[j + 1] {
                    support_levels.push((low[j], volume[j]));
                }
                // Swing high (resistance)
                if high[j] > high[j - 1] && high[j] > high[j + 1] {
                    resistance_levels.push((high[j], volume[j]));
                }
            }

            // Calculate support strength at current price
            let mut support_touches = 0;
            let mut support_vol_sum = 0.0;
            for (level, vol) in &support_levels {
                if (current_price - level).abs() <= tolerance {
                    support_touches += 1;
                    support_vol_sum += vol;
                }
            }

            // Calculate resistance strength at current price
            let mut resistance_touches = 0;
            let mut resistance_vol_sum = 0.0;
            for (level, vol) in &resistance_levels {
                if (current_price - level).abs() <= tolerance {
                    resistance_touches += 1;
                    resistance_vol_sum += vol;
                }
            }

            // Calculate average volume for normalization
            let avg_vol = volume[start..i].iter().sum::<f64>() / self.lookback as f64;

            // Calculate support strength
            if support_touches >= self.min_touches {
                let touch_score = (support_touches as f64 / self.min_touches as f64).min(2.0) / 2.0;
                let vol_score = if avg_vol > 1e-10 && support_touches > 0 {
                    ((support_vol_sum / support_touches as f64) / avg_vol).min(2.0) / 2.0
                } else {
                    0.5
                };

                // Check for recent bounces off this level
                let mut bounce_count = 0;
                for j in (i.saturating_sub(10))..i {
                    if low[j] <= current_price + tolerance && close[j] > current_price {
                        bounce_count += 1;
                    }
                }
                let bounce_score = (bounce_count as f64 / 3.0).min(1.0);

                let strength = (touch_score * 0.4 + vol_score * 0.3 + bounce_score * 0.3).min(1.0);
                result[i] = strength;
            }
            // Calculate resistance strength
            else if resistance_touches >= self.min_touches {
                let touch_score = (resistance_touches as f64 / self.min_touches as f64).min(2.0) / 2.0;
                let vol_score = if avg_vol > 1e-10 && resistance_touches > 0 {
                    ((resistance_vol_sum / resistance_touches as f64) / avg_vol).min(2.0) / 2.0
                } else {
                    0.5
                };

                // Check for recent rejections at this level
                let mut rejection_count = 0;
                for j in (i.saturating_sub(10))..i {
                    if high[j] >= current_price - tolerance && close[j] < current_price {
                        rejection_count += 1;
                    }
                }
                let rejection_score = (rejection_count as f64 / 3.0).min(1.0);

                let strength = (touch_score * 0.4 + vol_score * 0.3 + rejection_score * 0.3).min(1.0);
                result[i] = -strength;
            }
        }

        result
    }
}

impl TechnicalIndicator for SupportResistanceStrength {
    fn name(&self) -> &str {
        "Support Resistance Strength"
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close, &data.volume)))
    }
}

// ============================================================================
// PriceActionMomentum
// ============================================================================

/// Price Action Momentum - Measures momentum from price action patterns
///
/// Calculates momentum based on price action characteristics including
/// candle body strength, wick patterns, and close position within range.
#[derive(Debug, Clone)]
pub struct PriceActionMomentum {
    /// Lookback period for momentum calculation
    lookback: usize,
    /// Smoothing period for the momentum signal
    smoothing: usize,
}

impl PriceActionMomentum {
    /// Create a new PriceActionMomentum indicator.
    ///
    /// # Arguments
    /// * `lookback` - Lookback period for momentum calculation (5-50)
    /// * `smoothing` - Smoothing period for signal (1-20)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(lookback: usize, smoothing: usize) -> Result<Self> {
        if lookback < 5 || lookback > 50 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be between 5 and 50".to_string(),
            });
        }
        if smoothing < 1 || smoothing > 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing".to_string(),
                reason: "must be between 1 and 20".to_string(),
            });
        }
        if smoothing > lookback {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing".to_string(),
                reason: "cannot exceed lookback".to_string(),
            });
        }
        Ok(Self { lookback, smoothing })
    }

    /// Calculate price action momentum.
    ///
    /// Returns momentum between -1 and 1:
    /// * Positive values: Bullish momentum (strong buying pressure)
    /// * Negative values: Bearish momentum (strong selling pressure)
    /// * 0: Neutral or indecisive
    pub fn calculate(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut raw_momentum = vec![0.0; n];
        let mut result = vec![0.0; n];

        // Calculate raw price action momentum for each bar
        for i in 0..n {
            let range = high[i] - low[i];
            if range < 1e-10 {
                continue;
            }

            let body = close[i] - open[i];
            let body_size = body.abs();
            let is_bullish = body > 0.0;

            // Close position within range (0 = at low, 1 = at high)
            let close_position = (close[i] - low[i]) / range;

            // Body strength (body size relative to range)
            let body_strength = body_size / range;

            // Wick analysis
            let upper_wick = high[i] - close[i].max(open[i]);
            let lower_wick = close[i].min(open[i]) - low[i];

            // Bullish pressure: close near high, small upper wick
            let bullish_pressure = close_position * (1.0 - upper_wick / range);

            // Bearish pressure: close near low, small lower wick
            let bearish_pressure = (1.0 - close_position) * (1.0 - lower_wick / range);

            // Combined momentum
            let direction = if is_bullish { 1.0 } else { -1.0 };
            raw_momentum[i] = direction * body_strength * 0.5 +
                              (bullish_pressure - bearish_pressure) * 0.5;
        }

        // Calculate smoothed momentum over lookback period
        for i in self.lookback..n {
            let start = i - self.lookback;

            // Sum of raw momentum in lookback
            let momentum_sum: f64 = raw_momentum[start..=i].iter().sum();
            let avg_momentum = momentum_sum / (self.lookback + 1) as f64;

            // Apply additional smoothing
            if i >= self.lookback + self.smoothing - 1 {
                let smooth_start = i - self.smoothing + 1;
                let mut smooth_sum = 0.0;
                for j in smooth_start..=i {
                    let inner_start = j.saturating_sub(self.lookback);
                    let inner_sum: f64 = raw_momentum[inner_start..=j].iter().sum();
                    smooth_sum += inner_sum / (self.lookback + 1) as f64;
                }
                result[i] = (smooth_sum / self.smoothing as f64).max(-1.0).min(1.0);
            } else {
                result[i] = avg_momentum.max(-1.0).min(1.0);
            }
        }

        result
    }
}

impl TechnicalIndicator for PriceActionMomentum {
    fn name(&self) -> &str {
        "Price Action Momentum"
    }

    fn min_periods(&self) -> usize {
        self.lookback + self.smoothing
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.open, &data.high, &data.low, &data.close)))
    }
}

// ============================================================================
// CandleRangeAnalysis
// ============================================================================

/// Candle Range Analysis - Analyzes candle range patterns
///
/// Evaluates candle ranges to identify expansion, contraction, and range patterns
/// that indicate volatility changes and potential breakouts or consolidation.
#[derive(Debug, Clone)]
pub struct CandleRangeAnalysis {
    /// Period for average range calculation
    atr_period: usize,
    /// Threshold for narrow range (ATR multiple)
    narrow_threshold: f64,
    /// Threshold for wide range (ATR multiple)
    wide_threshold: f64,
}

impl CandleRangeAnalysis {
    /// Create a new CandleRangeAnalysis indicator.
    ///
    /// # Arguments
    /// * `atr_period` - Period for average range calculation (5-50)
    /// * `narrow_threshold` - Threshold for narrow range detection (0.3-0.8)
    /// * `wide_threshold` - Threshold for wide range detection (1.5-4.0)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(atr_period: usize, narrow_threshold: f64, wide_threshold: f64) -> Result<Self> {
        if atr_period < 5 || atr_period > 50 {
            return Err(IndicatorError::InvalidParameter {
                name: "atr_period".to_string(),
                reason: "must be between 5 and 50".to_string(),
            });
        }
        if narrow_threshold < 0.3 || narrow_threshold > 0.8 {
            return Err(IndicatorError::InvalidParameter {
                name: "narrow_threshold".to_string(),
                reason: "must be between 0.3 and 0.8".to_string(),
            });
        }
        if wide_threshold < 1.5 || wide_threshold > 4.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "wide_threshold".to_string(),
                reason: "must be between 1.5 and 4.0".to_string(),
            });
        }
        if narrow_threshold >= wide_threshold {
            return Err(IndicatorError::InvalidParameter {
                name: "narrow_threshold".to_string(),
                reason: "must be less than wide_threshold".to_string(),
            });
        }
        Ok(Self { atr_period, narrow_threshold, wide_threshold })
    }

    /// Calculate candle range analysis values.
    ///
    /// Returns range classification and strength:
    /// * Values > 1.0: Wide range (expansion) - value indicates strength
    /// * Values between -1.0 and 1.0: Normal range
    /// * Values < -1.0: Narrow range (contraction) - absolute value indicates strength
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.atr_period + 1 {
            return result;
        }

        // Calculate ATR
        let mut atr = vec![0.0; n];
        for i in 1..n {
            let tr = (high[i] - low[i])
                .max((high[i] - close[i - 1]).abs())
                .max((low[i] - close[i - 1]).abs());

            if i < self.atr_period {
                // Initial ATR using simple average
                let mut sum = 0.0;
                for j in 1..=i {
                    let tr_j = (high[j] - low[j])
                        .max((high[j] - close[j - 1]).abs())
                        .max((low[j] - close[j - 1]).abs());
                    sum += tr_j;
                }
                atr[i] = sum / i as f64;
            } else if i == self.atr_period {
                // First full ATR
                let mut sum = 0.0;
                for j in 1..=self.atr_period {
                    let tr_j = (high[j] - low[j])
                        .max((high[j] - close[j - 1]).abs())
                        .max((low[j] - close[j - 1]).abs());
                    sum += tr_j;
                }
                atr[i] = sum / self.atr_period as f64;
            } else {
                // Smoothed ATR
                atr[i] = (atr[i - 1] * (self.atr_period - 1) as f64 + tr) / self.atr_period as f64;
            }
        }

        // Analyze candle ranges
        for i in self.atr_period..n {
            let current_range = high[i] - low[i];
            let current_atr = atr[i];

            if current_atr < 1e-10 {
                continue;
            }

            let range_ratio = current_range / current_atr;

            // Wide range (expansion)
            if range_ratio >= self.wide_threshold {
                // Strength based on how much it exceeds threshold
                let strength = ((range_ratio - self.wide_threshold) / self.wide_threshold + 1.0).min(3.0);
                result[i] = strength;
            }
            // Narrow range (contraction)
            else if range_ratio <= self.narrow_threshold {
                // Strength based on how much below threshold
                let strength = ((self.narrow_threshold - range_ratio) / self.narrow_threshold + 1.0).min(3.0);
                result[i] = -strength;
            }
            // Normal range - output normalized ratio
            else {
                // Normalize to -1 to 1 range
                let mid_point = (self.narrow_threshold + self.wide_threshold) / 2.0;
                let half_range = (self.wide_threshold - self.narrow_threshold) / 2.0;
                result[i] = (range_ratio - mid_point) / half_range;
            }
        }

        result
    }
}

impl TechnicalIndicator for CandleRangeAnalysis {
    fn name(&self) -> &str {
        "Candle Range Analysis"
    }

    fn min_periods(&self) -> usize {
        self.atr_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

// ============================================================================
// TrendLineBreak
// ============================================================================

/// Trend Line Break - Detects trend line breaks
///
/// Identifies when price breaks through dynamically calculated trend lines
/// based on swing points, signaling potential trend reversals or continuations.
#[derive(Debug, Clone)]
pub struct TrendLineBreak {
    /// Lookback period for trend line calculation
    lookback: usize,
    /// Minimum swing points to form trend line
    min_points: usize,
    /// Breakout confirmation threshold (percentage)
    breakout_pct: f64,
}

impl TrendLineBreak {
    /// Create a new TrendLineBreak indicator.
    ///
    /// # Arguments
    /// * `lookback` - Lookback period for trend line calculation (10-100)
    /// * `min_points` - Minimum swing points for trend line (2-5)
    /// * `breakout_pct` - Breakout confirmation threshold (0.1-2.0%)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(lookback: usize, min_points: usize, breakout_pct: f64) -> Result<Self> {
        if lookback < 10 || lookback > 100 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be between 10 and 100".to_string(),
            });
        }
        if min_points < 2 || min_points > 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "min_points".to_string(),
                reason: "must be between 2 and 5".to_string(),
            });
        }
        if breakout_pct < 0.1 || breakout_pct > 2.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "breakout_pct".to_string(),
                reason: "must be between 0.1 and 2.0".to_string(),
            });
        }
        Ok(Self { lookback, min_points, breakout_pct })
    }

    /// Calculate trend line break signals.
    ///
    /// Returns:
    /// * +1.0: Bullish break (price breaks above downtrend line)
    /// * -1.0: Bearish break (price breaks below uptrend line)
    /// * 0.0: No trend line break
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.lookback + 2 {
            return result;
        }

        for i in self.lookback..n {
            let start = i - self.lookback;
            let avg_price = close[start..=i].iter().sum::<f64>() / (self.lookback + 1) as f64;

            if avg_price < 1e-10 {
                continue;
            }

            let breakout_threshold = avg_price * self.breakout_pct / 100.0;

            // Find swing highs and lows in lookback period
            let mut swing_highs: Vec<(usize, f64)> = Vec::new();
            let mut swing_lows: Vec<(usize, f64)> = Vec::new();

            for j in (start + 1)..(i - 1) {
                if high[j] > high[j - 1] && high[j] > high[j + 1] {
                    swing_highs.push((j, high[j]));
                }
                if low[j] < low[j - 1] && low[j] < low[j + 1] {
                    swing_lows.push((j, low[j]));
                }
            }

            // Check for downtrend line break (bullish)
            if swing_highs.len() >= self.min_points {
                // Calculate downtrend line using most recent swing highs
                let recent_highs: Vec<_> = swing_highs.iter()
                    .rev()
                    .take(self.min_points)
                    .collect();

                if recent_highs.len() >= 2 {
                    // Simple linear regression for trend line
                    let first = recent_highs.last().unwrap();
                    let last = recent_highs.first().unwrap();

                    if last.0 > first.0 {
                        let slope = (last.1 - first.1) / (last.0 - first.0) as f64;

                        // Only consider downtrend lines (negative or slightly positive slope)
                        if slope < 0.01 * avg_price {
                            let trend_line_value = last.1 + slope * (i - last.0) as f64;

                            // Check for break above trend line
                            if close[i] > trend_line_value + breakout_threshold {
                                // Confirm break wasn't already triggered
                                if i > 0 && close[i - 1] <= trend_line_value {
                                    result[i] = 1.0;
                                    continue;
                                }
                            }
                        }
                    }
                }
            }

            // Check for uptrend line break (bearish)
            if swing_lows.len() >= self.min_points {
                let recent_lows: Vec<_> = swing_lows.iter()
                    .rev()
                    .take(self.min_points)
                    .collect();

                if recent_lows.len() >= 2 {
                    let first = recent_lows.last().unwrap();
                    let last = recent_lows.first().unwrap();

                    if last.0 > first.0 {
                        let slope = (last.1 - first.1) / (last.0 - first.0) as f64;

                        // Only consider uptrend lines (positive or slightly negative slope)
                        if slope > -0.01 * avg_price {
                            let trend_line_value = last.1 + slope * (i - last.0) as f64;

                            // Check for break below trend line
                            if close[i] < trend_line_value - breakout_threshold {
                                // Confirm break wasn't already triggered
                                if i > 0 && close[i - 1] >= trend_line_value {
                                    result[i] = -1.0;
                                }
                            }
                        }
                    }
                }
            }
        }

        result
    }
}

impl TechnicalIndicator for TrendLineBreak {
    fn name(&self) -> &str {
        "Trend Line Break"
    }

    fn min_periods(&self) -> usize {
        self.lookback + 2
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

// ============================================================================
// PriceStructure
// ============================================================================

/// Price Structure - Identifies market structure (HH, HL, LH, LL)
///
/// Analyzes price structure to identify higher highs (HH), higher lows (HL),
/// lower highs (LH), and lower lows (LL), which define trend direction
/// and potential reversals according to market structure theory.
#[derive(Debug, Clone)]
pub struct PriceStructure {
    /// Lookback period for structure analysis
    lookback: usize,
    /// Swing detection strength (bars on each side)
    swing_strength: usize,
}

impl PriceStructure {
    /// Create a new PriceStructure indicator.
    ///
    /// # Arguments
    /// * `lookback` - Lookback period for structure analysis (10-100)
    /// * `swing_strength` - Bars on each side for swing detection (2-10)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(lookback: usize, swing_strength: usize) -> Result<Self> {
        if lookback < 10 || lookback > 100 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be between 10 and 100".to_string(),
            });
        }
        if swing_strength < 2 || swing_strength > 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "swing_strength".to_string(),
                reason: "must be between 2 and 10".to_string(),
            });
        }
        if swing_strength * 2 >= lookback {
            return Err(IndicatorError::InvalidParameter {
                name: "swing_strength".to_string(),
                reason: "swing_strength * 2 must be less than lookback".to_string(),
            });
        }
        Ok(Self { lookback, swing_strength })
    }

    /// Calculate price structure values.
    ///
    /// Returns structure encoding:
    /// * +2.0: Higher High (HH) - bullish continuation
    /// * +1.0: Higher Low (HL) - bullish continuation
    /// * -1.0: Lower High (LH) - bearish continuation
    /// * -2.0: Lower Low (LL) - bearish continuation
    /// * 0.0: No significant structure change
    pub fn calculate(&self, high: &[f64], low: &[f64]) -> Vec<f64> {
        let n = high.len();
        let mut result = vec![0.0; n];

        if n < self.lookback + self.swing_strength {
            return result;
        }

        // Track swing highs and lows with their indices
        let mut swing_highs: Vec<(usize, f64)> = Vec::new();
        let mut swing_lows: Vec<(usize, f64)> = Vec::new();

        for i in self.swing_strength..(n - self.swing_strength) {
            // Detect swing high
            let mut is_swing_high = true;
            for j in 1..=self.swing_strength {
                if high[i - j] >= high[i] || high[i + j] >= high[i] {
                    is_swing_high = false;
                    break;
                }
            }
            if is_swing_high {
                swing_highs.push((i, high[i]));
            }

            // Detect swing low
            let mut is_swing_low = true;
            for j in 1..=self.swing_strength {
                if low[i - j] <= low[i] || low[i + j] <= low[i] {
                    is_swing_low = false;
                    break;
                }
            }
            if is_swing_low {
                swing_lows.push((i, low[i]));
            }
        }

        // Analyze structure at each bar
        for i in self.lookback..n {
            // Find swing highs and lows within lookback period
            let recent_highs: Vec<_> = swing_highs.iter()
                .filter(|(idx, _)| *idx >= i.saturating_sub(self.lookback) && *idx <= i)
                .collect();

            let recent_lows: Vec<_> = swing_lows.iter()
                .filter(|(idx, _)| *idx >= i.saturating_sub(self.lookback) && *idx <= i)
                .collect();

            // Need at least 2 swing points of each type to compare
            if recent_highs.len() >= 2 {
                let last_high = recent_highs.last().unwrap();
                let prev_high = recent_highs[recent_highs.len() - 2];

                // Check for Higher High at current swing high
                if last_high.0 == i || (i > last_high.0 && i - last_high.0 <= self.swing_strength) {
                    if last_high.1 > prev_high.1 {
                        result[i] = 2.0; // Higher High
                    }
                }
                // Check for Lower High at current swing high
                if last_high.0 == i || (i > last_high.0 && i - last_high.0 <= self.swing_strength) {
                    if last_high.1 < prev_high.1 && result[i] == 0.0 {
                        result[i] = -1.0; // Lower High
                    }
                }
            }

            if recent_lows.len() >= 2 && result[i] == 0.0 {
                let last_low = recent_lows.last().unwrap();
                let prev_low = recent_lows[recent_lows.len() - 2];

                // Check for Higher Low at current swing low
                if last_low.0 == i || (i > last_low.0 && i - last_low.0 <= self.swing_strength) {
                    if last_low.1 > prev_low.1 {
                        result[i] = 1.0; // Higher Low
                    }
                }
                // Check for Lower Low at current swing low
                if last_low.0 == i || (i > last_low.0 && i - last_low.0 <= self.swing_strength) {
                    if last_low.1 < prev_low.1 && result[i] == 0.0 {
                        result[i] = -2.0; // Lower Low
                    }
                }
            }
        }

        result
    }
}

impl TechnicalIndicator for PriceStructure {
    fn name(&self) -> &str {
        "Price Structure"
    }

    fn min_periods(&self) -> usize {
        self.lookback + self.swing_strength
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low)))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> OHLCVSeries {
        // Create trending data with some consolidation periods
        let mut open = Vec::with_capacity(50);
        let mut high = Vec::with_capacity(50);
        let mut low = Vec::with_capacity(50);
        let mut close = Vec::with_capacity(50);
        let mut volume = Vec::with_capacity(50);

        for i in 0..50 {
            let base = 100.0 + (i as f64) * 0.5;
            let noise = ((i as f64) * 0.3).sin() * 0.5;

            open.push(base + noise);
            high.push(base + 1.5 + noise.abs());
            low.push(base - 1.0 - noise.abs());
            close.push(base + 0.3 + noise);
            volume.push(10000.0 + (i as f64) * 100.0);
        }

        OHLCVSeries { open, high, low, close, volume }
    }

    fn make_reversal_data() -> OHLCVSeries {
        // Data with reversal patterns
        let open = vec![
            100.0, 99.0, 98.0, 97.0, 96.0,  // Downtrend
            95.0,   // Pin bar (hammer): open near high
            96.0, 97.0, 98.0, 99.0,  // Uptrend
        ];
        let high = vec![
            101.0, 100.0, 99.0, 98.0, 97.0,
            95.5,  // Pin bar: small upper wick
            97.5, 98.5, 99.5, 100.5,
        ];
        let low = vec![
            99.0, 98.0, 97.0, 96.0, 95.0,
            92.0,  // Pin bar: long lower wick
            95.5, 96.5, 97.5, 98.5,
        ];
        let close = vec![
            99.5, 98.5, 97.5, 96.5, 95.5,
            95.3,  // Pin bar: close near high (bullish)
            97.0, 98.0, 99.0, 100.0,
        ];
        let volume = vec![1000.0; 10];

        OHLCVSeries { open, high, low, close, volume }
    }

    // ========== TrendContinuationPattern Tests ==========

    #[test]
    fn test_trend_continuation_new_valid() {
        let indicator = TrendContinuationPattern::new(10, 3);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_trend_continuation_invalid_lookback() {
        assert!(TrendContinuationPattern::new(3, 2).is_err());
        assert!(TrendContinuationPattern::new(60, 5).is_err());
    }

    #[test]
    fn test_trend_continuation_invalid_confirmations() {
        assert!(TrendContinuationPattern::new(10, 1).is_err());
        assert!(TrendContinuationPattern::new(10, 15).is_err());
        assert!(TrendContinuationPattern::new(10, 12).is_err()); // > lookback
    }

    #[test]
    fn test_trend_continuation_calculate() {
        let data = make_test_data();
        let indicator = TrendContinuationPattern::new(10, 3).unwrap();
        let result = indicator.calculate(&data.high, &data.low, &data.close);

        assert_eq!(result.len(), data.close.len());

        // Values should be -1, 0, or 1
        for val in &result {
            assert!(*val == -1.0 || *val == 0.0 || *val == 1.0);
        }
    }

    #[test]
    fn test_trend_continuation_min_periods() {
        let indicator = TrendContinuationPattern::new(15, 4).unwrap();
        assert_eq!(indicator.min_periods(), 16);
    }

    #[test]
    fn test_trend_continuation_name() {
        let indicator = TrendContinuationPattern::new(10, 3).unwrap();
        assert_eq!(indicator.name(), "Trend Continuation Pattern");
    }

    #[test]
    fn test_trend_continuation_compute() {
        let data = make_test_data();
        let indicator = TrendContinuationPattern::new(10, 3).unwrap();
        let output = indicator.compute(&data);
        assert!(output.is_ok());
    }

    // ========== ReversalCandlePattern Tests ==========

    #[test]
    fn test_reversal_candle_new_valid() {
        let indicator = ReversalCandlePattern::new(0.6, 2.0);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_reversal_candle_invalid_body_ratio() {
        assert!(ReversalCandlePattern::new(0.2, 2.0).is_err());
        assert!(ReversalCandlePattern::new(1.0, 2.0).is_err());
    }

    #[test]
    fn test_reversal_candle_invalid_wick_ratio() {
        assert!(ReversalCandlePattern::new(0.6, 1.0).is_err());
        assert!(ReversalCandlePattern::new(0.6, 6.0).is_err());
    }

    #[test]
    fn test_reversal_candle_calculate() {
        let data = make_reversal_data();
        let indicator = ReversalCandlePattern::new(0.5, 2.0).unwrap();
        let result = indicator.calculate(&data.open, &data.high, &data.low, &data.close);

        assert_eq!(result.len(), data.close.len());

        // Values should be -1, 0, or 1
        for val in &result {
            assert!(*val == -1.0 || *val == 0.0 || *val == 1.0);
        }
    }

    #[test]
    fn test_reversal_candle_min_periods() {
        let indicator = ReversalCandlePattern::new(0.6, 2.0).unwrap();
        assert_eq!(indicator.min_periods(), 2);
    }

    #[test]
    fn test_reversal_candle_name() {
        let indicator = ReversalCandlePattern::new(0.6, 2.0).unwrap();
        assert_eq!(indicator.name(), "Reversal Candle Pattern");
    }

    #[test]
    fn test_reversal_candle_compute() {
        let data = make_reversal_data();
        let indicator = ReversalCandlePattern::new(0.6, 2.0).unwrap();
        let output = indicator.compute(&data);
        assert!(output.is_ok());
    }

    // ========== VolumePricePattern Tests ==========

    #[test]
    fn test_volume_price_new_valid() {
        let indicator = VolumePricePattern::new(20, 2.0);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_volume_price_invalid_period() {
        assert!(VolumePricePattern::new(3, 2.0).is_err());
        assert!(VolumePricePattern::new(60, 2.0).is_err());
    }

    #[test]
    fn test_volume_price_invalid_multiplier() {
        assert!(VolumePricePattern::new(20, 1.0).is_err());
        assert!(VolumePricePattern::new(20, 6.0).is_err());
    }

    #[test]
    fn test_volume_price_calculate() {
        let data = make_test_data();
        let indicator = VolumePricePattern::new(10, 1.5).unwrap();
        let result = indicator.calculate(&data.open, &data.close, &data.volume);

        assert_eq!(result.len(), data.close.len());

        // Values should be -1, 0, or 1
        for val in &result {
            assert!(*val == -1.0 || *val == 0.0 || *val == 1.0);
        }
    }

    #[test]
    fn test_volume_price_min_periods() {
        let indicator = VolumePricePattern::new(20, 2.0).unwrap();
        assert_eq!(indicator.min_periods(), 21);
    }

    #[test]
    fn test_volume_price_name() {
        let indicator = VolumePricePattern::new(20, 2.0).unwrap();
        assert_eq!(indicator.name(), "Volume Price Pattern");
    }

    #[test]
    fn test_volume_price_compute() {
        let data = make_test_data();
        let indicator = VolumePricePattern::new(10, 1.5).unwrap();
        let output = indicator.compute(&data);
        assert!(output.is_ok());
    }

    // ========== MomentumPattern Tests ==========

    #[test]
    fn test_momentum_pattern_new_valid() {
        let indicator = MomentumPattern::new(5, 20, 1.0);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_momentum_pattern_invalid_fast_period() {
        assert!(MomentumPattern::new(1, 20, 1.0).is_err());
        assert!(MomentumPattern::new(25, 30, 1.0).is_err());
    }

    #[test]
    fn test_momentum_pattern_invalid_slow_period() {
        assert!(MomentumPattern::new(5, 5, 1.0).is_err());
        assert!(MomentumPattern::new(5, 60, 1.0).is_err());
    }

    #[test]
    fn test_momentum_pattern_fast_exceeds_slow() {
        assert!(MomentumPattern::new(15, 14, 1.0).is_err());
        assert!(MomentumPattern::new(15, 15, 1.0).is_err());
    }

    #[test]
    fn test_momentum_pattern_invalid_threshold() {
        assert!(MomentumPattern::new(5, 20, 0.3).is_err());
        assert!(MomentumPattern::new(5, 20, 6.0).is_err());
    }

    #[test]
    fn test_momentum_pattern_calculate() {
        let data = make_test_data();
        let indicator = MomentumPattern::new(5, 15, 1.0).unwrap();
        let result = indicator.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());

        // Values should be -1, 0, or 1
        for val in &result {
            assert!(*val == -1.0 || *val == 0.0 || *val == 1.0);
        }
    }

    #[test]
    fn test_momentum_pattern_min_periods() {
        let indicator = MomentumPattern::new(5, 20, 1.0).unwrap();
        assert_eq!(indicator.min_periods(), 21);
    }

    #[test]
    fn test_momentum_pattern_name() {
        let indicator = MomentumPattern::new(5, 20, 1.0).unwrap();
        assert_eq!(indicator.name(), "Momentum Pattern");
    }

    #[test]
    fn test_momentum_pattern_compute() {
        let data = make_test_data();
        let indicator = MomentumPattern::new(5, 15, 1.0).unwrap();
        let output = indicator.compute(&data);
        assert!(output.is_ok());
    }

    // ========== BreakoutPattern Tests ==========

    #[test]
    fn test_breakout_new_valid() {
        let indicator = BreakoutPattern::new(20, 0.5, true);
        assert!(indicator.is_ok());

        let indicator2 = BreakoutPattern::new(20, 0.5, false);
        assert!(indicator2.is_ok());
    }

    #[test]
    fn test_breakout_invalid_lookback() {
        assert!(BreakoutPattern::new(5, 0.5, true).is_err());
        assert!(BreakoutPattern::new(150, 0.5, true).is_err());
    }

    #[test]
    fn test_breakout_invalid_threshold() {
        assert!(BreakoutPattern::new(20, 0.05, true).is_err());
        assert!(BreakoutPattern::new(20, 2.0, true).is_err());
    }

    #[test]
    fn test_breakout_calculate() {
        let data = make_test_data();
        let indicator = BreakoutPattern::new(15, 0.3, false).unwrap();
        let result = indicator.calculate(&data.high, &data.low, &data.close, &data.volume);

        assert_eq!(result.len(), data.close.len());

        // Values should be -1, 0, or 1
        for val in &result {
            assert!(*val == -1.0 || *val == 0.0 || *val == 1.0);
        }
    }

    #[test]
    fn test_breakout_calculate_with_volume() {
        let data = make_test_data();
        let indicator = BreakoutPattern::new(15, 0.3, true).unwrap();
        let result = indicator.calculate(&data.high, &data.low, &data.close, &data.volume);

        assert_eq!(result.len(), data.close.len());
    }

    #[test]
    fn test_breakout_min_periods() {
        let indicator = BreakoutPattern::new(25, 0.5, true).unwrap();
        assert_eq!(indicator.min_periods(), 26);
    }

    #[test]
    fn test_breakout_name() {
        let indicator = BreakoutPattern::new(20, 0.5, true).unwrap();
        assert_eq!(indicator.name(), "Breakout Pattern");
    }

    #[test]
    fn test_breakout_compute() {
        let data = make_test_data();
        let indicator = BreakoutPattern::new(15, 0.3, true).unwrap();
        let output = indicator.compute(&data);
        assert!(output.is_ok());
    }

    // ========== ConsolidationBreak Tests ==========

    #[test]
    fn test_consolidation_break_new_valid() {
        let indicator = ConsolidationBreak::new(15, 5.0, 5);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_consolidation_break_invalid_period() {
        assert!(ConsolidationBreak::new(3, 5.0, 3).is_err());
        assert!(ConsolidationBreak::new(60, 5.0, 5).is_err());
    }

    #[test]
    fn test_consolidation_break_invalid_range_pct() {
        assert!(ConsolidationBreak::new(15, 0.5, 5).is_err());
        assert!(ConsolidationBreak::new(15, 15.0, 5).is_err());
    }

    #[test]
    fn test_consolidation_break_invalid_min_bars() {
        assert!(ConsolidationBreak::new(15, 5.0, 1).is_err());
        assert!(ConsolidationBreak::new(15, 5.0, 25).is_err());
        assert!(ConsolidationBreak::new(15, 5.0, 20).is_err()); // > period
    }

    #[test]
    fn test_consolidation_break_calculate() {
        let data = make_test_data();
        let indicator = ConsolidationBreak::new(10, 5.0, 3).unwrap();
        let result = indicator.calculate(&data.high, &data.low, &data.close);

        assert_eq!(result.len(), data.close.len());

        // Values should be -1, 0, or 1
        for val in &result {
            assert!(*val == -1.0 || *val == 0.0 || *val == 1.0);
        }
    }

    #[test]
    fn test_consolidation_break_min_periods() {
        let indicator = ConsolidationBreak::new(20, 5.0, 5).unwrap();
        assert_eq!(indicator.min_periods(), 21);
    }

    #[test]
    fn test_consolidation_break_name() {
        let indicator = ConsolidationBreak::new(15, 5.0, 5).unwrap();
        assert_eq!(indicator.name(), "Consolidation Break");
    }

    #[test]
    fn test_consolidation_break_compute() {
        let data = make_test_data();
        let indicator = ConsolidationBreak::new(10, 5.0, 3).unwrap();
        let output = indicator.compute(&data);
        assert!(output.is_ok());
    }

    // ========== Edge Case Tests ==========

    #[test]
    fn test_empty_data() {
        let empty_data = OHLCVSeries {
            open: vec![],
            high: vec![],
            low: vec![],
            close: vec![],
            volume: vec![],
        };

        let tcp = TrendContinuationPattern::new(10, 3).unwrap();
        let result = tcp.calculate(&empty_data.high, &empty_data.low, &empty_data.close);
        assert_eq!(result.len(), 0);

        let mp = MomentumPattern::new(5, 15, 1.0).unwrap();
        let result = mp.calculate(&empty_data.close);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_single_data_point() {
        let single_data = OHLCVSeries {
            open: vec![100.0],
            high: vec![101.0],
            low: vec![99.0],
            close: vec![100.5],
            volume: vec![1000.0],
        };

        let rcp = ReversalCandlePattern::new(0.6, 2.0).unwrap();
        let result = rcp.calculate(&single_data.open, &single_data.high, &single_data.low, &single_data.close);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 0.0);
    }

    #[test]
    fn test_insufficient_data_for_lookback() {
        let short_data = OHLCVSeries {
            open: vec![100.0, 101.0, 102.0],
            high: vec![101.0, 102.0, 103.0],
            low: vec![99.0, 100.0, 101.0],
            close: vec![100.5, 101.5, 102.5],
            volume: vec![1000.0, 1100.0, 1200.0],
        };

        let bp = BreakoutPattern::new(10, 0.5, false).unwrap();
        let result = bp.calculate(&short_data.high, &short_data.low, &short_data.close, &short_data.volume);

        // Should return zeros for insufficient data
        assert_eq!(result.len(), 3);
        for val in &result {
            assert_eq!(*val, 0.0);
        }
    }

    // ========== PatternStrength Tests ==========

    #[test]
    fn test_pattern_strength_new_valid() {
        let indicator = PatternStrength::new(14, 0.7);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_pattern_strength_invalid_lookback() {
        assert!(PatternStrength::new(2, 0.7).is_err());
        assert!(PatternStrength::new(150, 0.7).is_err());
    }

    #[test]
    fn test_pattern_strength_invalid_threshold() {
        assert!(PatternStrength::new(14, 0.05).is_err());
        assert!(PatternStrength::new(14, 1.5).is_err());
    }

    #[test]
    fn test_pattern_strength_calculate() {
        let data = make_test_data();
        let indicator = PatternStrength::new(14, 0.5).unwrap();
        let result = indicator.calculate(&data.high, &data.low, &data.close, &data.volume);

        assert_eq!(result.len(), data.close.len());

        // Values should be between 0 and 1
        for val in &result {
            assert!(*val >= 0.0 && *val <= 1.0);
        }
    }

    #[test]
    fn test_pattern_strength_min_periods() {
        let indicator = PatternStrength::new(20, 0.6).unwrap();
        assert_eq!(indicator.min_periods(), 21);
    }

    #[test]
    fn test_pattern_strength_name() {
        let indicator = PatternStrength::new(14, 0.7).unwrap();
        assert_eq!(indicator.name(), "Pattern Strength");
    }

    #[test]
    fn test_pattern_strength_compute() {
        let data = make_test_data();
        let indicator = PatternStrength::new(14, 0.5).unwrap();
        let output = indicator.compute(&data);
        assert!(output.is_ok());
    }

    // ========== PatternProbability Tests ==========

    #[test]
    fn test_pattern_probability_new_valid() {
        let indicator = PatternProbability::new(20, 5);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_pattern_probability_invalid_lookback() {
        assert!(PatternProbability::new(4, 3).is_err());
        assert!(PatternProbability::new(250, 10).is_err());
    }

    #[test]
    fn test_pattern_probability_invalid_min_occurrences() {
        assert!(PatternProbability::new(20, 0).is_err());
        assert!(PatternProbability::new(20, 25).is_err());
    }

    #[test]
    fn test_pattern_probability_calculate() {
        let data = make_test_data();
        let indicator = PatternProbability::new(15, 3).unwrap();
        let result = indicator.calculate(&data.high, &data.low, &data.close);

        assert_eq!(result.len(), data.close.len());

        // Values should be between 0 and 1
        for val in &result {
            assert!(*val >= 0.0 && *val <= 1.0);
        }
    }

    #[test]
    fn test_pattern_probability_min_periods() {
        let indicator = PatternProbability::new(25, 5).unwrap();
        assert_eq!(indicator.min_periods(), 26);
    }

    #[test]
    fn test_pattern_probability_name() {
        let indicator = PatternProbability::new(20, 5).unwrap();
        assert_eq!(indicator.name(), "Pattern Probability");
    }

    #[test]
    fn test_pattern_probability_compute() {
        let data = make_test_data();
        let indicator = PatternProbability::new(15, 3).unwrap();
        let output = indicator.compute(&data);
        assert!(output.is_ok());
    }

    // ========== MultiTimeframePattern Tests ==========

    #[test]
    fn test_multi_timeframe_pattern_new_valid() {
        let indicator = MultiTimeframePattern::new(5, 20, 3);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_multi_timeframe_pattern_invalid_short_period() {
        assert!(MultiTimeframePattern::new(1, 20, 3).is_err());
        assert!(MultiTimeframePattern::new(25, 30, 3).is_err());
    }

    #[test]
    fn test_multi_timeframe_pattern_invalid_long_period() {
        assert!(MultiTimeframePattern::new(5, 9, 3).is_err());
        assert!(MultiTimeframePattern::new(5, 250, 3).is_err());
    }

    #[test]
    fn test_multi_timeframe_pattern_invalid_min_agreement() {
        assert!(MultiTimeframePattern::new(5, 20, 0).is_err());
        assert!(MultiTimeframePattern::new(5, 20, 6).is_err());
    }

    #[test]
    fn test_multi_timeframe_pattern_calculate() {
        let data = make_test_data();
        let indicator = MultiTimeframePattern::new(5, 15, 2).unwrap();
        let result = indicator.calculate(&data.high, &data.low, &data.close);

        assert_eq!(result.len(), data.close.len());

        // Values should be -1, 0, or 1
        for val in &result {
            assert!(*val == -1.0 || *val == 0.0 || *val == 1.0);
        }
    }

    #[test]
    fn test_multi_timeframe_pattern_min_periods() {
        let indicator = MultiTimeframePattern::new(5, 25, 2).unwrap();
        assert_eq!(indicator.min_periods(), 26);
    }

    #[test]
    fn test_multi_timeframe_pattern_name() {
        let indicator = MultiTimeframePattern::new(5, 20, 3).unwrap();
        assert_eq!(indicator.name(), "Multi-Timeframe Pattern");
    }

    #[test]
    fn test_multi_timeframe_pattern_compute() {
        let data = make_test_data();
        let indicator = MultiTimeframePattern::new(5, 15, 2).unwrap();
        let output = indicator.compute(&data);
        assert!(output.is_ok());
    }

    // ========== PatternCluster Tests ==========

    #[test]
    fn test_pattern_cluster_new_valid() {
        let indicator = PatternCluster::new(10, 2.0, 3);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_pattern_cluster_invalid_lookback() {
        assert!(PatternCluster::new(2, 2.0, 2).is_err());
        assert!(PatternCluster::new(150, 2.0, 3).is_err());
    }

    #[test]
    fn test_pattern_cluster_invalid_distance() {
        assert!(PatternCluster::new(10, 0.05, 2).is_err());
        assert!(PatternCluster::new(10, 12.0, 3).is_err());
    }

    #[test]
    fn test_pattern_cluster_invalid_min_patterns() {
        assert!(PatternCluster::new(10, 2.0, 1).is_err());
        assert!(PatternCluster::new(10, 2.0, 15).is_err());
    }

    #[test]
    fn test_pattern_cluster_calculate() {
        let data = make_test_data();
        let indicator = PatternCluster::new(10, 2.0, 2).unwrap();
        let result = indicator.calculate(&data.high, &data.low, &data.close, &data.volume);

        assert_eq!(result.len(), data.close.len());

        // Values should be -1, 0, or 1
        for val in &result {
            assert!(*val == -1.0 || *val == 0.0 || *val == 1.0);
        }
    }

    #[test]
    fn test_pattern_cluster_min_periods() {
        let indicator = PatternCluster::new(15, 2.0, 3).unwrap();
        assert_eq!(indicator.min_periods(), 16);
    }

    #[test]
    fn test_pattern_cluster_name() {
        let indicator = PatternCluster::new(10, 2.0, 3).unwrap();
        assert_eq!(indicator.name(), "Pattern Cluster");
    }

    #[test]
    fn test_pattern_cluster_compute() {
        let data = make_test_data();
        let indicator = PatternCluster::new(10, 2.0, 2).unwrap();
        let output = indicator.compute(&data);
        assert!(output.is_ok());
    }

    // ========== SequentialPattern Tests ==========

    #[test]
    fn test_sequential_pattern_new_valid() {
        let indicator = SequentialPattern::new(9, 4);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_sequential_pattern_invalid_count() {
        assert!(SequentialPattern::new(2, 2).is_err());
        assert!(SequentialPattern::new(25, 5).is_err());
    }

    #[test]
    fn test_sequential_pattern_invalid_comparison() {
        assert!(SequentialPattern::new(9, 0).is_err());
        assert!(SequentialPattern::new(9, 12).is_err());
    }

    #[test]
    fn test_sequential_pattern_calculate() {
        let data = make_test_data();
        let indicator = SequentialPattern::new(9, 4).unwrap();
        let result = indicator.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());

        // Values should be between -1 and 1 (normalized count)
        for val in &result {
            assert!(*val >= -1.0 && *val <= 1.0);
        }
    }

    #[test]
    fn test_sequential_pattern_min_periods() {
        let indicator = SequentialPattern::new(13, 4).unwrap();
        assert_eq!(indicator.min_periods(), 17); // setup_count + comparison_bars
    }

    #[test]
    fn test_sequential_pattern_name() {
        let indicator = SequentialPattern::new(9, 4).unwrap();
        assert_eq!(indicator.name(), "Sequential Pattern");
    }

    #[test]
    fn test_sequential_pattern_compute() {
        let data = make_test_data();
        let indicator = SequentialPattern::new(9, 4).unwrap();
        let output = indicator.compute(&data);
        assert!(output.is_ok());
    }

    // ========== PatternBreakoutStrength Tests ==========

    #[test]
    fn test_pattern_breakout_strength_new_valid() {
        let indicator = PatternBreakoutStrength::new(20, 2.0, 1.5);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_pattern_breakout_strength_invalid_lookback() {
        assert!(PatternBreakoutStrength::new(4, 2.0, 1.5).is_err());
        assert!(PatternBreakoutStrength::new(150, 2.0, 1.5).is_err());
    }

    #[test]
    fn test_pattern_breakout_strength_invalid_atr_mult() {
        assert!(PatternBreakoutStrength::new(20, 0.3, 1.5).is_err());
        assert!(PatternBreakoutStrength::new(20, 8.0, 1.5).is_err());
    }

    #[test]
    fn test_pattern_breakout_strength_invalid_vol_mult() {
        assert!(PatternBreakoutStrength::new(20, 2.0, 0.5).is_err());
        assert!(PatternBreakoutStrength::new(20, 2.0, 8.0).is_err());
    }

    #[test]
    fn test_pattern_breakout_strength_calculate() {
        let data = make_test_data();
        let indicator = PatternBreakoutStrength::new(14, 1.5, 1.2).unwrap();
        let result = indicator.calculate(&data.high, &data.low, &data.close, &data.volume);

        assert_eq!(result.len(), data.close.len());

        // Strength values should be between -1 and 1
        for val in &result {
            assert!(*val >= -1.0 && *val <= 1.0);
        }
    }

    #[test]
    fn test_pattern_breakout_strength_min_periods() {
        let indicator = PatternBreakoutStrength::new(25, 2.0, 1.5).unwrap();
        assert_eq!(indicator.min_periods(), 26);
    }

    #[test]
    fn test_pattern_breakout_strength_name() {
        let indicator = PatternBreakoutStrength::new(20, 2.0, 1.5).unwrap();
        assert_eq!(indicator.name(), "Pattern Breakout Strength");
    }

    #[test]
    fn test_pattern_breakout_strength_compute() {
        let data = make_test_data();
        let indicator = PatternBreakoutStrength::new(14, 1.5, 1.2).unwrap();
        let output = indicator.compute(&data);
        assert!(output.is_ok());
    }

    // ========== PricePatternRecognizer Tests ==========

    #[test]
    fn test_price_pattern_recognizer_new_valid() {
        let indicator = PricePatternRecognizer::new(20, 4, 5.0);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_price_pattern_recognizer_invalid_lookback() {
        assert!(PricePatternRecognizer::new(5, 4, 5.0).is_err());
        assert!(PricePatternRecognizer::new(150, 4, 5.0).is_err());
    }

    #[test]
    fn test_price_pattern_recognizer_invalid_min_points() {
        assert!(PricePatternRecognizer::new(20, 1, 5.0).is_err());
        assert!(PricePatternRecognizer::new(20, 15, 5.0).is_err());
    }

    #[test]
    fn test_price_pattern_recognizer_invalid_tolerance() {
        assert!(PricePatternRecognizer::new(20, 4, 0.2).is_err());
        assert!(PricePatternRecognizer::new(20, 4, 15.0).is_err());
    }

    #[test]
    fn test_price_pattern_recognizer_calculate() {
        let data = make_test_data();
        let indicator = PricePatternRecognizer::new(15, 3, 5.0).unwrap();
        let result = indicator.calculate(&data.high, &data.low, &data.close);

        assert_eq!(result.len(), data.close.len());

        // Values should be pattern codes (0-5)
        for val in &result {
            assert!(*val >= 0.0 && *val <= 5.0);
        }
    }

    #[test]
    fn test_price_pattern_recognizer_min_periods() {
        let indicator = PricePatternRecognizer::new(25, 4, 5.0).unwrap();
        assert_eq!(indicator.min_periods(), 26);
    }

    #[test]
    fn test_price_pattern_recognizer_name() {
        let indicator = PricePatternRecognizer::new(20, 4, 5.0).unwrap();
        assert_eq!(indicator.name(), "Price Pattern Recognizer");
    }

    #[test]
    fn test_price_pattern_recognizer_compute() {
        let data = make_test_data();
        let indicator = PricePatternRecognizer::new(15, 3, 5.0).unwrap();
        let output = indicator.compute(&data);
        assert!(output.is_ok());
    }

    // ========== ConsolidationDetector Tests ==========

    #[test]
    fn test_consolidation_detector_new_valid() {
        let indicator = ConsolidationDetector::new(15, 8.0, 10);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_consolidation_detector_invalid_lookback() {
        assert!(ConsolidationDetector::new(3, 8.0, 10).is_err());
        assert!(ConsolidationDetector::new(60, 8.0, 10).is_err());
    }

    #[test]
    fn test_consolidation_detector_invalid_max_range() {
        assert!(ConsolidationDetector::new(15, 0.5, 10).is_err());
        assert!(ConsolidationDetector::new(15, 20.0, 10).is_err());
    }

    #[test]
    fn test_consolidation_detector_invalid_atr_period() {
        assert!(ConsolidationDetector::new(15, 8.0, 2).is_err());
        assert!(ConsolidationDetector::new(15, 8.0, 35).is_err());
    }

    #[test]
    fn test_consolidation_detector_calculate() {
        let data = make_test_data();
        let indicator = ConsolidationDetector::new(10, 10.0, 7).unwrap();
        let result = indicator.calculate(&data.high, &data.low, &data.close);

        assert_eq!(result.len(), data.close.len());

        // Values should be between 0 and 1
        for val in &result {
            assert!(*val >= 0.0 && *val <= 1.0);
        }
    }

    #[test]
    fn test_consolidation_detector_min_periods() {
        let indicator = ConsolidationDetector::new(20, 8.0, 10).unwrap();
        assert_eq!(indicator.min_periods(), 21);
    }

    #[test]
    fn test_consolidation_detector_name() {
        let indicator = ConsolidationDetector::new(15, 8.0, 10).unwrap();
        assert_eq!(indicator.name(), "Consolidation Detector");
    }

    #[test]
    fn test_consolidation_detector_compute() {
        let data = make_test_data();
        let indicator = ConsolidationDetector::new(10, 10.0, 7).unwrap();
        let output = indicator.compute(&data);
        assert!(output.is_ok());
    }

    // ========== BreakoutPatternStrength Tests ==========

    #[test]
    fn test_breakout_pattern_strength_new_valid() {
        let indicator = BreakoutPatternStrength::new(15, 2.0, 1.5);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_breakout_pattern_strength_invalid_lookback() {
        assert!(BreakoutPatternStrength::new(3, 2.0, 1.5).is_err());
        assert!(BreakoutPatternStrength::new(60, 2.0, 1.5).is_err());
    }

    #[test]
    fn test_breakout_pattern_strength_invalid_volume_mult() {
        assert!(BreakoutPatternStrength::new(15, 0.5, 1.5).is_err());
        assert!(BreakoutPatternStrength::new(15, 6.0, 1.5).is_err());
    }

    #[test]
    fn test_breakout_pattern_strength_invalid_atr_mult() {
        assert!(BreakoutPatternStrength::new(15, 2.0, 0.3).is_err());
        assert!(BreakoutPatternStrength::new(15, 2.0, 4.0).is_err());
    }

    #[test]
    fn test_breakout_pattern_strength_calculate() {
        let data = make_test_data();
        let indicator = BreakoutPatternStrength::new(10, 1.5, 1.0).unwrap();
        let result = indicator.calculate(&data.high, &data.low, &data.close, &data.volume);

        assert_eq!(result.len(), data.close.len());

        // Values should be between -1 and 1
        for val in &result {
            assert!(*val >= -1.0 && *val <= 1.0);
        }
    }

    #[test]
    fn test_breakout_pattern_strength_min_periods() {
        let indicator = BreakoutPatternStrength::new(20, 2.0, 1.5).unwrap();
        assert_eq!(indicator.min_periods(), 21);
    }

    #[test]
    fn test_breakout_pattern_strength_name() {
        let indicator = BreakoutPatternStrength::new(15, 2.0, 1.5).unwrap();
        assert_eq!(indicator.name(), "Breakout Pattern Strength");
    }

    #[test]
    fn test_breakout_pattern_strength_compute() {
        let data = make_test_data();
        let indicator = BreakoutPatternStrength::new(10, 1.5, 1.0).unwrap();
        let output = indicator.compute(&data);
        assert!(output.is_ok());
    }

    // ========== ReversalPatternScore Tests ==========

    #[test]
    fn test_reversal_pattern_score_new_valid() {
        let indicator = ReversalPatternScore::new(20, 8, 0.5);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_reversal_pattern_score_invalid_lookback() {
        assert!(ReversalPatternScore::new(5, 4, 0.5).is_err());
        assert!(ReversalPatternScore::new(60, 8, 0.5).is_err());
    }

    #[test]
    fn test_reversal_pattern_score_invalid_momentum_period() {
        assert!(ReversalPatternScore::new(20, 3, 0.5).is_err());
        assert!(ReversalPatternScore::new(20, 25, 0.5).is_err());
        assert!(ReversalPatternScore::new(20, 20, 0.5).is_err()); // >= lookback
    }

    #[test]
    fn test_reversal_pattern_score_invalid_min_score() {
        assert!(ReversalPatternScore::new(20, 8, 0.2).is_err());
        assert!(ReversalPatternScore::new(20, 8, 0.95).is_err());
    }

    #[test]
    fn test_reversal_pattern_score_calculate() {
        let data = make_test_data();
        let indicator = ReversalPatternScore::new(15, 7, 0.4).unwrap();
        let result = indicator.calculate(&data.high, &data.low, &data.close, &data.volume);

        assert_eq!(result.len(), data.close.len());

        // Values should be between -1 and 1
        for val in &result {
            assert!(*val >= -1.0 && *val <= 1.0);
        }
    }

    #[test]
    fn test_reversal_pattern_score_min_periods() {
        let indicator = ReversalPatternScore::new(25, 10, 0.5).unwrap();
        assert_eq!(indicator.min_periods(), 26);
    }

    #[test]
    fn test_reversal_pattern_score_name() {
        let indicator = ReversalPatternScore::new(20, 8, 0.5).unwrap();
        assert_eq!(indicator.name(), "Reversal Pattern Score");
    }

    #[test]
    fn test_reversal_pattern_score_compute() {
        let data = make_test_data();
        let indicator = ReversalPatternScore::new(15, 7, 0.4).unwrap();
        let output = indicator.compute(&data);
        assert!(output.is_ok());
    }

    // ========== PatternSymmetry Tests ==========

    #[test]
    fn test_pattern_symmetry_new_valid() {
        let indicator = PatternSymmetry::new(20, 10.0);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_pattern_symmetry_invalid_lookback() {
        assert!(PatternSymmetry::new(5, 10.0).is_err());
        assert!(PatternSymmetry::new(150, 10.0).is_err());
    }

    #[test]
    fn test_pattern_symmetry_invalid_tolerance() {
        assert!(PatternSymmetry::new(20, 0.5).is_err());
        assert!(PatternSymmetry::new(20, 25.0).is_err());
    }

    #[test]
    fn test_pattern_symmetry_calculate() {
        let data = make_test_data();
        let indicator = PatternSymmetry::new(15, 10.0).unwrap();
        let result = indicator.calculate(&data.high, &data.low, &data.close);

        assert_eq!(result.len(), data.close.len());

        // Values should be between 0 and 1
        for val in &result {
            assert!(*val >= 0.0 && *val <= 1.0);
        }
    }

    #[test]
    fn test_pattern_symmetry_min_periods() {
        let indicator = PatternSymmetry::new(25, 10.0).unwrap();
        assert_eq!(indicator.min_periods(), 26);
    }

    #[test]
    fn test_pattern_symmetry_name() {
        let indicator = PatternSymmetry::new(20, 10.0).unwrap();
        assert_eq!(indicator.name(), "Pattern Symmetry");
    }

    #[test]
    fn test_pattern_symmetry_compute() {
        let data = make_test_data();
        let indicator = PatternSymmetry::new(15, 10.0).unwrap();
        let output = indicator.compute(&data);
        assert!(output.is_ok());
    }

    // ========== TrendContinuationStrength Tests ==========

    #[test]
    fn test_trend_continuation_strength_new_valid() {
        let indicator = TrendContinuationStrength::new(20, 5, 30.0);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_trend_continuation_strength_invalid_lookback() {
        assert!(TrendContinuationStrength::new(5, 3, 30.0).is_err());
        assert!(TrendContinuationStrength::new(60, 5, 30.0).is_err());
    }

    #[test]
    fn test_trend_continuation_strength_invalid_short_period() {
        assert!(TrendContinuationStrength::new(20, 1, 30.0).is_err());
        assert!(TrendContinuationStrength::new(20, 18, 30.0).is_err());
        assert!(TrendContinuationStrength::new(20, 20, 30.0).is_err()); // >= lookback
    }

    #[test]
    fn test_trend_continuation_strength_invalid_max_pullback() {
        assert!(TrendContinuationStrength::new(20, 5, 5.0).is_err());
        assert!(TrendContinuationStrength::new(20, 5, 60.0).is_err());
    }

    #[test]
    fn test_trend_continuation_strength_calculate() {
        let data = make_test_data();
        let indicator = TrendContinuationStrength::new(15, 5, 35.0).unwrap();
        let result = indicator.calculate(&data.high, &data.low, &data.close, &data.volume);

        assert_eq!(result.len(), data.close.len());

        // Values should be between -1 and 1
        for val in &result {
            assert!(*val >= -1.0 && *val <= 1.0);
        }
    }

    #[test]
    fn test_trend_continuation_strength_min_periods() {
        let indicator = TrendContinuationStrength::new(25, 7, 30.0).unwrap();
        assert_eq!(indicator.min_periods(), 26);
    }

    #[test]
    fn test_trend_continuation_strength_name() {
        let indicator = TrendContinuationStrength::new(20, 5, 30.0).unwrap();
        assert_eq!(indicator.name(), "Trend Continuation Strength");
    }

    #[test]
    fn test_trend_continuation_strength_compute() {
        let data = make_test_data();
        let indicator = TrendContinuationStrength::new(15, 5, 35.0).unwrap();
        let output = indicator.compute(&data);
        assert!(output.is_ok());
    }

    // ========== Additional Edge Case Tests for New Indicators ==========

    #[test]
    fn test_new_indicators_empty_data() {
        let empty_data = OHLCVSeries {
            open: vec![],
            high: vec![],
            low: vec![],
            close: vec![],
            volume: vec![],
        };

        let ppr = PricePatternRecognizer::new(15, 3, 5.0).unwrap();
        assert_eq!(ppr.calculate(&empty_data.high, &empty_data.low, &empty_data.close).len(), 0);

        let cd = ConsolidationDetector::new(10, 8.0, 7).unwrap();
        assert_eq!(cd.calculate(&empty_data.high, &empty_data.low, &empty_data.close).len(), 0);

        let bps = BreakoutPatternStrength::new(10, 1.5, 1.0).unwrap();
        assert_eq!(bps.calculate(&empty_data.high, &empty_data.low, &empty_data.close, &empty_data.volume).len(), 0);

        let rps = ReversalPatternScore::new(15, 7, 0.4).unwrap();
        assert_eq!(rps.calculate(&empty_data.high, &empty_data.low, &empty_data.close, &empty_data.volume).len(), 0);

        let ps = PatternSymmetry::new(15, 10.0).unwrap();
        assert_eq!(ps.calculate(&empty_data.high, &empty_data.low, &empty_data.close).len(), 0);

        let tcs = TrendContinuationStrength::new(15, 5, 35.0).unwrap();
        assert_eq!(tcs.calculate(&empty_data.high, &empty_data.low, &empty_data.close, &empty_data.volume).len(), 0);
    }

    #[test]
    fn test_new_indicators_insufficient_data() {
        let short_data = OHLCVSeries {
            open: vec![100.0, 101.0, 102.0],
            high: vec![101.0, 102.0, 103.0],
            low: vec![99.0, 100.0, 101.0],
            close: vec![100.5, 101.5, 102.5],
            volume: vec![1000.0, 1100.0, 1200.0],
        };

        let ppr = PricePatternRecognizer::new(15, 3, 5.0).unwrap();
        let result = ppr.calculate(&short_data.high, &short_data.low, &short_data.close);
        assert_eq!(result.len(), 3);
        for val in &result {
            assert_eq!(*val, 0.0);
        }

        let ps = PatternSymmetry::new(15, 10.0).unwrap();
        let result = ps.calculate(&short_data.high, &short_data.low, &short_data.close);
        assert_eq!(result.len(), 3);
        for val in &result {
            assert_eq!(*val, 0.0);
        }
    }

    // ========== SwingPointDetector Tests ==========

    #[test]
    fn test_swing_point_detector_new_valid() {
        let indicator = SwingPointDetector::new(5, 1.0);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_swing_point_detector_invalid_strength() {
        assert!(SwingPointDetector::new(1, 1.0).is_err());
        assert!(SwingPointDetector::new(25, 1.0).is_err());
    }

    #[test]
    fn test_swing_point_detector_invalid_min_move() {
        assert!(SwingPointDetector::new(5, 0.05).is_err());
        assert!(SwingPointDetector::new(5, 6.0).is_err());
    }

    #[test]
    fn test_swing_point_detector_calculate() {
        let data = make_test_data();
        let indicator = SwingPointDetector::new(3, 0.5).unwrap();
        let result = indicator.calculate(&data.high, &data.low, &data.close);

        assert_eq!(result.len(), data.close.len());

        // Values should be -1, 0, or 1
        for val in &result {
            assert!(*val == -1.0 || *val == 0.0 || *val == 1.0);
        }
    }

    #[test]
    fn test_swing_point_detector_min_periods() {
        let indicator = SwingPointDetector::new(5, 1.0).unwrap();
        assert_eq!(indicator.min_periods(), 11); // 5 * 2 + 1
    }

    #[test]
    fn test_swing_point_detector_name() {
        let indicator = SwingPointDetector::new(5, 1.0).unwrap();
        assert_eq!(indicator.name(), "Swing Point Detector");
    }

    #[test]
    fn test_swing_point_detector_compute() {
        let data = make_test_data();
        let indicator = SwingPointDetector::new(3, 0.5).unwrap();
        let output = indicator.compute(&data);
        assert!(output.is_ok());
    }

    // ========== SupportResistanceStrength Tests ==========

    #[test]
    fn test_support_resistance_strength_new_valid() {
        let indicator = SupportResistanceStrength::new(50, 1.0, 3);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_support_resistance_strength_invalid_lookback() {
        assert!(SupportResistanceStrength::new(5, 1.0, 3).is_err());
        assert!(SupportResistanceStrength::new(250, 1.0, 3).is_err());
    }

    #[test]
    fn test_support_resistance_strength_invalid_tolerance() {
        assert!(SupportResistanceStrength::new(50, 0.05, 3).is_err());
        assert!(SupportResistanceStrength::new(50, 4.0, 3).is_err());
    }

    #[test]
    fn test_support_resistance_strength_invalid_min_touches() {
        assert!(SupportResistanceStrength::new(50, 1.0, 1).is_err());
        assert!(SupportResistanceStrength::new(50, 1.0, 15).is_err());
    }

    #[test]
    fn test_support_resistance_strength_calculate() {
        let data = make_test_data();
        let indicator = SupportResistanceStrength::new(20, 1.0, 2).unwrap();
        let result = indicator.calculate(&data.high, &data.low, &data.close, &data.volume);

        assert_eq!(result.len(), data.close.len());

        // Values should be between -1 and 1
        for val in &result {
            assert!(*val >= -1.0 && *val <= 1.0);
        }
    }

    #[test]
    fn test_support_resistance_strength_min_periods() {
        let indicator = SupportResistanceStrength::new(50, 1.0, 3).unwrap();
        assert_eq!(indicator.min_periods(), 51);
    }

    #[test]
    fn test_support_resistance_strength_name() {
        let indicator = SupportResistanceStrength::new(50, 1.0, 3).unwrap();
        assert_eq!(indicator.name(), "Support Resistance Strength");
    }

    #[test]
    fn test_support_resistance_strength_compute() {
        let data = make_test_data();
        let indicator = SupportResistanceStrength::new(20, 1.0, 2).unwrap();
        let output = indicator.compute(&data);
        assert!(output.is_ok());
    }

    // ========== PriceActionMomentum Tests ==========

    #[test]
    fn test_price_action_momentum_new_valid() {
        let indicator = PriceActionMomentum::new(14, 5);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_price_action_momentum_invalid_lookback() {
        assert!(PriceActionMomentum::new(3, 2).is_err());
        assert!(PriceActionMomentum::new(60, 5).is_err());
    }

    #[test]
    fn test_price_action_momentum_invalid_smoothing() {
        assert!(PriceActionMomentum::new(14, 0).is_err());
        assert!(PriceActionMomentum::new(14, 25).is_err());
    }

    #[test]
    fn test_price_action_momentum_smoothing_exceeds_lookback() {
        assert!(PriceActionMomentum::new(10, 15).is_err());
    }

    #[test]
    fn test_price_action_momentum_calculate() {
        let data = make_test_data();
        let indicator = PriceActionMomentum::new(10, 3).unwrap();
        let result = indicator.calculate(&data.open, &data.high, &data.low, &data.close);

        assert_eq!(result.len(), data.close.len());

        // Values should be between -1 and 1
        for val in &result {
            assert!(*val >= -1.0 && *val <= 1.0);
        }
    }

    #[test]
    fn test_price_action_momentum_min_periods() {
        let indicator = PriceActionMomentum::new(14, 5).unwrap();
        assert_eq!(indicator.min_periods(), 19); // lookback + smoothing
    }

    #[test]
    fn test_price_action_momentum_name() {
        let indicator = PriceActionMomentum::new(14, 5).unwrap();
        assert_eq!(indicator.name(), "Price Action Momentum");
    }

    #[test]
    fn test_price_action_momentum_compute() {
        let data = make_test_data();
        let indicator = PriceActionMomentum::new(10, 3).unwrap();
        let output = indicator.compute(&data);
        assert!(output.is_ok());
    }

    // ========== CandleRangeAnalysis Tests ==========

    #[test]
    fn test_candle_range_analysis_new_valid() {
        let indicator = CandleRangeAnalysis::new(14, 0.5, 2.0);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_candle_range_analysis_invalid_atr_period() {
        assert!(CandleRangeAnalysis::new(3, 0.5, 2.0).is_err());
        assert!(CandleRangeAnalysis::new(60, 0.5, 2.0).is_err());
    }

    #[test]
    fn test_candle_range_analysis_invalid_narrow_threshold() {
        assert!(CandleRangeAnalysis::new(14, 0.2, 2.0).is_err());
        assert!(CandleRangeAnalysis::new(14, 0.9, 2.0).is_err());
    }

    #[test]
    fn test_candle_range_analysis_invalid_wide_threshold() {
        assert!(CandleRangeAnalysis::new(14, 0.5, 1.2).is_err());
        assert!(CandleRangeAnalysis::new(14, 0.5, 5.0).is_err());
    }

    #[test]
    fn test_candle_range_analysis_narrow_exceeds_wide() {
        assert!(CandleRangeAnalysis::new(14, 0.7, 0.6).is_err());
    }

    #[test]
    fn test_candle_range_analysis_calculate() {
        let data = make_test_data();
        let indicator = CandleRangeAnalysis::new(10, 0.5, 2.0).unwrap();
        let result = indicator.calculate(&data.high, &data.low, &data.close);

        assert_eq!(result.len(), data.close.len());
    }

    #[test]
    fn test_candle_range_analysis_min_periods() {
        let indicator = CandleRangeAnalysis::new(14, 0.5, 2.0).unwrap();
        assert_eq!(indicator.min_periods(), 15);
    }

    #[test]
    fn test_candle_range_analysis_name() {
        let indicator = CandleRangeAnalysis::new(14, 0.5, 2.0).unwrap();
        assert_eq!(indicator.name(), "Candle Range Analysis");
    }

    #[test]
    fn test_candle_range_analysis_compute() {
        let data = make_test_data();
        let indicator = CandleRangeAnalysis::new(10, 0.5, 2.0).unwrap();
        let output = indicator.compute(&data);
        assert!(output.is_ok());
    }

    // ========== TrendLineBreak Tests ==========

    #[test]
    fn test_trend_line_break_new_valid() {
        let indicator = TrendLineBreak::new(30, 3, 0.5);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_trend_line_break_invalid_lookback() {
        assert!(TrendLineBreak::new(5, 3, 0.5).is_err());
        assert!(TrendLineBreak::new(150, 3, 0.5).is_err());
    }

    #[test]
    fn test_trend_line_break_invalid_min_points() {
        assert!(TrendLineBreak::new(30, 1, 0.5).is_err());
        assert!(TrendLineBreak::new(30, 7, 0.5).is_err());
    }

    #[test]
    fn test_trend_line_break_invalid_breakout_pct() {
        assert!(TrendLineBreak::new(30, 3, 0.05).is_err());
        assert!(TrendLineBreak::new(30, 3, 3.0).is_err());
    }

    #[test]
    fn test_trend_line_break_calculate() {
        let data = make_test_data();
        let indicator = TrendLineBreak::new(20, 2, 0.5).unwrap();
        let result = indicator.calculate(&data.high, &data.low, &data.close);

        assert_eq!(result.len(), data.close.len());

        // Values should be -1, 0, or 1
        for val in &result {
            assert!(*val == -1.0 || *val == 0.0 || *val == 1.0);
        }
    }

    #[test]
    fn test_trend_line_break_min_periods() {
        let indicator = TrendLineBreak::new(30, 3, 0.5).unwrap();
        assert_eq!(indicator.min_periods(), 32); // lookback + 2
    }

    #[test]
    fn test_trend_line_break_name() {
        let indicator = TrendLineBreak::new(30, 3, 0.5).unwrap();
        assert_eq!(indicator.name(), "Trend Line Break");
    }

    #[test]
    fn test_trend_line_break_compute() {
        let data = make_test_data();
        let indicator = TrendLineBreak::new(20, 2, 0.5).unwrap();
        let output = indicator.compute(&data);
        assert!(output.is_ok());
    }

    // ========== PriceStructure Tests ==========

    #[test]
    fn test_price_structure_new_valid() {
        let indicator = PriceStructure::new(30, 3);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_price_structure_invalid_lookback() {
        assert!(PriceStructure::new(5, 3).is_err());
        assert!(PriceStructure::new(150, 3).is_err());
    }

    #[test]
    fn test_price_structure_invalid_swing_strength() {
        assert!(PriceStructure::new(30, 1).is_err());
        assert!(PriceStructure::new(30, 15).is_err());
    }

    #[test]
    fn test_price_structure_swing_strength_exceeds_half_lookback() {
        assert!(PriceStructure::new(20, 10).is_err()); // 10 * 2 >= 20
    }

    #[test]
    fn test_price_structure_calculate() {
        let data = make_test_data();
        let indicator = PriceStructure::new(20, 3).unwrap();
        let result = indicator.calculate(&data.high, &data.low);

        assert_eq!(result.len(), data.close.len());

        // Values should be -2, -1, 0, 1, or 2
        for val in &result {
            assert!(*val == -2.0 || *val == -1.0 || *val == 0.0 || *val == 1.0 || *val == 2.0);
        }
    }

    #[test]
    fn test_price_structure_min_periods() {
        let indicator = PriceStructure::new(30, 3).unwrap();
        assert_eq!(indicator.min_periods(), 33); // lookback + swing_strength
    }

    #[test]
    fn test_price_structure_name() {
        let indicator = PriceStructure::new(30, 3).unwrap();
        assert_eq!(indicator.name(), "Price Structure");
    }

    #[test]
    fn test_price_structure_compute() {
        let data = make_test_data();
        let indicator = PriceStructure::new(20, 3).unwrap();
        let output = indicator.compute(&data);
        assert!(output.is_ok());
    }

    // ========== Edge Cases for New 6 Indicators ==========

    #[test]
    fn test_new_six_indicators_empty_data() {
        let empty_data = OHLCVSeries {
            open: vec![],
            high: vec![],
            low: vec![],
            close: vec![],
            volume: vec![],
        };

        let spd = SwingPointDetector::new(3, 0.5).unwrap();
        assert_eq!(spd.calculate(&empty_data.high, &empty_data.low, &empty_data.close).len(), 0);

        let srs = SupportResistanceStrength::new(20, 1.0, 2).unwrap();
        assert_eq!(srs.calculate(&empty_data.high, &empty_data.low, &empty_data.close, &empty_data.volume).len(), 0);

        let pam = PriceActionMomentum::new(10, 3).unwrap();
        assert_eq!(pam.calculate(&empty_data.open, &empty_data.high, &empty_data.low, &empty_data.close).len(), 0);

        let cra = CandleRangeAnalysis::new(10, 0.5, 2.0).unwrap();
        assert_eq!(cra.calculate(&empty_data.high, &empty_data.low, &empty_data.close).len(), 0);

        let tlb = TrendLineBreak::new(20, 2, 0.5).unwrap();
        assert_eq!(tlb.calculate(&empty_data.high, &empty_data.low, &empty_data.close).len(), 0);

        let ps = PriceStructure::new(20, 3).unwrap();
        assert_eq!(ps.calculate(&empty_data.high, &empty_data.low).len(), 0);
    }

    #[test]
    fn test_new_six_indicators_insufficient_data() {
        let short_data = OHLCVSeries {
            open: vec![100.0, 101.0, 102.0],
            high: vec![101.0, 102.0, 103.0],
            low: vec![99.0, 100.0, 101.0],
            close: vec![100.5, 101.5, 102.5],
            volume: vec![1000.0, 1100.0, 1200.0],
        };

        let spd = SwingPointDetector::new(3, 0.5).unwrap();
        let result = spd.calculate(&short_data.high, &short_data.low, &short_data.close);
        assert_eq!(result.len(), 3);
        for val in &result {
            assert_eq!(*val, 0.0);
        }

        let cra = CandleRangeAnalysis::new(10, 0.5, 2.0).unwrap();
        let result = cra.calculate(&short_data.high, &short_data.low, &short_data.close);
        assert_eq!(result.len(), 3);
        for val in &result {
            assert_eq!(*val, 0.0);
        }

        let ps = PriceStructure::new(20, 3).unwrap();
        let result = ps.calculate(&short_data.high, &short_data.low);
        assert_eq!(result.len(), 3);
        for val in &result {
            assert_eq!(*val, 0.0);
        }
    }
}
