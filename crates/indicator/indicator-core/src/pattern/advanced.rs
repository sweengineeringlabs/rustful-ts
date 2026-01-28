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
}
