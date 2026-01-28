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
}
