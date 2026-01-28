//! Minervini/Zanger Trading Pattern Indicators
//!
//! Pattern recognition tools based on Mark Minervini and Dan Zanger's methodologies.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};
use crate::SMA;

/// Trend Template - Minervini's 8-criteria trend qualification.
///
/// Checks if a stock meets all 8 criteria for a Stage 2 uptrend:
/// 1. Price above 150-day and 200-day MA
/// 2. 150-day MA above 200-day MA
/// 3. 200-day MA trending up for at least 1 month
/// 4. 50-day MA above 150-day and 200-day MA
/// 5. Price above 50-day MA
/// 6. Price at least 25% above 52-week low
/// 7. Price within 25% of 52-week high
/// 8. RS rating above 70 (approximated by relative performance)
#[derive(Debug, Clone)]
pub struct TrendTemplate {
    short_ma: usize,
    medium_ma: usize,
    long_ma: usize,
    trend_lookback: usize,
}

impl TrendTemplate {
    pub fn new() -> Self {
        Self {
            short_ma: 50,
            medium_ma: 150,
            long_ma: 200,
            trend_lookback: 252, // 52 weeks
        }
    }

    /// Calculate trend template score (0-8 based on criteria met).
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let min_required = self.trend_lookback.max(self.long_ma + 22);

        if n < min_required {
            return vec![f64::NAN; n];
        }

        let ma50 = SMA::new(self.short_ma).calculate(close);
        let ma150 = SMA::new(self.medium_ma).calculate(close);
        let ma200 = SMA::new(self.long_ma).calculate(close);

        let mut result = vec![f64::NAN; n];

        for i in min_required..n {
            let mut score = 0.0;

            // Skip if MAs not ready
            if ma50[i].is_nan() || ma150[i].is_nan() || ma200[i].is_nan() {
                continue;
            }

            // 1. Price above 150-day and 200-day MA
            if close[i] > ma150[i] && close[i] > ma200[i] {
                score += 1.0;
            }

            // 2. 150-day MA above 200-day MA
            if ma150[i] > ma200[i] {
                score += 1.0;
            }

            // 3. 200-day MA trending up (compare to 1 month ago)
            if i >= 22 && !ma200[i - 22].is_nan() && ma200[i] > ma200[i - 22] {
                score += 1.0;
            }

            // 4. 50-day MA above 150-day and 200-day MA
            if ma50[i] > ma150[i] && ma50[i] > ma200[i] {
                score += 1.0;
            }

            // 5. Price above 50-day MA
            if close[i] > ma50[i] {
                score += 1.0;
            }

            // 6. Price at least 25% above 52-week low
            let start_52w = i.saturating_sub(self.trend_lookback - 1);
            let low_52w = close[start_52w..=i].iter().cloned().fold(f64::INFINITY, f64::min);
            if close[i] >= low_52w * 1.25 {
                score += 1.0;
            }

            // 7. Price within 25% of 52-week high
            let high_52w = close[start_52w..=i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if close[i] >= high_52w * 0.75 {
                score += 1.0;
            }

            // 8. Relative strength (simplified: price above 90-day performance)
            if i >= 90 && close[i - 90] > 0.0 {
                let perf = (close[i] / close[i - 90] - 1.0) * 100.0;
                if perf > 0.0 {
                    score += 1.0;
                }
            }

            result[i] = score;
        }

        result
    }
}

impl Default for TrendTemplate {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for TrendTemplate {
    fn name(&self) -> &str {
        "Trend Template"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.trend_lookback.max(self.long_ma + 22);
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.close);
        Ok(IndicatorOutput::single(result))
    }

    fn min_periods(&self) -> usize {
        self.trend_lookback.max(self.long_ma + 22)
    }

    fn output_features(&self) -> usize {
        1
    }
}

impl SignalIndicator for TrendTemplate {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let scores = self.calculate(&data.close);

        if scores.is_empty() {
            return Ok(IndicatorSignal::Neutral);
        }

        let score = *scores.last().unwrap();
        if score.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        if score >= 7.0 {
            Ok(IndicatorSignal::Bullish)
        } else if score <= 3.0 {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let scores = self.calculate(&data.close);

        Ok(scores.iter().map(|&s| {
            if s.is_nan() {
                IndicatorSignal::Neutral
            } else if s >= 7.0 {
                IndicatorSignal::Bullish
            } else if s <= 3.0 {
                IndicatorSignal::Bearish
            } else {
                IndicatorSignal::Neutral
            }
        }).collect())
    }
}

/// Volatility Contraction Pattern (VCP) - Minervini's key pattern.
///
/// Detects contracting volatility bases with decreasing volume.
/// A valid VCP has progressively smaller price swings before breakout.
#[derive(Debug, Clone)]
pub struct VolatilityContractionPattern {
    lookback: usize,
    contraction_threshold: f64,
}

impl VolatilityContractionPattern {
    pub fn new(lookback: usize) -> Self {
        Self { lookback, contraction_threshold: 0.5 }
    }

    /// Calculate VCP score (0-100).
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < self.lookback + 10 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; n];

        for i in (self.lookback + 9)..n {
            let start = i - self.lookback;

            // Divide into 3 equal periods
            let period_len = self.lookback / 3;
            if period_len < 3 {
                continue;
            }

            // Calculate range for each period
            let mut ranges = Vec::new();
            let mut volumes = Vec::new();

            for p in 0..3 {
                let p_start = start + p * period_len;
                let p_end = (start + (p + 1) * period_len).min(i);

                let h_max = high[p_start..p_end].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let l_min = low[p_start..p_end].iter().cloned().fold(f64::INFINITY, f64::min);
                let avg_vol: f64 = volume[p_start..p_end].iter().sum::<f64>() / (p_end - p_start) as f64;

                ranges.push(h_max - l_min);
                volumes.push(avg_vol);
            }

            if ranges.len() < 3 {
                continue;
            }

            let mut score = 0.0;

            // Check for range contraction
            if ranges[1] < ranges[0] {
                score += 25.0;
            }
            if ranges[2] < ranges[1] {
                score += 25.0;
            }

            // Check for volume contraction
            if volumes[1] < volumes[0] {
                score += 15.0;
            }
            if volumes[2] < volumes[1] {
                score += 15.0;
            }

            // Check if price is near the high of the base
            let base_high = high[start..=i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if close[i] >= base_high * 0.95 {
                score += 20.0;
            }

            result[i] = score;
        }

        result
    }
}

impl TechnicalIndicator for VolatilityContractionPattern {
    fn name(&self) -> &str {
        "Volatility Contraction Pattern"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.lookback + 10 {
            return Err(IndicatorError::InsufficientData {
                required: self.lookback + 10,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::single(result))
    }

    fn min_periods(&self) -> usize {
        self.lookback + 10
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Pocket Pivot - Chris Kacher/Gil Morales volume signal.
///
/// Up day with volume greater than any down day's volume in the past 10 days.
#[derive(Debug, Clone)]
pub struct PocketPivot {
    lookback: usize,
}

impl PocketPivot {
    pub fn new() -> Self {
        Self { lookback: 10 }
    }

    pub fn with_lookback(lookback: usize) -> Self {
        Self { lookback }
    }

    /// Calculate pocket pivot signals.
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < self.lookback + 1 {
            return vec![0.0; n];
        }

        let mut result = vec![0.0; n];

        for i in self.lookback..n {
            // Check if today is an up day
            if close[i] <= close[i - 1] {
                continue;
            }

            // Find max down-day volume in lookback period
            let mut max_down_vol: f64 = 0.0;
            for j in (i - self.lookback)..i {
                if j > 0 && close[j] < close[j - 1] {
                    max_down_vol = max_down_vol.max(volume[j]);
                }
            }

            // Pocket pivot: today's volume > max down-day volume
            if volume[i] > max_down_vol && max_down_vol > 0.0 {
                result[i] = 1.0;
            }
        }

        result
    }
}

impl Default for PocketPivot {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for PocketPivot {
    fn name(&self) -> &str {
        "Pocket Pivot"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.lookback + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.lookback + 1,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.close, &data.volume);
        Ok(IndicatorOutput::single(result))
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn output_features(&self) -> usize {
        1
    }
}

impl SignalIndicator for PocketPivot {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let signals = self.calculate(&data.close, &data.volume);

        if signals.is_empty() {
            return Ok(IndicatorSignal::Neutral);
        }

        if *signals.last().unwrap() > 0.5 {
            Ok(IndicatorSignal::Bullish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let signals = self.calculate(&data.close, &data.volume);

        Ok(signals.iter().map(|&s| {
            if s > 0.5 {
                IndicatorSignal::Bullish
            } else {
                IndicatorSignal::Neutral
            }
        }).collect())
    }
}

/// Power Play - Strong momentum surge pattern.
///
/// Identifies stocks making powerful moves with strong volume.
#[derive(Debug, Clone)]
pub struct PowerPlay {
    lookback: usize,
    price_threshold: f64,
    volume_threshold: f64,
}

impl PowerPlay {
    pub fn new() -> Self {
        Self {
            lookback: 20,
            price_threshold: 0.25, // 25% gain
            volume_threshold: 2.0, // 2x average volume
        }
    }

    /// Calculate power play signals.
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < self.lookback + 1 {
            return vec![0.0; n];
        }

        let mut result = vec![0.0; n];

        for i in self.lookback..n {
            let start = i - self.lookback;

            // Calculate price gain from low of lookback
            let low = close[start..i].iter().cloned().fold(f64::INFINITY, f64::min);
            let gain = (close[i] - low) / low;

            // Calculate average volume
            let avg_vol: f64 = volume[start..i].iter().sum::<f64>() / self.lookback as f64;
            let vol_ratio = if avg_vol > 0.0 { volume[i] / avg_vol } else { 0.0 };

            // Power play: significant gain with volume confirmation
            if gain >= self.price_threshold && vol_ratio >= self.volume_threshold {
                result[i] = 1.0;
            }
        }

        result
    }
}

impl Default for PowerPlay {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for PowerPlay {
    fn name(&self) -> &str {
        "Power Play"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.lookback + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.lookback + 1,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.close, &data.volume);
        Ok(IndicatorOutput::single(result))
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Bull Flag Pattern - Consolidation after strong move.
///
/// Detects tight consolidation patterns following a sharp advance.
#[derive(Debug, Clone)]
pub struct BullFlag {
    pole_lookback: usize,
    flag_lookback: usize,
}

impl BullFlag {
    pub fn new() -> Self {
        Self {
            pole_lookback: 10,
            flag_lookback: 5,
        }
    }

    /// Calculate bull flag score (0-100).
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let total_lookback = self.pole_lookback + self.flag_lookback;

        if n < total_lookback + 1 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; n];

        for i in total_lookback..n {
            let pole_start = i - total_lookback;
            let flag_start = i - self.flag_lookback;

            // Check pole: strong advance
            let pole_low = close[pole_start];
            let pole_high = close[flag_start].max(high[flag_start]);
            let pole_gain = (pole_high - pole_low) / pole_low;

            if pole_gain < 0.05 {
                // Need at least 5% pole
                continue;
            }

            // Check flag: tight consolidation
            let flag_high = high[flag_start..=i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let flag_low = low[flag_start..=i].iter().cloned().fold(f64::INFINITY, f64::min);
            let flag_range = (flag_high - flag_low) / pole_high;

            let mut score = 0.0;

            // Strong pole
            if pole_gain >= 0.15 {
                score += 40.0;
            } else if pole_gain >= 0.10 {
                score += 30.0;
            } else if pole_gain >= 0.05 {
                score += 20.0;
            }

            // Tight flag (smaller range = better)
            if flag_range < 0.05 {
                score += 40.0;
            } else if flag_range < 0.08 {
                score += 30.0;
            } else if flag_range < 0.10 {
                score += 20.0;
            }

            // Price near flag high (ready to break out)
            if close[i] >= flag_high * 0.97 {
                score += 20.0;
            }

            result[i] = score;
        }

        result
    }
}

impl Default for BullFlag {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for BullFlag {
    fn name(&self) -> &str {
        "Bull Flag"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let total_lookback = self.pole_lookback + self.flag_lookback;
        if data.close.len() < total_lookback + 1 {
            return Err(IndicatorError::InsufficientData {
                required: total_lookback + 1,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::single(result))
    }

    fn min_periods(&self) -> usize {
        self.pole_lookback + self.flag_lookback + 1
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Cup Pattern Detection - Cup with handle base pattern.
///
/// Simplified detection of William O'Neil's cup with handle pattern.
#[derive(Debug, Clone)]
pub struct CupPattern {
    lookback: usize,
}

impl CupPattern {
    pub fn new(lookback: usize) -> Self {
        Self { lookback }
    }

    /// Calculate cup pattern score (0-100).
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < self.lookback + 10 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; n];

        for i in (self.lookback + 9)..n {
            let start = i - self.lookback;

            // Find the cup structure
            let left_rim = close[start];
            let right_rim = close[i];

            // Find the bottom of the cup (minimum in middle portion)
            let mid = start + self.lookback / 2;
            let cup_bottom = close[(start + 5)..(mid + 5)]
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);

            // Cup depth
            let rim_avg = (left_rim + right_rim) / 2.0;
            let cup_depth = (rim_avg - cup_bottom) / rim_avg;

            let mut score = 0.0;

            // Good cup depth (12-35% is ideal)
            if cup_depth >= 0.12 && cup_depth <= 0.35 {
                score += 40.0;
            } else if cup_depth >= 0.08 && cup_depth <= 0.40 {
                score += 25.0;
            }

            // Rims at similar levels
            let rim_diff = ((left_rim - right_rim) / left_rim).abs();
            if rim_diff < 0.05 {
                score += 30.0;
            } else if rim_diff < 0.10 {
                score += 20.0;
            }

            // Price near right rim (breakout potential)
            let recent_high = high[(i - 5)..=i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if close[i] >= recent_high * 0.98 {
                score += 30.0;
            }

            result[i] = score;
        }

        result
    }
}

impl TechnicalIndicator for CupPattern {
    fn name(&self) -> &str {
        "Cup Pattern"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.lookback + 10 {
            return Err(IndicatorError::InsufficientData {
                required: self.lookback + 10,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::single(result))
    }

    fn min_periods(&self) -> usize {
        self.lookback + 10
    }

    fn output_features(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let high: Vec<f64> = (0..300).map(|i| 105.0 + i as f64 * 0.05 + (i as f64 * 0.1).sin() * 3.0).collect();
        let low: Vec<f64> = (0..300).map(|i| 95.0 + i as f64 * 0.05 + (i as f64 * 0.1).sin() * 3.0).collect();
        let close: Vec<f64> = (0..300).map(|i| 100.0 + i as f64 * 0.05 + (i as f64 * 0.1).sin() * 3.0).collect();
        let volume: Vec<f64> = (0..300).map(|i| 1000000.0 + (i as f64 * 0.2).sin() * 200000.0).collect();
        (high, low, close, volume)
    }

    #[test]
    fn test_trend_template() {
        let (_, _, close, _) = make_test_data();
        let tt = TrendTemplate::new();
        let result = tt.calculate(&close);

        assert_eq!(result.len(), 300);

        // Scores should be 0-8
        for i in 252..300 {
            if !result[i].is_nan() {
                assert!(result[i] >= 0.0 && result[i] <= 8.0);
            }
        }
    }

    #[test]
    fn test_vcp() {
        let (high, low, close, volume) = make_test_data();
        let vcp = VolatilityContractionPattern::new(30);
        let result = vcp.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), 300);
    }

    #[test]
    fn test_pocket_pivot() {
        let (_, _, close, volume) = make_test_data();
        let pp = PocketPivot::new();
        let result = pp.calculate(&close, &volume);

        assert_eq!(result.len(), 300);
    }

    #[test]
    fn test_power_play() {
        let (_, _, close, volume) = make_test_data();
        let pp = PowerPlay::new();
        let result = pp.calculate(&close, &volume);

        assert_eq!(result.len(), 300);
    }

    #[test]
    fn test_bull_flag() {
        let (high, low, close, _) = make_test_data();
        let bf = BullFlag::new();
        let result = bf.calculate(&high, &low, &close);

        assert_eq!(result.len(), 300);
    }

    #[test]
    fn test_cup_pattern() {
        let (high, low, close, _) = make_test_data();
        let cup = CupPattern::new(50);
        let result = cup.calculate(&high, &low, &close);

        assert_eq!(result.len(), 300);
    }
}
