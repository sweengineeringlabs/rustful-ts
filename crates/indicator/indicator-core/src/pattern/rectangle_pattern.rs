//! Rectangle Pattern Indicator (IND-339)
//!
//! Range consolidation pattern with horizontal support and resistance.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Rectangle pattern breakout direction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RectangleBreakout {
    /// Bullish breakout above resistance.
    Bullish,
    /// Bearish breakout below support.
    Bearish,
    /// Currently within the rectangle.
    InRange,
    /// No rectangle pattern detected.
    None,
}

/// Configuration for Rectangle Pattern detection.
#[derive(Debug, Clone)]
pub struct RectanglePatternConfig {
    /// Minimum number of touches on support/resistance lines.
    pub min_touches: usize,
    /// Minimum length of rectangle in bars.
    pub min_length: usize,
    /// Maximum length of rectangle in bars.
    pub max_length: usize,
    /// Tolerance for support/resistance level matching (as percentage).
    pub level_tolerance: f64,
    /// Minimum range as percentage of average price.
    pub min_range: f64,
}

impl Default for RectanglePatternConfig {
    fn default() -> Self {
        Self {
            min_touches: 2,
            min_length: 10,
            max_length: 50,
            level_tolerance: 0.01,
            min_range: 0.02,
        }
    }
}

/// Rectangle Pattern indicator for range consolidation detection.
///
/// A rectangle pattern forms when price bounces between horizontal
/// support and resistance levels, indicating consolidation before
/// a potential breakout.
#[derive(Debug, Clone)]
pub struct RectanglePattern {
    config: RectanglePatternConfig,
}

impl RectanglePattern {
    /// Create a new Rectangle Pattern indicator with default settings.
    pub fn new() -> Self {
        Self {
            config: RectanglePatternConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: RectanglePatternConfig) -> Self {
        Self { config }
    }

    /// Create with custom parameters.
    pub fn with_params(min_touches: usize, min_length: usize, level_tolerance: f64) -> Self {
        Self {
            config: RectanglePatternConfig {
                min_touches,
                min_length,
                level_tolerance,
                ..Default::default()
            },
        }
    }

    /// Identify support and resistance levels in a range.
    fn identify_levels(&self, high: &[f64], low: &[f64]) -> Option<(f64, f64, usize, usize)> {
        if high.is_empty() || low.is_empty() {
            return None;
        }

        // Find potential resistance (highest highs)
        let max_high = high.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Find potential support (lowest lows)
        let min_low = low.iter().cloned().fold(f64::INFINITY, f64::min);

        // Count touches at each level
        let tolerance_high = max_high * self.config.level_tolerance;
        let tolerance_low = min_low * self.config.level_tolerance;

        let mut resistance_touches = 0;
        let mut support_touches = 0;

        for (h, l) in high.iter().zip(low.iter()) {
            if (*h - max_high).abs() <= tolerance_high {
                resistance_touches += 1;
            }
            if (*l - min_low).abs() <= tolerance_low {
                support_touches += 1;
            }
        }

        Some((max_high, min_low, resistance_touches, support_touches))
    }

    /// Check if the range is mostly horizontal (sideways movement).
    fn is_horizontal(&self, high: &[f64], low: &[f64]) -> bool {
        if high.len() < 3 {
            return false;
        }

        let n = high.len();

        // Calculate average high and low
        let avg_high: f64 = high.iter().sum::<f64>() / n as f64;
        let avg_low: f64 = low.iter().sum::<f64>() / n as f64;

        // Check variance - horizontal patterns have low variance
        let high_variance: f64 = high.iter()
            .map(|h| (h - avg_high).powi(2))
            .sum::<f64>() / n as f64;

        let low_variance: f64 = low.iter()
            .map(|l| (l - avg_low).powi(2))
            .sum::<f64>() / n as f64;

        // Coefficient of variation should be low for horizontal patterns
        let high_cv = high_variance.sqrt() / avg_high;
        let low_cv = low_variance.sqrt() / avg_low;

        high_cv < 0.03 && low_cv < 0.03
    }

    /// Detect rectangle pattern and breakout status.
    pub fn detect_pattern(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<RectangleBreakout> {
        let n = close.len();
        let mut patterns = vec![RectangleBreakout::None; n];

        if n < self.config.min_length {
            return patterns;
        }

        for end_idx in self.config.min_length..n {
            // Try different rectangle lengths
            for length in self.config.min_length..=self.config.max_length.min(end_idx) {
                let start_idx = end_idx - length;

                let high_slice = &high[start_idx..end_idx];
                let low_slice = &low[start_idx..end_idx];

                // Check if horizontal
                if !self.is_horizontal(high_slice, low_slice) {
                    continue;
                }

                // Identify levels
                if let Some((resistance, support, res_touches, sup_touches)) =
                    self.identify_levels(high_slice, low_slice) {

                    // Check minimum touches
                    if res_touches < self.config.min_touches ||
                       sup_touches < self.config.min_touches {
                        continue;
                    }

                    // Check minimum range
                    let avg_price = (resistance + support) / 2.0;
                    let range_pct = (resistance - support) / avg_price;

                    if range_pct < self.config.min_range {
                        continue;
                    }

                    // Check for breakout at current bar
                    let current_close = close[end_idx];
                    let tolerance = (resistance - support) * 0.1;

                    if current_close > resistance + tolerance {
                        patterns[end_idx] = RectangleBreakout::Bullish;
                    } else if current_close < support - tolerance {
                        patterns[end_idx] = RectangleBreakout::Bearish;
                    } else {
                        patterns[end_idx] = RectangleBreakout::InRange;
                    }
                }
            }
        }

        patterns
    }

    /// Calculate pattern signals.
    ///
    /// Returns a vector where:
    /// - Positive values (1.0) indicate bullish breakout
    /// - Negative values (-1.0) indicate bearish breakout
    /// - 0.5 indicates in-range (no breakout yet)
    /// - 0.0 indicates no pattern
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let patterns = self.detect_pattern(high, low, close);

        patterns.iter().map(|p| match p {
            RectangleBreakout::Bullish => 1.0,
            RectangleBreakout::Bearish => -1.0,
            RectangleBreakout::InRange => 0.5,
            RectangleBreakout::None => 0.0,
        }).collect()
    }

    /// Get support and resistance levels for the current rectangle.
    pub fn support_resistance(&self, high: &[f64], low: &[f64]) -> Vec<(f64, f64)> {
        let n = high.len();
        let mut levels = vec![(f64::NAN, f64::NAN); n];

        if n < self.config.min_length {
            return levels;
        }

        for i in self.config.min_length..n {
            let start = i.saturating_sub(self.config.max_length);
            let high_slice = &high[start..=i];
            let low_slice = &low[start..=i];

            if self.is_horizontal(high_slice, low_slice) {
                if let Some((resistance, support, res_touches, sup_touches)) =
                    self.identify_levels(high_slice, low_slice) {

                    if res_touches >= self.config.min_touches &&
                       sup_touches >= self.config.min_touches {
                        levels[i] = (support, resistance);
                    }
                }
            }
        }

        levels
    }

    /// Calculate range width as percentage of price.
    pub fn range_width(&self, high: &[f64], low: &[f64]) -> Vec<f64> {
        let n = high.len();
        let mut widths = vec![0.0; n];

        for i in self.config.min_length..n {
            let start = i.saturating_sub(self.config.min_length);
            let max_high = high[start..=i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_low = low[start..=i].iter().cloned().fold(f64::INFINITY, f64::min);

            let avg_price = (max_high + min_low) / 2.0;
            if avg_price > 0.0 {
                widths[i] = (max_high - min_low) / avg_price;
            }
        }

        widths
    }

    /// Detect if currently in a rectangle consolidation.
    pub fn in_consolidation(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<bool> {
        self.detect_pattern(high, low, close)
            .iter()
            .map(|p| *p == RectangleBreakout::InRange)
            .collect()
    }

    /// Detect bullish breakouts.
    pub fn bullish_breakout(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<bool> {
        self.detect_pattern(high, low, close)
            .iter()
            .map(|p| *p == RectangleBreakout::Bullish)
            .collect()
    }

    /// Detect bearish breakouts.
    pub fn bearish_breakout(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<bool> {
        self.detect_pattern(high, low, close)
            .iter()
            .map(|p| *p == RectangleBreakout::Bearish)
            .collect()
    }
}

impl Default for RectanglePattern {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for RectanglePattern {
    fn name(&self) -> &str {
        "RectanglePattern"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.config.min_length {
            return Err(IndicatorError::InsufficientData {
                required: self.config.min_length,
                got: data.close.len(),
            });
        }

        let signals = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::single(signals))
    }

    fn min_periods(&self) -> usize {
        self.config.min_length
    }
}

impl SignalIndicator for RectanglePattern {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let signals = self.calculate(&data.high, &data.low, &data.close);

        // Find the most recent signal
        for &s in signals.iter().rev() {
            if s > 0.5 {
                return Ok(IndicatorSignal::Bullish);
            } else if s < 0.0 {
                return Ok(IndicatorSignal::Bearish);
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.high, &data.low, &data.close);
        let signals = values.iter().map(|&s| {
            if s > 0.5 {
                IndicatorSignal::Bullish
            } else if s < 0.0 {
                IndicatorSignal::Bearish
            } else {
                IndicatorSignal::Neutral
            }
        }).collect();

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_rectangle_data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut high = Vec::new();
        let mut low = Vec::new();

        // Rectangle consolidation between 105 and 95
        for i in 0..20 {
            // Oscillate between support and resistance
            if i % 4 < 2 {
                high.push(105.0 + (i as f64 * 0.1).sin() * 0.5);
                low.push(99.0 + (i as f64 * 0.1).sin() * 0.5);
            } else {
                high.push(101.0 + (i as f64 * 0.1).sin() * 0.5);
                low.push(95.0 + (i as f64 * 0.1).sin() * 0.5);
            }
        }

        // Bullish breakout
        high.push(108.0);
        low.push(103.0);

        let close: Vec<f64> = high.iter().zip(low.iter())
            .map(|(h, l)| (h + l) / 2.0)
            .collect();

        (high, low, close)
    }

    fn create_horizontal_range() -> (Vec<f64>, Vec<f64>) {
        let high = vec![105.0, 105.5, 104.5, 105.0, 105.2, 104.8, 105.1, 104.9, 105.0, 105.1];
        let low = vec![95.0, 94.5, 95.5, 95.0, 94.8, 95.2, 94.9, 95.1, 95.0, 94.9];
        (high, low)
    }

    #[test]
    fn test_rectangle_pattern_creation() {
        let indicator = RectanglePattern::new();
        assert_eq!(indicator.config.min_touches, 2);
        assert_eq!(indicator.config.min_length, 10);
    }

    #[test]
    fn test_rectangle_pattern_with_params() {
        let indicator = RectanglePattern::with_params(3, 15, 0.02);
        assert_eq!(indicator.config.min_touches, 3);
        assert_eq!(indicator.config.min_length, 15);
        assert_eq!(indicator.config.level_tolerance, 0.02);
    }

    #[test]
    fn test_identify_levels() {
        let (high, low) = create_horizontal_range();
        let indicator = RectanglePattern::new();

        let result = indicator.identify_levels(&high, &low);
        assert!(result.is_some());

        let (resistance, support, _, _) = result.unwrap();
        assert!(resistance > support);
    }

    #[test]
    fn test_is_horizontal() {
        let (high, low) = create_horizontal_range();
        let indicator = RectanglePattern::new();

        let is_horiz = indicator.is_horizontal(&high, &low);
        assert!(is_horiz);

        // Non-horizontal (trending) data
        let trending_high = vec![100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0, 116.0, 118.0];
        let trending_low = vec![95.0, 97.0, 99.0, 101.0, 103.0, 105.0, 107.0, 109.0, 111.0, 113.0];

        let is_horiz_trend = indicator.is_horizontal(&trending_high, &trending_low);
        assert!(!is_horiz_trend);
    }

    #[test]
    fn test_detect_pattern() {
        let (high, low, close) = create_rectangle_data();
        let indicator = RectanglePattern::with_params(2, 10, 0.02);

        let patterns = indicator.detect_pattern(&high, &low, &close);
        assert_eq!(patterns.len(), high.len());
    }

    #[test]
    fn test_calculate() {
        let (high, low, close) = create_rectangle_data();
        let indicator = RectanglePattern::with_params(2, 10, 0.02);

        let signals = indicator.calculate(&high, &low, &close);
        assert_eq!(signals.len(), high.len());
    }

    #[test]
    fn test_support_resistance() {
        let (high, low) = create_horizontal_range();
        let indicator = RectanglePattern::with_params(2, 8, 0.02);

        let levels = indicator.support_resistance(&high, &low);
        assert_eq!(levels.len(), high.len());
    }

    #[test]
    fn test_range_width() {
        let (high, low) = create_horizontal_range();
        let indicator = RectanglePattern::with_params(2, 8, 0.02);

        let widths = indicator.range_width(&high, &low);
        assert_eq!(widths.len(), high.len());

        // Later widths should be positive
        assert!(widths.last().unwrap() > &0.0);
    }

    #[test]
    fn test_in_consolidation() {
        let (high, low, close) = create_rectangle_data();
        let indicator = RectanglePattern::with_params(2, 10, 0.02);

        let in_range = indicator.in_consolidation(&high, &low, &close);
        assert_eq!(in_range.len(), high.len());
    }

    #[test]
    fn test_min_periods() {
        let indicator = RectanglePattern::with_params(2, 15, 0.01);
        assert_eq!(indicator.min_periods(), 15);
    }

    #[test]
    fn test_insufficient_data() {
        let indicator = RectanglePattern::new();
        let data = OHLCVSeries {
            open: vec![100.0; 5],
            high: vec![105.0; 5],
            low: vec![95.0; 5],
            close: vec![102.0; 5],
            volume: vec![1000.0; 5],
        };

        let result = indicator.compute(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_breakout_enum() {
        assert_eq!(RectangleBreakout::Bullish, RectangleBreakout::Bullish);
        assert_ne!(RectangleBreakout::Bullish, RectangleBreakout::Bearish);
        assert_ne!(RectangleBreakout::InRange, RectangleBreakout::None);
    }
}
