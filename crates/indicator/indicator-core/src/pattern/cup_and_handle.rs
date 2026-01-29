//! Cup and Handle Pattern Indicator (IND-336)
//!
//! Continuation pattern identifying cup-shaped consolidation followed by a handle.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Configuration for Cup and Handle detection.
#[derive(Debug, Clone)]
pub struct CupAndHandleConfig {
    /// Minimum length of cup in bars.
    pub min_cup_length: usize,
    /// Maximum length of cup in bars.
    pub max_cup_length: usize,
    /// Maximum handle retracement (as percentage of cup depth, e.g., 0.5 = 50%).
    pub max_handle_retracement: f64,
    /// Maximum handle length as percentage of cup length.
    pub max_handle_length_ratio: f64,
    /// Minimum cup depth as percentage of left rim price.
    pub min_cup_depth: f64,
}

impl Default for CupAndHandleConfig {
    fn default() -> Self {
        Self {
            min_cup_length: 20,
            max_cup_length: 100,
            max_handle_retracement: 0.5,
            max_handle_length_ratio: 0.25,
            min_cup_depth: 0.10,
        }
    }
}

/// Cup and Handle indicator for continuation pattern detection.
///
/// The cup and handle is a bullish continuation pattern that resembles a cup
/// with a handle when viewed on a chart. It consists of:
/// 1. A U-shaped cup (rounded bottom)
/// 2. A handle (slight downward drift followed by breakout)
#[derive(Debug, Clone)]
pub struct CupAndHandle {
    config: CupAndHandleConfig,
}

impl CupAndHandle {
    /// Create a new Cup and Handle indicator with default settings.
    pub fn new() -> Self {
        Self {
            config: CupAndHandleConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: CupAndHandleConfig) -> Self {
        Self { config }
    }

    /// Create with custom parameters.
    pub fn with_params(
        min_cup_length: usize,
        max_cup_length: usize,
        max_handle_retracement: f64,
    ) -> Self {
        Self {
            config: CupAndHandleConfig {
                min_cup_length,
                max_cup_length,
                max_handle_retracement,
                ..Default::default()
            },
        }
    }

    /// Find the lowest point between two indices.
    fn find_cup_bottom(&self, low: &[f64], start: usize, end: usize) -> Option<(usize, f64)> {
        if start >= end || end > low.len() {
            return None;
        }

        let mut min_idx = start;
        let mut min_val = low[start];

        for i in start..end {
            if low[i] < min_val {
                min_val = low[i];
                min_idx = i;
            }
        }

        Some((min_idx, min_val))
    }

    /// Check if the cup has a U-shape (rounded bottom).
    fn is_u_shaped(&self, low: &[f64], start: usize, bottom: usize, end: usize) -> bool {
        if start >= bottom || bottom >= end {
            return false;
        }

        // Check left side is descending
        let left_len = bottom - start;
        if left_len < 2 {
            return true;
        }

        let mut left_descending = 0;
        for i in start..(bottom - 1) {
            if low[i + 1] <= low[i] {
                left_descending += 1;
            }
        }

        // Check right side is ascending
        let right_len = end - bottom;
        if right_len < 2 {
            return true;
        }

        let mut right_ascending = 0;
        for i in bottom..(end - 1) {
            if low[i + 1] >= low[i] {
                right_ascending += 1;
            }
        }

        // At least 50% of each side should follow the expected pattern
        let left_ratio = left_descending as f64 / left_len as f64;
        let right_ratio = right_ascending as f64 / right_len as f64;

        left_ratio >= 0.4 && right_ratio >= 0.4
    }

    /// Detect cup and handle patterns.
    ///
    /// Returns a vector where:
    /// - 1.0 indicates a cup and handle pattern detected (bullish)
    /// - 0.0 indicates no pattern
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut signals = vec![0.0; n];

        if n < self.config.min_cup_length + 5 {
            return signals;
        }

        // Scan for potential cup formations
        for cup_end in self.config.min_cup_length..n {
            // Try different cup lengths
            for cup_len in self.config.min_cup_length..=self.config.max_cup_length.min(cup_end) {
                let cup_start = cup_end - cup_len;

                // Left rim (high near start)
                let left_rim = high[cup_start..(cup_start + 3).min(cup_end)]
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);

                // Right rim (high near end)
                let right_rim = high[(cup_end.saturating_sub(3))..cup_end]
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);

                // Find cup bottom
                if let Some((bottom_idx, bottom_price)) = self.find_cup_bottom(low, cup_start, cup_end) {
                    // Check cup depth
                    let cup_depth = (left_rim.min(right_rim) - bottom_price) / left_rim;
                    if cup_depth < self.config.min_cup_depth {
                        continue;
                    }

                    // Check if rims are at similar levels (within 5%)
                    let rim_diff = (left_rim - right_rim).abs() / left_rim;
                    if rim_diff > 0.05 {
                        continue;
                    }

                    // Check U-shape
                    if !self.is_u_shaped(low, cup_start, bottom_idx, cup_end) {
                        continue;
                    }

                    // Look for handle (slight pullback after right rim)
                    let max_handle_len = (cup_len as f64 * self.config.max_handle_length_ratio) as usize;
                    let handle_end = (cup_end + max_handle_len).min(n);

                    if handle_end > cup_end {
                        // Check handle retracement
                        let handle_low = low[cup_end..handle_end]
                            .iter()
                            .cloned()
                            .fold(f64::INFINITY, f64::min);

                        let cup_height = left_rim.min(right_rim) - bottom_price;
                        let handle_retracement = (right_rim - handle_low) / cup_height;

                        if handle_retracement <= self.config.max_handle_retracement {
                            // Pattern confirmed at handle end
                            signals[handle_end - 1] = 1.0;
                        }
                    }
                }
            }
        }

        signals
    }

    /// Measure cup depth at each point.
    pub fn cup_depth(&self, high: &[f64], low: &[f64]) -> Vec<f64> {
        let n = high.len();
        let mut depths = vec![0.0; n];

        for i in self.config.min_cup_length..n {
            let start = i.saturating_sub(self.config.max_cup_length);
            let rim = high[start..i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let bottom = low[start..i].iter().cloned().fold(f64::INFINITY, f64::min);

            if rim > 0.0 {
                depths[i] = (rim - bottom) / rim;
            }
        }

        depths
    }

    /// Calculate pattern strength (0.0 to 1.0).
    pub fn pattern_strength(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let signals = self.calculate(high, low, close);
        let depths = self.cup_depth(high, low);

        signals.iter()
            .zip(depths.iter())
            .map(|(&s, &d)| {
                if s > 0.0 {
                    // Normalize depth to strength (deeper cup = stronger pattern)
                    (d * 2.0).min(1.0)
                } else {
                    0.0
                }
            })
            .collect()
    }
}

impl Default for CupAndHandle {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for CupAndHandle {
    fn name(&self) -> &str {
        "CupAndHandle"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.config.min_cup_length + 5;
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let signals = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::single(signals))
    }

    fn min_periods(&self) -> usize {
        self.config.min_cup_length + 5
    }
}

impl SignalIndicator for CupAndHandle {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let signals = self.calculate(&data.high, &data.low, &data.close);

        // Find the most recent signal
        for &s in signals.iter().rev() {
            if s > 0.0 {
                return Ok(IndicatorSignal::Bullish);
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.high, &data.low, &data.close);
        let signals = values.iter().map(|&s| {
            if s > 0.0 {
                IndicatorSignal::Bullish
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

    fn create_cup_and_handle_data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        // Create U-shaped cup with handle
        let mut high = Vec::new();
        let mut low = Vec::new();

        // Left side of cup (descending)
        for i in 0..10 {
            high.push(110.0 - i as f64 * 1.0);
            low.push(107.0 - i as f64 * 1.0);
        }

        // Bottom of cup
        for _ in 0..5 {
            high.push(100.0);
            low.push(97.0);
        }

        // Right side of cup (ascending)
        for i in 0..10 {
            high.push(100.0 + i as f64 * 1.0);
            low.push(97.0 + i as f64 * 1.0);
        }

        // Handle (slight pullback)
        for i in 0..5 {
            high.push(108.0 - i as f64 * 0.5);
            low.push(105.0 - i as f64 * 0.5);
        }

        let close: Vec<f64> = high.iter().zip(low.iter())
            .map(|(h, l)| (h + l) / 2.0)
            .collect();

        (high, low, close)
    }

    #[test]
    fn test_cup_and_handle_creation() {
        let indicator = CupAndHandle::new();
        assert_eq!(indicator.config.min_cup_length, 20);
        assert_eq!(indicator.config.max_cup_length, 100);
    }

    #[test]
    fn test_cup_and_handle_with_params() {
        let indicator = CupAndHandle::with_params(15, 80, 0.4);
        assert_eq!(indicator.config.min_cup_length, 15);
        assert_eq!(indicator.config.max_cup_length, 80);
        assert_eq!(indicator.config.max_handle_retracement, 0.4);
    }

    #[test]
    fn test_find_cup_bottom() {
        let indicator = CupAndHandle::new();
        let low = vec![100.0, 95.0, 90.0, 85.0, 90.0, 95.0, 100.0];

        let result = indicator.find_cup_bottom(&low, 0, low.len());
        assert!(result.is_some());
        let (idx, val) = result.unwrap();
        assert_eq!(idx, 3);
        assert_eq!(val, 85.0);
    }

    #[test]
    fn test_is_u_shaped() {
        let indicator = CupAndHandle::new();
        // Create U-shaped data
        let low = vec![100.0, 98.0, 95.0, 90.0, 85.0, 90.0, 95.0, 98.0, 100.0];

        let is_u = indicator.is_u_shaped(&low, 0, 4, 9);
        assert!(is_u);
    }

    #[test]
    fn test_cup_depth() {
        let (high, low, _) = create_cup_and_handle_data();
        let indicator = CupAndHandle::with_params(10, 50, 0.5);

        let depths = indicator.cup_depth(&high, &low);
        assert_eq!(depths.len(), high.len());

        // Later points should have measurable depth
        assert!(depths[depths.len() - 1] > 0.0);
    }

    #[test]
    fn test_calculate() {
        let (high, low, close) = create_cup_and_handle_data();
        let indicator = CupAndHandle::with_params(10, 50, 0.5);

        let signals = indicator.calculate(&high, &low, &close);
        assert_eq!(signals.len(), high.len());
    }

    #[test]
    fn test_pattern_strength() {
        let (high, low, close) = create_cup_and_handle_data();
        let indicator = CupAndHandle::with_params(10, 50, 0.5);

        let strength = indicator.pattern_strength(&high, &low, &close);
        assert_eq!(strength.len(), high.len());

        // All strength values should be between 0 and 1
        for s in &strength {
            assert!(*s >= 0.0 && *s <= 1.0);
        }
    }

    #[test]
    fn test_min_periods() {
        let indicator = CupAndHandle::with_params(25, 100, 0.5);
        assert_eq!(indicator.min_periods(), 30);
    }

    #[test]
    fn test_insufficient_data() {
        let indicator = CupAndHandle::new();
        let data = OHLCVSeries {
            open: vec![100.0; 10],
            high: vec![105.0; 10],
            low: vec![95.0; 10],
            close: vec![102.0; 10],
            volume: vec![1000.0; 10],
        };

        let result = indicator.compute(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_signal_indicator() {
        let (high, low, close) = create_cup_and_handle_data();
        let indicator = CupAndHandle::with_params(10, 50, 0.5);

        let data = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; 30],
        };

        let signals = indicator.signals(&data);
        assert!(signals.is_ok());
    }
}
