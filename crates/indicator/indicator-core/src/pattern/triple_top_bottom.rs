//! Triple Top/Bottom Pattern Indicator (IND-335)
//!
//! Extended reversal pattern identifying three peaks or troughs at similar levels.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Configuration for Triple Top/Bottom detection.
#[derive(Debug, Clone)]
pub struct TripleTopBottomConfig {
    /// Lookback period for finding peaks/troughs.
    pub lookback: usize,
    /// Tolerance for price level matching (as percentage, e.g., 0.02 = 2%).
    pub tolerance: f64,
    /// Minimum bars between peaks/troughs.
    pub min_separation: usize,
}

impl Default for TripleTopBottomConfig {
    fn default() -> Self {
        Self {
            lookback: 50,
            tolerance: 0.02,
            min_separation: 5,
        }
    }
}

/// Triple Top/Bottom indicator for extended reversal pattern detection.
///
/// A triple top is a bearish reversal pattern formed by three peaks at similar
/// price levels. A triple bottom is a bullish reversal pattern formed by three
/// troughs at similar price levels.
#[derive(Debug, Clone)]
pub struct TripleTopBottom {
    config: TripleTopBottomConfig,
}

impl TripleTopBottom {
    /// Create a new Triple Top/Bottom indicator with default settings.
    pub fn new() -> Self {
        Self {
            config: TripleTopBottomConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: TripleTopBottomConfig) -> Self {
        Self { config }
    }

    /// Create with custom parameters.
    pub fn with_params(lookback: usize, tolerance: f64, min_separation: usize) -> Self {
        Self {
            config: TripleTopBottomConfig {
                lookback,
                tolerance,
                min_separation,
            },
        }
    }

    /// Find local peaks in the price data.
    fn find_peaks(&self, high: &[f64], period: usize) -> Vec<(usize, f64)> {
        let mut peaks = Vec::new();
        let n = high.len();

        if n < period * 2 + 1 {
            return peaks;
        }

        for i in period..(n - period) {
            let mut is_peak = true;
            for j in 1..=period {
                if high[i - j] >= high[i] || high[i + j] >= high[i] {
                    is_peak = false;
                    break;
                }
            }
            if is_peak {
                peaks.push((i, high[i]));
            }
        }

        peaks
    }

    /// Find local troughs in the price data.
    fn find_troughs(&self, low: &[f64], period: usize) -> Vec<(usize, f64)> {
        let mut troughs = Vec::new();
        let n = low.len();

        if n < period * 2 + 1 {
            return troughs;
        }

        for i in period..(n - period) {
            let mut is_trough = true;
            for j in 1..=period {
                if low[i - j] <= low[i] || low[i + j] <= low[i] {
                    is_trough = false;
                    break;
                }
            }
            if is_trough {
                troughs.push((i, low[i]));
            }
        }

        troughs
    }

    /// Check if three price levels are approximately equal within tolerance.
    fn levels_match(&self, p1: f64, p2: f64, p3: f64) -> bool {
        let avg = (p1 + p2 + p3) / 3.0;
        let tolerance = avg * self.config.tolerance;

        (p1 - avg).abs() <= tolerance &&
        (p2 - avg).abs() <= tolerance &&
        (p3 - avg).abs() <= tolerance
    }

    /// Calculate triple top/bottom signals.
    ///
    /// Returns a vector where:
    /// - Positive values indicate triple bottom (bullish)
    /// - Negative values indicate triple top (bearish)
    /// - 0.0 indicates no pattern
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut signals = vec![0.0; n];

        if n < self.config.lookback {
            return signals;
        }

        let period = 2.min(self.config.min_separation / 2).max(1);
        let peaks = self.find_peaks(high, period);
        let troughs = self.find_troughs(low, period);

        // Check for triple tops
        for i in 0..peaks.len().saturating_sub(2) {
            let (idx1, p1) = peaks[i];
            let (idx2, p2) = peaks[i + 1];
            let (idx3, p3) = peaks[i + 2];

            // Check separation
            if idx2 - idx1 < self.config.min_separation || idx3 - idx2 < self.config.min_separation {
                continue;
            }

            // Check if peaks are at similar levels
            if self.levels_match(p1, p2, p3) {
                // Triple top confirmed at third peak
                if idx3 < n {
                    signals[idx3] = -1.0;
                }
            }
        }

        // Check for triple bottoms
        for i in 0..troughs.len().saturating_sub(2) {
            let (idx1, t1) = troughs[i];
            let (idx2, t2) = troughs[i + 1];
            let (idx3, t3) = troughs[i + 2];

            // Check separation
            if idx2 - idx1 < self.config.min_separation || idx3 - idx2 < self.config.min_separation {
                continue;
            }

            // Check if troughs are at similar levels
            if self.levels_match(t1, t2, t3) {
                // Triple bottom confirmed at third trough
                if idx3 < n {
                    signals[idx3] = 1.0;
                }
            }
        }

        signals
    }

    /// Detect triple top patterns.
    pub fn detect_triple_top(&self, high: &[f64], _close: &[f64]) -> Vec<bool> {
        let n = high.len();
        let mut patterns = vec![false; n];

        let period = 2.min(self.config.min_separation / 2).max(1);
        let peaks = self.find_peaks(high, period);

        for i in 0..peaks.len().saturating_sub(2) {
            let (idx1, p1) = peaks[i];
            let (idx2, p2) = peaks[i + 1];
            let (idx3, p3) = peaks[i + 2];

            if idx2 - idx1 >= self.config.min_separation &&
               idx3 - idx2 >= self.config.min_separation &&
               self.levels_match(p1, p2, p3) {
                if idx3 < n {
                    patterns[idx3] = true;
                }
            }
        }

        patterns
    }

    /// Detect triple bottom patterns.
    pub fn detect_triple_bottom(&self, low: &[f64], _close: &[f64]) -> Vec<bool> {
        let n = low.len();
        let mut patterns = vec![false; n];

        let period = 2.min(self.config.min_separation / 2).max(1);
        let troughs = self.find_troughs(low, period);

        for i in 0..troughs.len().saturating_sub(2) {
            let (idx1, t1) = troughs[i];
            let (idx2, t2) = troughs[i + 1];
            let (idx3, t3) = troughs[i + 2];

            if idx2 - idx1 >= self.config.min_separation &&
               idx3 - idx2 >= self.config.min_separation &&
               self.levels_match(t1, t2, t3) {
                if idx3 < n {
                    patterns[idx3] = true;
                }
            }
        }

        patterns
    }
}

impl Default for TripleTopBottom {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for TripleTopBottom {
    fn name(&self) -> &str {
        "TripleTopBottom"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.config.lookback {
            return Err(IndicatorError::InsufficientData {
                required: self.config.lookback,
                got: data.close.len(),
            });
        }

        let signals = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::single(signals))
    }

    fn min_periods(&self) -> usize {
        self.config.lookback
    }
}

impl SignalIndicator for TripleTopBottom {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let signals = self.calculate(&data.high, &data.low, &data.close);

        // Find the most recent signal
        for &s in signals.iter().rev() {
            if s > 0.0 {
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
            if s > 0.0 {
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

    fn create_triple_top_data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        // Create data with three peaks at similar levels around 110
        let high = vec![
            100.0, 102.0, 105.0, 108.0, 110.0, 108.0, 105.0, 102.0, 100.0, // First peak at index 4
            102.0, 105.0, 108.0, 110.5, 108.0, 105.0, 102.0, 100.0, // Second peak at index 12
            102.0, 105.0, 108.0, 109.5, 108.0, 105.0, 102.0, 100.0, // Third peak at index 21
        ];
        let low: Vec<f64> = high.iter().map(|h| h - 3.0).collect();
        let close: Vec<f64> = high.iter().map(|h| h - 1.5).collect();
        (high, low, close)
    }

    fn create_triple_bottom_data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        // Create data with three troughs at similar levels around 90
        let low = vec![
            100.0, 98.0, 95.0, 92.0, 90.0, 92.0, 95.0, 98.0, 100.0, // First trough at index 4
            98.0, 95.0, 92.0, 89.5, 92.0, 95.0, 98.0, 100.0, // Second trough at index 12
            98.0, 95.0, 92.0, 90.5, 92.0, 95.0, 98.0, 100.0, // Third trough at index 21
        ];
        let high: Vec<f64> = low.iter().map(|l| l + 3.0).collect();
        let close: Vec<f64> = low.iter().map(|l| l + 1.5).collect();
        (high, low, close)
    }

    #[test]
    fn test_triple_top_bottom_creation() {
        let indicator = TripleTopBottom::new();
        assert_eq!(indicator.config.lookback, 50);
        assert_eq!(indicator.config.tolerance, 0.02);
    }

    #[test]
    fn test_triple_top_bottom_with_params() {
        let indicator = TripleTopBottom::with_params(30, 0.03, 3);
        assert_eq!(indicator.config.lookback, 30);
        assert_eq!(indicator.config.tolerance, 0.03);
        assert_eq!(indicator.config.min_separation, 3);
    }

    #[test]
    fn test_find_peaks() {
        let indicator = TripleTopBottom::with_params(20, 0.02, 3);
        let high = vec![100.0, 105.0, 110.0, 105.0, 100.0, 105.0, 115.0, 105.0, 100.0];

        let peaks = indicator.find_peaks(&high, 2);
        assert!(!peaks.is_empty());
    }

    #[test]
    fn test_find_troughs() {
        let indicator = TripleTopBottom::with_params(20, 0.02, 3);
        let low = vec![100.0, 95.0, 90.0, 95.0, 100.0, 95.0, 85.0, 95.0, 100.0];

        let troughs = indicator.find_troughs(&low, 2);
        assert!(!troughs.is_empty());
    }

    #[test]
    fn test_levels_match() {
        let indicator = TripleTopBottom::with_params(20, 0.02, 3);

        // Levels within 2% tolerance
        assert!(indicator.levels_match(100.0, 101.0, 99.0));

        // Levels outside tolerance
        assert!(!indicator.levels_match(100.0, 110.0, 90.0));
    }

    #[test]
    fn test_triple_top_detection() {
        let (high, low, close) = create_triple_top_data();
        let indicator = TripleTopBottom::with_params(10, 0.02, 5);

        let patterns = indicator.detect_triple_top(&high, &close);
        assert_eq!(patterns.len(), high.len());
    }

    #[test]
    fn test_triple_bottom_detection() {
        let (high, low, close) = create_triple_bottom_data();
        let indicator = TripleTopBottom::with_params(10, 0.02, 5);

        let patterns = indicator.detect_triple_bottom(&low, &close);
        assert_eq!(patterns.len(), low.len());
    }

    #[test]
    fn test_calculate() {
        let (high, low, close) = create_triple_top_data();
        let indicator = TripleTopBottom::with_params(10, 0.02, 5);

        let signals = indicator.calculate(&high, &low, &close);
        assert_eq!(signals.len(), high.len());
    }

    #[test]
    fn test_min_periods() {
        let indicator = TripleTopBottom::with_params(30, 0.02, 5);
        assert_eq!(indicator.min_periods(), 30);
    }

    #[test]
    fn test_insufficient_data() {
        let indicator = TripleTopBottom::new();
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
}
