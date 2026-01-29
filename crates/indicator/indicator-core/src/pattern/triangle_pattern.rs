//! Triangle Pattern Indicator (IND-340)
//!
//! Symmetrical, ascending, and descending triangle patterns.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Triangle pattern type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TriangleType {
    /// Symmetrical triangle (converging trend lines, neutral bias).
    Symmetrical,
    /// Ascending triangle (flat top, rising bottom, bullish).
    Ascending,
    /// Descending triangle (flat bottom, falling top, bearish).
    Descending,
    /// No triangle pattern detected.
    None,
}

/// Configuration for Triangle Pattern detection.
#[derive(Debug, Clone)]
pub struct TrianglePatternConfig {
    /// Minimum length of triangle in bars.
    pub min_length: usize,
    /// Maximum length of triangle in bars.
    pub max_length: usize,
    /// Minimum convergence ratio.
    pub min_convergence: f64,
    /// Minimum touch points on each trend line.
    pub min_touches: usize,
    /// Tolerance for flat line detection (as percentage).
    pub flat_tolerance: f64,
}

impl Default for TrianglePatternConfig {
    fn default() -> Self {
        Self {
            min_length: 10,
            max_length: 50,
            min_convergence: 0.3,
            min_touches: 2,
            flat_tolerance: 0.02,
        }
    }
}

/// Triangle Pattern indicator for symmetrical/ascending/descending triangles.
///
/// Triangle patterns are consolidation patterns formed by converging trend lines:
/// - Symmetrical: Both lines converge equally (neutral, breaks in trend direction)
/// - Ascending: Flat resistance with rising support (bullish)
/// - Descending: Flat support with falling resistance (bearish)
#[derive(Debug, Clone)]
pub struct TrianglePattern {
    config: TrianglePatternConfig,
}

impl TrianglePattern {
    /// Create a new Triangle Pattern indicator with default settings.
    pub fn new() -> Self {
        Self {
            config: TrianglePatternConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: TrianglePatternConfig) -> Self {
        Self { config }
    }

    /// Create with custom parameters.
    pub fn with_params(min_length: usize, max_length: usize, min_convergence: f64) -> Self {
        Self {
            config: TrianglePatternConfig {
                min_length,
                max_length,
                min_convergence,
                ..Default::default()
            },
        }
    }

    /// Calculate linear regression slope.
    fn linear_regression_slope(&self, values: &[f64]) -> f64 {
        let n = values.len() as f64;
        if n < 2.0 {
            return 0.0;
        }

        let x_mean = (n - 1.0) / 2.0;
        let y_mean: f64 = values.iter().sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (i, &y) in values.iter().enumerate() {
            let x = i as f64;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean).powi(2);
        }

        if denominator.abs() < 1e-10 {
            return 0.0;
        }

        numerator / denominator
    }

    /// Find local highs.
    fn find_local_highs(&self, high: &[f64], window: usize) -> Vec<(usize, f64)> {
        let mut highs = Vec::new();
        let n = high.len();

        if n < window * 2 + 1 {
            return highs;
        }

        for i in window..(n - window) {
            let mut is_high = true;
            for j in 1..=window {
                if high[i - j] >= high[i] || high[i + j] >= high[i] {
                    is_high = false;
                    break;
                }
            }
            if is_high {
                highs.push((i, high[i]));
            }
        }

        highs
    }

    /// Find local lows.
    fn find_local_lows(&self, low: &[f64], window: usize) -> Vec<(usize, f64)> {
        let mut lows = Vec::new();
        let n = low.len();

        if n < window * 2 + 1 {
            return lows;
        }

        for i in window..(n - window) {
            let mut is_low = true;
            for j in 1..=window {
                if low[i - j] <= low[i] || low[i + j] <= low[i] {
                    is_low = false;
                    break;
                }
            }
            if is_low {
                lows.push((i, low[i]));
            }
        }

        lows
    }

    /// Check if a line is approximately flat.
    fn is_flat(&self, values: &[f64]) -> bool {
        if values.is_empty() {
            return false;
        }

        let avg = values.iter().sum::<f64>() / values.len() as f64;
        let max_deviation = values.iter()
            .map(|v| (v - avg).abs() / avg)
            .fold(0.0, f64::max);

        max_deviation <= self.config.flat_tolerance
    }

    /// Detect triangle type at each point.
    pub fn detect_triangle_type(&self, high: &[f64], low: &[f64]) -> Vec<TriangleType> {
        let n = high.len();
        let mut triangle_types = vec![TriangleType::None; n];

        if n < self.config.min_length {
            return triangle_types;
        }

        let window = 2;
        let local_highs = self.find_local_highs(high, window);
        let local_lows = self.find_local_lows(low, window);

        for end_idx in self.config.min_length..n {
            // Try different triangle lengths
            for length in self.config.min_length..=self.config.max_length.min(end_idx) {
                let start_idx = end_idx - length;

                // Get highs and lows within the range
                let range_highs: Vec<f64> = local_highs.iter()
                    .filter(|(idx, _)| *idx >= start_idx && *idx <= end_idx)
                    .map(|(_, val)| *val)
                    .collect();

                let range_lows: Vec<f64> = local_lows.iter()
                    .filter(|(idx, _)| *idx >= start_idx && *idx <= end_idx)
                    .map(|(_, val)| *val)
                    .collect();

                // Need minimum touches on each line
                if range_highs.len() < self.config.min_touches ||
                   range_lows.len() < self.config.min_touches {
                    continue;
                }

                // Calculate slopes
                let high_slope = self.linear_regression_slope(&range_highs);
                let low_slope = self.linear_regression_slope(&range_lows);

                // Check for convergence
                let initial_range = high[start_idx] - low[start_idx];
                let final_range = high[end_idx] - low[end_idx];

                if initial_range <= 0.0 || final_range <= 0.0 {
                    continue;
                }

                let convergence = 1.0 - (final_range / initial_range);

                if convergence < self.config.min_convergence {
                    continue;
                }

                // Determine triangle type
                let high_is_flat = self.is_flat(&range_highs);
                let low_is_flat = self.is_flat(&range_lows);

                if high_is_flat && low_slope > 0.0 {
                    // Ascending triangle: flat resistance, rising support
                    triangle_types[end_idx] = TriangleType::Ascending;
                } else if low_is_flat && high_slope < 0.0 {
                    // Descending triangle: flat support, falling resistance
                    triangle_types[end_idx] = TriangleType::Descending;
                } else if high_slope < 0.0 && low_slope > 0.0 {
                    // Symmetrical triangle: converging lines
                    triangle_types[end_idx] = TriangleType::Symmetrical;
                }
            }
        }

        triangle_types
    }

    /// Calculate pattern signals.
    ///
    /// Returns a vector where:
    /// - 1.0 indicates ascending triangle (bullish)
    /// - 0.5 indicates symmetrical triangle (neutral)
    /// - -1.0 indicates descending triangle (bearish)
    /// - 0.0 indicates no pattern
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let triangle_types = self.detect_triangle_type(high, low);

        triangle_types.iter().map(|tt| match tt {
            TriangleType::Ascending => 1.0,
            TriangleType::Symmetrical => 0.5,
            TriangleType::Descending => -1.0,
            TriangleType::None => 0.0,
        }).collect()
    }

    /// Calculate apex point (where trend lines would meet).
    pub fn apex_distance(&self, high: &[f64], low: &[f64]) -> Vec<f64> {
        let n = high.len();
        let mut distances = vec![f64::NAN; n];

        if n < self.config.min_length {
            return distances;
        }

        for i in self.config.min_length..n {
            let start = i.saturating_sub(self.config.min_length);

            let high_slice: Vec<f64> = high[start..=i].to_vec();
            let low_slice: Vec<f64> = low[start..=i].to_vec();

            let high_slope = self.linear_regression_slope(&high_slice);
            let low_slope = self.linear_regression_slope(&low_slice);

            // Calculate bars until apex (where lines would meet)
            if (high_slope - low_slope).abs() > 1e-10 {
                let current_range = high[i] - low[i];
                let convergence_rate = (low_slope - high_slope).abs();

                if convergence_rate > 0.0 && current_range > 0.0 {
                    let bars_to_apex = current_range / convergence_rate;
                    distances[i] = bars_to_apex;
                }
            }
        }

        distances
    }

    /// Calculate pattern completion percentage.
    pub fn completion_pct(&self, high: &[f64], low: &[f64]) -> Vec<f64> {
        let n = high.len();
        let mut completion = vec![0.0; n];

        if n < self.config.min_length {
            return completion;
        }

        for i in self.config.min_length..n {
            let start = i.saturating_sub(self.config.max_length);

            let initial_range = high[start] - low[start];
            let current_range = high[i] - low[i];

            if initial_range > 0.0 && current_range > 0.0 {
                // Completion based on how much the range has contracted
                let convergence = 1.0 - (current_range / initial_range);
                completion[i] = convergence.max(0.0).min(1.0);
            }
        }

        completion
    }

    /// Detect ascending triangles.
    pub fn detect_ascending(&self, high: &[f64], low: &[f64]) -> Vec<bool> {
        self.detect_triangle_type(high, low)
            .iter()
            .map(|tt| *tt == TriangleType::Ascending)
            .collect()
    }

    /// Detect descending triangles.
    pub fn detect_descending(&self, high: &[f64], low: &[f64]) -> Vec<bool> {
        self.detect_triangle_type(high, low)
            .iter()
            .map(|tt| *tt == TriangleType::Descending)
            .collect()
    }

    /// Detect symmetrical triangles.
    pub fn detect_symmetrical(&self, high: &[f64], low: &[f64]) -> Vec<bool> {
        self.detect_triangle_type(high, low)
            .iter()
            .map(|tt| *tt == TriangleType::Symmetrical)
            .collect()
    }

    /// Calculate breakout probability based on pattern characteristics.
    pub fn breakout_probability(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let triangle_types = self.detect_triangle_type(high, low);
        let completion = self.completion_pct(high, low);

        triangle_types.iter()
            .zip(completion.iter())
            .map(|(tt, &comp)| {
                match tt {
                    TriangleType::None => 0.0,
                    _ => {
                        // Higher completion = higher breakout probability
                        (comp * 0.8).min(0.9)
                    }
                }
            })
            .collect()
    }
}

impl Default for TrianglePattern {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for TrianglePattern {
    fn name(&self) -> &str {
        "TrianglePattern"
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

impl SignalIndicator for TrianglePattern {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let signals = self.calculate(&data.high, &data.low, &data.close);

        // Find the most recent signal
        for &s in signals.iter().rev() {
            if s > 0.5 {
                return Ok(IndicatorSignal::Bullish);
            } else if s < 0.0 {
                return Ok(IndicatorSignal::Bearish);
            } else if s > 0.0 {
                return Ok(IndicatorSignal::Neutral); // Symmetrical
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

    fn create_symmetrical_triangle_data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut high = Vec::new();
        let mut low = Vec::new();

        // Symmetrical triangle: converging highs and lows
        for i in 0..20 {
            // Highs descending
            high.push(110.0 - i as f64 * 0.5);
            // Lows ascending
            low.push(90.0 + i as f64 * 0.5);
        }

        let close: Vec<f64> = high.iter().zip(low.iter())
            .map(|(h, l)| (h + l) / 2.0)
            .collect();

        (high, low, close)
    }

    fn create_ascending_triangle_data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut high = Vec::new();
        let mut low = Vec::new();

        // Ascending triangle: flat highs, rising lows
        for i in 0..20 {
            high.push(110.0); // Flat resistance
            low.push(90.0 + i as f64 * 0.8); // Rising support
        }

        let close: Vec<f64> = high.iter().zip(low.iter())
            .map(|(h, l)| (h + l) / 2.0)
            .collect();

        (high, low, close)
    }

    fn create_descending_triangle_data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut high = Vec::new();
        let mut low = Vec::new();

        // Descending triangle: falling highs, flat lows
        for i in 0..20 {
            high.push(110.0 - i as f64 * 0.8); // Falling resistance
            low.push(90.0); // Flat support
        }

        let close: Vec<f64> = high.iter().zip(low.iter())
            .map(|(h, l)| (h + l) / 2.0)
            .collect();

        (high, low, close)
    }

    #[test]
    fn test_triangle_pattern_creation() {
        let indicator = TrianglePattern::new();
        assert_eq!(indicator.config.min_length, 10);
        assert_eq!(indicator.config.max_length, 50);
    }

    #[test]
    fn test_triangle_pattern_with_params() {
        let indicator = TrianglePattern::with_params(15, 40, 0.25);
        assert_eq!(indicator.config.min_length, 15);
        assert_eq!(indicator.config.max_length, 40);
        assert_eq!(indicator.config.min_convergence, 0.25);
    }

    #[test]
    fn test_linear_regression_slope() {
        let indicator = TrianglePattern::new();

        // Ascending values
        let ascending = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let slope = indicator.linear_regression_slope(&ascending);
        assert!(slope > 0.0);

        // Descending values
        let descending = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let slope = indicator.linear_regression_slope(&descending);
        assert!(slope < 0.0);
    }

    #[test]
    fn test_is_flat() {
        let indicator = TrianglePattern::new();

        // Flat values
        let flat = vec![100.0, 100.5, 99.5, 100.0, 100.2];
        assert!(indicator.is_flat(&flat));

        // Not flat values
        let not_flat = vec![100.0, 105.0, 110.0, 115.0, 120.0];
        assert!(!indicator.is_flat(&not_flat));
    }

    #[test]
    fn test_find_local_highs() {
        let indicator = TrianglePattern::new();
        let high = vec![100.0, 105.0, 110.0, 105.0, 100.0, 105.0, 115.0, 105.0, 100.0];

        let highs = indicator.find_local_highs(&high, 2);
        assert!(!highs.is_empty());
    }

    #[test]
    fn test_find_local_lows() {
        let indicator = TrianglePattern::new();
        let low = vec![100.0, 95.0, 90.0, 95.0, 100.0, 95.0, 85.0, 95.0, 100.0];

        let lows = indicator.find_local_lows(&low, 2);
        assert!(!lows.is_empty());
    }

    #[test]
    fn test_detect_ascending() {
        let (high, low, _) = create_ascending_triangle_data();
        let indicator = TrianglePattern::with_params(10, 30, 0.2);

        let ascending = indicator.detect_ascending(&high, &low);
        assert_eq!(ascending.len(), high.len());
    }

    #[test]
    fn test_detect_descending() {
        let (high, low, _) = create_descending_triangle_data();
        let indicator = TrianglePattern::with_params(10, 30, 0.2);

        let descending = indicator.detect_descending(&high, &low);
        assert_eq!(descending.len(), high.len());
    }

    #[test]
    fn test_detect_symmetrical() {
        let (high, low, _) = create_symmetrical_triangle_data();
        let indicator = TrianglePattern::with_params(10, 30, 0.2);

        let symmetrical = indicator.detect_symmetrical(&high, &low);
        assert_eq!(symmetrical.len(), high.len());
    }

    #[test]
    fn test_apex_distance() {
        let (high, low, _) = create_symmetrical_triangle_data();
        let indicator = TrianglePattern::with_params(10, 30, 0.2);

        let distances = indicator.apex_distance(&high, &low);
        assert_eq!(distances.len(), high.len());
    }

    #[test]
    fn test_completion_pct() {
        let (high, low, _) = create_symmetrical_triangle_data();
        let indicator = TrianglePattern::with_params(10, 30, 0.2);

        let completion = indicator.completion_pct(&high, &low);
        assert_eq!(completion.len(), high.len());

        // Later bars should show higher completion
        let last = completion.last().unwrap();
        assert!(*last >= 0.0 && *last <= 1.0);
    }

    #[test]
    fn test_breakout_probability() {
        let (high, low, close) = create_symmetrical_triangle_data();
        let indicator = TrianglePattern::with_params(10, 30, 0.2);

        let probs = indicator.breakout_probability(&high, &low, &close);
        assert_eq!(probs.len(), high.len());

        // All probabilities should be between 0 and 1
        for p in &probs {
            assert!(*p >= 0.0 && *p <= 1.0);
        }
    }

    #[test]
    fn test_calculate() {
        let (high, low, close) = create_symmetrical_triangle_data();
        let indicator = TrianglePattern::with_params(10, 30, 0.2);

        let signals = indicator.calculate(&high, &low, &close);
        assert_eq!(signals.len(), high.len());
    }

    #[test]
    fn test_min_periods() {
        let indicator = TrianglePattern::with_params(15, 40, 0.25);
        assert_eq!(indicator.min_periods(), 15);
    }

    #[test]
    fn test_insufficient_data() {
        let indicator = TrianglePattern::new();
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
    fn test_triangle_type_enum() {
        assert_eq!(TriangleType::Symmetrical, TriangleType::Symmetrical);
        assert_ne!(TriangleType::Ascending, TriangleType::Descending);
        assert_ne!(TriangleType::Symmetrical, TriangleType::None);
    }
}
