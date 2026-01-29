//! Diamond Pattern Indicator (IND-344)
//!
//! Detects diamond reversal patterns - rare formations that combine
//! broadening and converging price action.

use indicator_spi::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result,
    SignalIndicator, TechnicalIndicator,
};

/// Diamond Pattern detection for rare reversal formations.
///
/// A diamond pattern forms when price action first broadens (expanding
/// range) and then contracts (converging range), creating a diamond shape.
/// This is a rare but significant reversal pattern.
///
/// Pattern characteristics:
/// - First half: Broadening formation (higher highs, lower lows)
/// - Second half: Contracting formation (lower highs, higher lows)
/// - Diamond Top: Bearish reversal signal
/// - Diamond Bottom: Bullish reversal signal
#[derive(Debug, Clone)]
pub struct DiamondPattern {
    /// Lookback period for pattern detection
    lookback: usize,
    /// Minimum expansion ratio for broadening phase
    min_expansion: f64,
    /// Minimum contraction ratio for converging phase
    min_contraction: f64,
}

/// Diamond pattern type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DiamondType {
    None,
    Top,    // Bearish
    Bottom, // Bullish
}

/// Output from diamond pattern detection
#[derive(Debug, Clone)]
pub struct DiamondPatternOutput {
    /// Pattern type detected
    pub pattern_type: DiamondType,
    /// Pattern strength (0.0-1.0)
    pub strength: f64,
    /// Expansion ratio of broadening phase
    pub expansion_ratio: f64,
    /// Contraction ratio of converging phase
    pub contraction_ratio: f64,
    /// Pattern width in bars
    pub width: usize,
}

impl DiamondPattern {
    /// Create a new Diamond Pattern indicator.
    ///
    /// # Arguments
    /// * `lookback` - Number of periods to analyze (minimum 20)
    /// * `min_expansion` - Minimum expansion ratio (e.g., 1.5 = 50% expansion)
    /// * `min_contraction` - Minimum contraction ratio (e.g., 0.5 = 50% contraction)
    pub fn new(lookback: usize, min_expansion: f64, min_contraction: f64) -> Result<Self> {
        if lookback < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        if min_expansion <= 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "min_expansion".to_string(),
                reason: "must be greater than 1.0".to_string(),
            });
        }
        if min_contraction <= 0.0 || min_contraction >= 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "min_contraction".to_string(),
                reason: "must be between 0 and 1".to_string(),
            });
        }
        Ok(Self {
            lookback,
            min_expansion,
            min_contraction,
        })
    }

    /// Create with default parameters
    pub fn default_params() -> Self {
        Self {
            lookback: 30,
            min_expansion: 1.3,
            min_contraction: 0.7,
        }
    }

    /// Calculate diamond pattern detection.
    ///
    /// Returns: 1.0 = diamond bottom (bullish), -1.0 = diamond top (bearish), 0.0 = none
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = high.len().min(low.len()).min(close.len());
        let mut result = vec![0.0; n];

        for i in self.lookback..n {
            let start = i.saturating_sub(self.lookback);
            let window_high = &high[start..=i];
            let window_low = &low[start..=i];
            let window_close = &close[start..=i];

            if let Some((diamond_type, strength)) =
                self.detect_diamond(window_high, window_low, window_close)
            {
                result[i] = match diamond_type {
                    DiamondType::Bottom => strength,
                    DiamondType::Top => -strength,
                    DiamondType::None => 0.0,
                };
            }
        }

        result
    }

    /// Detect diamond pattern in window.
    fn detect_diamond(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Option<(DiamondType, f64)> {
        let n = high.len();
        if n < 10 {
            return None;
        }

        let mid_point = n / 2;

        // Calculate ranges for first half (should expand)
        let first_half_ranges = self.calculate_expanding_ranges(&high[..mid_point], &low[..mid_point]);

        // Calculate ranges for second half (should contract)
        let second_half_ranges = self.calculate_contracting_ranges(&high[mid_point..], &low[mid_point..]);

        // Check for valid expansion
        if first_half_ranges.is_empty() || second_half_ranges.is_empty() {
            return None;
        }

        let first_range = first_half_ranges.first()?;
        let mid_range_expand = first_half_ranges.last()?;
        let mid_range_contract = second_half_ranges.first()?;
        let last_range = second_half_ranges.last()?;

        // Expansion ratio
        let expansion_ratio = if *first_range > 0.0 {
            mid_range_expand / first_range
        } else {
            0.0
        };

        // Contraction ratio
        let contraction_ratio = if *mid_range_contract > 0.0 {
            last_range / mid_range_contract
        } else {
            1.0
        };

        // Validate diamond shape
        if expansion_ratio < self.min_expansion || contraction_ratio > self.min_contraction {
            return None;
        }

        // Determine if top or bottom
        let first_half_avg = close[..mid_point].iter().sum::<f64>() / mid_point as f64;
        let second_half_avg = close[mid_point..].iter().sum::<f64>() / (n - mid_point) as f64;

        let overall_trend = second_half_avg - first_half_avg;

        // Calculate strength based on pattern quality
        let strength = ((expansion_ratio - 1.0) + (1.0 - contraction_ratio)) / 2.0;
        let normalized_strength = strength.min(1.0);

        if overall_trend > 0.0 {
            // Price rising into pattern - diamond top (bearish)
            Some((DiamondType::Top, normalized_strength))
        } else {
            // Price falling into pattern - diamond bottom (bullish)
            Some((DiamondType::Bottom, normalized_strength))
        }
    }

    /// Calculate expanding ranges for first half.
    fn calculate_expanding_ranges(&self, high: &[f64], low: &[f64]) -> Vec<f64> {
        let n = high.len().min(low.len());
        if n < 3 {
            return vec![];
        }

        let window_size = 3;
        let mut ranges = Vec::new();

        for i in 0..(n - window_size + 1) {
            let window_high = high[i..i + window_size]
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let window_low = low[i..i + window_size]
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b));
            ranges.push(window_high - window_low);
        }

        ranges
    }

    /// Calculate contracting ranges for second half.
    fn calculate_contracting_ranges(&self, high: &[f64], low: &[f64]) -> Vec<f64> {
        self.calculate_expanding_ranges(high, low)
    }

    /// Calculate detailed diamond pattern output.
    pub fn calculate_detailed(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Vec<DiamondPatternOutput> {
        let n = high.len().min(low.len()).min(close.len());
        let mut results = Vec::with_capacity(n);

        for i in 0..n {
            if i < self.lookback {
                results.push(DiamondPatternOutput {
                    pattern_type: DiamondType::None,
                    strength: 0.0,
                    expansion_ratio: 0.0,
                    contraction_ratio: 0.0,
                    width: 0,
                });
                continue;
            }

            let start = i.saturating_sub(self.lookback);
            let window_high = &high[start..=i];
            let window_low = &low[start..=i];
            let window_close = &close[start..=i];

            let mid_point = window_high.len() / 2;

            let first_half_ranges =
                self.calculate_expanding_ranges(&window_high[..mid_point], &window_low[..mid_point]);
            let second_half_ranges = self
                .calculate_contracting_ranges(&window_high[mid_point..], &window_low[mid_point..]);

            if first_half_ranges.is_empty() || second_half_ranges.is_empty() {
                results.push(DiamondPatternOutput {
                    pattern_type: DiamondType::None,
                    strength: 0.0,
                    expansion_ratio: 0.0,
                    contraction_ratio: 0.0,
                    width: 0,
                });
                continue;
            }

            let first_range = first_half_ranges[0];
            let mid_range = *first_half_ranges.last().unwrap_or(&0.0);
            let last_range = *second_half_ranges.last().unwrap_or(&0.0);

            let expansion_ratio = if first_range > 0.0 {
                mid_range / first_range
            } else {
                0.0
            };

            let contraction_ratio = if mid_range > 0.0 {
                last_range / mid_range
            } else {
                1.0
            };

            if let Some((pattern_type, strength)) =
                self.detect_diamond(window_high, window_low, window_close)
            {
                results.push(DiamondPatternOutput {
                    pattern_type,
                    strength,
                    expansion_ratio,
                    contraction_ratio,
                    width: i - start + 1,
                });
            } else {
                results.push(DiamondPatternOutput {
                    pattern_type: DiamondType::None,
                    strength: 0.0,
                    expansion_ratio,
                    contraction_ratio,
                    width: 0,
                });
            }
        }

        results
    }

    /// Calculate the symmetry score of a potential diamond.
    pub fn symmetry_score(&self, high: &[f64], low: &[f64]) -> f64 {
        let n = high.len().min(low.len());
        if n < 10 {
            return 0.0;
        }

        let mid = n / 2;

        // Compare left and right sides
        let mut symmetry_sum = 0.0;
        let compare_len = mid.min(n - mid);

        for i in 0..compare_len {
            let left_range = high[mid - 1 - i] - low[mid - 1 - i];
            let right_range = high[mid + i] - low[mid + i];

            if left_range > 0.0 && right_range > 0.0 {
                let ratio = (left_range / right_range).min(right_range / left_range);
                symmetry_sum += ratio;
            }
        }

        if compare_len > 0 {
            symmetry_sum / compare_len as f64
        } else {
            0.0
        }
    }
}

impl TechnicalIndicator for DiamondPattern {
    fn name(&self) -> &str {
        "Diamond Pattern"
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let values = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::single(values))
    }
}

impl SignalIndicator for DiamondPattern {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.high, &data.low, &data.close);

        if let Some(&last) = values.last() {
            if last > 0.0 {
                return Ok(IndicatorSignal::Bullish); // Diamond bottom
            } else if last < 0.0 {
                return Ok(IndicatorSignal::Bearish); // Diamond top
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.high, &data.low, &data.close);
        let signals = values
            .iter()
            .map(|&v| {
                if v > 0.0 {
                    IndicatorSignal::Bullish
                } else if v < 0.0 {
                    IndicatorSignal::Bearish
                } else {
                    IndicatorSignal::Neutral
                }
            })
            .collect();

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_diamond_pattern() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        // Create diamond pattern: expand then contract
        let mut high = Vec::with_capacity(40);
        let mut low = Vec::with_capacity(40);
        let mut close = Vec::with_capacity(40);

        // First half: expanding
        for i in 0..20 {
            let expansion = i as f64 * 0.3;
            high.push(105.0 + expansion);
            low.push(95.0 - expansion);
            close.push(100.0 + (i % 2) as f64);
        }

        // Second half: contracting
        for i in 0..20 {
            let contraction = (20 - i) as f64 * 0.3;
            high.push(105.0 + contraction);
            low.push(95.0 - contraction);
            close.push(100.0 - (i % 2) as f64);
        }

        (high, low, close)
    }

    #[test]
    fn test_diamond_pattern_creation() {
        let dp = DiamondPattern::new(30, 1.3, 0.7);
        assert!(dp.is_ok());

        let dp_invalid = DiamondPattern::new(10, 1.3, 0.7);
        assert!(dp_invalid.is_err());
    }

    #[test]
    fn test_diamond_detection() {
        let (high, low, close) = make_diamond_pattern();
        let dp = DiamondPattern::new(25, 1.1, 0.9).unwrap();

        let result = dp.calculate(&high, &low, &close);
        assert_eq!(result.len(), high.len());
    }

    #[test]
    fn test_detailed_output() {
        let (high, low, close) = make_diamond_pattern();
        let dp = DiamondPattern::default_params();

        let detailed = dp.calculate_detailed(&high, &low, &close);
        assert_eq!(detailed.len(), high.len());
    }

    #[test]
    fn test_symmetry_score() {
        let dp = DiamondPattern::default_params();

        // Symmetric data
        let high = vec![105.0, 106.0, 107.0, 108.0, 107.0, 106.0, 105.0];
        let low = vec![95.0, 94.0, 93.0, 92.0, 93.0, 94.0, 95.0];

        let symmetry = dp.symmetry_score(&high, &low);
        assert!(symmetry > 0.0);
    }

    #[test]
    fn test_expanding_ranges() {
        let dp = DiamondPattern::default_params();

        let high = vec![105.0, 107.0, 110.0, 114.0, 119.0];
        let low = vec![95.0, 93.0, 90.0, 86.0, 81.0];

        let ranges = dp.calculate_expanding_ranges(&high, &low);
        assert!(!ranges.is_empty());

        // Ranges should be increasing
        for i in 1..ranges.len() {
            assert!(ranges[i] >= ranges[i - 1] - 0.001);
        }
    }
}
