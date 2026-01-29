//! Wedge Pattern Indicator (IND-337)
//!
//! Converging trend lines pattern (rising and falling wedges).

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Wedge pattern type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WedgeType {
    /// Rising wedge (bearish pattern).
    Rising,
    /// Falling wedge (bullish pattern).
    Falling,
    /// No wedge detected.
    None,
}

/// Configuration for Wedge Pattern detection.
#[derive(Debug, Clone)]
pub struct WedgePatternConfig {
    /// Minimum length of wedge in bars.
    pub min_length: usize,
    /// Maximum length of wedge in bars.
    pub max_length: usize,
    /// Minimum convergence ratio (how much the lines must converge).
    pub min_convergence: f64,
    /// Number of touch points required on each trend line.
    pub min_touches: usize,
}

impl Default for WedgePatternConfig {
    fn default() -> Self {
        Self {
            min_length: 10,
            max_length: 50,
            min_convergence: 0.3,
            min_touches: 2,
        }
    }
}

/// Wedge Pattern indicator for converging trend line detection.
///
/// A wedge pattern forms when price moves between two converging trend lines:
/// - Rising wedge: Both lines slope upward, but the lower line is steeper (bearish)
/// - Falling wedge: Both lines slope downward, but the upper line is steeper (bullish)
#[derive(Debug, Clone)]
pub struct WedgePattern {
    config: WedgePatternConfig,
}

impl WedgePattern {
    /// Create a new Wedge Pattern indicator with default settings.
    pub fn new() -> Self {
        Self {
            config: WedgePatternConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: WedgePatternConfig) -> Self {
        Self { config }
    }

    /// Create with custom parameters.
    pub fn with_params(min_length: usize, max_length: usize, min_convergence: f64) -> Self {
        Self {
            config: WedgePatternConfig {
                min_length,
                max_length,
                min_convergence,
                ..Default::default()
            },
        }
    }

    /// Calculate simple linear regression slope.
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

    /// Find local highs for upper trend line.
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

    /// Find local lows for lower trend line.
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

    /// Detect wedge type at each point.
    pub fn detect_wedge_type(&self, high: &[f64], low: &[f64]) -> Vec<WedgeType> {
        let n = high.len();
        let mut wedge_types = vec![WedgeType::None; n];

        if n < self.config.min_length {
            return wedge_types;
        }

        let window = 2;
        let local_highs = self.find_local_highs(high, window);
        let local_lows = self.find_local_lows(low, window);

        for end_idx in self.config.min_length..n {
            // Try different wedge lengths
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

                if initial_range <= 0.0 {
                    continue;
                }

                let convergence = 1.0 - (final_range / initial_range);

                if convergence < self.config.min_convergence {
                    continue;
                }

                // Determine wedge type
                if high_slope > 0.0 && low_slope > 0.0 && low_slope > high_slope * 0.5 {
                    // Rising wedge: both slopes positive, lower line steeper
                    wedge_types[end_idx] = WedgeType::Rising;
                } else if high_slope < 0.0 && low_slope < 0.0 && high_slope > low_slope * 0.5 {
                    // Falling wedge: both slopes negative, upper line steeper (less negative)
                    wedge_types[end_idx] = WedgeType::Falling;
                }
            }
        }

        wedge_types
    }

    /// Calculate wedge signals.
    ///
    /// Returns a vector where:
    /// - Positive values indicate falling wedge (bullish)
    /// - Negative values indicate rising wedge (bearish)
    /// - 0.0 indicates no wedge
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let wedge_types = self.detect_wedge_type(high, low);

        wedge_types.iter().map(|wt| match wt {
            WedgeType::Falling => 1.0,
            WedgeType::Rising => -1.0,
            WedgeType::None => 0.0,
        }).collect()
    }

    /// Calculate wedge angle (convergence angle in degrees).
    pub fn wedge_angle(&self, high: &[f64], low: &[f64]) -> Vec<f64> {
        let n = high.len();
        let mut angles = vec![0.0; n];

        if n < self.config.min_length {
            return angles;
        }

        for i in self.config.min_length..n {
            let start = i.saturating_sub(self.config.min_length);

            let high_slice: Vec<f64> = high[start..=i].to_vec();
            let low_slice: Vec<f64> = low[start..=i].to_vec();

            let high_slope = self.linear_regression_slope(&high_slice);
            let low_slope = self.linear_regression_slope(&low_slice);

            // Angle between the two lines
            let angle = ((high_slope - low_slope).atan() * 180.0 / std::f64::consts::PI).abs();
            angles[i] = angle;
        }

        angles
    }

    /// Detect rising wedge patterns.
    pub fn detect_rising_wedge(&self, high: &[f64], low: &[f64]) -> Vec<bool> {
        self.detect_wedge_type(high, low)
            .iter()
            .map(|wt| *wt == WedgeType::Rising)
            .collect()
    }

    /// Detect falling wedge patterns.
    pub fn detect_falling_wedge(&self, high: &[f64], low: &[f64]) -> Vec<bool> {
        self.detect_wedge_type(high, low)
            .iter()
            .map(|wt| *wt == WedgeType::Falling)
            .collect()
    }
}

impl Default for WedgePattern {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for WedgePattern {
    fn name(&self) -> &str {
        "WedgePattern"
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

impl SignalIndicator for WedgePattern {
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

    fn create_rising_wedge_data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        // Rising wedge: both highs and lows ascending, but converging
        let mut high = Vec::new();
        let mut low = Vec::new();

        for i in 0..20 {
            // Upper line rising slowly
            high.push(100.0 + i as f64 * 0.5);
            // Lower line rising faster (converging)
            low.push(90.0 + i as f64 * 0.7);
        }

        let close: Vec<f64> = high.iter().zip(low.iter())
            .map(|(h, l)| (h + l) / 2.0)
            .collect();

        (high, low, close)
    }

    fn create_falling_wedge_data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        // Falling wedge: both highs and lows descending, but converging
        let mut high = Vec::new();
        let mut low = Vec::new();

        for i in 0..20 {
            // Upper line falling slowly (converging)
            high.push(110.0 - i as f64 * 0.5);
            // Lower line falling faster
            low.push(100.0 - i as f64 * 0.7);
        }

        let close: Vec<f64> = high.iter().zip(low.iter())
            .map(|(h, l)| (h + l) / 2.0)
            .collect();

        (high, low, close)
    }

    #[test]
    fn test_wedge_pattern_creation() {
        let indicator = WedgePattern::new();
        assert_eq!(indicator.config.min_length, 10);
        assert_eq!(indicator.config.max_length, 50);
    }

    #[test]
    fn test_wedge_pattern_with_params() {
        let indicator = WedgePattern::with_params(15, 40, 0.25);
        assert_eq!(indicator.config.min_length, 15);
        assert_eq!(indicator.config.max_length, 40);
        assert_eq!(indicator.config.min_convergence, 0.25);
    }

    #[test]
    fn test_linear_regression_slope() {
        let indicator = WedgePattern::new();

        // Ascending values should have positive slope
        let ascending = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let slope = indicator.linear_regression_slope(&ascending);
        assert!(slope > 0.0);

        // Descending values should have negative slope
        let descending = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let slope = indicator.linear_regression_slope(&descending);
        assert!(slope < 0.0);

        // Flat values should have near-zero slope
        let flat = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let slope = indicator.linear_regression_slope(&flat);
        assert!(slope.abs() < 0.01);
    }

    #[test]
    fn test_find_local_highs() {
        let indicator = WedgePattern::new();
        let high = vec![100.0, 105.0, 110.0, 105.0, 100.0, 105.0, 115.0, 105.0, 100.0];

        let highs = indicator.find_local_highs(&high, 2);
        assert!(!highs.is_empty());
    }

    #[test]
    fn test_find_local_lows() {
        let indicator = WedgePattern::new();
        let low = vec![100.0, 95.0, 90.0, 95.0, 100.0, 95.0, 85.0, 95.0, 100.0];

        let lows = indicator.find_local_lows(&low, 2);
        assert!(!lows.is_empty());
    }

    #[test]
    fn test_wedge_angle() {
        let (high, low, _) = create_rising_wedge_data();
        let indicator = WedgePattern::with_params(10, 30, 0.2);

        let angles = indicator.wedge_angle(&high, &low);
        assert_eq!(angles.len(), high.len());
    }

    #[test]
    fn test_detect_rising_wedge() {
        let (high, low, _) = create_rising_wedge_data();
        let indicator = WedgePattern::with_params(10, 30, 0.2);

        let rising = indicator.detect_rising_wedge(&high, &low);
        assert_eq!(rising.len(), high.len());
    }

    #[test]
    fn test_detect_falling_wedge() {
        let (high, low, _) = create_falling_wedge_data();
        let indicator = WedgePattern::with_params(10, 30, 0.2);

        let falling = indicator.detect_falling_wedge(&high, &low);
        assert_eq!(falling.len(), high.len());
    }

    #[test]
    fn test_calculate() {
        let (high, low, close) = create_rising_wedge_data();
        let indicator = WedgePattern::with_params(10, 30, 0.2);

        let signals = indicator.calculate(&high, &low, &close);
        assert_eq!(signals.len(), high.len());
    }

    #[test]
    fn test_min_periods() {
        let indicator = WedgePattern::with_params(15, 40, 0.25);
        assert_eq!(indicator.min_periods(), 15);
    }

    #[test]
    fn test_insufficient_data() {
        let indicator = WedgePattern::new();
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
    fn test_wedge_type_enum() {
        assert_eq!(WedgeType::Rising, WedgeType::Rising);
        assert_ne!(WedgeType::Rising, WedgeType::Falling);
        assert_ne!(WedgeType::Falling, WedgeType::None);
    }
}
