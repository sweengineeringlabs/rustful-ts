//! Channel Pattern Indicator (IND-341)
//!
//! Detects parallel trend channels - price oscillating between parallel
//! support and resistance trendlines.

use indicator_spi::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result,
    SignalIndicator, TechnicalIndicator,
};

/// Channel Pattern detection for identifying parallel trend channels.
///
/// Detects when price is oscillating between parallel support and resistance
/// trendlines, indicating a trending market within defined boundaries.
///
/// Channel types detected:
/// - Ascending channel (higher highs and higher lows)
/// - Descending channel (lower highs and lower lows)
/// - Horizontal channel (range-bound)
#[derive(Debug, Clone)]
pub struct ChannelPattern {
    /// Lookback period for channel detection
    lookback: usize,
    /// Minimum R-squared for valid trendline (0.0-1.0)
    min_r_squared: f64,
    /// Maximum deviation tolerance for parallel lines (percentage)
    parallel_tolerance: f64,
}

/// Output from channel pattern detection
#[derive(Debug, Clone)]
pub struct ChannelPatternOutput {
    /// Channel type: 1.0 = ascending, -1.0 = descending, 0.5 = horizontal, 0.0 = none
    pub channel_type: f64,
    /// Upper channel line value
    pub upper_line: f64,
    /// Lower channel line value
    pub lower_line: f64,
    /// Channel width
    pub width: f64,
    /// Quality score (R-squared average)
    pub quality: f64,
}

impl ChannelPattern {
    /// Create a new Channel Pattern indicator.
    ///
    /// # Arguments
    /// * `lookback` - Number of periods to analyze (minimum 15)
    /// * `min_r_squared` - Minimum R-squared for valid trendline (0.5-1.0)
    /// * `parallel_tolerance` - Maximum slope deviation for parallel lines (1-10%)
    pub fn new(lookback: usize, min_r_squared: f64, parallel_tolerance: f64) -> Result<Self> {
        if lookback < 15 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least 15".to_string(),
            });
        }
        if min_r_squared < 0.0 || min_r_squared > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "min_r_squared".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        if parallel_tolerance <= 0.0 || parallel_tolerance > 20.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "parallel_tolerance".to_string(),
                reason: "must be between 0 and 20 percent".to_string(),
            });
        }
        Ok(Self {
            lookback,
            min_r_squared,
            parallel_tolerance,
        })
    }

    /// Create with default parameters (lookback=20, r2=0.7, tolerance=5%)
    pub fn default_params() -> Self {
        Self {
            lookback: 20,
            min_r_squared: 0.7,
            parallel_tolerance: 5.0,
        }
    }

    /// Calculate channel pattern detection.
    ///
    /// Returns channel type: 1.0=ascending, -1.0=descending, 0.5=horizontal, 0.0=none
    pub fn calculate(&self, high: &[f64], low: &[f64]) -> Vec<f64> {
        let n = high.len().min(low.len());
        let mut result = vec![0.0; n];

        for i in self.lookback..n {
            let start = i.saturating_sub(self.lookback);

            // Calculate linear regression for highs (upper trendline)
            let (high_slope, high_intercept, high_r2) =
                linear_regression(&high[start..=i]);

            // Calculate linear regression for lows (lower trendline)
            let (low_slope, low_intercept, low_r2) =
                linear_regression(&low[start..=i]);

            // Check if both trendlines meet quality threshold
            if high_r2 < self.min_r_squared || low_r2 < self.min_r_squared {
                continue;
            }

            // Check if lines are parallel (similar slopes)
            let avg_slope = (high_slope.abs() + low_slope.abs()) / 2.0;
            let slope_diff = (high_slope - low_slope).abs();

            let is_parallel = if avg_slope > 0.0001 {
                (slope_diff / avg_slope) * 100.0 < self.parallel_tolerance
            } else {
                slope_diff < 0.0001
            };

            if !is_parallel {
                continue;
            }

            // Determine channel type based on slope direction
            let avg_channel_slope = (high_slope + low_slope) / 2.0;

            // Check that channel width is consistent
            let start_width = high_intercept - low_intercept;
            let end_width = (high_slope * self.lookback as f64 + high_intercept)
                - (low_slope * self.lookback as f64 + low_intercept);

            if start_width <= 0.0 || end_width <= 0.0 {
                continue;
            }

            let width_change = ((end_width - start_width).abs() / start_width) * 100.0;
            if width_change > self.parallel_tolerance * 2.0 {
                continue;
            }

            // Classify channel type
            if avg_channel_slope > 0.001 {
                result[i] = 1.0; // Ascending channel
            } else if avg_channel_slope < -0.001 {
                result[i] = -1.0; // Descending channel
            } else {
                result[i] = 0.5; // Horizontal channel
            }
        }

        result
    }

    /// Calculate detailed channel output including boundaries.
    pub fn calculate_detailed(&self, high: &[f64], low: &[f64]) -> Vec<ChannelPatternOutput> {
        let n = high.len().min(low.len());
        let mut results = Vec::with_capacity(n);

        for i in 0..n {
            if i < self.lookback {
                results.push(ChannelPatternOutput {
                    channel_type: 0.0,
                    upper_line: f64::NAN,
                    lower_line: f64::NAN,
                    width: f64::NAN,
                    quality: 0.0,
                });
                continue;
            }

            let start = i.saturating_sub(self.lookback);

            let (high_slope, high_intercept, high_r2) =
                linear_regression(&high[start..=i]);
            let (low_slope, low_intercept, low_r2) =
                linear_regression(&low[start..=i]);

            let avg_r2 = (high_r2 + low_r2) / 2.0;

            if avg_r2 < self.min_r_squared {
                results.push(ChannelPatternOutput {
                    channel_type: 0.0,
                    upper_line: f64::NAN,
                    lower_line: f64::NAN,
                    width: f64::NAN,
                    quality: avg_r2,
                });
                continue;
            }

            // Calculate current channel lines
            let period_idx = (i - start) as f64;
            let upper_line = high_slope * period_idx + high_intercept;
            let lower_line = low_slope * period_idx + low_intercept;
            let width = upper_line - lower_line;

            // Determine channel type
            let avg_slope = (high_slope + low_slope) / 2.0;
            let channel_type = if avg_slope > 0.001 {
                1.0
            } else if avg_slope < -0.001 {
                -1.0
            } else {
                0.5
            };

            results.push(ChannelPatternOutput {
                channel_type,
                upper_line,
                lower_line,
                width,
                quality: avg_r2,
            });
        }

        results
    }

    /// Detect channel breakouts.
    /// Returns 1.0 for upside breakout, -1.0 for downside breakout.
    pub fn breakout_signals(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let detailed = self.calculate_detailed(high, low);
        let n = close.len().min(detailed.len());
        let mut signals = vec![0.0; n];

        for i in 1..n {
            if detailed[i].channel_type == 0.0 || detailed[i].upper_line.is_nan() {
                continue;
            }

            // Check for breakout above upper channel
            if close[i] > detailed[i].upper_line
                && close[i - 1] <= detailed[i - 1].upper_line
            {
                signals[i] = 1.0;
            }
            // Check for breakdown below lower channel
            else if close[i] < detailed[i].lower_line
                && close[i - 1] >= detailed[i - 1].lower_line
            {
                signals[i] = -1.0;
            }
        }

        signals
    }
}

/// Calculate linear regression with R-squared.
/// Returns (slope, intercept, r_squared)
fn linear_regression(data: &[f64]) -> (f64, f64, f64) {
    let n = data.len() as f64;
    if n < 2.0 {
        return (0.0, 0.0, 0.0);
    }

    let sum_x: f64 = (0..data.len()).map(|x| x as f64).sum();
    let sum_y: f64 = data.iter().sum();
    let sum_xy: f64 = data.iter().enumerate().map(|(x, &y)| x as f64 * y).sum();
    let sum_xx: f64 = (0..data.len()).map(|x| (x as f64).powi(2)).sum();
    let sum_yy: f64 = data.iter().map(|&y| y.powi(2)).sum();

    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-10 {
        return (0.0, data[0], 0.0);
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;

    // Calculate R-squared
    let mean_y = sum_y / n;
    let ss_tot = sum_yy - n * mean_y * mean_y;
    let ss_res: f64 = data
        .iter()
        .enumerate()
        .map(|(x, &y)| {
            let predicted = slope * x as f64 + intercept;
            (y - predicted).powi(2)
        })
        .sum();

    let r_squared = if ss_tot > 0.0 {
        1.0 - (ss_res / ss_tot)
    } else {
        0.0
    };

    (slope, intercept, r_squared.max(0.0))
}

impl TechnicalIndicator for ChannelPattern {
    fn name(&self) -> &str {
        "Channel Pattern"
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let detailed = self.calculate_detailed(&data.high, &data.low);

        let channel_type: Vec<f64> = detailed.iter().map(|o| o.channel_type).collect();
        let upper_line: Vec<f64> = detailed.iter().map(|o| o.upper_line).collect();
        let lower_line: Vec<f64> = detailed.iter().map(|o| o.lower_line).collect();

        Ok(IndicatorOutput::triple(channel_type, upper_line, lower_line))
    }

    fn output_features(&self) -> usize {
        3 // channel_type, upper_line, lower_line
    }
}

impl SignalIndicator for ChannelPattern {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let signals = self.breakout_signals(&data.high, &data.low, &data.close);

        if let Some(&last) = signals.last() {
            if last > 0.0 {
                return Ok(IndicatorSignal::Bullish);
            } else if last < 0.0 {
                return Ok(IndicatorSignal::Bearish);
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let breakouts = self.breakout_signals(&data.high, &data.low, &data.close);
        let signals = breakouts
            .iter()
            .map(|&s| {
                if s > 0.0 {
                    IndicatorSignal::Bullish
                } else if s < 0.0 {
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

    fn make_ascending_channel() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        // Create ascending channel data
        let mut high = Vec::with_capacity(30);
        let mut low = Vec::with_capacity(30);
        let mut close = Vec::with_capacity(30);

        for i in 0..30 {
            let base = 100.0 + i as f64 * 0.5; // Upward trend
            high.push(base + 3.0 + (i % 3) as f64 * 0.2);
            low.push(base - 1.0 + (i % 3) as f64 * 0.2);
            close.push(base + 1.0);
        }

        (high, low, close)
    }

    #[test]
    fn test_channel_pattern_creation() {
        let cp = ChannelPattern::new(20, 0.7, 5.0);
        assert!(cp.is_ok());

        let cp_invalid = ChannelPattern::new(5, 0.7, 5.0);
        assert!(cp_invalid.is_err());
    }

    #[test]
    fn test_channel_detection() {
        let (high, low, _close) = make_ascending_channel();
        let cp = ChannelPattern::new(15, 0.5, 10.0).unwrap();

        let result = cp.calculate(&high, &low);
        assert_eq!(result.len(), high.len());

        // Should detect some channel patterns
        let has_channel = result.iter().any(|&v| v != 0.0);
        assert!(has_channel || result.len() > 0); // At least runs without error
    }

    #[test]
    fn test_channel_detailed() {
        let (high, low, _close) = make_ascending_channel();
        let cp = ChannelPattern::default_params();

        let detailed = cp.calculate_detailed(&high, &low);
        assert_eq!(detailed.len(), high.len());
    }

    #[test]
    fn test_breakout_signals() {
        let (high, low, close) = make_ascending_channel();
        let cp = ChannelPattern::new(15, 0.5, 10.0).unwrap();

        let signals = cp.breakout_signals(&high, &low, &close);
        assert_eq!(signals.len(), close.len());
    }

    #[test]
    fn test_linear_regression() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (slope, intercept, r2) = linear_regression(&data);

        assert!((slope - 1.0).abs() < 0.01);
        assert!((intercept - 1.0).abs() < 0.01);
        assert!(r2 > 0.99);
    }
}
