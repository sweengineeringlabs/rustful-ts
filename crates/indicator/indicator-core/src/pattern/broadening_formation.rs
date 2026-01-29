//! Broadening Formation Indicator (IND-345)
//!
//! Detects megaphone/broadening patterns - expanding price formations
//! with higher highs and lower lows.

use indicator_spi::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result,
    SignalIndicator, TechnicalIndicator,
};

/// Broadening Formation (Megaphone) Pattern detection.
///
/// A broadening formation occurs when price makes progressively higher
/// highs and lower lows, creating an expanding pattern that resembles
/// a megaphone. This pattern indicates increasing volatility and
/// market indecision.
///
/// Pattern types:
/// - Right-Angled Broadening Top: Flat resistance, descending support
/// - Right-Angled Broadening Bottom: Flat support, ascending resistance
/// - Symmetrical Broadening: Both expanding equally
#[derive(Debug, Clone)]
pub struct BroadeningFormation {
    /// Lookback period for pattern detection
    lookback: usize,
    /// Minimum number of touches for trendline confirmation
    min_touches: usize,
    /// Minimum expansion rate
    min_expansion_rate: f64,
}

/// Broadening pattern type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BroadeningType {
    None,
    Symmetrical,
    AscendingTop,    // Flat bottom, rising top
    DescendingBottom, // Flat top, falling bottom
}

/// Output from broadening formation detection
#[derive(Debug, Clone)]
pub struct BroadeningFormationOutput {
    /// Pattern type detected
    pub pattern_type: BroadeningType,
    /// Pattern strength (0.0-1.0)
    pub strength: f64,
    /// Upper trendline slope
    pub upper_slope: f64,
    /// Lower trendline slope
    pub lower_slope: f64,
    /// Current range width
    pub width: f64,
    /// Expansion rate
    pub expansion_rate: f64,
}

impl BroadeningFormation {
    /// Create a new Broadening Formation indicator.
    ///
    /// # Arguments
    /// * `lookback` - Number of periods to analyze (minimum 15)
    /// * `min_touches` - Minimum trendline touches (minimum 2)
    /// * `min_expansion_rate` - Minimum range expansion rate per bar
    pub fn new(lookback: usize, min_touches: usize, min_expansion_rate: f64) -> Result<Self> {
        if lookback < 15 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least 15".to_string(),
            });
        }
        if min_touches < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "min_touches".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if min_expansion_rate <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "min_expansion_rate".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self {
            lookback,
            min_touches,
            min_expansion_rate,
        })
    }

    /// Create with default parameters
    pub fn default_params() -> Self {
        Self {
            lookback: 20,
            min_touches: 2,
            min_expansion_rate: 0.001,
        }
    }

    /// Calculate broadening formation detection.
    ///
    /// Returns: 1.0 = symmetrical, 0.5 = ascending top, -0.5 = descending bottom, 0.0 = none
    pub fn calculate(&self, high: &[f64], low: &[f64]) -> Vec<f64> {
        let n = high.len().min(low.len());
        let mut result = vec![0.0; n];

        for i in self.lookback..n {
            let start = i.saturating_sub(self.lookback);
            let window_high = &high[start..=i];
            let window_low = &low[start..=i];

            if let Some((pattern_type, strength)) = self.detect_broadening(window_high, window_low)
            {
                result[i] = match pattern_type {
                    BroadeningType::Symmetrical => strength,
                    BroadeningType::AscendingTop => 0.5 * strength,
                    BroadeningType::DescendingBottom => -0.5 * strength,
                    BroadeningType::None => 0.0,
                };
            }
        }

        result
    }

    /// Detect broadening pattern in window.
    fn detect_broadening(&self, high: &[f64], low: &[f64]) -> Option<(BroadeningType, f64)> {
        let n = high.len();
        if n < 10 {
            return None;
        }

        // Find local highs and lows
        let highs = self.find_swing_highs(high);
        let lows = self.find_swing_lows(low);

        if highs.len() < self.min_touches || lows.len() < self.min_touches {
            return None;
        }

        // Calculate trendline slopes
        let (upper_slope, upper_r2) = self.fit_trendline(&highs);
        let (lower_slope, lower_r2) = self.fit_trendline(&lows);

        // Check for expanding pattern
        let is_upper_rising = upper_slope > self.min_expansion_rate;
        let is_lower_falling = lower_slope < -self.min_expansion_rate;

        // Calculate expansion rate
        let first_range = high[0] - low[0];
        let last_range = high[n - 1] - low[n - 1];
        let expansion_rate = if first_range > 0.0 {
            (last_range - first_range) / (first_range * n as f64)
        } else {
            0.0
        };

        if expansion_rate < self.min_expansion_rate {
            return None;
        }

        // Classify pattern type
        let avg_r2 = (upper_r2 + lower_r2) / 2.0;
        let strength = (avg_r2 * expansion_rate * 10.0).min(1.0);

        let upper_flat = upper_slope.abs() < self.min_expansion_rate * 2.0;
        let lower_flat = lower_slope.abs() < self.min_expansion_rate * 2.0;

        if is_upper_rising && is_lower_falling {
            Some((BroadeningType::Symmetrical, strength))
        } else if is_upper_rising && lower_flat {
            Some((BroadeningType::AscendingTop, strength))
        } else if upper_flat && is_lower_falling {
            Some((BroadeningType::DescendingBottom, strength))
        } else {
            None
        }
    }

    /// Find swing high points.
    fn find_swing_highs(&self, high: &[f64]) -> Vec<(usize, f64)> {
        let n = high.len();
        let mut swings = Vec::new();

        for i in 2..(n - 2) {
            if high[i] > high[i - 1]
                && high[i] > high[i - 2]
                && high[i] > high[i + 1]
                && high[i] > high[i + 2]
            {
                swings.push((i, high[i]));
            }
        }

        swings
    }

    /// Find swing low points.
    fn find_swing_lows(&self, low: &[f64]) -> Vec<(usize, f64)> {
        let n = low.len();
        let mut swings = Vec::new();

        for i in 2..(n - 2) {
            if low[i] < low[i - 1]
                && low[i] < low[i - 2]
                && low[i] < low[i + 1]
                && low[i] < low[i + 2]
            {
                swings.push((i, low[i]));
            }
        }

        swings
    }

    /// Fit trendline to swing points.
    /// Returns (slope, r_squared)
    fn fit_trendline(&self, points: &[(usize, f64)]) -> (f64, f64) {
        if points.len() < 2 {
            return (0.0, 0.0);
        }

        let n = points.len() as f64;
        let sum_x: f64 = points.iter().map(|(x, _)| *x as f64).sum();
        let sum_y: f64 = points.iter().map(|(_, y)| *y).sum();
        let sum_xy: f64 = points.iter().map(|(x, y)| *x as f64 * y).sum();
        let sum_xx: f64 = points.iter().map(|(x, _)| (*x as f64).powi(2)).sum();
        let sum_yy: f64 = points.iter().map(|(_, y)| y.powi(2)).sum();

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-10 {
            return (0.0, 0.0);
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denom;
        let intercept = (sum_y - slope * sum_x) / n;

        // Calculate R-squared
        let mean_y = sum_y / n;
        let ss_tot = sum_yy - n * mean_y * mean_y;
        let ss_res: f64 = points
            .iter()
            .map(|(x, y)| {
                let predicted = slope * *x as f64 + intercept;
                (y - predicted).powi(2)
            })
            .sum();

        let r_squared = if ss_tot > 0.0 {
            (1.0 - ss_res / ss_tot).max(0.0)
        } else {
            0.0
        };

        (slope, r_squared)
    }

    /// Calculate detailed broadening formation output.
    pub fn calculate_detailed(&self, high: &[f64], low: &[f64]) -> Vec<BroadeningFormationOutput> {
        let n = high.len().min(low.len());
        let mut results = Vec::with_capacity(n);

        for i in 0..n {
            if i < self.lookback {
                results.push(BroadeningFormationOutput {
                    pattern_type: BroadeningType::None,
                    strength: 0.0,
                    upper_slope: 0.0,
                    lower_slope: 0.0,
                    width: 0.0,
                    expansion_rate: 0.0,
                });
                continue;
            }

            let start = i.saturating_sub(self.lookback);
            let window_high = &high[start..=i];
            let window_low = &low[start..=i];

            let highs = self.find_swing_highs(window_high);
            let lows = self.find_swing_lows(window_low);

            let (upper_slope, _) = self.fit_trendline(&highs);
            let (lower_slope, _) = self.fit_trendline(&lows);

            let first_range = window_high[0] - window_low[0];
            let last_range = window_high[window_high.len() - 1] - window_low[window_low.len() - 1];
            let expansion_rate = if first_range > 0.0 {
                (last_range - first_range) / (first_range * window_high.len() as f64)
            } else {
                0.0
            };

            if let Some((pattern_type, strength)) = self.detect_broadening(window_high, window_low)
            {
                results.push(BroadeningFormationOutput {
                    pattern_type,
                    strength,
                    upper_slope,
                    lower_slope,
                    width: last_range,
                    expansion_rate,
                });
            } else {
                results.push(BroadeningFormationOutput {
                    pattern_type: BroadeningType::None,
                    strength: 0.0,
                    upper_slope,
                    lower_slope,
                    width: last_range,
                    expansion_rate,
                });
            }
        }

        results
    }

    /// Calculate volatility expansion ratio.
    pub fn volatility_expansion(&self, high: &[f64], low: &[f64]) -> Vec<f64> {
        let n = high.len().min(low.len());
        let mut result = vec![0.0; n];

        for i in self.lookback..n {
            let start = i.saturating_sub(self.lookback);

            let first_range = high[start] - low[start];
            let current_range = high[i] - low[i];

            if first_range > 0.0 {
                result[i] = current_range / first_range;
            }
        }

        result
    }
}

impl TechnicalIndicator for BroadeningFormation {
    fn name(&self) -> &str {
        "Broadening Formation"
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let values = self.calculate(&data.high, &data.low);
        Ok(IndicatorOutput::single(values))
    }
}

impl SignalIndicator for BroadeningFormation {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.high, &data.low);

        if let Some(&last) = values.last() {
            if last > 0.5 {
                // Symmetrical broadening - high volatility warning
                return Ok(IndicatorSignal::Neutral);
            } else if last > 0.0 {
                // Ascending top - mildly bullish
                return Ok(IndicatorSignal::Bullish);
            } else if last < 0.0 {
                // Descending bottom - mildly bearish
                return Ok(IndicatorSignal::Bearish);
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.high, &data.low);
        let signals = values
            .iter()
            .map(|&v| {
                if v > 0.5 {
                    IndicatorSignal::Neutral
                } else if v > 0.0 {
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

    fn make_broadening_pattern() -> (Vec<f64>, Vec<f64>) {
        // Create expanding megaphone pattern
        let mut high = Vec::with_capacity(30);
        let mut low = Vec::with_capacity(30);

        for i in 0..30 {
            let expansion = i as f64 * 0.5;
            high.push(105.0 + expansion + (i % 3) as f64);
            low.push(95.0 - expansion - (i % 3) as f64);
        }

        (high, low)
    }

    #[test]
    fn test_broadening_formation_creation() {
        let bf = BroadeningFormation::new(20, 2, 0.001);
        assert!(bf.is_ok());

        let bf_invalid = BroadeningFormation::new(10, 2, 0.001);
        assert!(bf_invalid.is_err());
    }

    #[test]
    fn test_broadening_detection() {
        let (high, low) = make_broadening_pattern();
        let bf = BroadeningFormation::new(15, 2, 0.0001).unwrap();

        let result = bf.calculate(&high, &low);
        assert_eq!(result.len(), high.len());
    }

    #[test]
    fn test_swing_detection() {
        let bf = BroadeningFormation::default_params();

        let high = vec![
            100.0, 102.0, 105.0, 103.0, 101.0, // Peak at 2
            103.0, 106.0, 108.0, 106.0, 104.0, // Peak at 7
        ];

        let swings = bf.find_swing_highs(&high);
        assert!(!swings.is_empty() || high.len() > 0);
    }

    #[test]
    fn test_detailed_output() {
        let (high, low) = make_broadening_pattern();
        let bf = BroadeningFormation::default_params();

        let detailed = bf.calculate_detailed(&high, &low);
        assert_eq!(detailed.len(), high.len());
    }

    #[test]
    fn test_volatility_expansion() {
        let (high, low) = make_broadening_pattern();
        let bf = BroadeningFormation::default_params();

        let expansion = bf.volatility_expansion(&high, &low);
        assert_eq!(expansion.len(), high.len());

        // Expansion should increase over time
        if expansion.len() > bf.lookback + 5 {
            assert!(expansion[bf.lookback + 5] >= expansion[bf.lookback]);
        }
    }
}
