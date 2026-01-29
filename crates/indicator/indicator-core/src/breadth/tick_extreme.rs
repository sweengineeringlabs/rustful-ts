//! Tick Extreme - Extreme Tick Readings Detector (IND-398)

use super::BreadthIndicator;
use crate::{IndicatorError, IndicatorOutput, Result};

/// Tick Extreme Configuration
#[derive(Debug, Clone)]
pub struct TickExtremeConfig {
    /// Upper extreme threshold (default: 1000)
    pub upper_extreme: f64,
    /// Lower extreme threshold (default: -1000)
    pub lower_extreme: f64,
    /// Upper warning threshold (default: 800)
    pub upper_warning: f64,
    /// Lower warning threshold (default: -800)
    pub lower_warning: f64,
    /// Lookback period for counting extremes (default: 10)
    pub lookback_period: usize,
    /// Minimum consecutive readings for confirmation (default: 2)
    pub consecutive_threshold: usize,
}

impl Default for TickExtremeConfig {
    fn default() -> Self {
        Self {
            upper_extreme: 1000.0,
            lower_extreme: -1000.0,
            upper_warning: 800.0,
            lower_warning: -800.0,
            lookback_period: 10,
            consecutive_threshold: 2,
        }
    }
}

/// Tick Extreme Indicator
///
/// Detects and analyzes extreme tick readings that often signal short-term
/// market turning points or capitulation. Extreme readings indicate panic
/// or euphoria and often precede reversals.
///
/// # Interpretation
/// - TICK > +1000: Extreme buying pressure (potential short-term top)
/// - TICK < -1000: Extreme selling pressure (potential short-term bottom)
/// - Consecutive extremes: Strong directional conviction
/// - Multiple extremes in session: Trending day
///
/// # Use Cases
/// - Identifying intraday reversal points
/// - Detecting capitulation in selloffs
/// - Confirming breakout strength
/// - Mean reversion entry signals
#[derive(Debug, Clone)]
pub struct TickExtreme {
    config: TickExtremeConfig,
}

impl Default for TickExtreme {
    fn default() -> Self {
        Self::new()
    }
}

impl TickExtreme {
    pub fn new() -> Self {
        Self {
            config: TickExtremeConfig::default(),
        }
    }

    pub fn with_config(config: TickExtremeConfig) -> Self {
        Self { config }
    }

    pub fn with_thresholds(mut self, upper: f64, lower: f64) -> Self {
        self.config.upper_extreme = upper;
        self.config.lower_extreme = lower;
        self
    }

    pub fn with_lookback(mut self, period: usize) -> Self {
        self.config.lookback_period = period;
        self
    }

    /// Classify a single tick reading
    pub fn classify(&self, tick: f64) -> TickExtremeLevel {
        if tick.is_nan() {
            TickExtremeLevel::Unknown
        } else if tick >= self.config.upper_extreme {
            TickExtremeLevel::ExtremeHigh
        } else if tick >= self.config.upper_warning {
            TickExtremeLevel::WarningHigh
        } else if tick <= self.config.lower_extreme {
            TickExtremeLevel::ExtremeLow
        } else if tick <= self.config.lower_warning {
            TickExtremeLevel::WarningLow
        } else {
            TickExtremeLevel::Normal
        }
    }

    /// Calculate extreme readings from tick series
    pub fn calculate(&self, ticks: &[f64]) -> Vec<TickExtremeLevel> {
        ticks.iter().map(|&t| self.classify(t)).collect()
    }

    /// Calculate extreme count over lookback period
    pub fn calculate_extreme_count(&self, ticks: &[f64]) -> TickExtremeCount {
        let mut high_count = Vec::with_capacity(ticks.len());
        let mut low_count = Vec::with_capacity(ticks.len());

        for i in 0..ticks.len() {
            let start = if i >= self.config.lookback_period {
                i - self.config.lookback_period + 1
            } else {
                0
            };

            let mut high = 0;
            let mut low = 0;

            for j in start..=i {
                if !ticks[j].is_nan() {
                    if ticks[j] >= self.config.upper_extreme {
                        high += 1;
                    } else if ticks[j] <= self.config.lower_extreme {
                        low += 1;
                    }
                }
            }

            high_count.push(high as f64);
            low_count.push(low as f64);
        }

        TickExtremeCount {
            high_extreme_count: high_count,
            low_extreme_count: low_count,
        }
    }

    /// Detect consecutive extreme readings
    pub fn detect_consecutive(&self, ticks: &[f64]) -> Vec<ConsecutiveExtreme> {
        let mut result = Vec::new();
        let mut consecutive_high = 0;
        let mut consecutive_low = 0;

        for (i, &tick) in ticks.iter().enumerate() {
            if tick.is_nan() {
                consecutive_high = 0;
                consecutive_low = 0;
                continue;
            }

            if tick >= self.config.upper_extreme {
                consecutive_high += 1;
                consecutive_low = 0;
                if consecutive_high >= self.config.consecutive_threshold {
                    result.push(ConsecutiveExtreme {
                        index: i,
                        direction: ExtremeDirection::High,
                        count: consecutive_high,
                    });
                }
            } else if tick <= self.config.lower_extreme {
                consecutive_low += 1;
                consecutive_high = 0;
                if consecutive_low >= self.config.consecutive_threshold {
                    result.push(ConsecutiveExtreme {
                        index: i,
                        direction: ExtremeDirection::Low,
                        count: consecutive_low,
                    });
                }
            } else {
                consecutive_high = 0;
                consecutive_low = 0;
            }
        }

        result
    }

    /// Calculate normalized extreme indicator (scaled between -1 and 1)
    pub fn calculate_normalized(&self, ticks: &[f64]) -> Vec<f64> {
        ticks
            .iter()
            .map(|&tick| {
                if tick.is_nan() {
                    f64::NAN
                } else if tick >= 0.0 {
                    (tick / self.config.upper_extreme).min(1.0)
                } else {
                    (tick / self.config.lower_extreme.abs()).max(-1.0)
                }
            })
            .collect()
    }

    /// Analyze session for extreme patterns
    pub fn session_analysis(&self, ticks: &[f64]) -> TickExtremeAnalysis {
        if ticks.is_empty() {
            return TickExtremeAnalysis::default();
        }

        let classifications = self.calculate(ticks);
        let mut total_extreme_highs = 0;
        let mut total_extreme_lows = 0;
        let mut max_consecutive_high = 0;
        let mut max_consecutive_low = 0;
        let mut current_consecutive_high = 0;
        let mut current_consecutive_low = 0;
        let mut first_extreme_index: Option<usize> = None;
        let mut last_extreme_index: Option<usize> = None;

        for (i, level) in classifications.iter().enumerate() {
            match level {
                TickExtremeLevel::ExtremeHigh => {
                    total_extreme_highs += 1;
                    current_consecutive_high += 1;
                    current_consecutive_low = 0;
                    max_consecutive_high = max_consecutive_high.max(current_consecutive_high);
                    if first_extreme_index.is_none() {
                        first_extreme_index = Some(i);
                    }
                    last_extreme_index = Some(i);
                }
                TickExtremeLevel::ExtremeLow => {
                    total_extreme_lows += 1;
                    current_consecutive_low += 1;
                    current_consecutive_high = 0;
                    max_consecutive_low = max_consecutive_low.max(current_consecutive_low);
                    if first_extreme_index.is_none() {
                        first_extreme_index = Some(i);
                    }
                    last_extreme_index = Some(i);
                }
                _ => {
                    current_consecutive_high = 0;
                    current_consecutive_low = 0;
                }
            }
        }

        let session_bias = if total_extreme_highs > total_extreme_lows * 2 {
            SessionBias::StronglyBullish
        } else if total_extreme_highs > total_extreme_lows {
            SessionBias::Bullish
        } else if total_extreme_lows > total_extreme_highs * 2 {
            SessionBias::StronglyBearish
        } else if total_extreme_lows > total_extreme_highs {
            SessionBias::Bearish
        } else {
            SessionBias::Neutral
        };

        TickExtremeAnalysis {
            total_extreme_highs,
            total_extreme_lows,
            max_consecutive_high,
            max_consecutive_low,
            first_extreme_index,
            last_extreme_index,
            session_bias,
        }
    }
}

/// Tick extreme level classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TickExtremeLevel {
    /// Above upper extreme threshold
    ExtremeHigh,
    /// Above upper warning threshold
    WarningHigh,
    /// Within normal range
    Normal,
    /// Below lower warning threshold
    WarningLow,
    /// Below lower extreme threshold
    ExtremeLow,
    /// Invalid data
    Unknown,
}

/// Extreme count result
#[derive(Debug, Clone)]
pub struct TickExtremeCount {
    /// Count of high extremes in lookback
    pub high_extreme_count: Vec<f64>,
    /// Count of low extremes in lookback
    pub low_extreme_count: Vec<f64>,
}

/// Consecutive extreme event
#[derive(Debug, Clone)]
pub struct ConsecutiveExtreme {
    /// Index of the reading
    pub index: usize,
    /// Direction of extreme
    pub direction: ExtremeDirection,
    /// Consecutive count
    pub count: usize,
}

/// Direction of extreme reading
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtremeDirection {
    High,
    Low,
}

/// Session analysis results
#[derive(Debug, Clone, Default)]
pub struct TickExtremeAnalysis {
    /// Total extreme high readings
    pub total_extreme_highs: usize,
    /// Total extreme low readings
    pub total_extreme_lows: usize,
    /// Maximum consecutive high extremes
    pub max_consecutive_high: usize,
    /// Maximum consecutive low extremes
    pub max_consecutive_low: usize,
    /// Index of first extreme reading
    pub first_extreme_index: Option<usize>,
    /// Index of last extreme reading
    pub last_extreme_index: Option<usize>,
    /// Overall session bias
    pub session_bias: SessionBias,
}

/// Session bias from extreme readings
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SessionBias {
    StronglyBullish,
    Bullish,
    #[default]
    Neutral,
    Bearish,
    StronglyBearish,
}

impl BreadthIndicator for TickExtreme {
    fn name(&self) -> &str {
        "Tick Extreme"
    }

    fn compute_breadth(&self, data: &super::BreadthSeries) -> Result<IndicatorOutput> {
        if data.is_empty() {
            return Err(IndicatorError::InsufficientData {
                required: 1,
                got: 0,
            });
        }

        // Calculate net ticks from advances/declines
        let net_ticks: Vec<f64> = data
            .advances
            .iter()
            .zip(data.declines.iter())
            .map(|(a, d)| a - d)
            .collect();

        let normalized = self.calculate_normalized(&net_ticks);
        Ok(IndicatorOutput::single(normalized))
    }

    fn min_periods(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tick_extreme_classify() {
        let te = TickExtreme::new();

        assert_eq!(te.classify(1200.0), TickExtremeLevel::ExtremeHigh);
        assert_eq!(te.classify(900.0), TickExtremeLevel::WarningHigh);
        assert_eq!(te.classify(500.0), TickExtremeLevel::Normal);
        assert_eq!(te.classify(-900.0), TickExtremeLevel::WarningLow);
        assert_eq!(te.classify(-1200.0), TickExtremeLevel::ExtremeLow);
        assert_eq!(te.classify(f64::NAN), TickExtremeLevel::Unknown);
    }

    #[test]
    fn test_tick_extreme_calculate() {
        let te = TickExtreme::new();
        let ticks = vec![1200.0, 500.0, -1200.0, 900.0, -900.0];

        let result = te.calculate(&ticks);

        assert_eq!(result.len(), 5);
        assert_eq!(result[0], TickExtremeLevel::ExtremeHigh);
        assert_eq!(result[1], TickExtremeLevel::Normal);
        assert_eq!(result[2], TickExtremeLevel::ExtremeLow);
        assert_eq!(result[3], TickExtremeLevel::WarningHigh);
        assert_eq!(result[4], TickExtremeLevel::WarningLow);
    }

    #[test]
    fn test_extreme_count() {
        let te = TickExtreme::new().with_lookback(3);
        let ticks = vec![1200.0, 1100.0, 500.0, -1200.0, -1100.0];

        let result = te.calculate_extreme_count(&ticks);

        assert_eq!(result.high_extreme_count.len(), 5);
        assert_eq!(result.low_extreme_count.len(), 5);

        // First reading: 1 high extreme
        assert!((result.high_extreme_count[0] - 1.0).abs() < 1e-10);
        // Second reading: 2 high extremes
        assert!((result.high_extreme_count[1] - 2.0).abs() < 1e-10);
        // Third reading (lookback 3): 2 high, 0 low
        assert!((result.high_extreme_count[2] - 2.0).abs() < 1e-10);
        // Fourth reading: 1 high, 1 low
        assert!((result.low_extreme_count[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_consecutive_detection() {
        let te = TickExtreme::new();
        let ticks = vec![1200.0, 1100.0, 1050.0, 500.0, -1200.0, -1100.0];

        let result = te.detect_consecutive(&ticks);

        assert!(!result.is_empty());
        // Should detect consecutive highs starting at index 1 (2 consecutive)
        assert!(result.iter().any(|e| e.direction == ExtremeDirection::High));
        assert!(result.iter().any(|e| e.direction == ExtremeDirection::Low));
    }

    #[test]
    fn test_normalized() {
        let te = TickExtreme::new();
        let ticks = vec![1000.0, -1000.0, 500.0, -500.0, 2000.0];

        let result = te.calculate_normalized(&ticks);

        assert_eq!(result.len(), 5);
        assert!((result[0] - 1.0).abs() < 1e-10); // 1000/1000 = 1.0
        assert!((result[1] - (-1.0)).abs() < 1e-10); // -1000/1000 = -1.0
        assert!((result[2] - 0.5).abs() < 1e-10); // 500/1000 = 0.5
        assert!((result[3] - (-0.5)).abs() < 1e-10); // -500/1000 = -0.5
        assert!((result[4] - 1.0).abs() < 1e-10); // 2000/1000 capped at 1.0
    }

    #[test]
    fn test_session_analysis() {
        let te = TickExtreme::new();
        let ticks = vec![1200.0, 1100.0, 500.0, 1050.0, -200.0];

        let analysis = te.session_analysis(&ticks);

        assert_eq!(analysis.total_extreme_highs, 3);
        assert_eq!(analysis.total_extreme_lows, 0);
        assert_eq!(analysis.max_consecutive_high, 2);
        assert_eq!(analysis.first_extreme_index, Some(0));
        assert_eq!(analysis.last_extreme_index, Some(3));
        assert_eq!(analysis.session_bias, SessionBias::StronglyBullish);
    }

    #[test]
    fn test_custom_thresholds() {
        let te = TickExtreme::new().with_thresholds(500.0, -500.0);

        assert_eq!(te.classify(600.0), TickExtremeLevel::ExtremeHigh);
        assert_eq!(te.classify(-600.0), TickExtremeLevel::ExtremeLow);
    }

    #[test]
    fn test_breadth_indicator_trait() {
        use crate::breadth::{BreadthData, BreadthSeries};

        let te = TickExtreme::new();
        let mut series = BreadthSeries::new();
        series.push(BreadthData::from_ad(2000.0, 1000.0));
        series.push(BreadthData::from_ad(1500.0, 1500.0));

        let result = te.compute_breadth(&series);
        assert!(result.is_ok());
    }
}
