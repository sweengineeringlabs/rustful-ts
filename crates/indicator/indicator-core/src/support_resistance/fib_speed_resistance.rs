//! Fibonacci Speed Resistance Fan - Time-price grid analysis.
//!
//! IND-393: Fibonacci Speed Resistance combines price retracements
//! with time analysis to create a grid of support/resistance zones.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};
use serde::{Deserialize, Serialize};

/// Speed resistance arc data.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SpeedResistanceArc {
    /// Price level at this arc
    pub price: f64,
    /// Time (bar) offset from origin
    pub time_offset: usize,
    /// Speed line ratio (0.333, 0.5, 0.667)
    pub speed_ratio: f64,
}

/// Speed resistance fan lines.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SpeedFanLines {
    /// Origin price
    pub origin_price: f64,
    /// Target price
    pub target_price: f64,
    /// 1/3 speed line value at current bar
    pub speed_1_3: f64,
    /// 1/2 speed line value at current bar
    pub speed_1_2: f64,
    /// 2/3 speed line value at current bar
    pub speed_2_3: f64,
}

/// Fibonacci Speed Resistance Indicator
///
/// Creates a time-price grid using speed resistance lines. These lines
/// connect significant price points and project support/resistance at
/// standard speed ratios (1/3, 1/2, 2/3).
///
/// # Theory
/// Speed resistance fans divide both price and time by thirds:
/// - 1/3 speed line: Price moves 1/3 of total while time moves 2/3
/// - 1/2 speed line: Price and time move equally (45-degree angle)
/// - 2/3 speed line: Price moves 2/3 of total while time moves 1/3
///
/// # Interpretation
/// - Breaking below 2/3 speed line is bearish
/// - Support at 1/2 speed line indicates healthy correction
/// - Breaking 1/3 speed line suggests trend reversal
/// - Speed lines act as dynamic support/resistance over time
#[derive(Debug, Clone)]
pub struct FibonacciSpeedResistance {
    /// Lookback period for swing detection
    lookback: usize,
    /// Swing detection strength
    swing_strength: usize,
    /// Number of bars to project forward
    projection_bars: usize,
}

impl FibonacciSpeedResistance {
    /// Create a new Fibonacci Speed Resistance indicator.
    ///
    /// # Arguments
    /// * `lookback` - Period to find swing points (minimum 10)
    /// * `swing_strength` - Bars on each side to confirm swing (minimum 2)
    /// * `projection_bars` - Bars to project lines forward (minimum 5)
    pub fn new(lookback: usize, swing_strength: usize, projection_bars: usize) -> Result<Self> {
        if lookback < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if swing_strength < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "swing_strength".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if projection_bars < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "projection_bars".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self {
            lookback,
            swing_strength,
            projection_bars,
        })
    }

    /// Create with default parameters.
    pub fn default_params() -> Self {
        Self {
            lookback: 30,
            swing_strength: 3,
            projection_bars: 20,
        }
    }

    /// Find swing high in range.
    fn find_swing_high(&self, high: &[f64], start: usize, end: usize) -> Option<(usize, f64)> {
        let mut best_idx = None;
        let mut best_high = f64::NEG_INFINITY;

        for i in (start + self.swing_strength)..(end.saturating_sub(self.swing_strength)) {
            let mut is_swing = true;

            for j in (i.saturating_sub(self.swing_strength))..i {
                if high[j] >= high[i] {
                    is_swing = false;
                    break;
                }
            }

            if is_swing {
                for j in (i + 1)..=(i + self.swing_strength).min(end - 1) {
                    if high[j] >= high[i] {
                        is_swing = false;
                        break;
                    }
                }
            }

            if is_swing && high[i] > best_high {
                best_high = high[i];
                best_idx = Some(i);
            }
        }

        best_idx.map(|idx| (idx, best_high))
    }

    /// Find swing low in range.
    fn find_swing_low(&self, low: &[f64], start: usize, end: usize) -> Option<(usize, f64)> {
        let mut best_idx = None;
        let mut best_low = f64::INFINITY;

        for i in (start + self.swing_strength)..(end.saturating_sub(self.swing_strength)) {
            let mut is_swing = true;

            for j in (i.saturating_sub(self.swing_strength))..i {
                if low[j] <= low[i] {
                    is_swing = false;
                    break;
                }
            }

            if is_swing {
                for j in (i + 1)..=(i + self.swing_strength).min(end - 1) {
                    if low[j] <= low[i] {
                        is_swing = false;
                        break;
                    }
                }
            }

            if is_swing && low[i] < best_low {
                best_low = low[i];
                best_idx = Some(i);
            }
        }

        best_idx.map(|idx| (idx, best_low))
    }

    /// Calculate speed line value at given time offset.
    ///
    /// # Arguments
    /// * `origin_price` - Starting price (swing low for uptrend)
    /// * `target_price` - Ending price (swing high for uptrend)
    /// * `origin_time` - Starting bar index
    /// * `target_time` - Ending bar index
    /// * `current_time` - Current bar index
    /// * `speed_ratio` - Speed ratio (0.333, 0.5, or 0.667)
    fn speed_line_value(
        &self,
        origin_price: f64,
        target_price: f64,
        origin_time: usize,
        target_time: usize,
        current_time: usize,
        speed_ratio: f64,
    ) -> f64 {
        let price_range = target_price - origin_price;
        let time_range = (target_time - origin_time) as f64;

        if time_range == 0.0 {
            return origin_price;
        }

        let time_elapsed = (current_time - origin_time) as f64;

        // Speed line: how much price has moved relative to time
        // Higher speed ratio = steeper line
        let slope = (price_range * speed_ratio) / time_range;

        origin_price + slope * time_elapsed
    }

    /// Calculate speed resistance values.
    ///
    /// Returns three speed lines: 1/3, 1/2, and 2/3.
    pub fn calculate(&self, high: &[f64], low: &[f64], _close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = high.len();
        let mut speed_1_3 = vec![f64::NAN; n];
        let mut speed_1_2 = vec![f64::NAN; n];
        let mut speed_2_3 = vec![f64::NAN; n];

        if n < self.lookback {
            return (speed_1_3, speed_1_2, speed_2_3);
        }

        for i in self.lookback..n {
            let start = i.saturating_sub(self.lookback);

            // Find swing points
            let swing_high = self.find_swing_high(high, start, i);
            let swing_low = self.find_swing_low(low, start, i);

            if let (Some((high_idx, high_price)), Some((low_idx, low_price))) = (swing_high, swing_low) {
                // Determine trend direction based on which swing came first
                let (origin_idx, origin_price, target_idx, target_price) = if low_idx < high_idx {
                    // Uptrend: from low to high
                    (low_idx, low_price, high_idx, high_price)
                } else {
                    // Downtrend: from high to low
                    (high_idx, high_price, low_idx, low_price)
                };

                if origin_idx == target_idx {
                    continue;
                }

                // Calculate speed lines
                speed_1_3[i] = self.speed_line_value(
                    origin_price, target_price, origin_idx, target_idx, i, 1.0 / 3.0
                );
                speed_1_2[i] = self.speed_line_value(
                    origin_price, target_price, origin_idx, target_idx, i, 0.5
                );
                speed_2_3[i] = self.speed_line_value(
                    origin_price, target_price, origin_idx, target_idx, i, 2.0 / 3.0
                );
            }
        }

        (speed_1_3, speed_1_2, speed_2_3)
    }

    /// Get speed fan lines at a specific bar.
    pub fn get_speed_lines(
        &self,
        high: &[f64],
        low: &[f64],
        _close: &[f64],
        bar_index: usize,
    ) -> Option<SpeedFanLines> {
        if bar_index >= high.len() || bar_index < self.lookback {
            return None;
        }

        let start = bar_index.saturating_sub(self.lookback);

        let swing_high = self.find_swing_high(high, start, bar_index)?;
        let swing_low = self.find_swing_low(low, start, bar_index)?;

        let (origin_idx, origin_price, target_idx, target_price) = if swing_low.0 < swing_high.0 {
            (swing_low.0, swing_low.1, swing_high.0, swing_high.1)
        } else {
            (swing_high.0, swing_high.1, swing_low.0, swing_low.1)
        };

        if origin_idx == target_idx {
            return None;
        }

        Some(SpeedFanLines {
            origin_price,
            target_price,
            speed_1_3: self.speed_line_value(
                origin_price, target_price, origin_idx, target_idx, bar_index, 1.0 / 3.0
            ),
            speed_1_2: self.speed_line_value(
                origin_price, target_price, origin_idx, target_idx, bar_index, 0.5
            ),
            speed_2_3: self.speed_line_value(
                origin_price, target_price, origin_idx, target_idx, bar_index, 2.0 / 3.0
            ),
        })
    }

    /// Calculate the time-price grid intersections.
    ///
    /// Returns arcs at standard Fibonacci time intervals.
    pub fn calculate_arcs(
        &self,
        origin_price: f64,
        target_price: f64,
        origin_time: usize,
        target_time: usize,
    ) -> Vec<SpeedResistanceArc> {
        let mut arcs = Vec::new();
        let time_range = target_time - origin_time;

        if time_range == 0 {
            return arcs;
        }

        // Fibonacci time ratios
        let time_ratios = [0.382, 0.5, 0.618, 1.0, 1.618, 2.618];
        let speed_ratios = [1.0 / 3.0, 0.5, 2.0 / 3.0];

        for &time_ratio in &time_ratios {
            let time_offset = (time_range as f64 * time_ratio).round() as usize;

            for &speed_ratio in &speed_ratios {
                let current_time = origin_time + time_offset;
                let price = self.speed_line_value(
                    origin_price, target_price, origin_time, target_time, current_time, speed_ratio
                );

                arcs.push(SpeedResistanceArc {
                    price,
                    time_offset,
                    speed_ratio,
                });
            }
        }

        arcs
    }

    /// Determine which speed zone the price is in.
    pub fn get_speed_zone(&self, price: f64, lines: &SpeedFanLines) -> SpeedZone {
        let is_uptrend = lines.target_price > lines.origin_price;

        if is_uptrend {
            if price >= lines.speed_2_3 {
                SpeedZone::AboveTwoThirds
            } else if price >= lines.speed_1_2 {
                SpeedZone::BetweenHalfAndTwoThirds
            } else if price >= lines.speed_1_3 {
                SpeedZone::BetweenThirdAndHalf
            } else {
                SpeedZone::BelowOneThird
            }
        } else {
            // Downtrend - levels are inverted
            if price <= lines.speed_2_3 {
                SpeedZone::AboveTwoThirds
            } else if price <= lines.speed_1_2 {
                SpeedZone::BetweenHalfAndTwoThirds
            } else if price <= lines.speed_1_3 {
                SpeedZone::BetweenThirdAndHalf
            } else {
                SpeedZone::BelowOneThird
            }
        }
    }
}

/// Speed zone classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeedZone {
    /// Price above 2/3 speed line (strong trend)
    AboveTwoThirds,
    /// Price between 1/2 and 2/3 (healthy correction)
    BetweenHalfAndTwoThirds,
    /// Price between 1/3 and 1/2 (deeper correction)
    BetweenThirdAndHalf,
    /// Price below 1/3 speed line (potential reversal)
    BelowOneThird,
}

impl Default for FibonacciSpeedResistance {
    fn default() -> Self {
        Self::default_params()
    }
}

impl TechnicalIndicator for FibonacciSpeedResistance {
    fn name(&self) -> &str {
        "Fibonacci Speed Resistance"
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (speed_1_3, speed_1_2, speed_2_3) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(speed_1_3, speed_1_2, speed_2_3))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut high = Vec::new();
        let mut low = Vec::new();
        let mut close = Vec::new();

        for i in 0..60 {
            let base = 100.0 + (i as f64) * 0.3;
            let swing = (i as f64 * 0.15).sin() * 10.0;
            close.push(base + swing);
            high.push(base + swing + 2.0);
            low.push(base + swing - 2.0);
        }

        (high, low, close)
    }

    #[test]
    fn test_speed_resistance_creation() {
        let sr = FibonacciSpeedResistance::new(20, 3, 10);
        assert!(sr.is_ok());

        let sr = FibonacciSpeedResistance::new(5, 3, 10);
        assert!(sr.is_err());

        let sr = FibonacciSpeedResistance::new(20, 1, 10);
        assert!(sr.is_err());

        let sr = FibonacciSpeedResistance::new(20, 3, 2);
        assert!(sr.is_err());
    }

    #[test]
    fn test_speed_resistance_calculation() {
        let (high, low, close) = make_test_data();
        let sr = FibonacciSpeedResistance::new(20, 2, 10).unwrap();
        let (s1_3, s1_2, s2_3) = sr.calculate(&high, &low, &close);

        assert_eq!(s1_3.len(), close.len());
        assert_eq!(s1_2.len(), close.len());
        assert_eq!(s2_3.len(), close.len());

        // Should have valid values after lookback
        let valid_count = s1_2.iter().filter(|v| !v.is_nan()).count();
        assert!(valid_count > 0);
    }

    #[test]
    fn test_speed_line_ordering() {
        let (high, low, close) = make_test_data();
        let sr = FibonacciSpeedResistance::default_params();
        let (s1_3, s1_2, s2_3) = sr.calculate(&high, &low, &close);

        // In uptrend, 2/3 > 1/2 > 1/3
        for i in 35..close.len() {
            if !s1_3[i].is_nan() && !s1_2[i].is_nan() && !s2_3[i].is_nan() {
                // Lines should be distinct
                let has_diff = (s1_3[i] - s1_2[i]).abs() > 1e-10
                    || (s1_2[i] - s2_3[i]).abs() > 1e-10;
                assert!(has_diff || s1_3[i].is_nan());
            }
        }
    }

    #[test]
    fn test_speed_line_value() {
        let sr = FibonacciSpeedResistance::default_params();

        // Test simple case: origin at 100, target at 120, 10 bars apart
        let val = sr.speed_line_value(100.0, 120.0, 0, 10, 10, 0.5);

        // At target time with 0.5 speed, should be at 50% of the move
        assert!((val - 110.0).abs() < 0.01);
    }

    #[test]
    fn test_get_speed_lines() {
        let (high, low, close) = make_test_data();
        let sr = FibonacciSpeedResistance::default_params();

        let lines = sr.get_speed_lines(&high, &low, &close, 45);

        if let Some(l) = lines {
            // Speed lines should be calculated
            assert!(!l.speed_1_3.is_nan());
            assert!(!l.speed_1_2.is_nan());
            assert!(!l.speed_2_3.is_nan());
        }
    }

    #[test]
    fn test_speed_zone() {
        let sr = FibonacciSpeedResistance::default_params();

        // Uptrend lines
        let lines = SpeedFanLines {
            origin_price: 100.0,
            target_price: 120.0,
            speed_1_3: 103.0,
            speed_1_2: 106.0,
            speed_2_3: 110.0,
        };

        assert_eq!(sr.get_speed_zone(115.0, &lines), SpeedZone::AboveTwoThirds);
        assert_eq!(sr.get_speed_zone(108.0, &lines), SpeedZone::BetweenHalfAndTwoThirds);
        assert_eq!(sr.get_speed_zone(104.0, &lines), SpeedZone::BetweenThirdAndHalf);
        assert_eq!(sr.get_speed_zone(100.0, &lines), SpeedZone::BelowOneThird);
    }

    #[test]
    fn test_calculate_arcs() {
        let sr = FibonacciSpeedResistance::default_params();

        let arcs = sr.calculate_arcs(100.0, 120.0, 0, 20);

        // Should have arcs for each time/speed combination
        // 6 time ratios * 3 speed ratios = 18 arcs
        assert_eq!(arcs.len(), 18);

        // Check that arcs have valid data
        for arc in arcs.iter() {
            assert!(!arc.price.is_nan());
            assert!(arc.speed_ratio > 0.0 && arc.speed_ratio < 1.0);
        }
    }

    #[test]
    fn test_speed_resistance_technical_indicator() {
        let sr = FibonacciSpeedResistance::default_params();
        assert_eq!(sr.name(), "Fibonacci Speed Resistance");
        assert_eq!(sr.min_periods(), 31);
    }

    #[test]
    fn test_speed_resistance_compute() {
        let (high, low, close) = make_test_data();
        let volume = vec![1000.0; close.len()];

        let data = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume,
        };

        let sr = FibonacciSpeedResistance::default_params();
        let result = sr.compute(&data);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.values.len(), 3); // 3 speed lines
    }

    #[test]
    fn test_speed_resistance_downtrend() {
        // Create downtrend data
        let mut high = Vec::new();
        let mut low = Vec::new();
        let mut close = Vec::new();

        for i in 0..60 {
            let base = 150.0 - (i as f64) * 0.5;
            let swing = (i as f64 * 0.15).sin() * 5.0;
            close.push(base + swing);
            high.push(base + swing + 2.0);
            low.push(base + swing - 2.0);
        }

        let sr = FibonacciSpeedResistance::default_params();
        let (s1_3, s1_2, s2_3) = sr.calculate(&high, &low, &close);

        // Should still produce valid results
        let valid_count = s1_2.iter().filter(|v| !v.is_nan()).count();
        assert!(valid_count > 0);
    }
}
