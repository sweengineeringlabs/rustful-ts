//! Gann Fan Indicator (IND-323)
//!
//! Gann Fan creates trend lines at specific angles from a significant price point.
//! These angles represent the relationship between time and price movement.
//! The key angles include 1x1 (45 degrees), 2x1, 1x2, etc.

use crate::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result, SignalIndicator,
    TechnicalIndicator,
};
use serde::{Deserialize, Serialize};

/// Gann angle definition
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum GannAngle {
    /// 1x8 angle (82.5 degrees) - very steep uptrend
    OneByEight,
    /// 1x4 angle (75 degrees) - steep uptrend
    OneByFour,
    /// 1x3 angle (71.25 degrees)
    OneByThree,
    /// 1x2 angle (63.75 degrees)
    OneByTwo,
    /// 1x1 angle (45 degrees) - balanced trend
    OneByOne,
    /// 2x1 angle (26.25 degrees)
    TwoByOne,
    /// 3x1 angle (18.75 degrees)
    ThreeByOne,
    /// 4x1 angle (15 degrees) - shallow trend
    FourByOne,
    /// 8x1 angle (7.5 degrees) - very shallow trend
    EightByOne,
}

impl GannAngle {
    /// Get the slope ratio for this angle (time units per price unit)
    pub fn slope_ratio(&self) -> f64 {
        match self {
            GannAngle::OneByEight => 0.125,
            GannAngle::OneByFour => 0.25,
            GannAngle::OneByThree => 0.333,
            GannAngle::OneByTwo => 0.5,
            GannAngle::OneByOne => 1.0,
            GannAngle::TwoByOne => 2.0,
            GannAngle::ThreeByOne => 3.0,
            GannAngle::FourByOne => 4.0,
            GannAngle::EightByOne => 8.0,
        }
    }

    /// Get the angle in degrees
    pub fn degrees(&self) -> f64 {
        (1.0 / self.slope_ratio()).atan().to_degrees()
    }
}

/// Gann Fan output structure
#[derive(Debug, Clone)]
pub struct GannFanOutput {
    /// Base price (pivot point)
    pub base_price: f64,
    /// Base bar index
    pub base_bar: usize,
    /// 1x1 angle line values
    pub line_1x1: Vec<f64>,
    /// 2x1 angle line values (above 1x1)
    pub line_2x1: Vec<f64>,
    /// 1x2 angle line values (below 1x1)
    pub line_1x2: Vec<f64>,
    /// 4x1 angle line values
    pub line_4x1: Vec<f64>,
    /// 1x4 angle line values
    pub line_1x4: Vec<f64>,
    /// 8x1 angle line values
    pub line_8x1: Vec<f64>,
    /// 1x8 angle line values
    pub line_1x8: Vec<f64>,
    /// Current price position relative to 1x1 line
    pub price_vs_1x1: Vec<f64>,
    /// Trend strength based on angle position
    pub trend_strength: Vec<f64>,
}

/// Gann Fan configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GannFanConfig {
    /// Lookback period for pivot detection
    pub lookback: usize,
    /// Price scaling factor (points per time unit at 45 degrees)
    pub price_scale: f64,
    /// Use high/low for pivot detection vs close
    pub use_high_low: bool,
    /// Detect upward fans (from lows)
    pub detect_up: bool,
    /// Detect downward fans (from highs)
    pub detect_down: bool,
}

impl Default for GannFanConfig {
    fn default() -> Self {
        Self {
            lookback: 20,
            price_scale: 1.0,
            use_high_low: true,
            detect_up: true,
            detect_down: true,
        }
    }
}

/// Gann Fan Indicator
///
/// Creates fan lines at specific Gann angles from significant pivot points.
/// The 1x1 angle (45 degrees) represents balanced time/price movement.
/// Prices above the 1x1 line indicate strong trends, below indicates weakness.
///
/// # Gann Angles
/// - 1x8, 1x4, 1x3, 1x2: Steep angles (strong trends)
/// - 1x1: 45-degree balanced angle
/// - 2x1, 3x1, 4x1, 8x1: Shallow angles (weak trends)
///
/// # Trading Rules
/// - Price above 1x1: Bullish, use angle lines as support
/// - Price below 1x1: Bearish, use angle lines as resistance
/// - Breaking through angles indicates trend changes
#[derive(Debug, Clone)]
pub struct GannFan {
    config: GannFanConfig,
}

impl GannFan {
    pub fn new() -> Self {
        Self {
            config: GannFanConfig::default(),
        }
    }

    pub fn with_config(config: GannFanConfig) -> Self {
        Self { config }
    }

    pub fn with_lookback(mut self, lookback: usize) -> Self {
        self.config.lookback = lookback;
        self
    }

    pub fn with_price_scale(mut self, scale: f64) -> Self {
        self.config.price_scale = scale;
        self
    }

    /// Find the most significant low in the lookback period
    fn find_pivot_low(&self, data: &OHLCVSeries) -> (usize, f64) {
        let n = data.close.len();
        let lookback = self.config.lookback.min(n);

        let prices = if self.config.use_high_low {
            &data.low
        } else {
            &data.close
        };

        let mut min_idx = n.saturating_sub(lookback);
        let mut min_val = prices[min_idx];

        for i in (n.saturating_sub(lookback))..n {
            if prices[i] < min_val {
                min_val = prices[i];
                min_idx = i;
            }
        }

        (min_idx, min_val)
    }

    /// Find the most significant high in the lookback period
    fn find_pivot_high(&self, data: &OHLCVSeries) -> (usize, f64) {
        let n = data.close.len();
        let lookback = self.config.lookback.min(n);

        let prices = if self.config.use_high_low {
            &data.high
        } else {
            &data.close
        };

        let mut max_idx = n.saturating_sub(lookback);
        let mut max_val = prices[max_idx];

        for i in (n.saturating_sub(lookback))..n {
            if prices[i] > max_val {
                max_val = prices[i];
                max_idx = i;
            }
        }

        (max_idx, max_val)
    }

    /// Calculate fan line value at given bar from base point
    fn calc_line(&self, base_price: f64, base_bar: usize, current_bar: usize, angle: GannAngle, direction: i32) -> f64 {
        let bars_elapsed = current_bar as f64 - base_bar as f64;
        let price_change = bars_elapsed * self.config.price_scale / angle.slope_ratio();
        base_price + (direction as f64 * price_change)
    }

    /// Calculate Gann Fan from OHLCV data
    pub fn calculate(&self, data: &OHLCVSeries) -> GannFanOutput {
        let n = data.close.len();

        // Find pivot point (use low for upward fan)
        let (base_bar, base_price) = if self.config.detect_up {
            self.find_pivot_low(data)
        } else {
            self.find_pivot_high(data)
        };

        let direction = if self.config.detect_up { 1 } else { -1 };

        let mut line_1x1 = vec![f64::NAN; n];
        let mut line_2x1 = vec![f64::NAN; n];
        let mut line_1x2 = vec![f64::NAN; n];
        let mut line_4x1 = vec![f64::NAN; n];
        let mut line_1x4 = vec![f64::NAN; n];
        let mut line_8x1 = vec![f64::NAN; n];
        let mut line_1x8 = vec![f64::NAN; n];
        let mut price_vs_1x1 = vec![f64::NAN; n];
        let mut trend_strength = vec![f64::NAN; n];

        for i in base_bar..n {
            line_1x1[i] = self.calc_line(base_price, base_bar, i, GannAngle::OneByOne, direction);
            line_2x1[i] = self.calc_line(base_price, base_bar, i, GannAngle::TwoByOne, direction);
            line_1x2[i] = self.calc_line(base_price, base_bar, i, GannAngle::OneByTwo, direction);
            line_4x1[i] = self.calc_line(base_price, base_bar, i, GannAngle::FourByOne, direction);
            line_1x4[i] = self.calc_line(base_price, base_bar, i, GannAngle::OneByFour, direction);
            line_8x1[i] = self.calc_line(base_price, base_bar, i, GannAngle::EightByOne, direction);
            line_1x8[i] = self.calc_line(base_price, base_bar, i, GannAngle::OneByEight, direction);

            // Calculate price position relative to 1x1
            let current_1x1 = line_1x1[i];
            if current_1x1 > 0.0 {
                price_vs_1x1[i] = (data.close[i] - current_1x1) / current_1x1 * 100.0;
            }

            // Calculate trend strength based on which angle zone price is in
            let close = data.close[i];
            if direction > 0 {
                // Upward fan
                if close >= line_1x8[i] {
                    trend_strength[i] = 100.0;
                } else if close >= line_1x4[i] {
                    trend_strength[i] = 87.5;
                } else if close >= line_1x2[i] {
                    trend_strength[i] = 75.0;
                } else if close >= line_1x1[i] {
                    trend_strength[i] = 62.5;
                } else if close >= line_2x1[i] {
                    trend_strength[i] = 50.0;
                } else if close >= line_4x1[i] {
                    trend_strength[i] = 37.5;
                } else if close >= line_8x1[i] {
                    trend_strength[i] = 25.0;
                } else {
                    trend_strength[i] = 12.5;
                }
            } else {
                // Downward fan (inverted)
                if close <= line_1x8[i] {
                    trend_strength[i] = 100.0;
                } else if close <= line_1x4[i] {
                    trend_strength[i] = 87.5;
                } else if close <= line_1x2[i] {
                    trend_strength[i] = 75.0;
                } else if close <= line_1x1[i] {
                    trend_strength[i] = 62.5;
                } else if close <= line_2x1[i] {
                    trend_strength[i] = 50.0;
                } else if close <= line_4x1[i] {
                    trend_strength[i] = 37.5;
                } else if close <= line_8x1[i] {
                    trend_strength[i] = 25.0;
                } else {
                    trend_strength[i] = 12.5;
                }
            }
        }

        GannFanOutput {
            base_price,
            base_bar,
            line_1x1,
            line_2x1,
            line_1x2,
            line_4x1,
            line_1x4,
            line_8x1,
            line_1x8,
            price_vs_1x1,
            trend_strength,
        }
    }
}

impl Default for GannFan {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for GannFan {
    fn name(&self) -> &str {
        "Gann Fan"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.config.lookback {
            return Err(IndicatorError::InsufficientData {
                required: self.config.lookback,
                got: data.close.len(),
            });
        }

        let result = self.calculate(data);

        // Primary: 1x1 line, Secondary: price vs 1x1, Tertiary: trend strength
        Ok(IndicatorOutput::triple(
            result.line_1x1,
            result.price_vs_1x1,
            result.trend_strength,
        ))
    }

    fn min_periods(&self) -> usize {
        self.config.lookback
    }

    fn output_features(&self) -> usize {
        3
    }
}

impl SignalIndicator for GannFan {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let result = self.calculate(data);
        let n = result.trend_strength.len();

        if n == 0 {
            return Ok(IndicatorSignal::Neutral);
        }

        let strength = result.trend_strength[n - 1];
        let price_vs = result.price_vs_1x1[n - 1];

        if strength.is_nan() || price_vs.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Strong bullish if above 1x1 with high strength
        if price_vs > 0.0 && strength >= 62.5 {
            Ok(IndicatorSignal::Bullish)
        } else if price_vs < 0.0 && strength <= 37.5 {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let result = self.calculate(data);

        let signals: Vec<IndicatorSignal> = result
            .trend_strength
            .iter()
            .zip(result.price_vs_1x1.iter())
            .map(|(&strength, &price_vs)| {
                if strength.is_nan() || price_vs.is_nan() {
                    IndicatorSignal::Neutral
                } else if price_vs > 0.0 && strength >= 62.5 {
                    IndicatorSignal::Bullish
                } else if price_vs < 0.0 && strength <= 37.5 {
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

    fn create_uptrend_data(bars: usize) -> OHLCVSeries {
        let mut closes = Vec::with_capacity(bars);
        let base = 100.0;

        for i in 0..bars {
            closes.push(base + i as f64 * 1.5);
        }

        OHLCVSeries {
            open: closes.iter().map(|c| c - 0.5).collect(),
            high: closes.iter().map(|c| c + 2.0).collect(),
            low: closes.iter().map(|c| c - 2.0).collect(),
            close: closes,
            volume: vec![1000.0; bars],
        }
    }

    #[test]
    fn test_gann_fan_initialization() {
        let fan = GannFan::new();
        assert_eq!(fan.name(), "Gann Fan");
        assert_eq!(fan.min_periods(), 20);
        assert_eq!(fan.output_features(), 3);
    }

    #[test]
    fn test_gann_angle_degrees() {
        let angle = GannAngle::OneByOne;
        let degrees = angle.degrees();
        assert!((degrees - 45.0).abs() < 0.01);

        let angle_2x1 = GannAngle::TwoByOne;
        let degrees_2x1 = angle_2x1.degrees();
        assert!(degrees_2x1 < 45.0); // Shallower angle
    }

    #[test]
    fn test_gann_fan_calculation() {
        let data = create_uptrend_data(30);
        let fan = GannFan::new();
        let result = fan.calculate(&data);

        // Should have valid values after base bar
        for i in result.base_bar..30 {
            assert!(!result.line_1x1[i].is_nan());
            assert!(!result.trend_strength[i].is_nan());
        }
    }

    #[test]
    fn test_gann_fan_compute() {
        let data = create_uptrend_data(30);
        let fan = GannFan::new();
        let output = fan.compute(&data).unwrap();

        assert_eq!(output.primary.len(), 30);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    #[test]
    fn test_gann_fan_signals() {
        let data = create_uptrend_data(30);
        let fan = GannFan::new();
        let signals = fan.signals(&data).unwrap();

        assert_eq!(signals.len(), 30);
    }

    #[test]
    fn test_insufficient_data() {
        let data = OHLCVSeries {
            open: vec![100.0; 5],
            high: vec![102.0; 5],
            low: vec![98.0; 5],
            close: vec![100.0; 5],
            volume: vec![1000.0; 5],
        };

        let fan = GannFan::new();
        let result = fan.compute(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_gann_fan_custom_config() {
        let config = GannFanConfig {
            lookback: 10,
            price_scale: 2.0,
            use_high_low: false,
            detect_up: true,
            detect_down: false,
        };

        let fan = GannFan::with_config(config);
        assert_eq!(fan.min_periods(), 10);
    }
}
