//! Gann Square of 9 Indicator (IND-324)
//!
//! The Square of 9 is a mathematical tool developed by W.D. Gann that
//! identifies key price and time relationships through spiral calculations.
//! It helps identify support/resistance levels based on squaring of price and time.

use crate::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result, SignalIndicator,
    TechnicalIndicator,
};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Gann Square of 9 output structure
#[derive(Debug, Clone)]
pub struct GannSquareOf9Output {
    /// Current price square root
    pub sqrt_price: Vec<f64>,
    /// Cardinal cross levels (0, 90, 180, 270 degrees)
    pub cardinal_up: Vec<f64>,
    pub cardinal_down: Vec<f64>,
    /// Fixed cross levels (45, 135, 225, 315 degrees)
    pub fixed_up: Vec<f64>,
    pub fixed_down: Vec<f64>,
    /// Next resistance level
    pub resistance: Vec<f64>,
    /// Next support level
    pub support: Vec<f64>,
    /// Price position within current square (0-100%)
    pub position_in_square: Vec<f64>,
    /// Time squared value
    pub time_squared: Vec<f64>,
    /// Price/time convergence indicator
    pub convergence: Vec<f64>,
}

/// Gann Square of 9 configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GannSquareOf9Config {
    /// Angle increment for level calculation (degrees)
    pub angle_increment: f64,
    /// Number of levels to calculate above/below
    pub num_levels: usize,
    /// Starting point for time squaring
    pub time_base: usize,
    /// Price divisor for normalization
    pub price_divisor: f64,
}

impl Default for GannSquareOf9Config {
    fn default() -> Self {
        Self {
            angle_increment: 45.0,
            num_levels: 4,
            time_base: 0,
            price_divisor: 1.0,
        }
    }
}

/// Gann Square of 9 Indicator
///
/// Uses the mathematical relationship of square roots to identify
/// price and time harmonics. The square of 9 spirals outward from
/// a center point, with key levels at cardinal (0, 90, 180, 270)
/// and fixed (45, 135, 225, 315) degree angles.
///
/// # Calculation
/// 1. Take square root of price
/// 2. Add/subtract angle increments (in radians converted to price)
/// 3. Square the result to get target levels
///
/// # Trading Rules
/// - Cardinal cross levels provide strong support/resistance
/// - Fixed cross levels provide secondary support/resistance
/// - Time squaring identifies potential turning points
#[derive(Debug, Clone)]
pub struct GannSquareOf9 {
    config: GannSquareOf9Config,
}

impl GannSquareOf9 {
    pub fn new() -> Self {
        Self {
            config: GannSquareOf9Config::default(),
        }
    }

    pub fn with_config(config: GannSquareOf9Config) -> Self {
        Self { config }
    }

    pub fn with_angle_increment(mut self, increment: f64) -> Self {
        self.config.angle_increment = increment;
        self
    }

    pub fn with_num_levels(mut self, levels: usize) -> Self {
        self.config.num_levels = levels;
        self
    }

    /// Calculate price level from square root and angle
    fn calc_level(&self, sqrt_price: f64, angle_degrees: f64, direction: i32) -> f64 {
        // Convert angle to square root increment
        // One full rotation (360 degrees) = 2 units on the square root scale
        let sqrt_increment = (angle_degrees / 180.0) * direction as f64;
        let new_sqrt = sqrt_price + sqrt_increment;
        new_sqrt * new_sqrt
    }

    /// Calculate time squared value
    fn time_squared(&self, bar_index: usize) -> f64 {
        let time_from_base = (bar_index - self.config.time_base) as f64;
        time_from_base.sqrt().powi(2)
    }

    /// Calculate Gann Square of 9 from OHLCV data
    pub fn calculate(&self, data: &OHLCVSeries) -> GannSquareOf9Output {
        let n = data.close.len();

        let mut sqrt_price = vec![f64::NAN; n];
        let mut cardinal_up = vec![f64::NAN; n];
        let mut cardinal_down = vec![f64::NAN; n];
        let mut fixed_up = vec![f64::NAN; n];
        let mut fixed_down = vec![f64::NAN; n];
        let mut resistance = vec![f64::NAN; n];
        let mut support = vec![f64::NAN; n];
        let mut position_in_square = vec![f64::NAN; n];
        let mut time_squared = vec![f64::NAN; n];
        let mut convergence = vec![f64::NAN; n];

        for i in 0..n {
            let price = data.close[i] / self.config.price_divisor;
            if price <= 0.0 {
                continue;
            }

            let sqrt_p = price.sqrt();
            sqrt_price[i] = sqrt_p;

            // Calculate cardinal cross levels (90 degrees = 0.5 sqrt units)
            cardinal_up[i] = self.calc_level(sqrt_p, 90.0, 1);
            cardinal_down[i] = self.calc_level(sqrt_p, 90.0, -1);

            // Calculate fixed cross levels (45 degrees = 0.25 sqrt units)
            fixed_up[i] = self.calc_level(sqrt_p, 45.0, 1);
            fixed_down[i] = self.calc_level(sqrt_p, 45.0, -1);

            // Find nearest support and resistance
            // Resistance is next level up at the configured angle increment
            resistance[i] = self.calc_level(sqrt_p, self.config.angle_increment, 1);
            support[i] = self.calc_level(sqrt_p, self.config.angle_increment, -1);

            // Calculate position within current square
            // Based on where price is between support and resistance
            let range = resistance[i] - support[i];
            if range > 0.0 {
                position_in_square[i] = (price * self.config.price_divisor - support[i]) / range * 100.0;
            }

            // Time squared calculation
            time_squared[i] = self.time_squared(i);

            // Price/time convergence - how close sqrt(price) is to sqrt(time)
            let sqrt_time = ((i - self.config.time_base) as f64 + 1.0).sqrt();
            let diff = (sqrt_p - sqrt_time).abs();
            convergence[i] = 1.0 / (1.0 + diff); // Normalized 0-1, higher = more convergent
        }

        GannSquareOf9Output {
            sqrt_price,
            cardinal_up,
            cardinal_down,
            fixed_up,
            fixed_down,
            resistance,
            support,
            position_in_square,
            time_squared,
            convergence,
        }
    }

    /// Get all Gann levels for a specific price
    pub fn get_levels(&self, price: f64) -> Vec<f64> {
        let mut levels = Vec::new();
        let sqrt_p = (price / self.config.price_divisor).sqrt();

        // Calculate levels at multiple angle increments
        for i in 1..=self.config.num_levels {
            let angle = self.config.angle_increment * i as f64;
            levels.push(self.calc_level(sqrt_p, angle, 1));
            levels.push(self.calc_level(sqrt_p, angle, -1));
        }

        levels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        levels
    }
}

impl Default for GannSquareOf9 {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for GannSquareOf9 {
    fn name(&self) -> &str {
        "Gann Square of 9"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < 2 {
            return Err(IndicatorError::InsufficientData {
                required: 2,
                got: data.close.len(),
            });
        }

        let result = self.calculate(data);

        // Primary: position in square, Secondary: convergence, Tertiary: sqrt_price
        Ok(IndicatorOutput::triple(
            result.position_in_square,
            result.convergence,
            result.sqrt_price,
        ))
    }

    fn min_periods(&self) -> usize {
        2
    }

    fn output_features(&self) -> usize {
        3
    }
}

impl SignalIndicator for GannSquareOf9 {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let result = self.calculate(data);
        let n = result.position_in_square.len();

        if n < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let position = result.position_in_square[n - 1];
        let prev_position = result.position_in_square[n - 2];
        let convergence = result.convergence[n - 1];

        if position.is_nan() || prev_position.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Signal based on position within square and convergence
        // Near support (0-20%) with high convergence = bullish
        // Near resistance (80-100%) with high convergence = bearish
        if position < 20.0 && convergence > 0.5 && position > prev_position {
            Ok(IndicatorSignal::Bullish)
        } else if position > 80.0 && convergence > 0.5 && position < prev_position {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let result = self.calculate(data);
        let n = result.position_in_square.len();

        let mut signals = vec![IndicatorSignal::Neutral; n];

        for i in 1..n {
            let position = result.position_in_square[i];
            let prev_position = result.position_in_square[i - 1];
            let convergence = result.convergence[i];

            if position.is_nan() || prev_position.is_nan() || convergence.is_nan() {
                continue;
            }

            if position < 20.0 && convergence > 0.5 && position > prev_position {
                signals[i] = IndicatorSignal::Bullish;
            } else if position > 80.0 && convergence > 0.5 && position < prev_position {
                signals[i] = IndicatorSignal::Bearish;
            }
        }

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data(bars: usize) -> OHLCVSeries {
        let mut closes = Vec::with_capacity(bars);
        for i in 0..bars {
            closes.push(100.0 + (i as f64 * 0.5).sin() * 10.0);
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
    fn test_square_of_9_initialization() {
        let sq9 = GannSquareOf9::new();
        assert_eq!(sq9.name(), "Gann Square of 9");
        assert_eq!(sq9.min_periods(), 2);
        assert_eq!(sq9.output_features(), 3);
    }

    #[test]
    fn test_square_of_9_calculation() {
        let data = create_test_data(30);
        let sq9 = GannSquareOf9::new();
        let result = sq9.calculate(&data);

        assert_eq!(result.sqrt_price.len(), 30);
        assert_eq!(result.resistance.len(), 30);
        assert_eq!(result.support.len(), 30);

        // Verify sqrt_price calculation
        for i in 0..30 {
            if !result.sqrt_price[i].is_nan() {
                let expected = data.close[i].sqrt();
                assert!((result.sqrt_price[i] - expected).abs() < 0.001);
            }
        }
    }

    #[test]
    fn test_level_calculation() {
        let sq9 = GannSquareOf9::new();
        let sqrt_100 = 10.0;

        // 90 degrees up should give (10 + 0.5)^2 = 110.25
        let level_up = sq9.calc_level(sqrt_100, 90.0, 1);
        assert!((level_up - 110.25).abs() < 0.01);

        // 90 degrees down should give (10 - 0.5)^2 = 90.25
        let level_down = sq9.calc_level(sqrt_100, 90.0, -1);
        assert!((level_down - 90.25).abs() < 0.01);
    }

    #[test]
    fn test_get_levels() {
        let sq9 = GannSquareOf9::new();
        let levels = sq9.get_levels(100.0);

        // Should have num_levels * 2 levels (up and down)
        assert_eq!(levels.len(), sq9.config.num_levels * 2);

        // Levels should be sorted
        for i in 1..levels.len() {
            assert!(levels[i] >= levels[i - 1]);
        }
    }

    #[test]
    fn test_square_of_9_compute() {
        let data = create_test_data(30);
        let sq9 = GannSquareOf9::new();
        let output = sq9.compute(&data).unwrap();

        assert_eq!(output.primary.len(), 30);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    #[test]
    fn test_square_of_9_signals() {
        let data = create_test_data(30);
        let sq9 = GannSquareOf9::new();
        let signals = sq9.signals(&data).unwrap();

        assert_eq!(signals.len(), 30);
    }

    #[test]
    fn test_insufficient_data() {
        let data = OHLCVSeries {
            open: vec![100.0],
            high: vec![102.0],
            low: vec![98.0],
            close: vec![100.0],
            volume: vec![1000.0],
        };

        let sq9 = GannSquareOf9::new();
        let result = sq9.compute(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_custom_config() {
        let config = GannSquareOf9Config {
            angle_increment: 30.0,
            num_levels: 6,
            time_base: 0,
            price_divisor: 1.0,
        };

        let sq9 = GannSquareOf9::with_config(config);
        let levels = sq9.get_levels(100.0);
        assert_eq!(levels.len(), 12); // 6 * 2
    }
}
