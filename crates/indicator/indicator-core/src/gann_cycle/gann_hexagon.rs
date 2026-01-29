//! Gann Hexagon Indicator (IND-325)
//!
//! The Gann Hexagon chart creates price levels based on hexagonal
//! (6-sided) geometry. It divides the circle into 6 equal parts
//! (60-degree increments) to identify support and resistance levels.

use crate::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result, SignalIndicator,
    TechnicalIndicator,
};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Gann Hexagon output structure
#[derive(Debug, Clone)]
pub struct GannHexagonOutput {
    /// Current hexagon ring number
    pub ring_number: Vec<f64>,
    /// Position within current ring (0-6)
    pub ring_position: Vec<f64>,
    /// Hexagon resistance levels (6 levels at 60-degree intervals)
    pub hex_levels: Vec<Vec<f64>>,
    /// Nearest hexagon support
    pub hex_support: Vec<f64>,
    /// Nearest hexagon resistance
    pub hex_resistance: Vec<f64>,
    /// Distance to nearest hexagon vertex
    pub vertex_distance: Vec<f64>,
    /// Hexagon angle (0-360 degrees)
    pub hex_angle: Vec<f64>,
    /// Price position strength (how close to a key level)
    pub level_strength: Vec<f64>,
}

/// Gann Hexagon configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GannHexagonConfig {
    /// Base price for hexagon center
    pub base_price: Option<f64>,
    /// Price increment per hexagon cell
    pub price_increment: f64,
    /// Number of hexagon rings to calculate
    pub num_rings: usize,
    /// Use automatic base price detection
    pub auto_base: bool,
}

impl Default for GannHexagonConfig {
    fn default() -> Self {
        Self {
            base_price: None,
            price_increment: 1.0,
            num_rings: 10,
            auto_base: true,
        }
    }
}

/// Gann Hexagon Indicator
///
/// Creates hexagonal chart levels for support and resistance identification.
/// The hexagon divides price movements into 6 equal sectors (60 degrees each).
/// Each ring of the hexagon represents a price level.
///
/// # Hexagon Structure
/// - Center: Base price
/// - Ring 1: 6 vertices at 60-degree intervals
/// - Ring 2: 12 points (6 vertices + 6 midpoints)
/// - And so on...
///
/// # Key Angles
/// - 0/360: Top vertex
/// - 60: Upper right
/// - 120: Lower right
/// - 180: Bottom vertex
/// - 240: Lower left
/// - 300: Upper left
///
/// # Trading Rules
/// - Vertices provide strong support/resistance
/// - Movement between vertices indicates trend direction
/// - Ring changes indicate significant price levels
#[derive(Debug, Clone)]
pub struct GannHexagon {
    config: GannHexagonConfig,
}

impl GannHexagon {
    pub fn new() -> Self {
        Self {
            config: GannHexagonConfig::default(),
        }
    }

    pub fn with_config(config: GannHexagonConfig) -> Self {
        Self { config }
    }

    pub fn with_base_price(mut self, base: f64) -> Self {
        self.config.base_price = Some(base);
        self.config.auto_base = false;
        self
    }

    pub fn with_price_increment(mut self, increment: f64) -> Self {
        self.config.price_increment = increment;
        self
    }

    /// Calculate which hexagon ring a price falls into
    fn calc_ring(&self, price: f64, base_price: f64) -> f64 {
        let diff = (price - base_price).abs();
        (diff / self.config.price_increment).sqrt()
    }

    /// Calculate hexagon angle for a price
    fn calc_angle(&self, price: f64, base_price: f64) -> f64 {
        let diff = price - base_price;
        let ring = self.calc_ring(price, base_price);

        if ring < 0.5 {
            return 0.0;
        }

        // Simplified angle calculation based on price position
        // In a true hexagon spiral, this would be more complex
        let normalized = diff / (ring * self.config.price_increment);
        let angle = normalized.asin().to_degrees();

        if angle < 0.0 {
            360.0 + angle
        } else {
            angle
        }
    }

    /// Calculate hexagon level at specific angle and ring
    fn calc_level(&self, base_price: f64, ring: usize, angle_degrees: f64) -> f64 {
        let ring_distance = ring as f64 * self.config.price_increment;
        let angle_rad = angle_degrees.to_radians();
        base_price + ring_distance * angle_rad.cos()
    }

    /// Get all six vertex levels for a ring
    fn get_vertex_levels(&self, base_price: f64, ring: usize) -> Vec<f64> {
        (0..6)
            .map(|i| self.calc_level(base_price, ring, i as f64 * 60.0))
            .collect()
    }

    /// Calculate Gann Hexagon from OHLCV data
    pub fn calculate(&self, data: &OHLCVSeries) -> GannHexagonOutput {
        let n = data.close.len();

        // Determine base price
        let base_price = if let Some(bp) = self.config.base_price {
            bp
        } else if self.config.auto_base {
            // Use the minimum price in the data as base
            data.low.iter().cloned().fold(f64::INFINITY, f64::min)
        } else {
            data.close[0]
        };

        let mut ring_number = vec![f64::NAN; n];
        let mut ring_position = vec![f64::NAN; n];
        let mut hex_levels = vec![Vec::new(); n];
        let mut hex_support = vec![f64::NAN; n];
        let mut hex_resistance = vec![f64::NAN; n];
        let mut vertex_distance = vec![f64::NAN; n];
        let mut hex_angle = vec![f64::NAN; n];
        let mut level_strength = vec![f64::NAN; n];

        for i in 0..n {
            let price = data.close[i];

            // Calculate ring number
            let ring = self.calc_ring(price, base_price);
            ring_number[i] = ring;

            // Calculate angle
            let angle = self.calc_angle(price, base_price);
            hex_angle[i] = angle;

            // Calculate position within ring (0-6 based on which 60-degree sector)
            ring_position[i] = angle / 60.0;

            // Get current and adjacent ring levels
            let current_ring = ring.floor() as usize;
            let mut all_levels = Vec::new();

            for r in current_ring.saturating_sub(1)..=(current_ring + 2).min(self.config.num_rings) {
                let levels = self.get_vertex_levels(base_price, r);
                all_levels.extend(levels);
            }

            // Sort and remove duplicates
            all_levels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            all_levels.dedup();

            hex_levels[i] = all_levels.clone();

            // Find nearest support and resistance
            let mut nearest_support = f64::NEG_INFINITY;
            let mut nearest_resistance = f64::INFINITY;

            for &level in &all_levels {
                if level < price && level > nearest_support {
                    nearest_support = level;
                }
                if level > price && level < nearest_resistance {
                    nearest_resistance = level;
                }
            }

            if nearest_support.is_finite() {
                hex_support[i] = nearest_support;
            }
            if nearest_resistance.is_finite() {
                hex_resistance[i] = nearest_resistance;
            }

            // Calculate vertex distance (normalized)
            let mut min_distance = f64::INFINITY;
            for &level in &all_levels {
                let dist = (price - level).abs();
                if dist < min_distance {
                    min_distance = dist;
                }
            }
            vertex_distance[i] = min_distance / self.config.price_increment;

            // Level strength - inverse of distance (higher when close to a level)
            level_strength[i] = 1.0 / (1.0 + vertex_distance[i]);
        }

        GannHexagonOutput {
            ring_number,
            ring_position,
            hex_levels,
            hex_support,
            hex_resistance,
            vertex_distance,
            hex_angle,
            level_strength,
        }
    }
}

impl Default for GannHexagon {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for GannHexagon {
    fn name(&self) -> &str {
        "Gann Hexagon"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < 2 {
            return Err(IndicatorError::InsufficientData {
                required: 2,
                got: data.close.len(),
            });
        }

        let result = self.calculate(data);

        // Primary: ring_number, Secondary: level_strength, Tertiary: vertex_distance
        Ok(IndicatorOutput::triple(
            result.ring_number,
            result.level_strength,
            result.vertex_distance,
        ))
    }

    fn min_periods(&self) -> usize {
        2
    }

    fn output_features(&self) -> usize {
        3
    }
}

impl SignalIndicator for GannHexagon {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let result = self.calculate(data);
        let n = result.level_strength.len();

        if n < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let strength = result.level_strength[n - 1];
        let prev_ring = result.ring_number[n - 2];
        let curr_ring = result.ring_number[n - 1];

        if strength.is_nan() || prev_ring.is_nan() || curr_ring.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Signal when crossing ring boundaries with high level strength
        if curr_ring > prev_ring && strength > 0.7 {
            Ok(IndicatorSignal::Bullish) // Moving to outer ring (price increase)
        } else if curr_ring < prev_ring && strength > 0.7 {
            Ok(IndicatorSignal::Bearish) // Moving to inner ring (price decrease)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let result = self.calculate(data);
        let n = result.level_strength.len();

        let mut signals = vec![IndicatorSignal::Neutral; n];

        for i in 1..n {
            let strength = result.level_strength[i];
            let prev_ring = result.ring_number[i - 1];
            let curr_ring = result.ring_number[i];

            if strength.is_nan() || prev_ring.is_nan() || curr_ring.is_nan() {
                continue;
            }

            if curr_ring > prev_ring && strength > 0.7 {
                signals[i] = IndicatorSignal::Bullish;
            } else if curr_ring < prev_ring && strength > 0.7 {
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
            closes.push(100.0 + (i as f64 * 0.3).sin() * 15.0 + i as f64 * 0.5);
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
    fn test_hexagon_initialization() {
        let hex = GannHexagon::new();
        assert_eq!(hex.name(), "Gann Hexagon");
        assert_eq!(hex.min_periods(), 2);
        assert_eq!(hex.output_features(), 3);
    }

    #[test]
    fn test_hexagon_calculation() {
        let data = create_test_data(30);
        let hex = GannHexagon::new().with_price_increment(5.0);
        let result = hex.calculate(&data);

        assert_eq!(result.ring_number.len(), 30);
        assert_eq!(result.hex_support.len(), 30);
        assert_eq!(result.hex_resistance.len(), 30);

        // Verify ring numbers are calculated
        for i in 0..30 {
            assert!(!result.ring_number[i].is_nan());
            assert!(result.ring_number[i] >= 0.0);
        }
    }

    #[test]
    fn test_vertex_levels() {
        let hex = GannHexagon::new().with_price_increment(10.0);
        let levels = hex.get_vertex_levels(100.0, 1);

        assert_eq!(levels.len(), 6);

        // First vertex should be at base + increment * cos(0) = 110
        assert!((levels[0] - 110.0).abs() < 0.01);
    }

    #[test]
    fn test_hexagon_compute() {
        let data = create_test_data(30);
        let hex = GannHexagon::new();
        let output = hex.compute(&data).unwrap();

        assert_eq!(output.primary.len(), 30);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    #[test]
    fn test_hexagon_signals() {
        let data = create_test_data(30);
        let hex = GannHexagon::new().with_price_increment(5.0);
        let signals = hex.signals(&data).unwrap();

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

        let hex = GannHexagon::new();
        let result = hex.compute(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_custom_base_price() {
        let data = create_test_data(30);
        let hex = GannHexagon::new().with_base_price(50.0);
        let result = hex.calculate(&data);

        // All prices should be above base, so ring numbers should be positive
        for i in 0..30 {
            assert!(result.ring_number[i] > 0.0);
        }
    }

    #[test]
    fn test_level_strength_bounds() {
        let data = create_test_data(30);
        let hex = GannHexagon::new();
        let result = hex.calculate(&data);

        // Level strength should be between 0 and 1
        for i in 0..30 {
            if !result.level_strength[i].is_nan() {
                assert!(result.level_strength[i] >= 0.0);
                assert!(result.level_strength[i] <= 1.0);
            }
        }
    }
}
