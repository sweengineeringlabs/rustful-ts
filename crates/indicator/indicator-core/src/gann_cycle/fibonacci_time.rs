//! Fibonacci Time Zones Indicator (IND-327)
//!
//! Fibonacci Time Zones project Fibonacci numbers onto the time axis
//! to identify potential turning points. The zones mark bars at
//! Fibonacci intervals from a significant price event.

use crate::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result, SignalIndicator,
    TechnicalIndicator,
};
use serde::{Deserialize, Serialize};

/// Fibonacci Time Zones output structure
#[derive(Debug, Clone)]
pub struct FibonacciTimeZonesOutput {
    /// Fibonacci time zone markers (1.0 at zone, 0.0 otherwise)
    pub zone_markers: Vec<f64>,
    /// Distance to next Fibonacci zone (in bars)
    pub distance_to_zone: Vec<f64>,
    /// Current Fibonacci number at active zone
    pub current_fib_number: Vec<f64>,
    /// Zone strength (higher Fib numbers = stronger)
    pub zone_strength: Vec<f64>,
    /// Bars within zone window
    pub in_zone_window: Vec<bool>,
    /// Time extension levels
    pub extension_levels: Vec<Vec<usize>>,
}

/// Fibonacci Time Zones configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FibonacciTimeZonesConfig {
    /// Starting bar for Fibonacci projection
    pub start_bar: Option<usize>,
    /// Auto-detect start from significant pivot
    pub auto_start: bool,
    /// Lookback for pivot detection
    pub pivot_lookback: usize,
    /// Number of Fibonacci zones to project
    pub num_zones: usize,
    /// Zone window (bars before/after zone)
    pub zone_window: usize,
    /// Include extended Fibonacci ratios (1.618, 2.618, etc.)
    pub include_extensions: bool,
}

impl Default for FibonacciTimeZonesConfig {
    fn default() -> Self {
        Self {
            start_bar: None,
            auto_start: true,
            pivot_lookback: 20,
            num_zones: 15,
            zone_window: 2,
            include_extensions: true,
        }
    }
}

/// Fibonacci Time Zones Indicator
///
/// Projects Fibonacci numbers onto the time axis to identify
/// potential market turning points.
///
/// # Fibonacci Sequence
/// 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610...
///
/// # Time Extensions
/// When include_extensions is true, also calculates:
/// - 1.618x (Golden Ratio)
/// - 2.618x
/// - 4.236x
///
/// # Trading Rules
/// - Zones indicate potential reversal or continuation points
/// - Higher Fibonacci numbers suggest stronger potential reversals
/// - Combine with price Fibonacci levels for confluence
#[derive(Debug, Clone)]
pub struct FibonacciTimeZones {
    config: FibonacciTimeZonesConfig,
    fib_sequence: Vec<usize>,
}

impl FibonacciTimeZones {
    pub fn new() -> Self {
        let config = FibonacciTimeZonesConfig::default();
        let fib_sequence = Self::generate_fibonacci(config.num_zones);
        Self {
            config,
            fib_sequence,
        }
    }

    pub fn with_config(config: FibonacciTimeZonesConfig) -> Self {
        let fib_sequence = Self::generate_fibonacci(config.num_zones);
        Self {
            config,
            fib_sequence,
        }
    }

    pub fn with_start_bar(mut self, bar: usize) -> Self {
        self.config.start_bar = Some(bar);
        self.config.auto_start = false;
        self
    }

    pub fn with_num_zones(mut self, zones: usize) -> Self {
        self.config.num_zones = zones;
        self.fib_sequence = Self::generate_fibonacci(zones);
        self
    }

    /// Generate Fibonacci sequence
    fn generate_fibonacci(count: usize) -> Vec<usize> {
        let mut fibs = Vec::with_capacity(count);
        if count == 0 {
            return fibs;
        }

        fibs.push(1);
        if count == 1 {
            return fibs;
        }

        fibs.push(1);
        while fibs.len() < count {
            let next = fibs[fibs.len() - 1] + fibs[fibs.len() - 2];
            fibs.push(next);
        }

        fibs
    }

    /// Find significant pivot low
    fn find_pivot_low(&self, data: &OHLCVSeries) -> usize {
        let n = data.close.len();
        let lookback = self.config.pivot_lookback.min(n);

        let start = n.saturating_sub(lookback);
        let mut min_idx = start;
        let mut min_val = data.low[start];

        for i in start..n {
            if data.low[i] < min_val {
                min_val = data.low[i];
                min_idx = i;
            }
        }

        min_idx
    }

    /// Get time extension levels from start bar
    fn get_extension_levels(&self, start: usize, max_bar: usize) -> Vec<usize> {
        let mut levels = Vec::new();

        // Add standard Fibonacci numbers
        for &fib in &self.fib_sequence {
            let bar = start + fib;
            if bar < max_bar {
                levels.push(bar);
            }
        }

        // Add extension levels if enabled
        if self.config.include_extensions {
            let extensions = [1.618, 2.618, 4.236];

            for &fib in &self.fib_sequence {
                for &ext in &extensions {
                    let bar = start + (fib as f64 * ext).round() as usize;
                    if bar < max_bar && !levels.contains(&bar) {
                        levels.push(bar);
                    }
                }
            }
        }

        levels.sort();
        levels.dedup();
        levels
    }

    /// Calculate Fibonacci Time Zones from OHLCV data
    pub fn calculate(&self, data: &OHLCVSeries) -> FibonacciTimeZonesOutput {
        let n = data.close.len();

        // Determine start bar
        let start_bar = if let Some(sb) = self.config.start_bar {
            sb.min(n.saturating_sub(1))
        } else if self.config.auto_start {
            self.find_pivot_low(data)
        } else {
            0
        };

        // Get all Fibonacci zone bars
        let zone_bars = self.get_extension_levels(start_bar, n);

        let mut zone_markers = vec![0.0; n];
        let mut distance_to_zone = vec![f64::NAN; n];
        let mut current_fib_number = vec![f64::NAN; n];
        let mut zone_strength = vec![f64::NAN; n];
        let mut in_zone_window = vec![false; n];
        let mut extension_levels = vec![Vec::new(); n];

        // Mark zone bars
        for &zone_bar in &zone_bars {
            if zone_bar < n {
                zone_markers[zone_bar] = 1.0;

                // Find which Fibonacci number this corresponds to
                let bars_from_start = zone_bar - start_bar;
                for (idx, &fib) in self.fib_sequence.iter().enumerate() {
                    if fib == bars_from_start {
                        current_fib_number[zone_bar] = fib as f64;
                        // Strength increases with Fibonacci index
                        zone_strength[zone_bar] = (idx as f64 + 1.0) / self.config.num_zones as f64;
                        break;
                    }
                }

                // Mark zone window
                let window_start = zone_bar.saturating_sub(self.config.zone_window);
                let window_end = (zone_bar + self.config.zone_window + 1).min(n);
                for j in window_start..window_end {
                    in_zone_window[j] = true;
                }
            }
        }

        // Calculate distance to next zone for each bar
        for i in 0..n {
            let mut min_distance = f64::INFINITY;
            let mut nearest_zone = None;

            for &zone_bar in &zone_bars {
                if zone_bar >= i {
                    let dist = (zone_bar - i) as f64;
                    if dist < min_distance {
                        min_distance = dist;
                        nearest_zone = Some(zone_bar);
                    }
                }
            }

            if min_distance.is_finite() {
                distance_to_zone[i] = min_distance;
            }

            // Store extension levels for this bar
            extension_levels[i] = zone_bars.iter().filter(|&&b| b >= i).cloned().collect();
        }

        FibonacciTimeZonesOutput {
            zone_markers,
            distance_to_zone,
            current_fib_number,
            zone_strength,
            in_zone_window,
            extension_levels,
        }
    }

    /// Get the Fibonacci sequence
    pub fn fibonacci_sequence(&self) -> &[usize] {
        &self.fib_sequence
    }
}

impl Default for FibonacciTimeZones {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for FibonacciTimeZones {
    fn name(&self) -> &str {
        "Fibonacci Time Zones"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.config.pivot_lookback {
            return Err(IndicatorError::InsufficientData {
                required: self.config.pivot_lookback,
                got: data.close.len(),
            });
        }

        let result = self.calculate(data);

        // Primary: zone_markers, Secondary: distance_to_zone, Tertiary: zone_strength
        Ok(IndicatorOutput::triple(
            result.zone_markers,
            result.distance_to_zone,
            result.zone_strength,
        ))
    }

    fn min_periods(&self) -> usize {
        self.config.pivot_lookback
    }

    fn output_features(&self) -> usize {
        3
    }
}

impl SignalIndicator for FibonacciTimeZones {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let result = self.calculate(data);
        let n = result.zone_markers.len();

        if n < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        // Signal when entering a zone window
        let in_zone = result.in_zone_window[n - 1];
        let was_in_zone = result.in_zone_window[n - 2];

        if in_zone && !was_in_zone {
            // Entering a Fibonacci time zone - potential reversal
            // Use price action to determine direction
            let price_rising = data.close[n - 1] > data.close[n - 2];

            if price_rising {
                // Price rising into zone - potential top
                Ok(IndicatorSignal::Bearish)
            } else {
                // Price falling into zone - potential bottom
                Ok(IndicatorSignal::Bullish)
            }
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let result = self.calculate(data);
        let n = result.zone_markers.len();

        let mut signals = vec![IndicatorSignal::Neutral; n];

        for i in 1..n {
            let in_zone = result.in_zone_window[i];
            let was_in_zone = result.in_zone_window[i - 1];

            if in_zone && !was_in_zone {
                let price_rising = data.close[i] > data.close[i - 1];
                signals[i] = if price_rising {
                    IndicatorSignal::Bearish
                } else {
                    IndicatorSignal::Bullish
                };
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
            closes.push(100.0 + (i as f64 * 0.2).sin() * 10.0);
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
    fn test_fib_time_initialization() {
        let ftz = FibonacciTimeZones::new();
        assert_eq!(ftz.name(), "Fibonacci Time Zones");
        assert_eq!(ftz.output_features(), 3);
    }

    #[test]
    fn test_fibonacci_sequence_generation() {
        let fibs = FibonacciTimeZones::generate_fibonacci(10);
        assert_eq!(fibs, vec![1, 1, 2, 3, 5, 8, 13, 21, 34, 55]);
    }

    #[test]
    fn test_fib_time_calculation() {
        let data = create_test_data(100);
        let ftz = FibonacciTimeZones::new().with_start_bar(0);
        let result = ftz.calculate(&data);

        assert_eq!(result.zone_markers.len(), 100);
        assert_eq!(result.distance_to_zone.len(), 100);

        // Check that Fibonacci bars are marked
        // Starting at 0, zones should be at 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89
        assert_eq!(result.zone_markers[1], 1.0);
        assert_eq!(result.zone_markers[2], 1.0);
        assert_eq!(result.zone_markers[3], 1.0);
        assert_eq!(result.zone_markers[5], 1.0);
        assert_eq!(result.zone_markers[8], 1.0);
        assert_eq!(result.zone_markers[13], 1.0);
        assert_eq!(result.zone_markers[21], 1.0);
        assert_eq!(result.zone_markers[34], 1.0);
        assert_eq!(result.zone_markers[55], 1.0);
        assert_eq!(result.zone_markers[89], 1.0);
    }

    #[test]
    fn test_zone_window() {
        let data = create_test_data(100);
        let ftz = FibonacciTimeZones::new().with_start_bar(0);
        let result = ftz.calculate(&data);

        // Zone at bar 13 with window of 2 should mark bars 11-15
        // (default zone_window is 2)
        assert!(result.in_zone_window[11]);
        assert!(result.in_zone_window[12]);
        assert!(result.in_zone_window[13]);
        assert!(result.in_zone_window[14]);
        assert!(result.in_zone_window[15]);
    }

    #[test]
    fn test_distance_to_zone() {
        let data = create_test_data(100);
        let ftz = FibonacciTimeZones::new().with_start_bar(0);
        let result = ftz.calculate(&data);

        // At bar 0, distance to first zone (bar 1) should be 1
        assert!((result.distance_to_zone[0] - 1.0).abs() < 0.001);

        // At bar 1 (which is a zone), distance to next zone (bar 2) should be 1
        assert!((result.distance_to_zone[1] - 0.0).abs() < 0.001 ||
                (result.distance_to_zone[1] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_fib_time_compute() {
        let data = create_test_data(50);
        let ftz = FibonacciTimeZones::new();
        let output = ftz.compute(&data).unwrap();

        assert_eq!(output.primary.len(), 50);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    #[test]
    fn test_fib_time_signals() {
        let data = create_test_data(50);
        let ftz = FibonacciTimeZones::new();
        let signals = ftz.signals(&data).unwrap();

        assert_eq!(signals.len(), 50);
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

        let ftz = FibonacciTimeZones::new();
        let result = ftz.compute(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_extension_levels() {
        let ftz = FibonacciTimeZones::new();
        let levels = ftz.get_extension_levels(0, 200);

        // Should include both standard Fib numbers and extensions
        assert!(levels.contains(&1));
        assert!(levels.contains(&2));
        assert!(levels.contains(&3));
        assert!(levels.contains(&5));
        assert!(levels.contains(&8));

        // Should be sorted
        for i in 1..levels.len() {
            assert!(levels[i] >= levels[i - 1]);
        }
    }
}
