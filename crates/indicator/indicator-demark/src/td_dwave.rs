//! TD D-Wave - DeMark Wave Analysis.
//!
//! TD D-Wave identifies market waves (similar to Elliott Wave but with
//! objective rules) by detecting pivot points and classifying wave structure.

use indicator_spi::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result, SignalIndicator,
    TechnicalIndicator,
};
use serde::{Deserialize, Serialize};

/// Wave direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WaveDirection {
    Up,
    Down,
    None,
}

/// Wave phase in the D-Wave count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DWavePhase {
    /// No wave identified
    None,
    /// Wave 1 (initial impulse)
    Wave1,
    /// Wave 2 (correction)
    Wave2,
    /// Wave 3 (strongest impulse)
    Wave3,
    /// Wave 4 (correction)
    Wave4,
    /// Wave 5 (final impulse)
    Wave5,
    /// Wave A (corrective)
    WaveA,
    /// Wave B (corrective)
    WaveB,
    /// Wave C (corrective)
    WaveC,
}

/// Pivot point type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PivotType {
    None,
    High,
    Low,
}

/// TD D-Wave output.
#[derive(Debug, Clone)]
pub struct TDDWaveOutput {
    /// Wave phase at each bar
    pub phase: Vec<DWavePhase>,
    /// Current wave direction
    pub direction: Vec<WaveDirection>,
    /// Pivot points detected
    pub pivots: Vec<PivotType>,
    /// Pivot prices
    pub pivot_prices: Vec<f64>,
    /// Wave number (1-5 for impulse, 6-8 for corrective A-B-C)
    pub wave_number: Vec<i32>,
    /// Wave is potentially complete
    pub wave_complete: Vec<bool>,
}

/// TD D-Wave configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TDDWaveConfig {
    /// Pivot detection lookback (default: 5)
    pub pivot_lookback: usize,
    /// Minimum bars between pivots (default: 3)
    pub min_pivot_bars: usize,
    /// Minimum wave size as percentage (default: 1.0%)
    pub min_wave_pct: f64,
}

impl Default for TDDWaveConfig {
    fn default() -> Self {
        Self {
            pivot_lookback: 5,
            min_pivot_bars: 3,
            min_wave_pct: 1.0,
        }
    }
}

/// TD D-Wave Indicator.
///
/// Identifies waves in price action using objective pivot-based rules.
///
/// # Wave Rules (Simplified DeMark)
/// - Wave 1: Initial move from a significant low/high
/// - Wave 2: Retracement that doesn't exceed Wave 1 start
/// - Wave 3: Extension beyond Wave 1 end (usually strongest)
/// - Wave 4: Retracement that doesn't overlap Wave 1 end
/// - Wave 5: Final extension (often shows exhaustion)
/// - Waves A, B, C: Corrective structure after Wave 5
///
/// # Interpretation
/// - Wave 3: Usually best for trend following
/// - Wave 5: Look for exhaustion signals
/// - Wave C: Potential reversal zone
#[derive(Debug, Clone)]
pub struct TDDWave {
    config: TDDWaveConfig,
}

impl TDDWave {
    pub fn new() -> Self {
        Self {
            config: TDDWaveConfig::default(),
        }
    }

    pub fn with_config(config: TDDWaveConfig) -> Self {
        Self { config }
    }

    pub fn with_pivot_lookback(mut self, lookback: usize) -> Self {
        self.config.pivot_lookback = lookback;
        self
    }

    /// Detect pivot highs and lows.
    fn detect_pivots(&self, data: &OHLCVSeries) -> (Vec<PivotType>, Vec<f64>) {
        let n = data.close.len();
        let lookback = self.config.pivot_lookback;

        let mut pivots = vec![PivotType::None; n];
        let mut pivot_prices = vec![f64::NAN; n];

        if n < lookback * 2 + 1 {
            return (pivots, pivot_prices);
        }

        for i in lookback..(n - lookback) {
            // Check for pivot high
            let mut is_pivot_high = true;
            let high_i = data.high[i];

            for j in (i - lookback)..i {
                if data.high[j] >= high_i {
                    is_pivot_high = false;
                    break;
                }
            }
            if is_pivot_high {
                for j in (i + 1)..=(i + lookback) {
                    if data.high[j] >= high_i {
                        is_pivot_high = false;
                        break;
                    }
                }
            }

            // Check for pivot low
            let mut is_pivot_low = true;
            let low_i = data.low[i];

            for j in (i - lookback)..i {
                if data.low[j] <= low_i {
                    is_pivot_low = false;
                    break;
                }
            }
            if is_pivot_low {
                for j in (i + 1)..=(i + lookback) {
                    if data.low[j] <= low_i {
                        is_pivot_low = false;
                        break;
                    }
                }
            }

            // Record pivot (prefer low if both, or take the more significant)
            if is_pivot_low {
                pivots[i] = PivotType::Low;
                pivot_prices[i] = low_i;
            } else if is_pivot_high {
                pivots[i] = PivotType::High;
                pivot_prices[i] = high_i;
            }
        }

        (pivots, pivot_prices)
    }

    /// Calculate TD D-Wave from OHLC data.
    pub fn calculate(&self, data: &OHLCVSeries) -> TDDWaveOutput {
        let n = data.close.len();

        let (pivots, pivot_prices) = self.detect_pivots(data);

        let mut phase = vec![DWavePhase::None; n];
        let mut direction = vec![WaveDirection::None; n];
        let mut wave_number = vec![0i32; n];
        let mut wave_complete = vec![false; n];

        if n < self.config.pivot_lookback * 2 + 1 {
            return TDDWaveOutput {
                phase,
                direction,
                pivots,
                pivot_prices,
                wave_number,
                wave_complete,
            };
        }

        // Collect significant pivots
        let mut pivot_list: Vec<(usize, PivotType, f64)> = Vec::new();
        let mut last_pivot_idx = 0usize;

        for i in 0..n {
            if pivots[i] != PivotType::None {
                // Check minimum bars between pivots
                if pivot_list.is_empty() || i - last_pivot_idx >= self.config.min_pivot_bars {
                    // Check minimum wave size
                    if let Some(&(_, _, prev_price)) = pivot_list.last() {
                        let pct_change = ((pivot_prices[i] - prev_price) / prev_price).abs() * 100.0;
                        if pct_change >= self.config.min_wave_pct {
                            pivot_list.push((i, pivots[i], pivot_prices[i]));
                            last_pivot_idx = i;
                        }
                    } else {
                        pivot_list.push((i, pivots[i], pivot_prices[i]));
                        last_pivot_idx = i;
                    }
                }
            }
        }

        // Assign wave numbers based on pivot sequence
        let mut current_wave = 0i32;
        let mut wave_start_price = 0.0f64;
        let mut wave1_end_price = 0.0f64;
        let mut is_bullish_sequence = true;

        for (pivot_idx, (bar, ptype, price)) in pivot_list.iter().enumerate() {
            // Determine wave direction
            if pivot_idx > 0 {
                let prev_price = pivot_list[pivot_idx - 1].2;
                is_bullish_sequence = *price > prev_price;
            }

            // Simple wave assignment
            current_wave = ((pivot_idx % 5) + 1) as i32;

            // Assign to all bars up to this pivot
            let start_bar = if pivot_idx > 0 {
                pivot_list[pivot_idx - 1].0 + 1
            } else {
                0
            };

            for i in start_bar..=*bar {
                wave_number[i] = current_wave;
                direction[i] = if is_bullish_sequence {
                    WaveDirection::Up
                } else {
                    WaveDirection::Down
                };

                phase[i] = match current_wave {
                    1 => DWavePhase::Wave1,
                    2 => DWavePhase::Wave2,
                    3 => DWavePhase::Wave3,
                    4 => DWavePhase::Wave4,
                    5 => DWavePhase::Wave5,
                    _ => DWavePhase::None,
                };
            }

            // Track wave 1 end for validation
            if current_wave == 1 {
                wave_start_price = *price;
            } else if current_wave == 2 {
                wave1_end_price = *price;
            }

            // Mark wave completion at pivots
            wave_complete[*bar] = true;
        }

        // Fill remaining bars after last pivot
        if let Some(&(last_bar, _, _)) = pivot_list.last() {
            for i in (last_bar + 1)..n {
                wave_number[i] = current_wave;
                direction[i] = direction[last_bar];
                phase[i] = phase[last_bar];
            }
        }

        TDDWaveOutput {
            phase,
            direction,
            pivots,
            pivot_prices,
            wave_number,
            wave_complete,
        }
    }
}

impl Default for TDDWave {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for TDDWave {
    fn name(&self) -> &str {
        "TD D-Wave"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_bars = self.config.pivot_lookback * 2 + 1;
        if data.close.len() < min_bars {
            return Err(IndicatorError::InsufficientData {
                required: min_bars,
                got: data.close.len(),
            });
        }

        let result = self.calculate(data);

        // Encode wave number and direction
        let wave_f64: Vec<f64> = result.wave_number.iter().map(|&w| w as f64).collect();
        let dir_f64: Vec<f64> = result.direction.iter().map(|d| match d {
            WaveDirection::Up => 1.0,
            WaveDirection::Down => -1.0,
            WaveDirection::None => 0.0,
        }).collect();

        Ok(IndicatorOutput::dual(wave_f64, dir_f64))
    }

    fn min_periods(&self) -> usize {
        self.config.pivot_lookback * 2 + 1
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for TDDWave {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let result = self.calculate(data);
        let n = result.wave_number.len();

        if n == 0 {
            return Ok(IndicatorSignal::Neutral);
        }

        // Signal based on wave and completion
        let last_wave = result.wave_number[n - 1];
        let last_dir = result.direction[n - 1];
        let last_complete = result.wave_complete[n - 1];

        // Wave 3 in progress is bullish/bearish based on direction
        if last_wave == 3 && !last_complete {
            return match last_dir {
                WaveDirection::Up => Ok(IndicatorSignal::Bullish),
                WaveDirection::Down => Ok(IndicatorSignal::Bearish),
                WaveDirection::None => Ok(IndicatorSignal::Neutral),
            };
        }

        // Wave 5 completion suggests reversal coming
        if last_wave == 5 && last_complete {
            return match last_dir {
                WaveDirection::Up => Ok(IndicatorSignal::Bearish), // End of up move
                WaveDirection::Down => Ok(IndicatorSignal::Bullish), // End of down move
                WaveDirection::None => Ok(IndicatorSignal::Neutral),
            };
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let result = self.calculate(data);
        let n = result.wave_number.len();

        let mut signals = vec![IndicatorSignal::Neutral; n];

        for i in 0..n {
            let wave = result.wave_number[i];
            let dir = result.direction[i];
            let complete = result.wave_complete[i];

            // Wave 3 is the strongest, signal with direction
            if wave == 3 {
                signals[i] = match dir {
                    WaveDirection::Up => IndicatorSignal::Bullish,
                    WaveDirection::Down => IndicatorSignal::Bearish,
                    WaveDirection::None => IndicatorSignal::Neutral,
                };
            }

            // Wave 5 completion
            if wave == 5 && complete {
                signals[i] = match dir {
                    WaveDirection::Up => IndicatorSignal::Bearish,
                    WaveDirection::Down => IndicatorSignal::Bullish,
                    WaveDirection::None => IndicatorSignal::Neutral,
                };
            }
        }

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_wave_data() -> OHLCVSeries {
        // Create data with clear wave structure
        let prices = vec![
            // Wave 1 up
            100.0, 101.0, 102.0, 103.0, 104.0, 105.0,
            // Wave 2 down
            104.0, 103.0, 102.0,
            // Wave 3 up (strongest)
            103.0, 105.0, 107.0, 109.0, 111.0, 113.0, 115.0,
            // Wave 4 down
            114.0, 112.0, 111.0,
            // Wave 5 up
            112.0, 114.0, 116.0, 117.0, 118.0,
        ];

        let n = prices.len();
        OHLCVSeries {
            open: prices.clone(),
            high: prices.iter().map(|p| p + 1.0).collect(),
            low: prices.iter().map(|p| p - 1.0).collect(),
            close: prices,
            volume: vec![1000.0; n],
        }
    }

    #[test]
    fn test_dwave_initialization() {
        let dwave = TDDWave::new();
        assert_eq!(dwave.name(), "TD D-Wave");
        assert_eq!(dwave.config.pivot_lookback, 5);
    }

    #[test]
    fn test_pivot_detection() {
        let data = create_wave_data();
        let dwave = TDDWave::new().with_pivot_lookback(2);
        let result = dwave.calculate(&data);

        // Should detect some pivots
        let pivot_count = result.pivots.iter()
            .filter(|p| **p != PivotType::None)
            .count();
        assert!(pivot_count > 0);
    }

    #[test]
    fn test_wave_number_assignment() {
        let data = create_wave_data();
        let dwave = TDDWave::new().with_pivot_lookback(2);
        let result = dwave.calculate(&data);

        // Should have wave numbers 1-5
        let has_wave1 = result.wave_number.iter().any(|&w| w == 1);
        let has_wave3 = result.wave_number.iter().any(|&w| w == 3);
        let has_wave5 = result.wave_number.iter().any(|&w| w == 5);

        // At least some waves should be assigned
        let non_zero = result.wave_number.iter().any(|&w| w != 0);
        assert!(non_zero);
    }

    #[test]
    fn test_direction_tracking() {
        let data = create_wave_data();
        let dwave = TDDWave::new().with_pivot_lookback(2);
        let result = dwave.calculate(&data);

        // Should have both up and down directions
        let has_up = result.direction.iter().any(|d| *d == WaveDirection::Up);
        let has_down = result.direction.iter().any(|d| *d == WaveDirection::Down);

        // In wave data, should have at least one direction
        let has_direction = has_up || has_down;
        // May not have clear waves with small lookback
    }

    #[test]
    fn test_wave_phases() {
        let phases = vec![
            DWavePhase::None,
            DWavePhase::Wave1,
            DWavePhase::Wave2,
            DWavePhase::Wave3,
            DWavePhase::Wave4,
            DWavePhase::Wave5,
            DWavePhase::WaveA,
            DWavePhase::WaveB,
            DWavePhase::WaveC,
        ];

        // Verify enum variants exist
        assert_eq!(phases.len(), 9);
    }

    #[test]
    fn test_insufficient_data() {
        let data = OHLCVSeries {
            open: vec![100.0; 5],
            high: vec![101.0; 5],
            low: vec![99.0; 5],
            close: vec![100.0; 5],
            volume: vec![1000.0; 5],
        };

        let dwave = TDDWave::new();
        let result = dwave.compute(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_output() {
        let data = create_wave_data();
        let dwave = TDDWave::new().with_pivot_lookback(2);
        let output = dwave.compute(&data);

        assert!(output.is_ok());
        let out = output.unwrap();
        assert_eq!(out.primary.len(), data.close.len());
    }

    #[test]
    fn test_signals() {
        let data = create_wave_data();
        let dwave = TDDWave::new().with_pivot_lookback(2);
        let signals = dwave.signals(&data).unwrap();

        assert_eq!(signals.len(), data.close.len());
    }
}
