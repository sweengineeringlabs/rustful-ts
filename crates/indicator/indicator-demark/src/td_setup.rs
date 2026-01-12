//! TD Setup - The first phase of Tom DeMark's Sequential indicator.
//!
//! TD Setup identifies potential trend exhaustion by counting consecutive bars
//! where the close is higher (buy setup) or lower (sell setup) than the close
//! 4 bars earlier. A complete setup is 9 consecutive qualifying bars.

use indicator_spi::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result, SignalIndicator,
    TechnicalIndicator,
};
use serde::{Deserialize, Serialize};

/// TD Setup phase type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SetupPhase {
    /// No active setup
    None,
    /// Buy setup (bearish trend potentially exhausting)
    BuySetup,
    /// Sell setup (bullish trend potentially exhausting)
    SellSetup,
}

/// TD Setup output containing setup counts and phase information.
#[derive(Debug, Clone)]
pub struct TDSetupOutput {
    /// Setup count for each bar (1-9 for active setup, 0 otherwise)
    pub count: Vec<i32>,
    /// Phase at each bar
    pub phase: Vec<SetupPhase>,
    /// True when a 9-count setup completes (perfected or not)
    pub setup_complete: Vec<bool>,
    /// True when setup is "perfected" (bar 8 or 9 low <= low of bars 6 and 7 for buy)
    pub perfected: Vec<bool>,
    /// True Range Maximum of the setup (TDST support/resistance)
    pub tdst_level: Vec<f64>,
}

/// TD Setup configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TDSetupConfig {
    /// Lookback period for price comparison (default: 4)
    pub lookback: usize,
    /// Required count for complete setup (default: 9)
    pub setup_count: usize,
}

impl Default for TDSetupConfig {
    fn default() -> Self {
        Self {
            lookback: 4,
            setup_count: 9,
        }
    }
}

/// TD Setup Indicator.
///
/// Counts consecutive closes above (sell setup) or below (buy setup) the close
/// from 4 bars earlier. A complete setup requires 9 consecutive qualifying bars.
///
/// # Setup Rules
/// - Buy Setup: Close < Close[4] for 9 consecutive bars
/// - Sell Setup: Close > Close[4] for 9 consecutive bars
/// - Setup is "perfected" when bar 8 or 9's low is <= low of bars 6 and 7 (for buy)
/// - TDST (TD Setup Trend) level is the True High/Low of the setup range
#[derive(Debug, Clone)]
pub struct TDSetup {
    config: TDSetupConfig,
}

impl TDSetup {
    pub fn new() -> Self {
        Self {
            config: TDSetupConfig::default(),
        }
    }

    pub fn with_config(config: TDSetupConfig) -> Self {
        Self { config }
    }

    /// Calculate TD Setup from OHLC data.
    pub fn calculate(&self, data: &OHLCVSeries) -> TDSetupOutput {
        let n = data.close.len();
        let lookback = self.config.lookback;
        let setup_count = self.config.setup_count as i32;

        let mut count = vec![0i32; n];
        let mut phase = vec![SetupPhase::None; n];
        let mut setup_complete = vec![false; n];
        let mut perfected = vec![false; n];
        let mut tdst_level = vec![f64::NAN; n];

        if n <= lookback {
            return TDSetupOutput {
                count,
                phase,
                setup_complete,
                perfected,
                tdst_level,
            };
        }

        let mut current_count = 0i32;
        let mut current_phase = SetupPhase::None;
        let mut setup_start_idx = 0usize;

        for i in lookback..n {
            let close = data.close[i];
            let close_lookback = data.close[i - lookback];

            // Determine if current bar qualifies for buy or sell setup
            let is_buy_bar = close < close_lookback;
            let is_sell_bar = close > close_lookback;

            // Handle setup continuation or reset
            match current_phase {
                SetupPhase::None => {
                    if is_buy_bar {
                        current_phase = SetupPhase::BuySetup;
                        current_count = 1;
                        setup_start_idx = i;
                    } else if is_sell_bar {
                        current_phase = SetupPhase::SellSetup;
                        current_count = 1;
                        setup_start_idx = i;
                    }
                }
                SetupPhase::BuySetup => {
                    if is_buy_bar {
                        current_count += 1;
                    } else {
                        // Setup broken - check if sell setup starts
                        if is_sell_bar {
                            current_phase = SetupPhase::SellSetup;
                            current_count = 1;
                            setup_start_idx = i;
                        } else {
                            current_phase = SetupPhase::None;
                            current_count = 0;
                        }
                    }
                }
                SetupPhase::SellSetup => {
                    if is_sell_bar {
                        current_count += 1;
                    } else {
                        // Setup broken - check if buy setup starts
                        if is_buy_bar {
                            current_phase = SetupPhase::BuySetup;
                            current_count = 1;
                            setup_start_idx = i;
                        } else {
                            current_phase = SetupPhase::None;
                            current_count = 0;
                        }
                    }
                }
            }

            // Record current state
            count[i] = current_count;
            phase[i] = current_phase;

            // Check for setup completion
            if current_count == setup_count {
                setup_complete[i] = true;

                // Calculate TDST level (True Range of setup)
                let setup_highs: Vec<f64> = (setup_start_idx..=i)
                    .map(|j| data.high[j])
                    .collect();
                let setup_lows: Vec<f64> = (setup_start_idx..=i)
                    .map(|j| data.low[j])
                    .collect();

                match current_phase {
                    SetupPhase::BuySetup => {
                        // TDST resistance is the highest high of the buy setup
                        tdst_level[i] = setup_highs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                        // Check perfection: bar 8 or 9 low <= low of bars 6 and 7
                        if i >= 2 {
                            let bar_8_low = data.low[i - 1];
                            let bar_9_low = data.low[i];
                            let bar_6_low = data.low[i - 3];
                            let bar_7_low = data.low[i - 2];
                            let reference_low = bar_6_low.min(bar_7_low);
                            perfected[i] = bar_8_low <= reference_low || bar_9_low <= reference_low;
                        }
                    }
                    SetupPhase::SellSetup => {
                        // TDST support is the lowest low of the sell setup
                        tdst_level[i] = setup_lows.iter().cloned().fold(f64::INFINITY, f64::min);

                        // Check perfection: bar 8 or 9 high >= high of bars 6 and 7
                        if i >= 2 {
                            let bar_8_high = data.high[i - 1];
                            let bar_9_high = data.high[i];
                            let bar_6_high = data.high[i - 3];
                            let bar_7_high = data.high[i - 2];
                            let reference_high = bar_6_high.max(bar_7_high);
                            perfected[i] = bar_8_high >= reference_high || bar_9_high >= reference_high;
                        }
                    }
                    SetupPhase::None => {}
                }

                // Reset for next setup after completion
                current_phase = SetupPhase::None;
                current_count = 0;
            }
        }

        TDSetupOutput {
            count,
            phase,
            setup_complete,
            perfected,
            tdst_level,
        }
    }
}

impl Default for TDSetup {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for TDSetup {
    fn name(&self) -> &str {
        "TD Setup"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() <= self.config.lookback {
            return Err(IndicatorError::InsufficientData {
                required: self.config.lookback + 1,
                got: data.close.len(),
            });
        }

        let result = self.calculate(data);
        // Convert count to f64, phase encoded as: 0 = None, 1 = Buy, -1 = Sell
        let count_f64: Vec<f64> = result.count.iter().map(|&c| c as f64).collect();
        let phase_f64: Vec<f64> = result.phase.iter().map(|p| match p {
            SetupPhase::None => 0.0,
            SetupPhase::BuySetup => 1.0,
            SetupPhase::SellSetup => -1.0,
        }).collect();

        Ok(IndicatorOutput::dual(count_f64, phase_f64))
    }

    fn min_periods(&self) -> usize {
        self.config.lookback + 1
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for TDSetup {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let result = self.calculate(data);
        let n = result.count.len();

        if n == 0 {
            return Ok(IndicatorSignal::Neutral);
        }

        // Signal on setup completion
        if result.setup_complete[n - 1] {
            match result.phase[n - 1] {
                SetupPhase::BuySetup => Ok(IndicatorSignal::Bullish),
                SetupPhase::SellSetup => Ok(IndicatorSignal::Bearish),
                SetupPhase::None => Ok(IndicatorSignal::Neutral),
            }
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let result = self.calculate(data);
        let signals = result.setup_complete.iter().zip(result.phase.iter())
            .map(|(&complete, &p)| {
                if complete {
                    match p {
                        SetupPhase::BuySetup => IndicatorSignal::Bullish,
                        SetupPhase::SellSetup => IndicatorSignal::Bearish,
                        SetupPhase::None => IndicatorSignal::Neutral,
                    }
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

    fn create_test_data(closes: Vec<f64>) -> OHLCVSeries {
        let n = closes.len();
        OHLCVSeries {
            open: closes.clone(),
            high: closes.iter().map(|c| c + 1.0).collect(),
            low: closes.iter().map(|c| c - 1.0).collect(),
            close: closes,
            volume: vec![1000.0; n],
        }
    }

    #[test]
    fn test_buy_setup_detection() {
        // Create declining prices to trigger buy setup
        // Each close needs to be less than close 4 bars ago
        let closes = vec![
            100.0, 99.0, 98.0, 97.0,  // Initial bars (0-3)
            95.0,  // Bar 4: 95 < 100 (bar 0) - count 1
            94.0,  // Bar 5: 94 < 99 (bar 1) - count 2
            93.0,  // Bar 6: 93 < 98 (bar 2) - count 3
            92.0,  // Bar 7: 92 < 97 (bar 3) - count 4
            90.0,  // Bar 8: 90 < 95 (bar 4) - count 5
            89.0,  // Bar 9: 89 < 94 (bar 5) - count 6
            88.0,  // Bar 10: 88 < 93 (bar 6) - count 7
            87.0,  // Bar 11: 87 < 92 (bar 7) - count 8
            85.0,  // Bar 12: 85 < 90 (bar 8) - count 9 COMPLETE
        ];

        let data = create_test_data(closes);
        let setup = TDSetup::new();
        let result = setup.calculate(&data);

        // Check that setup completes at bar 12
        assert!(result.setup_complete[12]);
        assert_eq!(result.phase[12], SetupPhase::BuySetup);
        assert_eq!(result.count[12], 9);
    }

    #[test]
    fn test_sell_setup_detection() {
        // Create rising prices to trigger sell setup
        let closes = vec![
            100.0, 101.0, 102.0, 103.0,  // Initial bars
            105.0,  // Bar 4: 105 > 100 - count 1
            106.0,  // Bar 5: 106 > 101 - count 2
            107.0,  // Bar 6: 107 > 102 - count 3
            108.0,  // Bar 7: 108 > 103 - count 4
            110.0,  // Bar 8: 110 > 105 - count 5
            111.0,  // Bar 9: 111 > 106 - count 6
            112.0,  // Bar 10: 112 > 107 - count 7
            113.0,  // Bar 11: 113 > 108 - count 8
            115.0,  // Bar 12: 115 > 110 - count 9 COMPLETE
        ];

        let data = create_test_data(closes);
        let setup = TDSetup::new();
        let result = setup.calculate(&data);

        assert!(result.setup_complete[12]);
        assert_eq!(result.phase[12], SetupPhase::SellSetup);
        assert_eq!(result.count[12], 9);
    }

    #[test]
    fn test_setup_interruption() {
        // Setup gets interrupted before completing
        let closes = vec![
            100.0, 99.0, 98.0, 97.0,  // Initial bars
            95.0,  // count 1
            94.0,  // count 2
            93.0,  // count 3
            92.0,  // count 4
            95.0,  // BREAK: 95 > 94 (bar 5) - resets
            96.0,  // Start new sell setup
        ];

        let data = create_test_data(closes);
        let setup = TDSetup::new();
        let result = setup.calculate(&data);

        // No setup should complete
        assert!(!result.setup_complete.iter().any(|&x| x));

        // Check that setup was broken
        assert_eq!(result.count[7], 4);
        assert_eq!(result.count[8], 0); // Reset
    }

    #[test]
    fn test_signal_generation() {
        let closes = vec![
            100.0, 101.0, 102.0, 103.0,
            105.0, 106.0, 107.0, 108.0, 110.0,
            111.0, 112.0, 113.0, 115.0,
        ];

        let data = create_test_data(closes);
        let setup = TDSetup::new();
        let signals = setup.signals(&data).unwrap();

        // Sell setup completes at bar 12
        assert_eq!(signals[12], IndicatorSignal::Bearish);

        // Other bars should be neutral
        assert_eq!(signals[8], IndicatorSignal::Neutral);
    }

    #[test]
    fn test_insufficient_data() {
        let closes = vec![100.0, 99.0, 98.0]; // Only 3 bars
        let data = create_test_data(closes);
        let setup = TDSetup::new();

        let result = setup.compute(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_tdst_level() {
        let closes = vec![
            100.0, 101.0, 102.0, 103.0,
            105.0, 106.0, 107.0, 108.0, 110.0,
            111.0, 112.0, 113.0, 115.0,
        ];

        let data = create_test_data(closes);
        let setup = TDSetup::new();
        let result = setup.calculate(&data);

        // TDST level should be set at completion
        assert!(!result.tdst_level[12].is_nan());
    }
}
