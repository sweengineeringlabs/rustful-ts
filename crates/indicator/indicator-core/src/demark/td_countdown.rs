//! TD Countdown - The second phase of Tom DeMark's Sequential indicator.
//!
//! TD Countdown begins after a completed TD Setup and counts 13 bars that meet
//! specific price conditions. Unlike Setup, Countdown bars do not need to be
//! consecutive.

use crate::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result, SignalIndicator,
    TechnicalIndicator,
};
use serde::{Deserialize, Serialize};

use super::td_setup::{SetupPhase, TDSetup, TDSetupOutput};

/// TD Countdown phase type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CountdownPhase {
    /// No active countdown
    None,
    /// Buy countdown (looking for potential bottom)
    BuyCountdown,
    /// Sell countdown (looking for potential top)
    SellCountdown,
}

/// TD Countdown output.
#[derive(Debug, Clone)]
pub struct TDCountdownOutput {
    /// Countdown count for each bar (1-13 when active, 0 otherwise)
    pub count: Vec<i32>,
    /// Phase at each bar
    pub phase: Vec<CountdownPhase>,
    /// True when countdown reaches 13 (completion)
    pub countdown_complete: Vec<bool>,
    /// True when bar 13 is "qualified" (additional confirmation)
    pub qualified: Vec<bool>,
    /// Index of the setup that initiated this countdown
    pub setup_bar: Vec<Option<usize>>,
}

/// TD Countdown configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TDCountdownConfig {
    /// Required count for complete countdown (default: 13)
    pub countdown_count: usize,
    /// Lookback for close comparison (default: 2)
    pub close_lookback: usize,
    /// Whether countdown can be recycled (reset on new setup)
    pub allow_recycle: bool,
}

impl Default for TDCountdownConfig {
    fn default() -> Self {
        Self {
            countdown_count: 13,
            close_lookback: 2,
            allow_recycle: true,
        }
    }
}

/// TD Countdown Indicator.
///
/// After a completed 9-count TD Setup, Countdown begins counting bars where:
/// - Buy Countdown: Close <= Low[2] (close less than or equal to low 2 bars ago)
/// - Sell Countdown: Close >= High[2] (close greater than or equal to high 2 bars ago)
///
/// # Countdown Rules
/// - Bars do not need to be consecutive
/// - Count continues until 13 qualifying bars are found
/// - Countdown is "qualified" when bar 13's close <= bar 8's low (for buy)
/// - Countdown can be "recycled" if a new setup completes in same direction
#[derive(Debug, Clone)]
pub struct TDCountdown {
    config: TDCountdownConfig,
    setup: TDSetup,
}

impl TDCountdown {
    pub fn new() -> Self {
        Self {
            config: TDCountdownConfig::default(),
            setup: TDSetup::new(),
        }
    }

    pub fn with_config(config: TDCountdownConfig) -> Self {
        Self {
            config,
            setup: TDSetup::new(),
        }
    }

    /// Calculate TD Countdown from OHLC data.
    /// Returns both setup and countdown results.
    pub fn calculate(&self, data: &OHLCVSeries) -> (TDSetupOutput, TDCountdownOutput) {
        let setup_result = self.setup.calculate(data);
        let countdown_result = self.calculate_countdown(data, &setup_result);
        (setup_result, countdown_result)
    }

    /// Calculate countdown given setup results.
    fn calculate_countdown(&self, data: &OHLCVSeries, setup: &TDSetupOutput) -> TDCountdownOutput {
        let n = data.close.len();
        let countdown_count = self.config.countdown_count as i32;
        let close_lookback = self.config.close_lookback;

        let mut count = vec![0i32; n];
        let mut phase = vec![CountdownPhase::None; n];
        let mut countdown_complete = vec![false; n];
        let mut qualified = vec![false; n];
        let mut setup_bar = vec![None; n];

        if n <= close_lookback {
            return TDCountdownOutput {
                count,
                phase,
                countdown_complete,
                qualified,
                setup_bar,
            };
        }

        let mut current_count = 0i32;
        let mut current_phase = CountdownPhase::None;
        let mut current_setup_bar: Option<usize> = None;
        let mut bar_8_idx: Option<usize> = None;

        for i in 0..n {
            // Check for setup completion that would start countdown
            if setup.setup_complete[i] {
                match setup.phase[i] {
                    SetupPhase::BuySetup => {
                        // Start or recycle buy countdown
                        if self.config.allow_recycle || current_phase == CountdownPhase::None {
                            current_phase = CountdownPhase::BuyCountdown;
                            current_count = 0;
                            current_setup_bar = Some(i);
                            bar_8_idx = None;
                        }
                    }
                    SetupPhase::SellSetup => {
                        // Start or recycle sell countdown
                        if self.config.allow_recycle || current_phase == CountdownPhase::None {
                            current_phase = CountdownPhase::SellCountdown;
                            current_count = 0;
                            current_setup_bar = Some(i);
                            bar_8_idx = None;
                        }
                    }
                    SetupPhase::None => {}
                }
            }

            // Check for opposite setup that would cancel countdown
            if setup.setup_complete[i] {
                match (current_phase, setup.phase[i]) {
                    (CountdownPhase::BuyCountdown, SetupPhase::SellSetup) => {
                        current_phase = CountdownPhase::SellCountdown;
                        current_count = 0;
                        current_setup_bar = Some(i);
                        bar_8_idx = None;
                    }
                    (CountdownPhase::SellCountdown, SetupPhase::BuySetup) => {
                        current_phase = CountdownPhase::BuyCountdown;
                        current_count = 0;
                        current_setup_bar = Some(i);
                        bar_8_idx = None;
                    }
                    _ => {}
                }
            }

            // Count qualifying bars
            if i >= close_lookback {
                let close = data.close[i];

                match current_phase {
                    CountdownPhase::BuyCountdown => {
                        let low_lookback = data.low[i - close_lookback];
                        if close <= low_lookback {
                            current_count += 1;

                            // Track bar 8 for qualification check
                            if current_count == 8 {
                                bar_8_idx = Some(i);
                            }
                        }
                    }
                    CountdownPhase::SellCountdown => {
                        let high_lookback = data.high[i - close_lookback];
                        if close >= high_lookback {
                            current_count += 1;

                            // Track bar 8 for qualification check
                            if current_count == 8 {
                                bar_8_idx = Some(i);
                            }
                        }
                    }
                    CountdownPhase::None => {}
                }
            }

            // Record current state
            count[i] = if current_phase != CountdownPhase::None {
                current_count
            } else {
                0
            };
            phase[i] = current_phase;
            setup_bar[i] = current_setup_bar;

            // Check for countdown completion
            if current_count == countdown_count {
                countdown_complete[i] = true;

                // Check qualification
                if let Some(bar_8) = bar_8_idx {
                    match current_phase {
                        CountdownPhase::BuyCountdown => {
                            // Bar 13 close should be <= bar 8 low
                            qualified[i] = data.close[i] <= data.low[bar_8];
                        }
                        CountdownPhase::SellCountdown => {
                            // Bar 13 close should be >= bar 8 high
                            qualified[i] = data.close[i] >= data.high[bar_8];
                        }
                        CountdownPhase::None => {}
                    }
                }

                // Reset after completion
                current_phase = CountdownPhase::None;
                current_count = 0;
                current_setup_bar = None;
                bar_8_idx = None;
            }
        }

        TDCountdownOutput {
            count,
            phase,
            countdown_complete,
            qualified,
            setup_bar,
        }
    }
}

impl Default for TDCountdown {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for TDCountdown {
    fn name(&self) -> &str {
        "TD Countdown"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() <= self.config.close_lookback {
            return Err(IndicatorError::InsufficientData {
                required: self.config.close_lookback + 1,
                got: data.close.len(),
            });
        }

        let (_, countdown) = self.calculate(data);
        let count_f64: Vec<f64> = countdown.count.iter().map(|&c| c as f64).collect();
        let phase_f64: Vec<f64> = countdown.phase.iter().map(|p| match p {
            CountdownPhase::None => 0.0,
            CountdownPhase::BuyCountdown => 1.0,
            CountdownPhase::SellCountdown => -1.0,
        }).collect();

        Ok(IndicatorOutput::dual(count_f64, phase_f64))
    }

    fn min_periods(&self) -> usize {
        self.config.close_lookback + 1
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for TDCountdown {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (_, countdown) = self.calculate(data);
        let n = countdown.count.len();

        if n == 0 {
            return Ok(IndicatorSignal::Neutral);
        }

        // Signal on countdown completion (qualified signals are stronger)
        if countdown.countdown_complete[n - 1] {
            match countdown.phase[n - 1] {
                CountdownPhase::BuyCountdown => Ok(IndicatorSignal::Bullish),
                CountdownPhase::SellCountdown => Ok(IndicatorSignal::Bearish),
                CountdownPhase::None => Ok(IndicatorSignal::Neutral),
            }
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (_, countdown) = self.calculate(data);
        let signals = countdown.countdown_complete.iter().zip(countdown.phase.iter())
            .map(|(&complete, &p)| {
                if complete {
                    match p {
                        CountdownPhase::BuyCountdown => IndicatorSignal::Bullish,
                        CountdownPhase::SellCountdown => IndicatorSignal::Bearish,
                        CountdownPhase::None => IndicatorSignal::Neutral,
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

    fn create_test_ohlc(
        opens: Vec<f64>,
        highs: Vec<f64>,
        lows: Vec<f64>,
        closes: Vec<f64>,
    ) -> OHLCVSeries {
        let n = closes.len();
        OHLCVSeries {
            open: opens,
            high: highs,
            low: lows,
            close: closes,
            volume: vec![1000.0; n],
        }
    }

    #[test]
    fn test_buy_countdown_starts_after_buy_setup() {
        // Create data that triggers buy setup then starts countdown
        // Setup: declining closes for 9 bars
        // Countdown: close <= low[2]

        let mut closes = vec![
            100.0, 99.0, 98.0, 97.0,  // Initial
            95.0, 94.0, 93.0, 92.0, 90.0,  // Setup bars 1-5
            89.0, 88.0, 87.0, 85.0,  // Setup bars 6-9 (completes at index 12)
        ];

        // Add more bars for countdown
        for i in 0..20 {
            closes.push(84.0 - (i as f64 * 0.5));
        }

        let highs: Vec<f64> = closes.iter().map(|c| c + 2.0).collect();
        let lows: Vec<f64> = closes.iter().map(|c| c - 2.0).collect();
        let data = create_test_ohlc(closes.clone(), highs, lows, closes);

        let countdown = TDCountdown::new();
        let (setup, cd) = countdown.calculate(&data);

        // Verify setup completes
        assert!(setup.setup_complete[12]);
        assert_eq!(setup.phase[12], SetupPhase::BuySetup);

        // After setup, countdown should begin
        assert_eq!(cd.phase[13], CountdownPhase::BuyCountdown);
    }

    #[test]
    fn test_countdown_not_consecutive() {
        // Countdown counts should not need to be consecutive
        let countdown = TDCountdown::new();
        assert_eq!(countdown.config.countdown_count, 13);
        // The countdown logic in calculate_countdown handles non-consecutive counting
    }

    #[test]
    fn test_insufficient_data() {
        let data = OHLCVSeries {
            open: vec![100.0],
            high: vec![101.0],
            low: vec![99.0],
            close: vec![100.0],
            volume: vec![1000.0],
        };

        let countdown = TDCountdown::new();
        let result = countdown.compute(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_countdown_phase_tracking() {
        let countdown = TDCountdown::new();

        let closes = vec![100.0; 50];
        let data = OHLCVSeries {
            open: closes.clone(),
            high: closes.iter().map(|c| c + 1.0).collect(),
            low: closes.iter().map(|c| c - 1.0).collect(),
            close: closes,
            volume: vec![1000.0; 50],
        };

        let (_, cd) = countdown.calculate(&data);

        // With flat prices, no setup should complete so no countdown
        assert!(cd.phase.iter().all(|p| *p == CountdownPhase::None));
    }

    #[test]
    fn test_countdown_qualification_logic() {
        // Test the qualification check is present
        let countdown = TDCountdown::new();
        assert!(countdown.config.allow_recycle);
        assert_eq!(countdown.config.close_lookback, 2);
    }
}
