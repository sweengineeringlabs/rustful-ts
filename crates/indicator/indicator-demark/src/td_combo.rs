//! TD Combo - Alternative DeMark counting method.
//!
//! TD Combo uses different conditions for counting compared to TD Sequential.
//! Instead of comparing to close[4], TD Combo compares to close[2] for setup
//! and has different countdown conditions.

use indicator_spi::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result, SignalIndicator,
    TechnicalIndicator,
};
use serde::{Deserialize, Serialize};

/// TD Combo phase type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComboPhase {
    /// No active combo
    None,
    /// Buy combo (bearish trend potentially exhausting)
    BuyCombo,
    /// Sell combo (bullish trend potentially exhausting)
    SellCombo,
}

/// TD Combo state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComboState {
    /// No pattern active
    Idle,
    /// Setup phase in progress
    Setup,
    /// Countdown phase in progress
    Countdown,
    /// Pattern complete
    Complete,
}

/// TD Combo output.
#[derive(Debug, Clone)]
pub struct TDComboOutput {
    /// Setup count (1-9)
    pub setup_count: Vec<i32>,
    /// Countdown count (1-13)
    pub countdown_count: Vec<i32>,
    /// Current phase
    pub phase: Vec<ComboPhase>,
    /// Current state
    pub state: Vec<ComboState>,
    /// True when pattern completes
    pub complete: Vec<bool>,
    /// True when setup is perfected
    pub setup_perfected: Vec<bool>,
}

/// TD Combo configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TDComboConfig {
    /// Lookback for setup comparison (default: 2)
    pub setup_lookback: usize,
    /// Required setup count (default: 9)
    pub setup_count: usize,
    /// Required countdown count (default: 13)
    pub countdown_count: usize,
    /// Countdown can run concurrent with setup
    pub concurrent_countdown: bool,
}

impl Default for TDComboConfig {
    fn default() -> Self {
        Self {
            setup_lookback: 2,
            setup_count: 9,
            countdown_count: 13,
            concurrent_countdown: true,
        }
    }
}

/// TD Combo Indicator.
///
/// TD Combo differs from TD Sequential in several ways:
/// - Setup compares close to close[2] instead of close[4]
/// - Countdown can begin before setup completes (concurrent)
/// - Countdown conditions: close < low[2] (buy) or close > high[2] (sell)
///
/// # Combo Rules
/// - Buy Combo Setup: Close < Close[2] for 9 bars
/// - Sell Combo Setup: Close > Close[2] for 9 bars
/// - Buy Combo Countdown: Close < Low[2] (not necessarily consecutive)
/// - Sell Combo Countdown: Close > High[2]
/// - Countdown can accumulate alongside setup
#[derive(Debug, Clone)]
pub struct TDCombo {
    config: TDComboConfig,
}

impl TDCombo {
    pub fn new() -> Self {
        Self {
            config: TDComboConfig::default(),
        }
    }

    pub fn with_config(config: TDComboConfig) -> Self {
        Self { config }
    }

    /// Calculate TD Combo from OHLC data.
    pub fn calculate(&self, data: &OHLCVSeries) -> TDComboOutput {
        let n = data.close.len();
        let setup_lookback = self.config.setup_lookback;
        let setup_target = self.config.setup_count as i32;
        let countdown_target = self.config.countdown_count as i32;

        let mut setup_count = vec![0i32; n];
        let mut countdown_count = vec![0i32; n];
        let mut phase = vec![ComboPhase::None; n];
        let mut state = vec![ComboState::Idle; n];
        let mut complete = vec![false; n];
        let mut setup_perfected = vec![false; n];

        if n <= setup_lookback {
            return TDComboOutput {
                setup_count,
                countdown_count,
                phase,
                state,
                complete,
                setup_perfected,
            };
        }

        // Track active combo
        let mut current_phase = ComboPhase::None;
        let mut current_setup_count = 0i32;
        let mut current_countdown_count = 0i32;
        let mut setup_complete_bar: Option<usize> = None;

        for i in setup_lookback..n {
            let close = data.close[i];
            let close_lookback = data.close[i - setup_lookback];
            let low_lookback = data.low[i - setup_lookback];
            let high_lookback = data.high[i - setup_lookback];

            // Check for setup conditions
            let is_buy_setup_bar = close < close_lookback;
            let is_sell_setup_bar = close > close_lookback;

            // Check for countdown conditions
            let is_buy_countdown_bar = close < low_lookback;
            let is_sell_countdown_bar = close > high_lookback;

            // Handle setup logic
            match current_phase {
                ComboPhase::None => {
                    if is_buy_setup_bar {
                        current_phase = ComboPhase::BuyCombo;
                        current_setup_count = 1;
                        current_countdown_count = 0;
                        setup_complete_bar = None;
                    } else if is_sell_setup_bar {
                        current_phase = ComboPhase::SellCombo;
                        current_setup_count = 1;
                        current_countdown_count = 0;
                        setup_complete_bar = None;
                    }
                }
                ComboPhase::BuyCombo => {
                    // Continue or break setup
                    if is_buy_setup_bar && current_setup_count < setup_target {
                        current_setup_count += 1;
                    } else if is_sell_setup_bar {
                        // Flip to sell
                        current_phase = ComboPhase::SellCombo;
                        current_setup_count = 1;
                        current_countdown_count = 0;
                        setup_complete_bar = None;
                    }

                    // Concurrent countdown (can start before setup completes)
                    if self.config.concurrent_countdown && is_buy_countdown_bar {
                        current_countdown_count += 1;
                    } else if !self.config.concurrent_countdown && setup_complete_bar.is_some() && is_buy_countdown_bar {
                        current_countdown_count += 1;
                    }

                    // Check setup completion
                    if current_setup_count == setup_target && setup_complete_bar.is_none() {
                        setup_complete_bar = Some(i);

                        // Check perfection
                        if i >= 3 {
                            let bar_8_low = data.low[i - 1];
                            let bar_9_low = data.low[i];
                            let bar_6_low = data.low[i - 3];
                            let bar_7_low = data.low[i - 2];
                            let reference = bar_6_low.min(bar_7_low);
                            setup_perfected[i] = bar_8_low <= reference || bar_9_low <= reference;
                        }
                    }
                }
                ComboPhase::SellCombo => {
                    // Continue or break setup
                    if is_sell_setup_bar && current_setup_count < setup_target {
                        current_setup_count += 1;
                    } else if is_buy_setup_bar {
                        // Flip to buy
                        current_phase = ComboPhase::BuyCombo;
                        current_setup_count = 1;
                        current_countdown_count = 0;
                        setup_complete_bar = None;
                    }

                    // Concurrent countdown
                    if self.config.concurrent_countdown && is_sell_countdown_bar {
                        current_countdown_count += 1;
                    } else if !self.config.concurrent_countdown && setup_complete_bar.is_some() && is_sell_countdown_bar {
                        current_countdown_count += 1;
                    }

                    // Check setup completion
                    if current_setup_count == setup_target && setup_complete_bar.is_none() {
                        setup_complete_bar = Some(i);

                        // Check perfection
                        if i >= 3 {
                            let bar_8_high = data.high[i - 1];
                            let bar_9_high = data.high[i];
                            let bar_6_high = data.high[i - 3];
                            let bar_7_high = data.high[i - 2];
                            let reference = bar_6_high.max(bar_7_high);
                            setup_perfected[i] = bar_8_high >= reference || bar_9_high >= reference;
                        }
                    }
                }
            }

            // Record state
            setup_count[i] = current_setup_count.min(setup_target);
            countdown_count[i] = current_countdown_count.min(countdown_target);
            phase[i] = current_phase;

            // Determine state
            if current_countdown_count >= countdown_target {
                state[i] = ComboState::Complete;
                complete[i] = true;

                // Reset for new pattern
                current_phase = ComboPhase::None;
                current_setup_count = 0;
                current_countdown_count = 0;
                setup_complete_bar = None;
            } else if setup_complete_bar.is_some() {
                state[i] = ComboState::Countdown;
            } else if current_setup_count > 0 {
                state[i] = ComboState::Setup;
            }
        }

        TDComboOutput {
            setup_count,
            countdown_count,
            phase,
            state,
            complete,
            setup_perfected,
        }
    }
}

impl Default for TDCombo {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for TDCombo {
    fn name(&self) -> &str {
        "TD Combo"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() <= self.config.setup_lookback {
            return Err(IndicatorError::InsufficientData {
                required: self.config.setup_lookback + 1,
                got: data.close.len(),
            });
        }

        let result = self.calculate(data);

        // Encode: primary = setup count, secondary = countdown count
        let setup_f64: Vec<f64> = result.setup_count.iter().map(|&c| c as f64).collect();
        let countdown_f64: Vec<f64> = result.countdown_count.iter().map(|&c| c as f64).collect();

        Ok(IndicatorOutput::dual(setup_f64, countdown_f64))
    }

    fn min_periods(&self) -> usize {
        self.config.setup_lookback + 1
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for TDCombo {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let result = self.calculate(data);
        let n = result.phase.len();

        if n == 0 {
            return Ok(IndicatorSignal::Neutral);
        }

        if result.complete[n - 1] {
            match result.phase[n - 1] {
                ComboPhase::BuyCombo => Ok(IndicatorSignal::Bullish),
                ComboPhase::SellCombo => Ok(IndicatorSignal::Bearish),
                ComboPhase::None => Ok(IndicatorSignal::Neutral),
            }
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let result = self.calculate(data);

        let signals = result.complete.iter().zip(result.phase.iter())
            .map(|(&complete, &p)| {
                if complete {
                    match p {
                        ComboPhase::BuyCombo => IndicatorSignal::Bullish,
                        ComboPhase::SellCombo => IndicatorSignal::Bearish,
                        ComboPhase::None => IndicatorSignal::Neutral,
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
            high: closes.iter().map(|c| c + 2.0).collect(),
            low: closes.iter().map(|c| c - 2.0).collect(),
            close: closes,
            volume: vec![1000.0; n],
        }
    }

    #[test]
    fn test_combo_initialization() {
        let combo = TDCombo::new();
        assert_eq!(combo.name(), "TD Combo");
        assert_eq!(combo.config.setup_lookback, 2);
        assert_eq!(combo.config.setup_count, 9);
        assert_eq!(combo.config.countdown_count, 13);
    }

    #[test]
    fn test_buy_combo_detection() {
        // Declining prices - compare to close[2]
        let closes: Vec<f64> = (0..20).map(|i| 100.0 - (i as f64)).collect();
        let data = create_test_data(closes);

        let combo = TDCombo::new();
        let result = combo.calculate(&data);

        // Should build buy combo
        let has_buy = result.phase.iter().any(|p| *p == ComboPhase::BuyCombo);
        assert!(has_buy);
    }

    #[test]
    fn test_sell_combo_detection() {
        // Rising prices
        let closes: Vec<f64> = (0..20).map(|i| 100.0 + (i as f64)).collect();
        let data = create_test_data(closes);

        let combo = TDCombo::new();
        let result = combo.calculate(&data);

        // Should build sell combo
        let has_sell = result.phase.iter().any(|p| *p == ComboPhase::SellCombo);
        assert!(has_sell);
    }

    #[test]
    fn test_concurrent_countdown() {
        let config = TDComboConfig {
            concurrent_countdown: true,
            ..Default::default()
        };
        let combo = TDCombo::with_config(config);

        // Strong downtrend where countdown conditions can be met during setup
        let closes: Vec<f64> = (0..25).map(|i| 100.0 - (i as f64 * 2.0)).collect();
        let data = create_test_data(closes);

        let result = combo.calculate(&data);

        // Countdown should start accumulating
        let has_countdown = result.countdown_count.iter().any(|&c| c > 0);
        assert!(has_countdown);
    }

    #[test]
    fn test_state_tracking() {
        let closes: Vec<f64> = (0..15).map(|i| 100.0 + (i as f64)).collect();
        let data = create_test_data(closes);

        let combo = TDCombo::new();
        let result = combo.calculate(&data);

        // Should have setup state
        let has_setup_state = result.state.iter().any(|s| *s == ComboState::Setup);
        assert!(has_setup_state);
    }

    #[test]
    fn test_insufficient_data() {
        let data = create_test_data(vec![100.0, 101.0]);
        let combo = TDCombo::new();
        let result = combo.compute(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_output() {
        let closes: Vec<f64> = (0..15).map(|i| 100.0 + (i as f64)).collect();
        let data = create_test_data(closes);

        let combo = TDCombo::new();
        let output = combo.compute(&data).unwrap();

        assert_eq!(output.primary.len(), 15);
        assert!(output.secondary.is_some());
    }

    #[test]
    fn test_signals_method() {
        let closes: Vec<f64> = (0..15).map(|i| 100.0 + (i as f64)).collect();
        let data = create_test_data(closes);

        let combo = TDCombo::new();
        let signals = combo.signals(&data).unwrap();

        assert_eq!(signals.len(), 15);
    }
}
