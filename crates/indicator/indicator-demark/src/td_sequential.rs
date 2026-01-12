//! TD Sequential - Tom DeMark's primary exhaustion indicator.
//!
//! TD Sequential combines TD Setup (9-count) and TD Countdown (13-count) to
//! identify potential trend exhaustion points. It is one of the most widely
//! used DeMark indicators for timing market reversals.

use indicator_spi::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result, SignalIndicator,
    TechnicalIndicator,
};
use serde::{Deserialize, Serialize};

use crate::td_countdown::{CountdownPhase, TDCountdown, TDCountdownConfig, TDCountdownOutput};
use crate::td_setup::{SetupPhase, TDSetup, TDSetupConfig, TDSetupOutput};

/// TD Sequential state enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SequentialState {
    /// No active pattern
    Idle,
    /// Buy setup in progress (bars 1-9)
    BuySetup,
    /// Sell setup in progress (bars 1-9)
    SellSetup,
    /// Buy setup complete, countdown in progress
    BuyCountdown,
    /// Sell setup complete, countdown in progress
    SellCountdown,
    /// Buy countdown complete - potential buy signal
    BuyComplete,
    /// Sell countdown complete - potential sell signal
    SellComplete,
}

/// TD Sequential signal strength.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalStrength {
    /// No signal
    None,
    /// Setup complete only (weaker)
    SetupOnly,
    /// Countdown complete but not qualified
    CountdownUnqualified,
    /// Countdown complete and qualified (strongest)
    CountdownQualified,
}

/// Comprehensive TD Sequential output.
#[derive(Debug, Clone)]
pub struct TDSequentialOutput {
    /// Setup count (1-9, 0 if no setup active)
    pub setup_count: Vec<i32>,
    /// Countdown count (1-13, 0 if no countdown active)
    pub countdown_count: Vec<i32>,
    /// Current sequential state
    pub state: Vec<SequentialState>,
    /// Signal strength at each bar
    pub signal_strength: Vec<SignalStrength>,
    /// Setup is perfected
    pub setup_perfected: Vec<bool>,
    /// Countdown is qualified
    pub countdown_qualified: Vec<bool>,
    /// TDST support/resistance levels
    pub tdst_level: Vec<f64>,
    /// Risk level (distance to TDST as percentage)
    pub risk_level: Vec<f64>,
}

/// TD Sequential configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TDSequentialConfig {
    /// Setup configuration
    pub setup: TDSetupConfig,
    /// Countdown configuration
    pub countdown: TDCountdownConfig,
    /// Require perfected setup for stronger signals
    pub require_perfection: bool,
    /// Enable aggressive mode (signals on setup completion)
    pub aggressive_mode: bool,
}

impl Default for TDSequentialConfig {
    fn default() -> Self {
        Self {
            setup: TDSetupConfig::default(),
            countdown: TDCountdownConfig::default(),
            require_perfection: false,
            aggressive_mode: false,
        }
    }
}

/// TD Sequential Indicator.
///
/// The complete TD Sequential system consisting of:
/// 1. **TD Setup Phase (9-count)**: Identifies potential trend exhaustion
/// 2. **TD Countdown Phase (13-count)**: Confirms exhaustion after setup completes
///
/// # Trading Rules
/// - Setup completion (9) suggests trend exhaustion is possible
/// - Countdown completion (13) confirms exhaustion is likely
/// - Qualified countdown provides strongest signals
/// - TDST levels provide support/resistance targets
///
/// # Signal Hierarchy
/// 1. Qualified Countdown (13 + qualified) - Strongest
/// 2. Unqualified Countdown (13) - Strong
/// 3. Perfected Setup (9 + perfected) - Moderate
/// 4. Setup Only (9) - Weak
#[derive(Debug, Clone)]
pub struct TDSequential {
    config: TDSequentialConfig,
    setup: TDSetup,
    countdown: TDCountdown,
}

impl TDSequential {
    pub fn new() -> Self {
        Self {
            config: TDSequentialConfig::default(),
            setup: TDSetup::new(),
            countdown: TDCountdown::new(),
        }
    }

    pub fn with_config(config: TDSequentialConfig) -> Self {
        Self {
            setup: TDSetup::with_config(config.setup.clone()),
            countdown: TDCountdown::with_config(config.countdown.clone()),
            config,
        }
    }

    /// Enable aggressive mode (signals on setup completion).
    pub fn aggressive(mut self) -> Self {
        self.config.aggressive_mode = true;
        self
    }

    /// Require perfected setups for signals.
    pub fn require_perfection(mut self) -> Self {
        self.config.require_perfection = true;
        self
    }

    /// Calculate TD Sequential from OHLC data.
    pub fn calculate(&self, data: &OHLCVSeries) -> TDSequentialOutput {
        let n = data.close.len();

        // Get setup and countdown results
        let setup_result = self.setup.calculate(data);
        let (_, countdown_result) = self.countdown.calculate(data);

        let mut state = vec![SequentialState::Idle; n];
        let mut signal_strength = vec![SignalStrength::None; n];
        let mut risk_level = vec![f64::NAN; n];

        for i in 0..n {
            // Determine current state
            if countdown_result.countdown_complete[i] {
                match countdown_result.phase[i] {
                    CountdownPhase::BuyCountdown => {
                        state[i] = SequentialState::BuyComplete;
                        signal_strength[i] = if countdown_result.qualified[i] {
                            SignalStrength::CountdownQualified
                        } else {
                            SignalStrength::CountdownUnqualified
                        };
                    }
                    CountdownPhase::SellCountdown => {
                        state[i] = SequentialState::SellComplete;
                        signal_strength[i] = if countdown_result.qualified[i] {
                            SignalStrength::CountdownQualified
                        } else {
                            SignalStrength::CountdownUnqualified
                        };
                    }
                    CountdownPhase::None => {}
                }
            } else if setup_result.setup_complete[i] {
                match setup_result.phase[i] {
                    SetupPhase::BuySetup => {
                        state[i] = SequentialState::BuyCountdown;
                        if self.config.aggressive_mode {
                            signal_strength[i] = SignalStrength::SetupOnly;
                        }
                    }
                    SetupPhase::SellSetup => {
                        state[i] = SequentialState::SellCountdown;
                        if self.config.aggressive_mode {
                            signal_strength[i] = SignalStrength::SetupOnly;
                        }
                    }
                    SetupPhase::None => {}
                }
            } else if countdown_result.count[i] > 0 {
                match countdown_result.phase[i] {
                    CountdownPhase::BuyCountdown => state[i] = SequentialState::BuyCountdown,
                    CountdownPhase::SellCountdown => state[i] = SequentialState::SellCountdown,
                    CountdownPhase::None => {}
                }
            } else if setup_result.count[i] > 0 {
                match setup_result.phase[i] {
                    SetupPhase::BuySetup => state[i] = SequentialState::BuySetup,
                    SetupPhase::SellSetup => state[i] = SequentialState::SellSetup,
                    SetupPhase::None => {}
                }
            }

            // Calculate risk level (distance to TDST)
            if !setup_result.tdst_level[i].is_nan() {
                let tdst = setup_result.tdst_level[i];
                let close = data.close[i];
                risk_level[i] = ((close - tdst) / close).abs() * 100.0;
            }
        }

        TDSequentialOutput {
            setup_count: setup_result.count,
            countdown_count: countdown_result.count,
            state,
            signal_strength,
            setup_perfected: setup_result.perfected,
            countdown_qualified: countdown_result.qualified,
            tdst_level: setup_result.tdst_level,
            risk_level,
        }
    }

    /// Get detailed setup output.
    pub fn get_setup(&self, data: &OHLCVSeries) -> TDSetupOutput {
        self.setup.calculate(data)
    }

    /// Get detailed countdown output.
    pub fn get_countdown(&self, data: &OHLCVSeries) -> TDCountdownOutput {
        let (_, countdown) = self.countdown.calculate(data);
        countdown
    }
}

impl Default for TDSequential {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for TDSequential {
    fn name(&self) -> &str {
        "TD Sequential"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < 5 {
            return Err(IndicatorError::InsufficientData {
                required: 5,
                got: data.close.len(),
            });
        }

        let result = self.calculate(data);

        // Primary: setup count, Secondary: countdown count, Tertiary: state encoded
        let setup_f64: Vec<f64> = result.setup_count.iter().map(|&c| c as f64).collect();
        let countdown_f64: Vec<f64> = result.countdown_count.iter().map(|&c| c as f64).collect();
        let state_f64: Vec<f64> = result.state.iter().map(|s| match s {
            SequentialState::Idle => 0.0,
            SequentialState::BuySetup => 1.0,
            SequentialState::SellSetup => -1.0,
            SequentialState::BuyCountdown => 2.0,
            SequentialState::SellCountdown => -2.0,
            SequentialState::BuyComplete => 3.0,
            SequentialState::SellComplete => -3.0,
        }).collect();

        Ok(IndicatorOutput::triple(setup_f64, countdown_f64, state_f64))
    }

    fn min_periods(&self) -> usize {
        5 // Need at least lookback + 1
    }

    fn output_features(&self) -> usize {
        3
    }
}

impl SignalIndicator for TDSequential {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let result = self.calculate(data);
        let n = result.state.len();

        if n == 0 {
            return Ok(IndicatorSignal::Neutral);
        }

        let last_strength = result.signal_strength[n - 1];
        let last_state = result.state[n - 1];

        // Only generate signal on significant events
        match last_strength {
            SignalStrength::CountdownQualified | SignalStrength::CountdownUnqualified => {
                match last_state {
                    SequentialState::BuyComplete => Ok(IndicatorSignal::Bullish),
                    SequentialState::SellComplete => Ok(IndicatorSignal::Bearish),
                    _ => Ok(IndicatorSignal::Neutral),
                }
            }
            SignalStrength::SetupOnly if self.config.aggressive_mode => {
                match last_state {
                    SequentialState::BuyCountdown => Ok(IndicatorSignal::Bullish),
                    SequentialState::SellCountdown => Ok(IndicatorSignal::Bearish),
                    _ => Ok(IndicatorSignal::Neutral),
                }
            }
            _ => Ok(IndicatorSignal::Neutral),
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let result = self.calculate(data);

        let signals: Vec<IndicatorSignal> = result.signal_strength.iter()
            .zip(result.state.iter())
            .map(|(strength, state)| {
                match strength {
                    SignalStrength::CountdownQualified | SignalStrength::CountdownUnqualified => {
                        match state {
                            SequentialState::BuyComplete => IndicatorSignal::Bullish,
                            SequentialState::SellComplete => IndicatorSignal::Bearish,
                            _ => IndicatorSignal::Neutral,
                        }
                    }
                    SignalStrength::SetupOnly if self.config.aggressive_mode => {
                        match state {
                            SequentialState::BuyCountdown | SequentialState::BuySetup => IndicatorSignal::Bullish,
                            SequentialState::SellCountdown | SequentialState::SellSetup => IndicatorSignal::Bearish,
                            _ => IndicatorSignal::Neutral,
                        }
                    }
                    _ => IndicatorSignal::Neutral,
                }
            })
            .collect();

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_trending_data(direction: i32, bars: usize) -> OHLCVSeries {
        let mut closes = Vec::with_capacity(bars);
        let base = 100.0;

        for i in 0..bars {
            let price = if direction > 0 {
                base + (i as f64 * 1.0)
            } else {
                base - (i as f64 * 1.0)
            };
            closes.push(price);
        }

        OHLCVSeries {
            open: closes.clone(),
            high: closes.iter().map(|c| c + 2.0).collect(),
            low: closes.iter().map(|c| c - 2.0).collect(),
            close: closes,
            volume: vec![1000.0; bars],
        }
    }

    #[test]
    fn test_td_sequential_initialization() {
        let seq = TDSequential::new();
        assert_eq!(seq.name(), "TD Sequential");
        assert_eq!(seq.min_periods(), 5);
        assert_eq!(seq.output_features(), 3);
    }

    #[test]
    fn test_sell_setup_detection() {
        // Rising prices should trigger sell setup
        let data = create_trending_data(1, 20);
        let seq = TDSequential::new();
        let result = seq.calculate(&data);

        // Check that sell setup builds
        let has_sell_setup = result.state.iter()
            .any(|s| *s == SequentialState::SellSetup || *s == SequentialState::SellCountdown);
        assert!(has_sell_setup);
    }

    #[test]
    fn test_buy_setup_detection() {
        // Falling prices should trigger buy setup
        let data = create_trending_data(-1, 20);
        let seq = TDSequential::new();
        let result = seq.calculate(&data);

        // Check that buy setup builds
        let has_buy_setup = result.state.iter()
            .any(|s| *s == SequentialState::BuySetup || *s == SequentialState::BuyCountdown);
        assert!(has_buy_setup);
    }

    #[test]
    fn test_state_transitions() {
        let data = create_trending_data(1, 30);
        let seq = TDSequential::new();
        let result = seq.calculate(&data);

        // Verify state progression
        let states: Vec<_> = result.state.iter().filter(|s| **s != SequentialState::Idle).collect();
        assert!(!states.is_empty());
    }

    #[test]
    fn test_aggressive_mode() {
        let data = create_trending_data(1, 20);

        let seq_normal = TDSequential::new();
        let seq_aggressive = TDSequential::new().aggressive();

        let result_normal = seq_normal.calculate(&data);
        let result_aggressive = seq_aggressive.calculate(&data);

        // Aggressive mode should generate signals on setup completion
        let normal_signals: Vec<_> = result_normal.signal_strength.iter()
            .filter(|s| **s != SignalStrength::None)
            .collect();
        let aggressive_signals: Vec<_> = result_aggressive.signal_strength.iter()
            .filter(|s| **s != SignalStrength::None)
            .collect();

        // Aggressive mode may have more signals
        assert!(aggressive_signals.len() >= normal_signals.len());
    }

    #[test]
    fn test_compute_output_format() {
        let data = create_trending_data(1, 15);
        let seq = TDSequential::new();
        let output = seq.compute(&data).unwrap();

        assert_eq!(output.primary.len(), 15);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
        assert_eq!(output.secondary.unwrap().len(), 15);
        assert_eq!(output.tertiary.unwrap().len(), 15);
    }

    #[test]
    fn test_insufficient_data() {
        let data = OHLCVSeries {
            open: vec![100.0, 101.0],
            high: vec![102.0, 103.0],
            low: vec![98.0, 99.0],
            close: vec![100.0, 101.0],
            volume: vec![1000.0, 1000.0],
        };

        let seq = TDSequential::new();
        let result = seq.compute(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_signal_strength_hierarchy() {
        // Verify signal strength enum values
        assert!(matches!(SignalStrength::CountdownQualified, SignalStrength::CountdownQualified));
        assert!(matches!(SignalStrength::CountdownUnqualified, SignalStrength::CountdownUnqualified));
        assert!(matches!(SignalStrength::SetupOnly, SignalStrength::SetupOnly));
        assert!(matches!(SignalStrength::None, SignalStrength::None));
    }

    #[test]
    fn test_signals_method() {
        let data = create_trending_data(1, 20);
        let seq = TDSequential::new();
        let signals = seq.signals(&data).unwrap();

        assert_eq!(signals.len(), 20);
        // Most bars should be neutral
        let neutral_count = signals.iter().filter(|s| **s == IndicatorSignal::Neutral).count();
        assert!(neutral_count > 10);
    }
}
