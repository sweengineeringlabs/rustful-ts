//! Tom DeMark Technical Indicators
//!
//! Implementation of Tom DeMark's technical analysis indicators including
//! TD Sequential, TD Combo, TD REI, and more.

pub mod td_setup;
pub mod td_countdown;
pub mod td_sequential;
pub mod td_combo;
pub mod td_rei;
pub mod td_poq;
pub mod td_pressure;
pub mod td_dwave;
pub mod td_trend_factor;
pub mod extended;

// Re-exports from td_setup
pub use td_setup::{TDSetup, TDSetupOutput, TDSetupConfig, SetupPhase};

// Re-exports from td_countdown
pub use td_countdown::{TDCountdown, TDCountdownOutput, TDCountdownConfig, CountdownPhase};

// Re-exports from td_sequential
pub use td_sequential::{TDSequential, TDSequentialOutput, TDSequentialConfig, SequentialState, SignalStrength};

// Re-exports from td_combo
pub use td_combo::{TDCombo, TDComboOutput, TDComboConfig, ComboPhase, ComboState};

// Re-exports from td_rei
pub use td_rei::{TDREI, TDREIOutput, TDREIConfig};

// Re-exports from td_poq
pub use td_poq::{TDPOQ, TDPOQOutput, TDPOQConfig};

// Re-exports from td_pressure
pub use td_pressure::{TDPressure, TDPressureOutput, TDPressureConfig};

// Re-exports from td_dwave
pub use td_dwave::{TDDWave, TDDWaveOutput, TDDWaveConfig, WaveDirection, DWavePhase, PivotType};

// Re-exports from td_trend_factor
pub use td_trend_factor::{TDTrendFactor, TDTrendFactorOutput, TDTrendFactorConfig, TrendState};

// Re-exports from extended
pub use extended::{
    TDCamouflage, TDCLOP, TDMovingAverageQualifier,
    TDRiskLevel, TDMomentum, TDDifferential,
};
