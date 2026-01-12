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

// Re-exports
pub use td_setup::TDSetup;
pub use td_countdown::TDCountdown;
pub use td_sequential::TDSequential;
pub use td_combo::TDCombo;
pub use td_rei::TDREI;
pub use td_poq::TDPOQ;
pub use td_pressure::TDPressure;
pub use td_dwave::TDDWave;
pub use td_trend_factor::TDTrendFactor;

// Re-export SPI types
pub use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCV, OHLCVSeries,
};
