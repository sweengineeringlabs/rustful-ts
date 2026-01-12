//! Band and Channel Indicators
//!
//! Price band and channel indicators.

pub mod starc;
pub mod envelope;
pub mod chandelier;
pub mod price_channel;
pub mod std_error_bands;
pub mod acceleration_bands;
pub mod tirone;
pub mod projection_bands;
pub mod high_low_bands;

// Re-exports
pub use starc::STARCBands;
pub use envelope::Envelope;
pub use chandelier::ChandelierExit;
pub use price_channel::PriceChannel;
pub use std_error_bands::StandardErrorBands;
pub use acceleration_bands::AccelerationBands;
pub use tirone::TironeLevels;
pub use projection_bands::ProjectionBands;
pub use high_low_bands::HighLowBands;

// Re-export SPI types
pub use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCV, OHLCVSeries,
};
