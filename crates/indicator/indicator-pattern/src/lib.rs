//! Pattern Recognition Indicators
//!
//! Pattern detection and candlestick analysis.

pub mod zigzag;
pub mod heikin_ashi;
pub mod darvas;
pub mod fractals;
pub mod doji;
pub mod hammer;
pub mod engulfing;
pub mod harami;
pub mod morning_star;
pub mod three_soldiers;
pub mod marubozu;

// Re-exports
pub use zigzag::ZigZag;
pub use heikin_ashi::{HeikinAshi, HeikinAshiOutput};
pub use darvas::DarvasBox;
pub use fractals::Fractals;
pub use doji::Doji;
pub use hammer::Hammer;
pub use engulfing::Engulfing;
pub use harami::Harami;
pub use morning_star::MorningStar;
pub use three_soldiers::ThreeSoldiers;
pub use marubozu::Marubozu;

// Re-export SPI types
pub use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCV, OHLCVSeries,
};
