//! Risk Metrics
//!
//! Risk and performance measurement indicators.

pub mod sharpe;
pub mod sortino;
pub mod calmar;
pub mod max_drawdown;
pub mod var;
pub mod cvar;
pub mod beta;
pub mod alpha;
pub mod treynor;
pub mod information_ratio;
pub mod omega;
pub mod gain_loss;

// Re-exports
pub use sharpe::SharpeRatio;
pub use sortino::SortinoRatio;
pub use calmar::CalmarRatio;
pub use max_drawdown::MaxDrawdown;
pub use var::ValueAtRisk;
pub use cvar::ConditionalVaR;
pub use beta::Beta;
pub use alpha::Alpha;
pub use treynor::TreynorRatio;
pub use information_ratio::InformationRatio;
pub use omega::OmegaRatio;
pub use gain_loss::GainLossRatio;

// Re-export SPI types
pub use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCV, OHLCVSeries,
};
