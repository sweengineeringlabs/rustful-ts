//! Statistical Indicators
//!
//! Statistical measures and analysis tools.

pub mod std_dev;
pub mod variance;
pub mod zscore;
pub mod linear_regression;
pub mod correlation;
pub mod spread;
pub mod ratio;
pub mod zscore_spread;
pub mod autocorrelation;
pub mod skewness;
pub mod kurtosis;

// Re-exports
pub use std_dev::StandardDeviation;
pub use variance::Variance;
pub use zscore::ZScore;
pub use linear_regression::{LinearRegression, LinearRegressionOutput};
pub use correlation::Correlation;
pub use spread::Spread;
pub use ratio::Ratio;
pub use zscore_spread::ZScoreSpread;
pub use autocorrelation::Autocorrelation;
pub use skewness::Skewness;
pub use kurtosis::Kurtosis;

// Re-export SPI types
pub use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCV, OHLCVSeries,
};
