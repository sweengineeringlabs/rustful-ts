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
pub mod fractal_dimension;
pub mod hurst;
pub mod dfa;
pub mod entropy;

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
pub use fractal_dimension::{FractalDimension, FractalDimensionMethod};
pub use hurst::{HurstExponent, HurstMethod};
pub use dfa::DetrendedFluctuationAnalysis;
pub use entropy::{MarketEntropy, EntropyMethod};
