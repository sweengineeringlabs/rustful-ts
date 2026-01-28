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
pub mod mean;
pub mod median;
pub mod descriptive;
pub mod tests_stat;
pub mod diagnostics;
pub mod stationarity;
pub mod seasonal;
pub mod extended;
pub mod advanced;

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
pub use mean::Mean;
pub use median::Median;
pub use descriptive::{
    Mode, Range, CoefficientOfVariation, Percentile, Quartiles, QuartilesOutput, IQR,
};
pub use tests_stat::{
    MAD, TStatistic, PValue, ConfidenceInterval, ConfidenceIntervalOutput,
    RSquared, AdjustedRSquared,
};
pub use diagnostics::{
    FStatistic, AIC, BIC, DurbinWatson, JarqueBera, ShapiroWilk,
};
pub use stationarity::{
    KolmogorovSmirnov, AndersonDarling, AugmentedDickeyFuller, KPSS, PhillipsPerron,
    TRINMovingAverage,
};
pub use seasonal::{
    SeasonalStrength, DayOfWeekEffect, TurnOfMonthEffect, HolidayEffect,
    QuarterlyEffect, JanuaryEffect,
};
pub use extended::{
    RollingVariance, RollingSkewness, RollingKurtosis,
    PriceDistribution, ReturnDistribution, TailRiskIndicator,
};
pub use advanced::{
    RollingCovariance, SerialCorrelation, RunsTest,
    MeanReversionStrength, DistributionMoments, DistributionMomentsOutput,
    OutlierDetector, RollingBeta, RollingAlpha, InformationCoefficient,
    RankCorrelation, TailDependence, TailDependenceOutput, CopulaCorrelation,
    ZScoreExtreme, PercentileRank, StatisticalRegime, StatisticalRegimeOutput,
    AutocorrelationIndex, HurstExponentMA, EntropyMeasure,
};
