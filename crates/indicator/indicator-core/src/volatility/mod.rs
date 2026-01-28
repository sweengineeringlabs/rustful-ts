//! Technical Volatility Indicators
//!
//! Indicators for measuring market volatility and risk.

pub mod atr;
pub mod historical;
pub mod chaikin_vol;
pub mod mass_index;
pub mod parkinson;
pub mod garman_klass;
pub mod rogers_satchell;
pub mod yang_zhang;
pub mod realized;
pub mod normalized_atr;
pub mod choppiness;
pub mod ulcer;
pub mod keltner_original;
pub mod volatility_cone;
pub mod close_to_close;
pub mod market_thermometer;
pub mod kase_dev_stops;
pub mod vix_derived;
pub mod implied_vol;
pub mod extended;

// Re-exports
pub use atr::ATR;
pub use historical::HistoricalVolatility;
pub use chaikin_vol::ChaikinVolatility;
pub use mass_index::MassIndex;
pub use parkinson::ParkinsonVolatility;
pub use garman_klass::GarmanKlassVolatility;
pub use rogers_satchell::RogersSatchellVolatility;
pub use yang_zhang::YangZhangVolatility;
pub use realized::RealizedVolatility;
pub use normalized_atr::NormalizedATR;
pub use choppiness::ChoppinessIndex;
pub use ulcer::UlcerIndex;
pub use keltner_original::KeltnerOriginal;
pub use volatility_cone::{VolatilityCone, VolatilityConeOutput};
pub use close_to_close::CloseToCloseVolatility;
pub use market_thermometer::MarketThermometer;
pub use kase_dev_stops::{KaseDevStops, KaseDevStopsOutput};
pub use vix_derived::{
    VIXTermStructure, VolatilityOfVolatility, VolatilitySkew,
    PutCallProxy, VolatilityPercentile, VolatilityRegime, VolRegime,
};
pub use implied_vol::{
    IVRank, IVPercentile, IVSkewSlope, TermStructureSlope,
    VolOfVol, RiskReversal,
};
pub use extended::{
    VolatilityRatio, RangeExpansionIndex, IntradayIntensityVolatility,
    NormalizedVolatility, VolatilityBreakout, VolatilityRegimeClassifier,
};
