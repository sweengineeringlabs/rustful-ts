//! Moving Average Indicators
//!
//! Various moving average implementations for trend following and smoothing.

pub mod sma;
pub mod ema;
pub mod wma;
pub mod dema;
pub mod tema;
pub mod hma;
pub mod kama;
pub mod zlema;
pub mod smma;
pub mod alma;
pub mod frama;
pub mod vidya;
pub mod t3;
pub mod triangular;
pub mod gmma;
pub mod sine_wma;
pub mod jurik_ma;
pub mod evwma;
pub mod extended;
pub mod advanced;

// Re-exports
pub use sma::SMA;
pub use ema::EMA;
pub use wma::WMA;
pub use dema::DEMA;
pub use tema::TEMA;
pub use hma::HMA;
pub use kama::KAMA;
pub use zlema::ZLEMA;
pub use smma::SMMA;
pub use alma::ALMA;
pub use frama::FRAMA;
pub use vidya::VIDYA;
pub use t3::T3;
pub use triangular::TRIMA;
pub use gmma::GMMA;
pub use sine_wma::SineWMA;
pub use jurik_ma::JurikMA;
pub use evwma::EVWMA;
pub use extended::{
    VolumeAdjustedMA, RangeWeightedMA, MomentumWeightedMA,
    AdaptiveMA, DoubleSmoothedMA, TripleSmoothedMA,
};
pub use advanced::{
    FractalAdaptiveMA, VolumeAdaptiveMA, TrendAdaptiveMA,
    NoiseAdaptiveMA, MomentumAdaptiveMA, EfficiencyAdaptiveMA,
    VolatilityAdaptiveMA, CycleAdaptiveMA,
    RegimeAdaptiveMA, VolumePriceMA, MomentumFilteredMA,
    TrendStrengthMA, CycleAdjustedMA, AdaptiveLagMA,
};
