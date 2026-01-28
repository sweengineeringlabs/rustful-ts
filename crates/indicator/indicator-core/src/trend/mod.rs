//! Technical Trend Indicators
//!
//! Indicators for identifying and measuring trend direction and strength.

pub mod macd;
pub mod adx;
pub mod ichimoku;
pub mod supertrend;
pub mod parabolic_sar;
pub mod alligator;
pub mod aroon;
pub mod coppock;
pub mod dpo;
pub mod efficiency_ratio;
pub mod gator;
pub mod kase_cd;
pub mod mcginley;
pub mod rainbow;
pub mod rwi;
pub mod tdi;
pub mod tii;
pub mod vhf;
pub mod vortex;
pub mod safezone_stop;
pub mod weinstein;
pub mod extended;
pub mod advanced;

// Re-exports
pub use macd::MACD;
pub use adx::ADX;
pub use ichimoku::{Ichimoku, IchimokuOutput};
pub use supertrend::SuperTrend;
pub use parabolic_sar::ParabolicSAR;
pub use alligator::{Alligator, AlligatorOutput};
pub use aroon::{Aroon, AroonOutput};
pub use coppock::CoppockCurve;
pub use dpo::DPO;
pub use efficiency_ratio::EfficiencyRatio;
pub use gator::GatorOscillator;
pub use kase_cd::KaseCD;
pub use mcginley::McGinleyDynamic;
pub use rainbow::RainbowMA;
pub use rwi::RandomWalkIndex;
pub use tdi::TrendDetectionIndex;
pub use tii::TrendIntensityIndex;
pub use vhf::VerticalHorizontalFilter;
pub use vortex::VortexIndicator;
pub use safezone_stop::{SafeZoneStop, SafeZoneStopOutput};
pub use weinstein::{
    StageAnalysis, StageAnalysisOutput, WeinStage, WeinsteinMA,
    MansfieldRS, RelativePriceStrength, VolumeConfirmation,
    SupportResistanceLevels, BreakoutValidation, TrendScore,
};
pub use extended::{
    CompositeTrendScore, TrendPersistence, PriceChannelPosition,
    TrendExhaustion, DirectionalMovementQuality, MultiTimeframeTrend,
};
pub use advanced::{
    TrendAcceleration, TrendConsistency, AdaptiveTrendLine,
    TrendStrengthMeter, TrendChangeDetector, MultiScaleTrend,
    AdaptiveTrendFollower, TrendQualityIndex, TrendBreakoutStrength,
    TrendPersistenceMetric, TrendCycleFinder, TrendVolatilityRatio,
    TrendContinuity, TrendMomentumConvergence, AdaptiveTrendStrength,
    TrendDirectionIndex, TrendMaturity, MultiPeriodTrendAlignment,
    // New trend indicators
    TrendAccelerationIndex, TrendConsistencyRating, TrendMaturityScore,
    TrendReversalEstimator, DynamicAdaptiveTrendLine, TrendMomentumDivergence,
    // Newest trend indicators
    TrendPersistenceIndex, TrendStrengthOscillator, MultiScaleTrendIndex,
    TrendEfficiencyRatio, TrendVelocityIndex, TrendRegimeDetector,
    // 6 additional trend indicators
    TrendAngle, TrendChannel, TrendCurvature, TrendVolatilityBand,
    TrendQualityRating, TrendExhaustionSignal,
};
