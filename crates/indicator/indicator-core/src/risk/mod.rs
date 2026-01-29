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
pub mod extended;
pub mod advanced;
pub mod tail_dependence;
pub mod jump_detection;
pub mod extreme_value_index;
pub mod black_swan_index;
pub mod crisis_alpha;
pub mod stress_index;

// Re-exports
pub use sharpe::SharpeRatio;
pub use sortino::SortinoRatio;
pub use calmar::CalmarRatio;
pub use max_drawdown::MaxDrawdown;
pub use var::{ValueAtRisk, VaRMethod};
pub use cvar::ConditionalVaR;
pub use beta::Beta;
pub use alpha::Alpha;
pub use treynor::TreynorRatio;
pub use information_ratio::InformationRatio;
pub use omega::OmegaRatio;
pub use gain_loss::GainLossRatio;
pub use extended::{
    SterlingRatio, BurkeRatio, UlcerPerformanceIndex,
    PainIndex, RecoveryFactor, TailRatio,
    ConditionalDrawdown, RiskAdjustedReturn, ReturnVariance,
    DrawdownDuration, RecoveryRatio, VolatilityRiskRatio,
};
pub use advanced::{
    DownsideDeviation, UpsidePotentialRatio, KappaRatio,
    WinRate, ProfitFactor, Expectancy,
    ConditionalBeta, TailVaR, StressTestMetric,
    LiquidityAdjustedVaR, CorrelationVaR, RegimeAwareRisk,
    ConditionalDrawdownAtRisk, UpsideDownsideRatio, RiskAdjustedReturnMetric,
    MaxDrawdownDuration, DrawdownRecoveryFactor, RiskRegimeIndicator,
    // New risk indicators
    SortinoRatioAdvanced, CalmarRatioAdvanced, OmegaRatioAdvanced,
    PainRatio, UlcerIndex, KellyFraction,
    // Newest risk indicators
    TailRiskRatio, VaRBreachRate, VolatilityOfVolatility,
    AsymmetricBeta, RiskParityScore, TrackingErrorVariance,
    // 6 NEW risk indicators
    RollingVaR, DrawdownVelocity, RiskMomentum,
    ConditionalBetaRegime, TailDependence, RiskRegimeIndicatorAdvanced,
};
pub use tail_dependence::{TailDependenceCopula, TailDependenceConfig};
pub use jump_detection::{JumpDetection, JumpDetectionConfig};
pub use extreme_value_index::{ExtremeValueIndex, ExtremeValueIndexConfig, EVIMethod};
pub use black_swan_index::{BlackSwanIndex, BlackSwanIndexConfig};
pub use crisis_alpha::{CrisisAlpha, CrisisAlphaConfig};
pub use stress_index::{StressIndex, StressIndexConfig, StressComponents, StressRegime};
