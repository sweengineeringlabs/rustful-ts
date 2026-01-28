//! Technical Indicator Composite - Multi-Indicator Systems
//!
//! This module provides composite indicators that combine multiple technical
//! indicators to generate more robust trading signals.
//!
//! # Composite Indicators
//!
//! - [`TTMSqueeze`]: TTM Squeeze (Bollinger inside Keltner + momentum)
//! - [`ElderImpulse`]: Elder Impulse System (EMA + MACD histogram)
//! - [`SchaffTrendCycle`]: Schaff Trend Cycle (MACD + Stochastic smoothing)
//! - [`ElderTripleScreen`]: Elder Triple Screen trading system
//! - [`CommoditySelectionIndex`]: Commodity Selection Index (ADXR + ATR)
//! - [`SqueezeMomentum`]: LazyBear's Squeeze Momentum
//! - [`TrendStrengthIndex`]: Composite trend strength measurement
//! - [`RegimeDetector`]: Market regime detection (trending/ranging)

// ============================================================================
// Module Declarations
// ============================================================================

pub mod commodity_selection;
pub mod elder_impulse;
pub mod elder_ray;
pub mod elder_triple_screen;
pub mod regime_detector;
pub mod schaff;
pub mod squeeze_momentum;
pub mod trend_strength;
pub mod ttm_squeeze;
pub mod extended;
pub mod multi_factor;
pub mod advanced;

// ============================================================================
// Re-exports
// ============================================================================

// TTM Squeeze
pub use ttm_squeeze::{TTMSqueeze, TTMSqueezeConfig, TTMSqueezeOutput};

// Elder Impulse System
pub use elder_impulse::{ElderImpulse, ElderImpulseConfig, ElderImpulseOutput};

// Schaff Trend Cycle
pub use schaff::{SchaffTrendCycle, SchaffConfig, SchaffOutput};

// Elder Triple Screen
pub use elder_triple_screen::{ElderTripleScreen, ElderTripleScreenConfig, ElderTripleScreenOutput};

// Commodity Selection Index
pub use commodity_selection::{CommoditySelectionIndex, CommoditySelectionConfig, CommoditySelectionOutput};

// Squeeze Momentum (LazyBear)
pub use squeeze_momentum::{SqueezeMomentum, SqueezeMomentumConfig, SqueezeMomentumOutput};

// Trend Strength Index
pub use trend_strength::{TrendStrengthIndex, TrendStrengthConfig, TrendStrengthOutput, TrendComponents};

// Regime Detector
pub use regime_detector::{RegimeDetector, RegimeDetectorConfig, RegimeDetectorOutput, MarketRegime};

// Elder Ray
pub use elder_ray::ElderRay;

// Extended composites
pub use extended::{
    TrendMomentumScore, VolatilityTrendCombo, MultiPeriodMomentum,
    MomentumStrengthIndex, MarketConditionScore, PriceActionScore,
};

// Multi-factor composites
pub use multi_factor::{
    QualityMomentumFactor, ValueMomentumComposite, RiskAdjustedTrend,
    BreakoutStrengthIndex, TrendReversalProbability, MultiFactorAlphaScore,
};

// Advanced composites
pub use advanced::{
    TrendVolatilityIndex, TrendVolatilityIndexConfig, TrendVolatilityIndexOutput,
    MomentumQualityScore, MomentumQualityScoreConfig, MomentumQualityScoreOutput,
    MarketPhaseIndicator, MarketPhaseIndicatorConfig, MarketPhaseIndicatorOutput, MarketPhase,
    PriceTrendStrength, PriceTrendStrengthConfig, PriceTrendStrengthOutput,
    AdaptiveMarketIndicator, AdaptiveMarketIndicatorConfig, AdaptiveMarketIndicatorOutput,
    CompositeSignalStrength, CompositeSignalStrengthConfig, CompositeSignalStrengthOutput,
    AdaptiveCompositeScore, AdaptiveCompositeScoreConfig, AdaptiveCompositeScoreOutput,
    MultiFactorMomentum, MultiFactorMomentumConfig, MultiFactorMomentumOutput,
    TrendQualityComposite, TrendQualityCompositeConfig, TrendQualityCompositeOutput,
    RiskOnRiskOff, RiskOnRiskOffConfig, RiskOnRiskOffOutput,
    MarketBreadthComposite, MarketBreadthCompositeConfig, MarketBreadthCompositeOutput,
    SentimentTrendComposite, SentimentTrendCompositeConfig, SentimentTrendCompositeOutput,
    MarketStrengthIndex, MarketStrengthIndexConfig, MarketStrengthIndexOutput,
    TrendMomentumComposite, TrendMomentumCompositeConfig, TrendMomentumCompositeOutput,
    VolatilityTrendIndex, VolatilityTrendIndexConfig, VolatilityTrendIndexOutput,
    MultiFactorSignal, MultiFactorSignalConfig, MultiFactorSignalOutput,
    AdaptiveMarketScore, AdaptiveMarketScoreConfig, AdaptiveMarketScoreOutput,
    CompositeLeadingIndicator, CompositeLeadingIndicatorConfig, CompositeLeadingIndicatorOutput,
    // New indicators
    TechnicalScorecard, TechnicalScorecardConfig, TechnicalScorecardOutput,
    TrendMomentumVolume, TrendMomentumVolumeConfig, TrendMomentumVolumeOutput,
    MultiIndicatorConfluence, MultiIndicatorConfluenceConfig, MultiIndicatorConfluenceOutput,
    MarketConditionIndex, MarketConditionIndexConfig, MarketConditionIndexOutput,
    SignalStrengthComposite, SignalStrengthCompositeConfig, SignalStrengthCompositeOutput,
    AdaptiveCompositeMA, AdaptiveCompositeMAConfig, AdaptiveCompositeMAOutput,
    // Additional advanced indicators
    TrendStrengthComposite, TrendStrengthCompositeConfig, TrendStrengthCompositeOutput,
    MomentumQualityComposite, MomentumQualityCompositeConfig, MomentumQualityCompositeOutput,
    VolatilityAdjustedSignal, VolatilityAdjustedSignalConfig, VolatilityAdjustedSignalOutput,
    MultiFactorMomentumV2, MultiFactorMomentumV2Config, MultiFactorMomentumV2Output,
    TechnicalRating, TechnicalRatingConfig, TechnicalRatingOutput,
    MarketPhaseDetector, MarketPhaseDetectorConfig, MarketPhaseDetectorOutput, DetectedPhase,
};
