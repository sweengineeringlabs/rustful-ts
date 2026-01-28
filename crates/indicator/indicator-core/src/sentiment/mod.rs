//! Sentiment Indicators
//!
//! Indicators for measuring market sentiment and crowd behavior.

pub mod market_sentiment;
pub mod extended;
pub mod advanced;

// Re-exports
pub use market_sentiment::{
    FearIndex, GreedIndex, CrowdPsychology, MarketEuphoria,
    Capitulation, SmartMoneyConfidence,
};
pub use extended::{
    MarketMomentumSentiment, VolatilitySentiment, TrendSentiment,
    ReversalSentiment, ExtremeReadings, SentimentOscillator,
};
pub use advanced::{
    PriceActionSentiment, VolumeBasedSentiment, MomentumSentiment,
    ExtremeSentiment, SentimentDivergence, CompositeSentimentScore,
    SentimentWeights,
    // New advanced sentiment indicators
    SentimentMomentum, SentimentExtremeDetector, SentimentTrendFollower,
    SentimentContrarianSignal, SentimentVolatility, SentimentCycle,
    // Extended advanced sentiment indicators
    SentimentStrength, SentimentAcceleration, SentimentMeanReversion,
    CrowdBehaviorIndex, SentimentRegimeDetector, ContraSentimentSignal,
    // New sentiment indicators (6 new)
    PriceBasedSentiment, VolumeSentimentPattern, MomentumSentimentIndex,
    ExtremeSentimentDetector, SentimentOscillator as AdvancedSentimentOscillator,
    CompositeSentimentIndex,
    // Additional NEW sentiment indicators (6 new)
    FearGreedProxy, MarketPanicIndex, EuphoriaDetector,
    SentimentStrengthIndicator, CrowdBehaviorIndicator, SmartMoneySentiment,
    // NEW sentiment indicators (6 new)
    BullBearRatio, SentimentScore, MarketMoodIndex,
    SpeculativeIndex, RiskAppetiteIndex, ContraryIndicator,
};
