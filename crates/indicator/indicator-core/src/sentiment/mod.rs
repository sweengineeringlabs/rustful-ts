//! Sentiment Indicators
//!
//! Indicators for measuring market sentiment and crowd behavior.

pub mod market_sentiment;
pub mod extended;
pub mod advanced;
pub mod platform_activity;
pub mod commitment_of_traders;
pub mod smart_money_sentiment;
pub mod aaii_sentiment;
pub mod vix_term_structure;
pub mod insider_trading;
pub mod social_volume;
pub mod social_sentiment;
pub mod news_sentiment;
pub mod google_trends;

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
// Platform activity indicators (IND-281)
pub use platform_activity::{RedditTwitterActivity, RedditTwitterActivityConfig};
// Commitment of Traders indicators (IND-282)
pub use commitment_of_traders::{CommitmentOfTraders, CommitmentOfTradersConfig};
// Smart Money Index Sentiment indicators (IND-283)
pub use smart_money_sentiment::{SmartMoneyIndexSentiment, SmartMoneyIndexSentimentConfig};
// AAII Sentiment indicators (IND-284)
pub use aaii_sentiment::{AAIISentiment, AAIISentimentConfig};
// VIX Term Structure indicators (IND-285)
pub use vix_term_structure::{VIXTermStructure, VIXTermStructureConfig};
// Insider Trading Ratio indicators (IND-286)
pub use insider_trading::{InsiderTradingRatio, InsiderTradingRatioConfig};
pub use social_volume::{SocialVolume, SocialVolumeOutput, SocialVolumeSignal};
pub use social_sentiment::{SocialSentiment, SocialSentimentOutput, SocialSentimentSignal};
pub use news_sentiment::{NewsSentiment, NewsSentimentOutput, NewsSentimentSignal};
pub use google_trends::{GoogleTrends, GoogleTrendsOutput, GoogleTrendsSignal};
