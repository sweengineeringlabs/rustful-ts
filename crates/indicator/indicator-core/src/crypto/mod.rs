//! Crypto & On-Chain Indicators
//!
//! Cryptocurrency-specific indicators including on-chain metrics and valuation ratios.

pub mod nvt;
pub mod mvrv;
pub mod sopr;
pub mod hash_ribbons;
pub mod fear_greed;
pub mod advanced;
pub mod extended;

// Re-exports
pub use nvt::{NVTRatio, NVTRatioOutput, NVTSignal};
pub use mvrv::{MVRVRatio, MVRVOutput, MVRVSignal};
pub use sopr::{SOPR, SOPROutput, SOPRSignal};
pub use hash_ribbons::{HashRibbons, HashRibbonsOutput, HashRibbonsPhase};
pub use fear_greed::{FearGreedIndex, FearGreedOutput, FearGreedLevel, FearGreedWeights};
pub use advanced::{
    HashRateMomentum, MinerCapitulation, WhaleAccumulation,
    RetailSentimentProxy, InstitutionalFlowProxy, NetworkActivityProxy,
    OnChainMomentum, NetworkHealthIndex, HodlerBehaviorIndex,
    ExchangeFlowMomentum, MinerBehaviorIndex, InstitutionalFlowIndex,
    CryptoTrendStrength, VolatilityRegimeIndex, CryptoMomentumRank,
    MarketSentimentProxy, CryptoCyclePhase, AdaptiveCryptoMA,
};
pub use extended::{
    ActiveAddressesProxy, ExchangeFlowProxy, HODLBehaviorProxy,
    NetworkValueMomentum, TransactionVelocityProxy, CryptoMomentumScore,
};
