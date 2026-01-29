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
pub mod whale_transactions;
pub mod hodl_waves;
pub mod realized_cap;
pub mod thermocap_ratio;
pub mod puell_multiple;
pub mod reserve_risk;
pub mod active_addresses;
pub mod transaction_count;
pub mod transfer_volume;
pub mod exchange_flow;
pub mod stablecoin_supply_ratio;
pub mod stock_to_flow;

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
    CryptoVolatilityRank, AltcoinSeasonIndex, CryptoMarketBreadth,
    CryptoMomentumWave, CryptoTrendPhase, CryptoRiskIndex,
    // New 6 crypto indicators
    CryptoStrengthIndex, CryptoVolatilityScore, CryptoTrendScore,
    CryptoOverboughtOversold, CryptoMomentumIndex, CryptoRiskScore,
};
pub use extended::{
    ActiveAddressesProxy, ExchangeFlowProxy, HODLBehaviorProxy,
    NetworkValueMomentum, TransactionVelocityProxy, CryptoMomentumScore,
};
pub use whale_transactions::{
    WhaleTransactions, WhaleTransactionsConfig, WhaleTransactionsOutput, WhaleSignal,
};
pub use hodl_waves::{
    HODLWaves, HODLWavesConfig, HODLWavesOutput, HODLPhase,
};
pub use realized_cap::{
    RealizedCap, RealizedCapConfig, RealizedCapOutput, RealizedCapPhase,
};
pub use thermocap_ratio::{
    ThermocapRatio, ThermocapRatioConfig, ThermocapRatioOutput, ThermocapSignal,
};
pub use puell_multiple::{
    PuellMultiple, PuellMultipleConfig, PuellMultipleOutput, PuellSignal,
};
pub use reserve_risk::{
    ReserveRisk, ReserveRiskConfig, ReserveRiskOutput, ReserveRiskSignal,
};
pub use active_addresses::{ActiveAddresses, ActiveAddressesOutput, ActivityLevel};
pub use transaction_count::{TransactionCount, TransactionCountOutput, TransactionActivity};
pub use transfer_volume::{TransferVolume, TransferVolumeOutput, VolumeLevel};
pub use exchange_flow::{ExchangeInflow, ExchangeFlowOutput, FlowSignal};
pub use stablecoin_supply_ratio::{StablecoinSupplyRatio, StablecoinSupplyRatioOutput, SSRSignal};
pub use stock_to_flow::{StockToFlow, StockToFlowOutput, StockToFlowSignal};
