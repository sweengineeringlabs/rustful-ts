//! Forex-Specific Indicators
//!
//! Indicators designed specifically for foreign exchange markets including
//! COT analysis, central bank policy, and cross-rate arbitrage detection.

pub mod cot_forex;
pub mod central_bank_policy;
pub mod cross_rate_arbitrage;

// Re-exports
pub use cot_forex::{COTForex, COTForexOutput};
pub use central_bank_policy::{CentralBankPolicy, CentralBankPolicyOutput, PolicyRegime};
pub use cross_rate_arbitrage::{CrossRateArbitrage, CrossRateArbitrageOutput, ArbitrageDirection};
