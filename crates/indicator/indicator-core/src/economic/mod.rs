//! Economic Indicators
//!
//! Macro-economic indicators for analyzing economic conditions including
//! employment data, housing, consumer confidence, and financial conditions.

pub mod unemployment_claims;
pub mod housing_starts;
pub mod consumer_confidence;
pub mod ism_new_orders;
pub mod real_m2_growth;
pub mod financial_conditions;

// Re-exports
pub use unemployment_claims::{UnemploymentClaimsTrend, UnemploymentClaimsTrendConfig};
pub use housing_starts::{HousingStartsTrend, HousingStartsTrendConfig};
pub use consumer_confidence::{ConsumerConfidenceDelta, ConsumerConfidenceDeltaConfig};
pub use ism_new_orders::{ISMNewOrders, ISMNewOrdersConfig};
pub use real_m2_growth::{RealM2Growth, RealM2GrowthConfig};
pub use financial_conditions::{FinancialConditionsIndex, FinancialConditionsIndexConfig, FinancialConditionsComponents};
