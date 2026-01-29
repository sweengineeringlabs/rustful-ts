//! Factor Indicators
//!
//! Quantitative factor-based indicators for systematic trading strategies.
//! Includes mean reversion, momentum factors, and composite factor combinations.

pub mod reversal_factor;
pub mod composite_factor;
pub mod size_factor;
pub mod quality_factor;
pub mod low_volatility_factor;
pub mod dividend_yield_factor;
pub mod growth_factor;
pub mod liquidity_factor;

// Re-exports
pub use reversal_factor::{ReversalFactor, ReversalFactorOutput, ReversalSignal};
pub use composite_factor::{CompositeFactorScore, CompositeFactorOutput, FactorWeights};
pub use size_factor::{SizeFactor, SizeFactorConfig};
pub use quality_factor::{QualityFactor, QualityFactorConfig};
pub use low_volatility_factor::{LowVolatilityFactor, LowVolatilityFactorConfig, VolatilityType};
pub use dividend_yield_factor::{DividendYieldFactor, DividendYieldFactorConfig};
pub use growth_factor::{GrowthFactor, GrowthFactorConfig};
pub use liquidity_factor::{LiquidityFactor, LiquidityFactorConfig};
