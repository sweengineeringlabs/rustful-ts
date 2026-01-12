//! Financial Analytics Service Provider Interface
//!
//! Defines traits and types for financial analytics including
//! portfolio management, backtesting, trading signals, and risk metrics.

pub mod contract;
pub mod error;
pub mod model;

// Re-export all public items at crate root for convenience
pub use contract::*;
pub use error::*;
pub use model::*;
