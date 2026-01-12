//! Financial Analytics Core
//!
//! Implementations for portfolio management, backtesting, signals, and risk metrics.

pub mod portfolio;
pub mod backtesting;
pub mod signals;
pub mod risk;

pub use portfolio::*;
pub use backtesting::*;
pub use signals::*;
pub use risk::*;
