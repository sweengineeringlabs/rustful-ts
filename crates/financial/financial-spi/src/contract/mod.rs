//! Trait contracts for financial analytics.

mod backtester;
mod portfolio_manager;
mod risk_calculator;
mod signal_generator;

pub use backtester::*;
pub use portfolio_manager::*;
pub use risk_calculator::*;
pub use signal_generator::*;
