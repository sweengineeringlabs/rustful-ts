//! # rustful-financial
//!
//! Financial analytics module for rustful-ts.
//! Provides portfolio management, backtesting, trading signals, and risk metrics.

pub mod portfolio;
pub mod backtesting;
pub mod signals;
pub mod risk;

pub use portfolio::*;
pub use backtesting::*;
pub use signals::*;
