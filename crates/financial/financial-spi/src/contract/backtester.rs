//! Backtester trait.

use crate::error::Result;
use crate::model::{BacktestResult, Signal, Trade};

/// Backtest engine trait.
pub trait Backtester: Send + Sync {
    /// Run backtest with given signals and price data.
    fn run(&self, signals: &[Signal], prices: &[f64]) -> Result<BacktestResult>;

    /// Get all trades from the last backtest.
    fn trades(&self) -> &[Trade];
}
