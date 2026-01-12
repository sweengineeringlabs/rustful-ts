//! Portfolio manager trait.

use crate::error::Result;
use crate::model::Position;
use std::collections::HashMap;

/// Portfolio trait for position management.
pub trait PortfolioManager: Send + Sync {
    /// Get current cash balance.
    fn cash(&self) -> f64;

    /// Get all positions.
    fn positions(&self) -> &HashMap<String, Position>;

    /// Calculate total portfolio value given current prices.
    fn value(&self, prices: &HashMap<String, f64>) -> f64;

    /// Add a position.
    fn add_position(&mut self, position: Position) -> Result<()>;

    /// Remove a position.
    fn remove_position(&mut self, symbol: &str) -> Result<Option<Position>>;
}
