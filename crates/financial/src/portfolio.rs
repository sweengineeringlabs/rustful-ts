//! Portfolio management

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single position in the portfolio
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub quantity: f64,
    pub entry_price: f64,
    pub entry_timestamp: i64,
}

/// Portfolio of positions
#[derive(Debug, Clone, Default)]
pub struct Portfolio {
    pub positions: HashMap<String, Position>,
    pub cash: f64,
}

impl Portfolio {
    /// Create a new portfolio with initial cash
    pub fn new(initial_cash: f64) -> Self {
        Self {
            positions: HashMap::new(),
            cash: initial_cash,
        }
    }

    /// Get total portfolio value given current prices
    pub fn value(&self, prices: &HashMap<String, f64>) -> f64 {
        let positions_value: f64 = self
            .positions
            .iter()
            .map(|(symbol, pos)| {
                prices.get(symbol).unwrap_or(&pos.entry_price) * pos.quantity
            })
            .sum();
        self.cash + positions_value
    }
}
