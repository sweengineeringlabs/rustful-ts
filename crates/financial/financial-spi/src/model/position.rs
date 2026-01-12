//! Portfolio position model.

use serde::{Deserialize, Serialize};

/// A single position in the portfolio.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub quantity: f64,
    pub entry_price: f64,
    pub entry_timestamp: i64,
}
