//! Trading signal model.

use serde::{Deserialize, Serialize};

/// Trading signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Signal {
    Buy,
    Sell,
    Hold,
}

impl Signal {
    /// Convert to numeric signal: Buy = 1, Sell = -1, Hold = 0.
    pub fn to_numeric(&self) -> f64 {
        match self {
            Signal::Buy => 1.0,
            Signal::Sell => -1.0,
            Signal::Hold => 0.0,
        }
    }
}
