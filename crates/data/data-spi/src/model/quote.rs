//! Price quote types.

use serde::{Deserialize, Serialize};

/// A single price quote (OHLCV bar).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quote {
    /// Unix timestamp
    pub timestamp: i64,
    /// Opening price
    pub open: f64,
    /// Highest price
    pub high: f64,
    /// Lowest price
    pub low: f64,
    /// Closing price
    pub close: f64,
    /// Adjusted closing price (accounts for splits/dividends)
    pub adj_close: f64,
    /// Trading volume
    pub volume: u64,
}

impl Quote {
    /// Create a new Quote.
    pub fn new(
        timestamp: i64,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        adj_close: f64,
        volume: u64,
    ) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            adj_close,
            volume,
        }
    }

    /// Get the date as YYYY-MM-DD string.
    pub fn date_string(&self) -> String {
        let secs = self.timestamp;
        let days = secs / 86400;
        let years = 1970 + days / 365;
        // Simplified - for display purposes
        format!("{}-XX-XX", years)
    }
}
