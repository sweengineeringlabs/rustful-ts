//! Trading signals

use serde::{Deserialize, Serialize};

/// Trading signal
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Signal {
    Buy,
    Sell,
    Hold,
}

/// Signal generator trait
pub trait SignalGenerator {
    /// Generate a signal based on current data
    fn generate(&self, data: &[f64]) -> Signal;

    /// Generate signals for a series
    fn generate_series(&self, data: &[f64]) -> Vec<Signal> {
        (1..=data.len())
            .map(|i| self.generate(&data[..i]))
            .collect()
    }
}
