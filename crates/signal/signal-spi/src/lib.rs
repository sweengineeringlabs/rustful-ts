//! Signal Service Provider Interface
//!
//! Defines traits for trading signal generators.

use serde::{Deserialize, Serialize};

/// Trading signal
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Signal {
    Buy,
    Sell,
    Hold,
}

/// Common trait for signal generators
pub trait SignalGenerator: Send + Sync {
    /// Generate a signal based on current data
    fn generate(&self, data: &[f64]) -> Signal;

    /// Generate signals for a series
    fn generate_series(&self, data: &[f64]) -> Vec<Signal> {
        (1..=data.len())
            .map(|i| self.generate(&data[..i]))
            .collect()
    }
}
