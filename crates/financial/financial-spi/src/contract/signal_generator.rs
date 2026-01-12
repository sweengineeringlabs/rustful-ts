//! Signal generator trait.

use crate::model::Signal;

/// Signal generator trait.
pub trait SignalGenerator: Send + Sync {
    /// Generate a signal based on current data.
    fn generate(&self, data: &[f64]) -> Signal;

    /// Generate signals for a series.
    fn generate_series(&self, data: &[f64]) -> Vec<Signal> {
        (1..=data.len())
            .map(|i| self.generate(&data[..i]))
            .collect()
    }
}
