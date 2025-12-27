//! Trading signal implementations
//!
//! This crate provides implementations of various signal generators:
//!
//! - [`SMACrossover`]: Simple Moving Average crossover strategy

mod sma_crossover;

// Re-export from core
pub use signal_core::{Result, SignalError};

// Re-export traits from SPI
pub use signal_spi::{Signal, SignalGenerator};

// Re-export implementations
pub use sma_crossover::SMACrossover;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::SignalGenerator;
    pub use crate::{Signal, SMACrossover};
    pub use crate::{Result, SignalError};
}
