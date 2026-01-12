//! Technical Indicator Facade
//!
//! Unified re-exports for the indicator module.

// Re-export everything from SPI
pub use indicator_spi::*;

// Re-export everything from API
pub use indicator_api::*;

// Re-export everything from Core
pub use indicator_core::*;

// Re-export everything from Oscillators
pub use indicator_oscillators::*;

// Re-export everything from Trend
pub use indicator_trend::*;

// Re-export everything from Volatility
pub use indicator_volatility::*;

// Re-export everything from Volume
pub use indicator_volume::*;

// Re-export everything from Statistical
pub use indicator_statistical::*;

// Re-export everything from Pattern
pub use indicator_pattern::*;

// Re-export everything from Risk
pub use indicator_risk::*;

// Re-export everything from Bands
pub use indicator_bands::*;
