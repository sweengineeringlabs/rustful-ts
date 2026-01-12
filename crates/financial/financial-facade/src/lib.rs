//! Financial Analytics Facade
//!
//! Unified re-exports for the financial module.
//!
//! This facade provides access to all financial analytics components:
//! - `portfolio` - Portfolio management (Position, Portfolio)
//! - `backtesting` - Backtesting engine (BacktestResult, Trade, SimpleBacktester)
//! - `signals` - Trading signals (Signal, SignalGenerator, SMACrossover, etc.)
//! - `risk` - Risk metrics (VaR, Sharpe, Sortino, max drawdown, etc.)

// Re-export everything from SPI (traits, errors, types)
pub use financial_spi::*;

// Re-export everything from API (configs)
pub use financial_api::*;

// Re-export everything from Core (implementations)
pub use financial_core::*;
