//! Data Source Facade
//!
//! Unified re-exports for the data module.
//!
//! This facade provides a single entry point for all data source functionality:
//! - `data_spi` - Traits, types, and errors for data sources
//! - `data_api` - Configuration types and builders
//! - `data_core` - Implementations (Yahoo Finance)
//!
//! # Example
//!
//! ```rust,ignore
//! use data_facade::{fetch_stock, Interval, Quote, closing_prices};
//!
//! #[tokio::main]
//! async fn main() {
//!     let quotes = fetch_stock("AAPL", "2024-01-01", "2024-12-01", Interval::Daily)
//!         .await
//!         .unwrap();
//!
//!     let prices = closing_prices(&quotes);
//!     println!("Got {} price points", prices.len());
//! }
//! ```

// Re-export everything from SPI
pub use data_spi::*;

// Re-export everything from API
pub use data_api::*;

// Re-export everything from Core
pub use data_core::*;
