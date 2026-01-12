//! Data Source Core
//!
//! Implementations for data sources.

pub mod yahoo;

pub use yahoo::YahooFinance;

#[cfg(feature = "fetch")]
pub use yahoo::{fetch_stock, fetch_stock_sync};
