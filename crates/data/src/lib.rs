//! Data fetching utilities for time series analysis
//!
//! Provides functions to retrieve financial data from Yahoo Finance.

pub mod yahoo;

pub use yahoo::{
    YahooFinance, Quote, Interval, YahooError2,
    closing_prices, adj_closing_prices, volumes,
    daily_returns, log_returns,
};

#[cfg(feature = "fetch")]
pub use yahoo::{fetch_stock, fetch_stock_sync};
