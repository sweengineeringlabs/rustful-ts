//! Utility functions for working with quotes and prices.

use super::Quote;

/// Extract closing prices from quotes.
pub fn closing_prices(quotes: &[Quote]) -> Vec<f64> {
    quotes.iter().map(|q| q.close).collect()
}

/// Extract adjusted closing prices from quotes.
pub fn adj_closing_prices(quotes: &[Quote]) -> Vec<f64> {
    quotes.iter().map(|q| q.adj_close).collect()
}

/// Extract volumes from quotes.
pub fn volumes(quotes: &[Quote]) -> Vec<f64> {
    quotes.iter().map(|q| q.volume as f64).collect()
}

/// Calculate daily returns from prices.
pub fn daily_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return Vec::new();
    }

    prices
        .windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect()
}

/// Calculate log returns from prices.
pub fn log_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return Vec::new();
    }

    prices
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect()
}
