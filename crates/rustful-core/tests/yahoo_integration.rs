//! Integration tests for Yahoo Finance data fetching
//!
//! These tests make real HTTP requests to Yahoo Finance API.
//! Run with: cargo test --features fetch --test yahoo_integration
//!
//! Note: These tests are ignored by default to avoid network dependencies in CI.
//! Run ignored tests with: cargo test --features fetch --test yahoo_integration -- --ignored

use rustful_core::data::{
    adj_closing_prices, closing_prices, daily_returns, fetch_stock_sync, log_returns, volumes,
    Interval, YahooFinance,
};

#[test]
#[ignore] // Requires network access
fn test_fetch_aapl_daily() {
    let quotes = fetch_stock_sync("AAPL", "2024-01-01", "2024-06-01", Interval::Daily)
        .expect("Failed to fetch AAPL data");

    assert!(!quotes.is_empty(), "Should return some quotes");
    assert!(quotes.len() > 50, "Should have at least 50 trading days");

    // Verify quote structure
    for quote in &quotes {
        assert!(quote.open > 0.0, "Open price should be positive");
        assert!(quote.high >= quote.low, "High should be >= low");
        assert!(quote.close > 0.0, "Close price should be positive");
        assert!(quote.volume > 0, "Volume should be positive");
        assert!(quote.timestamp > 0, "Timestamp should be positive");
    }

    // Verify prices are in reasonable range for AAPL (2024)
    let prices = closing_prices(&quotes);
    for price in &prices {
        assert!(*price > 100.0 && *price < 300.0, "AAPL price should be in reasonable range");
    }
}

#[test]
#[ignore] // Requires network access
fn test_fetch_weekly_data() {
    let quotes = fetch_stock_sync("MSFT", "2024-01-01", "2024-06-01", Interval::Weekly)
        .expect("Failed to fetch MSFT weekly data");

    assert!(!quotes.is_empty(), "Should return some quotes");
    // ~22 weeks in 5 months
    assert!(quotes.len() >= 15 && quotes.len() <= 30, "Should have reasonable number of weekly bars");
}

#[test]
#[ignore] // Requires network access
fn test_fetch_monthly_data() {
    let quotes = fetch_stock_sync("GOOGL", "2023-01-01", "2024-01-01", Interval::Monthly)
        .expect("Failed to fetch GOOGL monthly data");

    assert!(!quotes.is_empty(), "Should return some quotes");
    assert!(quotes.len() >= 10 && quotes.len() <= 14, "Should have ~12 monthly bars");
}

#[test]
#[ignore] // Requires network access
fn test_helper_functions_with_real_data() {
    let quotes = fetch_stock_sync("AAPL", "2024-01-01", "2024-03-01", Interval::Daily)
        .expect("Failed to fetch AAPL data");

    // Test closing prices extraction
    let prices = closing_prices(&quotes);
    assert_eq!(prices.len(), quotes.len());

    // Test adjusted closing prices
    let adj_prices = adj_closing_prices(&quotes);
    assert_eq!(adj_prices.len(), quotes.len());

    // Test volumes
    let vols = volumes(&quotes);
    assert_eq!(vols.len(), quotes.len());
    assert!(vols.iter().all(|v| *v > 0.0), "All volumes should be positive");

    // Test daily returns
    let returns = daily_returns(&prices);
    assert_eq!(returns.len(), prices.len() - 1);
    // Returns should be small (typically < 10% per day)
    assert!(
        returns.iter().all(|r| r.abs() < 0.2),
        "Daily returns should be reasonable"
    );

    // Test log returns
    let log_ret = log_returns(&prices);
    assert_eq!(log_ret.len(), prices.len() - 1);
}

#[test]
#[ignore] // Requires network access
fn test_fetch_invalid_symbol() {
    let result = fetch_stock_sync("INVALID_SYMBOL_XYZ123", "2024-01-01", "2024-06-01", Interval::Daily);

    // Should return an error for invalid symbol
    assert!(result.is_err(), "Invalid symbol should return error");
}

#[test]
#[ignore] // Requires network access
fn test_fetch_etf() {
    // Test with an ETF symbol
    let quotes = fetch_stock_sync("SPY", "2024-01-01", "2024-03-01", Interval::Daily)
        .expect("Failed to fetch SPY data");

    assert!(!quotes.is_empty(), "Should return some quotes for ETF");
}

#[test]
#[ignore] // Requires network access
fn test_fetch_international_stock() {
    // Test with a stock that has a dot in the symbol (e.g., Toronto Stock Exchange)
    let result = fetch_stock_sync("RY.TO", "2024-01-01", "2024-03-01", Interval::Daily);

    // This may or may not work depending on Yahoo's support
    // Just verify it doesn't panic
    match result {
        Ok(quotes) => assert!(!quotes.is_empty()),
        Err(_) => {} // Some international symbols may not be available
    }
}

#[test]
#[ignore] // Requires network access
fn test_yahoo_finance_client_directly() {
    let client = YahooFinance::new();
    let quotes = client
        .fetch_blocking("NVDA", "2024-01-01", "2024-02-01", Interval::Daily)
        .expect("Failed to fetch NVDA data");

    assert!(!quotes.is_empty());

    // Verify we got valid price data
    let prices = closing_prices(&quotes);
    assert!(prices.iter().all(|p| *p > 0.0), "All prices should be positive");
}

#[cfg(feature = "fetch")]
#[tokio::test]
#[ignore] // Requires network access
async fn test_async_fetch() {
    use rustful_core::data::fetch_stock;

    let quotes = fetch_stock("AMZN", "2024-01-01", "2024-02-01", Interval::Daily)
        .await
        .expect("Failed to fetch AMZN data");

    assert!(!quotes.is_empty(), "Should return some quotes");
    assert!(quotes.len() > 15, "Should have at least 15 trading days in January");
}
