//! Unit tests for data crate
//!
//! Extracted from source modules for better organization.
//! Tests for private methods remain in source files.

use data::{
    Quote, Interval, YahooFinance, YahooError2,
    closing_prices, adj_closing_prices, volumes,
    daily_returns, log_returns,
};

// ============================================================================
// Price Extraction Tests
// ============================================================================

#[test]
fn test_closing_prices() {
    let quotes = vec![
        Quote {
            timestamp: 1,
            open: 100.0,
            high: 105.0,
            low: 99.0,
            close: 102.0,
            adj_close: 102.0,
            volume: 1000,
        },
        Quote {
            timestamp: 2,
            open: 102.0,
            high: 108.0,
            low: 101.0,
            close: 107.0,
            adj_close: 107.0,
            volume: 1100,
        },
    ];

    let prices = closing_prices(&quotes);
    assert_eq!(prices, vec![102.0, 107.0]);
}

#[test]
fn test_adj_closing_prices() {
    let quotes = vec![
        Quote {
            timestamp: 1,
            open: 100.0,
            high: 105.0,
            low: 99.0,
            close: 102.0,
            adj_close: 101.0,
            volume: 1000,
        },
        Quote {
            timestamp: 2,
            open: 102.0,
            high: 108.0,
            low: 101.0,
            close: 107.0,
            adj_close: 106.0,
            volume: 1100,
        },
    ];

    let prices = adj_closing_prices(&quotes);
    assert_eq!(prices, vec![101.0, 106.0]);
}

#[test]
fn test_volumes() {
    let quotes = vec![
        Quote {
            timestamp: 1,
            open: 100.0,
            high: 105.0,
            low: 99.0,
            close: 102.0,
            adj_close: 102.0,
            volume: 1000000,
        },
        Quote {
            timestamp: 2,
            open: 102.0,
            high: 108.0,
            low: 101.0,
            close: 107.0,
            adj_close: 107.0,
            volume: 2000000,
        },
    ];

    let vols = volumes(&quotes);
    assert_eq!(vols, vec![1000000.0, 2000000.0]);
}

#[test]
fn test_closing_prices_empty() {
    let quotes: Vec<Quote> = vec![];
    let prices = closing_prices(&quotes);
    assert!(prices.is_empty());
}

// ============================================================================
// Returns Calculation Tests
// ============================================================================

#[test]
fn test_daily_returns() {
    let prices = vec![100.0, 110.0, 105.0];
    let returns = daily_returns(&prices);
    assert_eq!(returns.len(), 2);
    assert!((returns[0] - 0.1).abs() < 1e-10);
    assert!((returns[1] - (-0.0454545)).abs() < 1e-5);
}

#[test]
fn test_daily_returns_single_price() {
    let prices = vec![100.0];
    let returns = daily_returns(&prices);
    assert!(returns.is_empty());
}

#[test]
fn test_daily_returns_empty() {
    let prices: Vec<f64> = vec![];
    let returns = daily_returns(&prices);
    assert!(returns.is_empty());
}

#[test]
fn test_log_returns() {
    let prices = vec![100.0, 110.0];
    let returns = log_returns(&prices);
    assert_eq!(returns.len(), 1);
    assert!((returns[0] - 0.09531).abs() < 1e-4);
}

#[test]
fn test_log_returns_multiple() {
    let prices = vec![100.0, 105.0, 110.25]; // 5% then 5%
    let returns = log_returns(&prices);
    assert_eq!(returns.len(), 2);
    // ln(1.05) ≈ 0.04879
    assert!((returns[0] - 0.04879).abs() < 1e-4);
    assert!((returns[1] - 0.04879).abs() < 1e-4);
}

#[test]
fn test_log_returns_empty() {
    let prices: Vec<f64> = vec![];
    let returns = log_returns(&prices);
    assert!(returns.is_empty());
}

// ============================================================================
// Client Tests
// ============================================================================

#[test]
fn test_yahoo_finance_default() {
    let client1 = YahooFinance::new();
    let client2 = YahooFinance::default();

    // Both should be debuggable
    let _ = format!("{:?}", client1);
    let _ = format!("{:?}", client2);
}

// ============================================================================
// Quote Tests
// ============================================================================

#[test]
fn test_quote_date_string() {
    let quote = Quote {
        timestamp: 1704067200, // 2024-01-01
        open: 100.0,
        high: 105.0,
        low: 99.0,
        close: 102.0,
        adj_close: 102.0,
        volume: 1000,
    };

    let date_str = quote.date_string();
    assert!(date_str.starts_with("2024"));
}

// ============================================================================
// Error Display Tests
// ============================================================================

#[test]
fn test_error_display() {
    let err1 = YahooError2::RequestFailed("connection error".to_string());
    assert!(err1.to_string().contains("connection error"));

    let err2 = YahooError2::ParseError("invalid json".to_string());
    assert!(err2.to_string().contains("invalid json"));

    let err3 = YahooError2::InvalidDate("bad date".to_string());
    assert!(err3.to_string().contains("bad date"));

    let err4 = YahooError2::NoData;
    assert!(err4.to_string().contains("No data"));

    let err5 = YahooError2::ApiError {
        code: "404".to_string(),
        description: "Not found".to_string(),
    };
    assert!(err5.to_string().contains("404"));
    assert!(err5.to_string().contains("Not found"));
}

// ============================================================================
// Interval Tests
// ============================================================================

#[test]
fn test_interval_clone_and_debug() {
    let intervals = vec![
        Interval::Minute1,
        Interval::Minute5,
        Interval::Minute15,
        Interval::Minute30,
        Interval::Hour1,
        Interval::Daily,
        Interval::Weekly,
        Interval::Monthly,
    ];

    for interval in &intervals {
        let _ = *interval; // Copy
        let _ = interval.clone(); // Clone
        let debug = format!("{:?}", interval);
        assert!(!debug.is_empty());
    }
}
