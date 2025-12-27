//! Integration tests for data crate

use data::{
    closing_prices, adj_closing_prices, volumes,
    daily_returns, log_returns, Quote, Interval, YahooFinance,
};

fn sample_quotes() -> Vec<Quote> {
    vec![
        Quote {
            timestamp: 1704067200,
            open: 185.0,
            high: 186.5,
            low: 184.0,
            close: 185.5,
            adj_close: 184.0,
            volume: 50000000,
        },
        Quote {
            timestamp: 1704153600,
            open: 186.0,
            high: 188.0,
            low: 185.5,
            close: 187.5,
            adj_close: 186.0,
            volume: 55000000,
        },
        Quote {
            timestamp: 1704240000,
            open: 187.0,
            high: 189.0,
            low: 186.0,
            close: 188.0,
            adj_close: 186.5,
            volume: 48000000,
        },
        Quote {
            timestamp: 1704326400,
            open: 188.0,
            high: 190.5,
            low: 187.5,
            close: 190.0,
            adj_close: 188.5,
            volume: 62000000,
        },
        Quote {
            timestamp: 1704412800,
            open: 189.5,
            high: 191.0,
            low: 188.0,
            close: 189.5,
            adj_close: 188.0,
            volume: 51000000,
        },
    ]
}

#[test]
fn test_quote_struct_fields() {
    let quote = Quote {
        timestamp: 1704067200,
        open: 185.0,
        high: 186.5,
        low: 184.0,
        close: 185.5,
        adj_close: 184.0,
        volume: 50000000,
    };

    assert_eq!(quote.open, 185.0);
    assert_eq!(quote.high, 186.5);
    assert_eq!(quote.low, 184.0);
    assert_eq!(quote.close, 185.5);
    assert_eq!(quote.adj_close, 184.0);
    assert_eq!(quote.volume, 50000000);
}

#[test]
fn test_closing_prices_extraction() {
    let quotes = sample_quotes();
    let prices = closing_prices(&quotes);

    assert_eq!(prices.len(), 5);
    assert_eq!(prices[0], 185.5);
    assert_eq!(prices[1], 187.5);
    assert_eq!(prices[4], 189.5);
}

#[test]
fn test_adj_closing_prices_extraction() {
    let quotes = sample_quotes();
    let prices = adj_closing_prices(&quotes);

    assert_eq!(prices.len(), 5);
    assert_eq!(prices[0], 184.0);
    assert_eq!(prices[1], 186.0);
}

#[test]
fn test_volumes_extraction() {
    let quotes = sample_quotes();
    let vols = volumes(&quotes);

    assert_eq!(vols.len(), 5);
    assert_eq!(vols[0], 50000000.0);
    assert_eq!(vols[3], 62000000.0);
}

#[test]
fn test_daily_returns_calculation() {
    let quotes = sample_quotes();
    let prices = closing_prices(&quotes);
    let returns = daily_returns(&prices);

    // Should have n-1 returns for n prices
    assert_eq!(returns.len(), prices.len() - 1);

    // First return: (187.5 - 185.5) / 185.5 = 0.01078
    assert!((returns[0] - 0.01078).abs() < 0.001);
}

#[test]
fn test_log_returns_calculation() {
    let quotes = sample_quotes();
    let prices = closing_prices(&quotes);
    let returns = log_returns(&prices);

    assert_eq!(returns.len(), prices.len() - 1);

    // First log return: ln(187.5 / 185.5) = 0.01072
    assert!((returns[0] - 0.01072).abs() < 0.001);
}

#[test]
fn test_returns_edge_cases() {
    // Single price
    let single = vec![100.0];
    assert!(daily_returns(&single).is_empty());
    assert!(log_returns(&single).is_empty());

    // Empty
    let empty: Vec<f64> = vec![];
    assert!(daily_returns(&empty).is_empty());
    assert!(log_returns(&empty).is_empty());
}

#[test]
fn test_interval_variants() {
    // Test all interval variants exist
    let _m1 = Interval::Minute1;
    let _m5 = Interval::Minute5;
    let _m15 = Interval::Minute15;
    let _m30 = Interval::Minute30;
    let _h1 = Interval::Hour1;
    let _d = Interval::Daily;
    let _w = Interval::Weekly;
    let _mo = Interval::Monthly;
}

#[test]
fn test_yahoo_finance_client_creation() {
    let client = YahooFinance::new();
    let client_default = YahooFinance::default();

    // Both should create valid clients
    let _ = format!("{:?}", client);
    let _ = format!("{:?}", client_default);
}

#[test]
fn test_quote_clone_and_debug() {
    let quote = Quote {
        timestamp: 1704067200,
        open: 185.0,
        high: 186.5,
        low: 184.0,
        close: 185.5,
        adj_close: 184.0,
        volume: 50000000,
    };

    let cloned = quote.clone();
    assert_eq!(cloned.close, quote.close);

    // Debug should work
    let debug_str = format!("{:?}", quote);
    assert!(debug_str.contains("Quote"));
}

#[test]
fn test_price_analysis_workflow() {
    // Simulate a typical workflow
    let quotes = sample_quotes();

    // 1. Extract prices
    let close_prices = closing_prices(&quotes);
    let adj_prices = adj_closing_prices(&quotes);

    // 2. Calculate returns
    let simple_returns = daily_returns(&close_prices);
    let lreturns = log_returns(&close_prices);

    // 3. Verify consistency
    assert_eq!(close_prices.len(), adj_prices.len());
    assert_eq!(simple_returns.len(), lreturns.len());
    assert_eq!(simple_returns.len(), close_prices.len() - 1);

    // 4. Returns should be similar for small changes
    for (sr, lr) in simple_returns.iter().zip(lreturns.iter()) {
        // For small returns, simple and log returns should be close
        assert!((sr - lr).abs() < 0.01);
    }
}

#[test]
fn test_volumes_as_f64() {
    let quotes = sample_quotes();
    let vols = volumes(&quotes);

    // Volumes should be converted correctly to f64
    for vol in &vols {
        assert!(*vol > 0.0);
        assert!(*vol < 1e12); // Reasonable upper bound
    }
}
