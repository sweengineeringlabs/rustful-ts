//! End-to-end tests for data crate
//!
//! Tests complete data processing workflows using only this crate's API.

use data::{
    Quote, Interval, YahooFinance,
    closing_prices, adj_closing_prices, volumes,
    daily_returns, log_returns,
};

fn sample_quotes() -> Vec<Quote> {
    vec![
        Quote {
            timestamp: 1704067200, // 2024-01-01
            open: 100.0,
            high: 105.0,
            low: 99.0,
            close: 102.0,
            adj_close: 101.5,
            volume: 1000000,
        },
        Quote {
            timestamp: 1704153600, // 2024-01-02
            open: 102.0,
            high: 108.0,
            low: 101.0,
            close: 107.0,
            adj_close: 106.5,
            volume: 1200000,
        },
        Quote {
            timestamp: 1704240000, // 2024-01-03
            open: 107.0,
            high: 110.0,
            low: 105.0,
            close: 109.0,
            adj_close: 108.5,
            volume: 1100000,
        },
        Quote {
            timestamp: 1704326400, // 2024-01-04
            open: 109.0,
            high: 112.0,
            low: 108.0,
            close: 111.0,
            adj_close: 110.5,
            volume: 900000,
        },
        Quote {
            timestamp: 1704412800, // 2024-01-05
            open: 111.0,
            high: 115.0,
            low: 110.0,
            close: 114.0,
            adj_close: 113.5,
            volume: 1500000,
        },
    ]
}

#[test]
fn e2e_price_extraction_workflow() {
    let quotes = sample_quotes();

    // Extract different price series
    let close = closing_prices(&quotes);
    let adj_close = adj_closing_prices(&quotes);
    let vol = volumes(&quotes);

    // Verify extraction
    assert_eq!(close.len(), 5);
    assert_eq!(adj_close.len(), 5);
    assert_eq!(vol.len(), 5);

    // Close prices
    assert_eq!(close, vec![102.0, 107.0, 109.0, 111.0, 114.0]);

    // Adjusted close prices
    assert_eq!(adj_close, vec![101.5, 106.5, 108.5, 110.5, 113.5]);

    // Volumes
    assert_eq!(vol, vec![1000000.0, 1200000.0, 1100000.0, 900000.0, 1500000.0]);
}

#[test]
fn e2e_returns_calculation_workflow() {
    let quotes = sample_quotes();
    let prices = closing_prices(&quotes);

    // Calculate returns
    let returns = daily_returns(&prices);
    let log_rets = log_returns(&prices);

    // Should have one fewer return than prices
    assert_eq!(returns.len(), prices.len() - 1);
    assert_eq!(log_rets.len(), prices.len() - 1);

    // First return: (107 - 102) / 102 ≈ 0.049
    assert!((returns[0] - 0.049).abs() < 0.001);

    // Log returns should be close to simple returns for small changes
    for (r, lr) in returns.iter().zip(log_rets.iter()) {
        // For small returns, ln(1+r) ≈ r
        assert!((r - lr).abs() < 0.01);
    }
}

#[test]
fn e2e_log_returns_vs_daily_returns() {
    // Use prices with known returns
    let prices = vec![100.0, 110.0, 121.0, 133.1]; // 10% each period

    let daily = daily_returns(&prices);
    let log = log_returns(&prices);

    assert_eq!(daily.len(), 3);
    assert_eq!(log.len(), 3);

    // Daily returns should all be ~0.10
    for &r in &daily {
        assert!((r - 0.10).abs() < 0.001);
    }

    // Log returns should all be ln(1.10) ≈ 0.0953
    for &lr in &log {
        assert!((lr - 0.0953).abs() < 0.001);
    }
}

#[test]
fn e2e_interval_variants() {
    // Test all interval variants are usable
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

    // All should be clonable and copyable
    for interval in &intervals {
        let _ = *interval; // Copy
        let _ = interval.clone(); // Clone
    }

    // All should be debuggable
    for interval in &intervals {
        let debug = format!("{:?}", interval);
        assert!(!debug.is_empty());
    }
}

#[test]
fn e2e_yahoo_finance_client_creation() {
    let client = YahooFinance::new();
    let default_client = YahooFinance::default();

    // Both should be valid
    let _ = format!("{:?}", client);
    let _ = format!("{:?}", default_client);
}

#[test]
fn e2e_empty_data_handling() {
    let empty: Vec<Quote> = vec![];

    // All extraction functions should handle empty data
    assert!(closing_prices(&empty).is_empty());
    assert!(adj_closing_prices(&empty).is_empty());
    assert!(volumes(&empty).is_empty());

    // Returns calculations should handle empty data
    assert!(daily_returns(&[]).is_empty());
    assert!(log_returns(&[]).is_empty());

    // Single price should return empty returns
    assert!(daily_returns(&[100.0]).is_empty());
    assert!(log_returns(&[100.0]).is_empty());
}

#[test]
fn e2e_quote_fields_access() {
    let quote = Quote {
        timestamp: 1704067200,
        open: 100.0,
        high: 105.0,
        low: 99.0,
        close: 102.0,
        adj_close: 101.5,
        volume: 1000000,
    };

    // All fields should be accessible
    assert_eq!(quote.timestamp, 1704067200);
    assert_eq!(quote.open, 100.0);
    assert_eq!(quote.high, 105.0);
    assert_eq!(quote.low, 99.0);
    assert_eq!(quote.close, 102.0);
    assert_eq!(quote.adj_close, 101.5);
    assert_eq!(quote.volume, 1000000);

    // Quote should be clonable and debuggable
    let cloned = quote.clone();
    assert_eq!(cloned.close, quote.close);
    let _ = format!("{:?}", quote);
}

#[test]
fn e2e_ohlcv_analysis_workflow() {
    let quotes = sample_quotes();

    // Calculate typical OHLCV-based metrics
    let mut highs = Vec::new();
    let mut lows = Vec::new();
    let mut ranges = Vec::new();

    for q in &quotes {
        highs.push(q.high);
        lows.push(q.low);
        ranges.push(q.high - q.low);
    }

    // Verify data relationships
    for q in &quotes {
        assert!(q.high >= q.low);
        assert!(q.high >= q.open);
        assert!(q.high >= q.close);
        assert!(q.low <= q.open);
        assert!(q.low <= q.close);
    }

    // All ranges should be positive
    for &r in &ranges {
        assert!(r >= 0.0);
    }
}

#[test]
fn e2e_cumulative_returns_workflow() {
    let quotes = sample_quotes();
    let prices = closing_prices(&quotes);
    let returns = daily_returns(&prices);

    // Calculate cumulative returns
    let mut cumulative = 1.0;
    for r in &returns {
        cumulative *= 1.0 + r;
    }

    // Should match direct calculation
    let expected = prices.last().unwrap() / prices.first().unwrap();
    assert!((cumulative - expected).abs() < 0.001);
}

#[test]
fn e2e_volume_analysis_workflow() {
    let quotes = sample_quotes();
    let vol = volumes(&quotes);

    // Calculate volume statistics
    let total_volume: f64 = vol.iter().sum();
    let avg_volume = total_volume / vol.len() as f64;
    let max_volume = vol.iter().cloned().fold(0.0, f64::max);
    let min_volume = vol.iter().cloned().fold(f64::INFINITY, f64::min);

    assert!(total_volume > 0.0);
    assert!(avg_volume > 0.0);
    assert!(max_volume >= min_volume);
    assert_eq!(max_volume, 1500000.0);
    assert_eq!(min_volume, 900000.0);
}

#[test]
fn e2e_price_momentum_workflow() {
    let quotes = sample_quotes();
    let prices = closing_prices(&quotes);

    // Calculate momentum (price change over N periods)
    let lookback = 2;
    let mut momentum = Vec::new();

    for i in lookback..prices.len() {
        momentum.push(prices[i] - prices[i - lookback]);
    }

    assert_eq!(momentum.len(), prices.len() - lookback);

    // With steadily increasing prices, all momentum should be positive
    for &m in &momentum {
        assert!(m > 0.0);
    }
}
