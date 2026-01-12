//! Data Source Service Provider Interface
//!
//! Defines traits and types for fetching financial time series data.

pub mod contract;
pub mod error;
pub mod model;

// Re-export all public items at crate root for convenience
pub use contract::DataSource;
pub use error::{DataError, Result};
pub use model::{
    adj_closing_prices, closing_prices, daily_returns, log_returns, volumes, Interval, Quote,
};

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_as_yahoo_str() {
        assert_eq!(Interval::Minute1.as_yahoo_str(), "1m");
        assert_eq!(Interval::Minute5.as_yahoo_str(), "5m");
        assert_eq!(Interval::Minute15.as_yahoo_str(), "15m");
        assert_eq!(Interval::Minute30.as_yahoo_str(), "30m");
        assert_eq!(Interval::Hour1.as_yahoo_str(), "1h");
        assert_eq!(Interval::Daily.as_yahoo_str(), "1d");
        assert_eq!(Interval::Weekly.as_yahoo_str(), "1wk");
        assert_eq!(Interval::Monthly.as_yahoo_str(), "1mo");
    }

    #[test]
    fn test_quote_new() {
        let quote = Quote::new(1704067200, 185.0, 186.0, 184.0, 185.5, 185.5, 1000000);
        assert_eq!(quote.timestamp, 1704067200);
        assert_eq!(quote.open, 185.0);
        assert_eq!(quote.high, 186.0);
        assert_eq!(quote.low, 184.0);
        assert_eq!(quote.close, 185.5);
        assert_eq!(quote.adj_close, 185.5);
        assert_eq!(quote.volume, 1000000);
    }

    #[test]
    fn test_closing_prices() {
        let quotes = vec![
            Quote::new(1, 100.0, 105.0, 95.0, 102.0, 102.0, 1000),
            Quote::new(2, 102.0, 108.0, 100.0, 106.0, 106.0, 1200),
            Quote::new(3, 106.0, 110.0, 104.0, 108.0, 108.0, 1100),
        ];
        let closes = closing_prices(&quotes);
        assert_eq!(closes, vec![102.0, 106.0, 108.0]);
    }

    #[test]
    fn test_adj_closing_prices() {
        let quotes = vec![
            Quote::new(1, 100.0, 105.0, 95.0, 102.0, 101.0, 1000),
            Quote::new(2, 102.0, 108.0, 100.0, 106.0, 105.0, 1200),
        ];
        let adj = adj_closing_prices(&quotes);
        assert_eq!(adj, vec![101.0, 105.0]);
    }

    #[test]
    fn test_volumes() {
        let quotes = vec![
            Quote::new(1, 100.0, 105.0, 95.0, 102.0, 102.0, 1000),
            Quote::new(2, 102.0, 108.0, 100.0, 106.0, 106.0, 1200),
        ];
        let vols = volumes(&quotes);
        assert_eq!(vols, vec![1000.0, 1200.0]);
    }

    #[test]
    fn test_daily_returns() {
        let prices = vec![100.0, 110.0, 105.0];
        let returns = daily_returns(&prices);
        assert_eq!(returns.len(), 2);
        assert!((returns[0] - 0.1).abs() < 1e-10);
        assert!((returns[1] - (-0.0454545454545)).abs() < 1e-6);
    }

    #[test]
    fn test_daily_returns_empty() {
        assert!(daily_returns(&[]).is_empty());
        assert!(daily_returns(&[100.0]).is_empty());
    }

    #[test]
    fn test_log_returns() {
        let prices = vec![100.0, 110.0];
        let returns = log_returns(&prices);
        assert_eq!(returns.len(), 1);
        assert!((returns[0] - (1.1_f64).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_log_returns_empty() {
        assert!(log_returns(&[]).is_empty());
        assert!(log_returns(&[100.0]).is_empty());
    }
}
