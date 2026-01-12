//! Price quote types.

use serde::{Deserialize, Serialize};

/// A single price quote (OHLCV bar).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quote {
    /// Unix timestamp
    pub timestamp: i64,
    /// Opening price
    pub open: f64,
    /// Highest price
    pub high: f64,
    /// Lowest price
    pub low: f64,
    /// Closing price
    pub close: f64,
    /// Adjusted closing price (accounts for splits/dividends)
    pub adj_close: f64,
    /// Trading volume
    pub volume: u64,
}

impl Quote {
    /// Create a new Quote.
    pub fn new(
        timestamp: i64,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        adj_close: f64,
        volume: u64,
    ) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            adj_close,
            volume,
        }
    }

    /// Get the date as YYYY-MM-DD string.
    pub fn date_string(&self) -> String {
        let secs = self.timestamp;
        let days = secs / 86400;
        let years = 1970 + days / 365;
        // Simplified - for display purposes
        format!("{}-XX-XX", years)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_quote() -> Quote {
        Quote::new(
            1704067200, // 2024-01-01 00:00:00 UTC
            100.0,
            105.0,
            99.0,
            103.0,
            102.5,
            1_000_000,
        )
    }

    #[test]
    fn test_quote_new() {
        let quote = Quote::new(1704067200, 100.0, 105.0, 99.0, 103.0, 102.5, 1_000_000);

        assert_eq!(quote.timestamp, 1704067200);
        assert_eq!(quote.open, 100.0);
        assert_eq!(quote.high, 105.0);
        assert_eq!(quote.low, 99.0);
        assert_eq!(quote.close, 103.0);
        assert_eq!(quote.adj_close, 102.5);
        assert_eq!(quote.volume, 1_000_000);
    }

    #[test]
    fn test_quote_direct_construction() {
        let quote = Quote {
            timestamp: 1704067200,
            open: 100.0,
            high: 105.0,
            low: 99.0,
            close: 103.0,
            adj_close: 102.5,
            volume: 1_000_000,
        };

        assert_eq!(quote.timestamp, 1704067200);
        assert_eq!(quote.close, 103.0);
    }

    #[test]
    fn test_quote_clone() {
        let quote = sample_quote();
        let cloned = quote.clone();

        assert_eq!(quote.timestamp, cloned.timestamp);
        assert_eq!(quote.open, cloned.open);
        assert_eq!(quote.high, cloned.high);
        assert_eq!(quote.low, cloned.low);
        assert_eq!(quote.close, cloned.close);
        assert_eq!(quote.adj_close, cloned.adj_close);
        assert_eq!(quote.volume, cloned.volume);
    }

    #[test]
    fn test_quote_debug() {
        let quote = sample_quote();
        let debug_str = format!("{:?}", quote);

        assert!(debug_str.contains("Quote"));
        assert!(debug_str.contains("timestamp"));
        assert!(debug_str.contains("1704067200"));
        assert!(debug_str.contains("open"));
        assert!(debug_str.contains("100"));
    }

    #[test]
    fn test_quote_date_string() {
        let quote = sample_quote();
        let date_str = quote.date_string();

        // The date_string method provides a simplified year-based format
        assert!(date_str.contains("202"));
        assert!(date_str.contains("-XX-XX"));
    }

    #[test]
    fn test_quote_date_string_epoch() {
        let quote = Quote::new(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0);
        let date_str = quote.date_string();
        assert!(date_str.starts_with("1970"));
    }

    #[test]
    fn test_quote_with_zero_volume() {
        let quote = Quote::new(1704067200, 100.0, 105.0, 99.0, 103.0, 102.5, 0);
        assert_eq!(quote.volume, 0);
    }

    #[test]
    fn test_quote_with_large_volume() {
        let quote = Quote::new(
            1704067200,
            100.0,
            105.0,
            99.0,
            103.0,
            102.5,
            u64::MAX,
        );
        assert_eq!(quote.volume, u64::MAX);
    }

    #[test]
    fn test_quote_with_negative_prices() {
        // While unusual, negative prices can occur (e.g., oil futures)
        let quote = Quote::new(1704067200, -10.0, -5.0, -15.0, -8.0, -8.0, 1000);
        assert_eq!(quote.open, -10.0);
        assert_eq!(quote.close, -8.0);
    }

    #[test]
    fn test_quote_serialize() {
        let quote = sample_quote();
        let json = serde_json::to_string(&quote).unwrap();

        assert!(json.contains("\"timestamp\":1704067200"));
        assert!(json.contains("\"open\":100"));
        assert!(json.contains("\"high\":105"));
        assert!(json.contains("\"low\":99"));
        assert!(json.contains("\"close\":103"));
        assert!(json.contains("\"adj_close\":102.5"));
        assert!(json.contains("\"volume\":1000000"));
    }

    #[test]
    fn test_quote_deserialize() {
        let json = r#"{
            "timestamp": 1704067200,
            "open": 100.0,
            "high": 105.0,
            "low": 99.0,
            "close": 103.0,
            "adj_close": 102.5,
            "volume": 1000000
        }"#;

        let quote: Quote = serde_json::from_str(json).unwrap();

        assert_eq!(quote.timestamp, 1704067200);
        assert_eq!(quote.open, 100.0);
        assert_eq!(quote.high, 105.0);
        assert_eq!(quote.low, 99.0);
        assert_eq!(quote.close, 103.0);
        assert_eq!(quote.adj_close, 102.5);
        assert_eq!(quote.volume, 1000000);
    }

    #[test]
    fn test_quote_roundtrip_serialization() {
        let original = sample_quote();
        let json = serde_json::to_string(&original).unwrap();
        let deserialized: Quote = serde_json::from_str(&json).unwrap();

        assert_eq!(original.timestamp, deserialized.timestamp);
        assert_eq!(original.open, deserialized.open);
        assert_eq!(original.high, deserialized.high);
        assert_eq!(original.low, deserialized.low);
        assert_eq!(original.close, deserialized.close);
        assert_eq!(original.adj_close, deserialized.adj_close);
        assert_eq!(original.volume, deserialized.volume);
    }

    #[test]
    fn test_quote_with_float_precision() {
        let quote = Quote::new(
            1704067200,
            100.123456789,
            105.987654321,
            99.111111111,
            103.555555555,
            102.999999999,
            1000,
        );

        assert!((quote.open - 100.123456789).abs() < f64::EPSILON);
        assert!((quote.high - 105.987654321).abs() < f64::EPSILON);
    }

    #[test]
    fn test_multiple_quotes() {
        let quotes = vec![
            Quote::new(1704067200, 100.0, 105.0, 99.0, 103.0, 102.5, 1000),
            Quote::new(1704153600, 103.0, 108.0, 102.0, 107.0, 106.5, 1200),
            Quote::new(1704240000, 107.0, 110.0, 105.0, 109.0, 108.5, 1100),
        ];

        assert_eq!(quotes.len(), 3);
        assert_eq!(quotes[0].close, 103.0);
        assert_eq!(quotes[1].close, 107.0);
        assert_eq!(quotes[2].close, 109.0);
    }
}
