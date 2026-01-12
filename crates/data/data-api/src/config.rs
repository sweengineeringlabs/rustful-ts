//! Data source configuration types.

use data_spi::Interval;
use serde::{Deserialize, Serialize};

/// Configuration for fetching data from a data source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FetchConfig {
    /// Stock symbol (e.g., "AAPL", "MSFT")
    pub symbol: String,
    /// Start date in YYYY-MM-DD format
    pub start_date: String,
    /// End date in YYYY-MM-DD format
    pub end_date: String,
    /// Time interval for bars
    pub interval: Interval,
}

impl FetchConfig {
    /// Create a new fetch configuration.
    pub fn new(symbol: &str, start_date: &str, end_date: &str, interval: Interval) -> Self {
        Self {
            symbol: symbol.to_string(),
            start_date: start_date.to_string(),
            end_date: end_date.to_string(),
            interval,
        }
    }

    /// Create a daily fetch configuration.
    pub fn daily(symbol: &str, start_date: &str, end_date: &str) -> Self {
        Self::new(symbol, start_date, end_date, Interval::Daily)
    }

    /// Create a weekly fetch configuration.
    pub fn weekly(symbol: &str, start_date: &str, end_date: &str) -> Self {
        Self::new(symbol, start_date, end_date, Interval::Weekly)
    }

    /// Create a monthly fetch configuration.
    pub fn monthly(symbol: &str, start_date: &str, end_date: &str) -> Self {
        Self::new(symbol, start_date, end_date, Interval::Monthly)
    }
}

/// Builder for FetchConfig.
#[derive(Debug, Default)]
pub struct FetchConfigBuilder {
    symbol: Option<String>,
    start_date: Option<String>,
    end_date: Option<String>,
    interval: Option<Interval>,
}

impl FetchConfigBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the symbol.
    pub fn symbol(mut self, symbol: &str) -> Self {
        self.symbol = Some(symbol.to_string());
        self
    }

    /// Set the start date.
    pub fn start_date(mut self, start_date: &str) -> Self {
        self.start_date = Some(start_date.to_string());
        self
    }

    /// Set the end date.
    pub fn end_date(mut self, end_date: &str) -> Self {
        self.end_date = Some(end_date.to_string());
        self
    }

    /// Set the interval.
    pub fn interval(mut self, interval: Interval) -> Self {
        self.interval = Some(interval);
        self
    }

    /// Build the configuration.
    pub fn build(self) -> Result<FetchConfig, &'static str> {
        Ok(FetchConfig {
            symbol: self.symbol.ok_or("symbol is required")?,
            start_date: self.start_date.ok_or("start_date is required")?,
            end_date: self.end_date.ok_or("end_date is required")?,
            interval: self.interval.unwrap_or(Interval::Daily),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fetch_config_new() {
        let config = FetchConfig::new("AAPL", "2024-01-01", "2024-12-31", Interval::Daily);
        assert_eq!(config.symbol, "AAPL");
        assert_eq!(config.start_date, "2024-01-01");
        assert_eq!(config.end_date, "2024-12-31");
        assert_eq!(config.interval, Interval::Daily);
    }

    #[test]
    fn test_fetch_config_daily() {
        let config = FetchConfig::daily("MSFT", "2024-01-01", "2024-06-30");
        assert_eq!(config.symbol, "MSFT");
        assert_eq!(config.interval, Interval::Daily);
    }

    #[test]
    fn test_fetch_config_weekly() {
        let config = FetchConfig::weekly("GOOG", "2024-01-01", "2024-06-30");
        assert_eq!(config.interval, Interval::Weekly);
    }

    #[test]
    fn test_fetch_config_monthly() {
        let config = FetchConfig::monthly("AMZN", "2024-01-01", "2024-12-31");
        assert_eq!(config.interval, Interval::Monthly);
    }

    #[test]
    fn test_builder_success() {
        let config = FetchConfigBuilder::new()
            .symbol("AAPL")
            .start_date("2024-01-01")
            .end_date("2024-12-31")
            .interval(Interval::Weekly)
            .build()
            .unwrap();

        assert_eq!(config.symbol, "AAPL");
        assert_eq!(config.interval, Interval::Weekly);
    }

    #[test]
    fn test_builder_default_interval() {
        let config = FetchConfigBuilder::new()
            .symbol("AAPL")
            .start_date("2024-01-01")
            .end_date("2024-12-31")
            .build()
            .unwrap();

        assert_eq!(config.interval, Interval::Daily);
    }

    #[test]
    fn test_builder_missing_symbol() {
        let result = FetchConfigBuilder::new()
            .start_date("2024-01-01")
            .end_date("2024-12-31")
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_builder_missing_dates() {
        let result = FetchConfigBuilder::new()
            .symbol("AAPL")
            .build();

        assert!(result.is_err());
    }
}
