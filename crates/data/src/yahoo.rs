//! Yahoo Finance data fetcher
//!
//! Fetches historical stock data from Yahoo Finance API.
//!
//! # Example
//!
//! ```rust,no_run
//! use data::{fetch_stock, Interval};
//!
//! #[tokio::main]
//! async fn main() {
//!     let quotes = fetch_stock("AAPL", "2024-01-01", "2024-12-01", Interval::Daily)
//!         .await
//!         .unwrap();
//!
//!     let prices: Vec<f64> = quotes.iter().map(|q| q.close).collect();
//!     println!("Got {} price points", prices.len());
//! }
//! ```

use serde::{Deserialize, Serialize};

/// Time interval for historical data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Interval {
    /// 1 minute bars
    Minute1,
    /// 5 minute bars
    Minute5,
    /// 15 minute bars
    Minute15,
    /// 30 minute bars
    Minute30,
    /// 1 hour bars
    Hour1,
    /// Daily bars
    Daily,
    /// Weekly bars
    Weekly,
    /// Monthly bars
    Monthly,
}

impl Interval {
    fn as_str(&self) -> &'static str {
        match self {
            Interval::Minute1 => "1m",
            Interval::Minute5 => "5m",
            Interval::Minute15 => "15m",
            Interval::Minute30 => "30m",
            Interval::Hour1 => "1h",
            Interval::Daily => "1d",
            Interval::Weekly => "1wk",
            Interval::Monthly => "1mo",
        }
    }
}

/// A single price quote
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
    /// Get the date as YYYY-MM-DD string
    pub fn date_string(&self) -> String {
        let secs = self.timestamp;
        let days = secs / 86400;
        let years = 1970 + days / 365;
        // Simplified - for display purposes
        format!("{}-XX-XX", years)
    }
}

/// Yahoo Finance API response structures
#[derive(Debug, Deserialize)]
struct YahooResponse {
    chart: ChartResult,
}

#[derive(Debug, Deserialize)]
struct ChartResult {
    result: Option<Vec<ChartData>>,
    error: Option<YahooError>,
}

#[derive(Debug, Deserialize)]
struct YahooError {
    code: String,
    description: String,
}

#[derive(Debug, Deserialize)]
struct ChartData {
    timestamp: Vec<i64>,
    indicators: Indicators,
}

#[derive(Debug, Deserialize)]
struct Indicators {
    quote: Vec<QuoteData>,
    adjclose: Option<Vec<AdjClose>>,
}

#[derive(Debug, Deserialize)]
struct QuoteData {
    open: Vec<Option<f64>>,
    high: Vec<Option<f64>>,
    low: Vec<Option<f64>>,
    close: Vec<Option<f64>>,
    volume: Vec<Option<u64>>,
}

#[derive(Debug, Deserialize)]
struct AdjClose {
    adjclose: Vec<Option<f64>>,
}

/// Error types for Yahoo Finance operations
#[derive(Debug)]
pub enum YahooError2 {
    /// HTTP request failed
    RequestFailed(String),
    /// Failed to parse response
    ParseError(String),
    /// Invalid date format
    InvalidDate(String),
    /// No data returned
    NoData,
    /// API error
    ApiError { code: String, description: String },
}

impl std::fmt::Display for YahooError2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            YahooError2::RequestFailed(msg) => write!(f, "Request failed: {}", msg),
            YahooError2::ParseError(msg) => write!(f, "Parse error: {}", msg),
            YahooError2::InvalidDate(msg) => write!(f, "Invalid date: {}", msg),
            YahooError2::NoData => write!(f, "No data returned"),
            YahooError2::ApiError { code, description } => {
                write!(f, "API error [{}]: {}", code, description)
            }
        }
    }
}

impl std::error::Error for YahooError2 {}

/// Yahoo Finance client
#[derive(Debug, Clone)]
pub struct YahooFinance {
    base_url: String,
}

impl Default for YahooFinance {
    fn default() -> Self {
        Self::new()
    }
}

impl YahooFinance {
    /// Create a new Yahoo Finance client
    pub fn new() -> Self {
        Self {
            base_url: "https://query1.finance.yahoo.com/v8/finance/chart".to_string(),
        }
    }

    /// Parse date string (YYYY-MM-DD) to Unix timestamp
    fn parse_date(date: &str) -> Result<i64, YahooError2> {
        let parts: Vec<&str> = date.split('-').collect();
        if parts.len() != 3 {
            return Err(YahooError2::InvalidDate(format!(
                "Expected YYYY-MM-DD, got: {}",
                date
            )));
        }

        let year: i32 = parts[0]
            .parse()
            .map_err(|_| YahooError2::InvalidDate(format!("Invalid year: {}", parts[0])))?;
        let month: i32 = parts[1]
            .parse()
            .map_err(|_| YahooError2::InvalidDate(format!("Invalid month: {}", parts[1])))?;
        let day: i32 = parts[2]
            .parse()
            .map_err(|_| YahooError2::InvalidDate(format!("Invalid day: {}", parts[2])))?;

        // Simple Unix timestamp calculation (approximate, ignores leap years properly)
        let days_since_epoch = (year - 1970) * 365
            + (year - 1969) / 4  // leap years
            + Self::days_before_month(month, year)
            + day
            - 1;

        Ok(days_since_epoch as i64 * 86400)
    }

    fn days_before_month(month: i32, year: i32) -> i32 {
        let is_leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
        let days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
        let d = days[(month - 1) as usize];
        if is_leap && month > 2 {
            d + 1
        } else {
            d
        }
    }

    /// Build the API URL
    fn build_url(&self, symbol: &str, start: i64, end: i64, interval: Interval) -> String {
        format!(
            "{}/{}?period1={}&period2={}&interval={}",
            self.base_url,
            symbol,
            start,
            end,
            interval.as_str()
        )
    }

    /// Fetch historical data (async)
    #[cfg(feature = "fetch")]
    pub async fn fetch(
        &self,
        symbol: &str,
        start_date: &str,
        end_date: &str,
        interval: Interval,
    ) -> Result<Vec<Quote>, YahooError2> {
        let start = Self::parse_date(start_date)?;
        let end = Self::parse_date(end_date)?;
        let url = self.build_url(symbol, start, end, interval);

        let client = reqwest::Client::builder()
            .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
            .build()
            .map_err(|e| YahooError2::RequestFailed(e.to_string()))?;

        let response = client
            .get(&url)
            .send()
            .await
            .map_err(|e| YahooError2::RequestFailed(e.to_string()))?;

        let text = response
            .text()
            .await
            .map_err(|e| YahooError2::RequestFailed(e.to_string()))?;

        self.parse_response(&text)
    }

    /// Fetch historical data (blocking)
    #[cfg(feature = "fetch")]
    pub fn fetch_blocking(
        &self,
        symbol: &str,
        start_date: &str,
        end_date: &str,
        interval: Interval,
    ) -> Result<Vec<Quote>, YahooError2> {
        let start = Self::parse_date(start_date)?;
        let end = Self::parse_date(end_date)?;
        let url = self.build_url(symbol, start, end, interval);

        let client = reqwest::blocking::Client::builder()
            .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
            .build()
            .map_err(|e| YahooError2::RequestFailed(e.to_string()))?;

        let response = client
            .get(&url)
            .send()
            .map_err(|e| YahooError2::RequestFailed(e.to_string()))?;

        let text = response
            .text()
            .map_err(|e| YahooError2::RequestFailed(e.to_string()))?;

        self.parse_response(&text)
    }

    /// Parse Yahoo Finance API response
    fn parse_response(&self, json: &str) -> Result<Vec<Quote>, YahooError2> {
        let response: YahooResponse =
            serde_json::from_str(json).map_err(|e| YahooError2::ParseError(e.to_string()))?;

        if let Some(error) = response.chart.error {
            return Err(YahooError2::ApiError {
                code: error.code,
                description: error.description,
            });
        }

        let results = response.chart.result.ok_or(YahooError2::NoData)?;
        let data = results.first().ok_or(YahooError2::NoData)?;

        let quote_data = data.indicators.quote.first().ok_or(YahooError2::NoData)?;
        let adj_close_data = data.indicators.adjclose.as_ref().and_then(|a| a.first());

        let mut quotes = Vec::new();

        for i in 0..data.timestamp.len() {
            let open = quote_data.open.get(i).and_then(|v| *v);
            let high = quote_data.high.get(i).and_then(|v| *v);
            let low = quote_data.low.get(i).and_then(|v| *v);
            let close = quote_data.close.get(i).and_then(|v| *v);
            let volume = quote_data.volume.get(i).and_then(|v| *v);
            let adj_close = adj_close_data
                .and_then(|a| a.adjclose.get(i))
                .and_then(|v| *v);

            // Skip if any required field is missing
            if let (Some(o), Some(h), Some(l), Some(c), Some(v)) = (open, high, low, close, volume)
            {
                quotes.push(Quote {
                    timestamp: data.timestamp[i],
                    open: o,
                    high: h,
                    low: l,
                    close: c,
                    adj_close: adj_close.unwrap_or(c),
                    volume: v,
                });
            }
        }

        if quotes.is_empty() {
            return Err(YahooError2::NoData);
        }

        Ok(quotes)
    }
}

/// Convenience function to fetch stock data (async)
#[cfg(feature = "fetch")]
pub async fn fetch_stock(
    symbol: &str,
    start_date: &str,
    end_date: &str,
    interval: Interval,
) -> Result<Vec<Quote>, YahooError2> {
    YahooFinance::new()
        .fetch(symbol, start_date, end_date, interval)
        .await
}

/// Convenience function to fetch stock data (blocking)
#[cfg(feature = "fetch")]
pub fn fetch_stock_sync(
    symbol: &str,
    start_date: &str,
    end_date: &str,
    interval: Interval,
) -> Result<Vec<Quote>, YahooError2> {
    YahooFinance::new().fetch_blocking(symbol, start_date, end_date, interval)
}

/// Extract closing prices from quotes
pub fn closing_prices(quotes: &[Quote]) -> Vec<f64> {
    quotes.iter().map(|q| q.close).collect()
}

/// Extract adjusted closing prices from quotes
pub fn adj_closing_prices(quotes: &[Quote]) -> Vec<f64> {
    quotes.iter().map(|q| q.adj_close).collect()
}

/// Extract volumes from quotes
pub fn volumes(quotes: &[Quote]) -> Vec<f64> {
    quotes.iter().map(|q| q.volume as f64).collect()
}

/// Calculate daily returns from prices
pub fn daily_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return Vec::new();
    }

    prices
        .windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect()
}

/// Calculate log returns from prices
pub fn log_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return Vec::new();
    }

    prices
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect()
}

// Private method tests must stay here
#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Date Parsing Tests (private method) ====================

    #[test]
    fn test_parse_date_valid() {
        let ts = YahooFinance::parse_date("2024-01-01").unwrap();
        assert!(ts > 0);

        let ts2 = YahooFinance::parse_date("2024-06-15").unwrap();
        assert!(ts2 > ts);
    }

    #[test]
    fn test_parse_date_leap_year() {
        let feb28 = YahooFinance::parse_date("2024-02-28").unwrap();
        let feb29 = YahooFinance::parse_date("2024-02-29").unwrap();
        let mar01 = YahooFinance::parse_date("2024-03-01").unwrap();

        assert_eq!(feb29 - feb28, 86400);
        assert_eq!(mar01 - feb29, 86400);
    }

    #[test]
    fn test_parse_date_non_leap_year() {
        let feb28 = YahooFinance::parse_date("2023-02-28").unwrap();
        let mar01 = YahooFinance::parse_date("2023-03-01").unwrap();

        assert_eq!(mar01 - feb28, 86400);
    }

    #[test]
    fn test_parse_date_invalid_format() {
        assert!(YahooFinance::parse_date("2024/01/01").is_err());
        assert!(YahooFinance::parse_date("2024").is_err());
        assert!(YahooFinance::parse_date("2024-01").is_err());
        assert!(YahooFinance::parse_date("2024-1-1").is_ok());
    }

    #[test]
    fn test_parse_date_invalid_values() {
        assert!(YahooFinance::parse_date("abcd-01-01").is_err());
        assert!(YahooFinance::parse_date("2024-ab-01").is_err());
        assert!(YahooFinance::parse_date("2024-01-ab").is_err());
    }

    // ==================== Interval Tests (private method) ====================

    #[test]
    fn test_interval_str_all() {
        assert_eq!(Interval::Minute1.as_str(), "1m");
        assert_eq!(Interval::Minute5.as_str(), "5m");
        assert_eq!(Interval::Minute15.as_str(), "15m");
        assert_eq!(Interval::Minute30.as_str(), "30m");
        assert_eq!(Interval::Hour1.as_str(), "1h");
        assert_eq!(Interval::Daily.as_str(), "1d");
        assert_eq!(Interval::Weekly.as_str(), "1wk");
        assert_eq!(Interval::Monthly.as_str(), "1mo");
    }

    // ==================== URL Building Tests (private method) ====================

    #[test]
    fn test_build_url() {
        let client = YahooFinance::new();
        let url = client.build_url("AAPL", 1704067200, 1733011200, Interval::Daily);

        assert!(url.contains("AAPL"));
        assert!(url.contains("period1=1704067200"));
        assert!(url.contains("period2=1733011200"));
        assert!(url.contains("interval=1d"));
    }

    #[test]
    fn test_build_url_special_symbols() {
        let client = YahooFinance::new();
        let url1 = client.build_url("MSFT", 0, 100, Interval::Daily);
        assert!(url1.contains("MSFT"));

        let url2 = client.build_url("BRK.B", 0, 100, Interval::Daily);
        assert!(url2.contains("BRK.B"));
    }

    // ==================== Response Parsing Tests (private method) ====================

    #[test]
    fn test_parse_response_valid() {
        let client = YahooFinance::new();
        let json = r#"{"chart":{"result":[{"timestamp":[1704067200,1704153600,1704240000],"indicators":{"quote":[{"open":[185.0,186.0,187.0],"high":[186.0,187.0,188.0],"low":[184.0,185.0,186.0],"close":[185.5,186.5,187.5],"volume":[1000000,1100000,1200000]}],"adjclose":[{"adjclose":[185.5,186.5,187.5]}]}}],"error":null}}"#;
        let quotes = client.parse_response(json).unwrap();
        assert_eq!(quotes.len(), 3);
        assert_eq!(quotes[0].close, 185.5);
    }

    #[test]
    fn test_parse_response_with_nulls() {
        let client = YahooFinance::new();
        let json = r#"{"chart":{"result":[{"timestamp":[1704067200,1704153600,1704240000],"indicators":{"quote":[{"open":[185.0,null,187.0],"high":[186.0,null,188.0],"low":[184.0,null,186.0],"close":[185.5,null,187.5],"volume":[1000000,null,1200000]}],"adjclose":[{"adjclose":[185.5,null,187.5]}]}}],"error":null}}"#;
        let quotes = client.parse_response(json).unwrap();
        assert_eq!(quotes.len(), 2);
    }

    #[test]
    fn test_parse_response_api_error() {
        let client = YahooFinance::new();
        let json = r#"{"chart":{"result":null,"error":{"code":"Not Found","description":"No data found"}}}"#;
        let result = client.parse_response(json);
        assert!(matches!(result, Err(YahooError2::ApiError { .. })));
    }

    #[test]
    fn test_parse_response_no_data() {
        let client = YahooFinance::new();
        let json = r#"{"chart":{"result":[],"error":null}}"#;
        assert!(matches!(client.parse_response(json), Err(YahooError2::NoData)));
    }

    #[test]
    fn test_parse_response_invalid_json() {
        let client = YahooFinance::new();
        assert!(matches!(client.parse_response("not json"), Err(YahooError2::ParseError(_))));
    }

    // ==================== Private field/method tests ====================

    #[test]
    fn test_yahoo_finance_base_url() {
        let client1 = YahooFinance::new();
        let client2 = YahooFinance::default();
        assert_eq!(client1.base_url, client2.base_url);
        assert!(client1.base_url.contains("yahoo"));
    }

    #[test]
    fn test_days_before_month() {
        assert_eq!(YahooFinance::days_before_month(1, 2023), 0);
        assert_eq!(YahooFinance::days_before_month(2, 2023), 31);
        assert_eq!(YahooFinance::days_before_month(3, 2023), 59);
        assert_eq!(YahooFinance::days_before_month(12, 2023), 334);

        assert_eq!(YahooFinance::days_before_month(2, 2024), 31);
        assert_eq!(YahooFinance::days_before_month(3, 2024), 60);
        assert_eq!(YahooFinance::days_before_month(12, 2024), 335);
    }
}
