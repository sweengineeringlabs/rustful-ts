//! Test Fixtures - Real Market Data
//!
//! This module provides real market data for indicator testing.
//! Data is sourced from Yahoo Finance and stored as JSON fixtures.
//!
//! Available fixtures:
//! - SPY (S&P 500 ETF) - Daily 2024
//! - BTC-USD (Bitcoin) - Daily 2024
//! - AAPL (Apple) - Daily 2024
//! - GLD (Gold ETF) - Daily 2024
//! - SPY 1H - Hourly December 2024

use serde::Deserialize;
use std::fs;
use std::path::PathBuf;

/// OHLCV bar data
#[derive(Debug, Clone)]
pub struct OHLCVData {
    pub symbol: String,
    pub timestamps: Vec<i64>,
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Vec<f64>,
}

impl OHLCVData {
    /// Get number of bars
    pub fn len(&self) -> usize {
        self.close.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.close.is_empty()
    }

    /// Get typical price (HLC/3)
    pub fn typical_price(&self) -> Vec<f64> {
        self.high
            .iter()
            .zip(self.low.iter())
            .zip(self.close.iter())
            .map(|((h, l), c)| (h + l + c) / 3.0)
            .collect()
    }
}

// Yahoo Finance JSON response structures
#[derive(Debug, Deserialize)]
struct YahooResponse {
    chart: ChartResult,
}

#[derive(Debug, Deserialize)]
struct ChartResult {
    result: Option<Vec<ChartData>>,
}

#[derive(Debug, Deserialize)]
struct ChartData {
    meta: Meta,
    timestamp: Option<Vec<i64>>,
    indicators: Indicators,
}

#[derive(Debug, Deserialize)]
struct Meta {
    symbol: String,
}

#[derive(Debug, Deserialize)]
struct Indicators {
    quote: Vec<QuoteData>,
}

#[derive(Debug, Deserialize)]
struct QuoteData {
    open: Option<Vec<Option<f64>>>,
    high: Option<Vec<Option<f64>>>,
    low: Option<Vec<Option<f64>>>,
    close: Option<Vec<Option<f64>>>,
    volume: Option<Vec<Option<u64>>>,
}

/// Get the fixtures directory path
fn fixtures_dir() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests");
    path.push("fixtures");
    path
}

/// Load fixture from JSON file
fn load_fixture(filename: &str) -> Option<OHLCVData> {
    let path = fixtures_dir().join(filename);
    let content = fs::read_to_string(&path).ok()?;
    let response: YahooResponse = serde_json::from_str(&content).ok()?;

    let result = response.chart.result?.into_iter().next()?;
    let timestamps = result.timestamp?;
    let quote = result.indicators.quote.into_iter().next()?;

    let opens = quote.open?;
    let highs = quote.high?;
    let lows = quote.low?;
    let closes = quote.close?;
    let volumes = quote.volume?;

    // Filter out None values and build vectors
    let mut open = Vec::new();
    let mut high = Vec::new();
    let mut low = Vec::new();
    let mut close = Vec::new();
    let mut volume = Vec::new();
    let mut valid_timestamps = Vec::new();

    for i in 0..timestamps.len() {
        if let (Some(o), Some(h), Some(l), Some(c), Some(v)) = (
            opens.get(i).and_then(|x| *x),
            highs.get(i).and_then(|x| *x),
            lows.get(i).and_then(|x| *x),
            closes.get(i).and_then(|x| *x),
            volumes.get(i).and_then(|x| *x),
        ) {
            valid_timestamps.push(timestamps[i]);
            open.push(o);
            high.push(h);
            low.push(l);
            close.push(c);
            volume.push(v as f64);
        }
    }

    Some(OHLCVData {
        symbol: result.meta.symbol,
        timestamps: valid_timestamps,
        open,
        high,
        low,
        close,
        volume,
    })
}

/// Load SPY daily data (S&P 500 ETF)
pub fn spy_daily() -> OHLCVData {
    load_fixture("spy_1d.json").expect("Failed to load SPY daily fixture")
}

/// Load BTC-USD daily data (Bitcoin)
pub fn btc_daily() -> OHLCVData {
    load_fixture("btc-usd_1d.json").expect("Failed to load BTC-USD daily fixture")
}

/// Load AAPL daily data (Apple)
pub fn aapl_daily() -> OHLCVData {
    load_fixture("aapl_1d.json").expect("Failed to load AAPL daily fixture")
}

/// Load GLD daily data (Gold ETF)
pub fn gld_daily() -> OHLCVData {
    load_fixture("gld_1d.json").expect("Failed to load GLD daily fixture")
}

/// Load SPY hourly data
pub fn spy_hourly() -> OHLCVData {
    load_fixture("spy_1h.json").expect("Failed to load SPY hourly fixture")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_spy_daily() {
        let data = spy_daily();
        assert!(!data.is_empty());
        assert!(data.len() > 200); // Should have ~252 trading days
        println!("SPY daily: {} bars", data.len());
    }

    #[test]
    fn test_load_btc_daily() {
        let data = btc_daily();
        assert!(!data.is_empty());
        assert!(data.len() > 300); // BTC trades 365 days
        println!("BTC daily: {} bars", data.len());
    }

    #[test]
    fn test_load_aapl_daily() {
        let data = aapl_daily();
        assert!(!data.is_empty());
        println!("AAPL daily: {} bars", data.len());
    }

    #[test]
    fn test_load_gld_daily() {
        let data = gld_daily();
        assert!(!data.is_empty());
        println!("GLD daily: {} bars", data.len());
    }

    #[test]
    fn test_load_spy_hourly() {
        let data = spy_hourly();
        assert!(!data.is_empty());
        println!("SPY hourly: {} bars", data.len());
    }

    #[test]
    fn test_data_integrity() {
        let data = spy_daily();

        // Check all arrays have same length
        assert_eq!(data.open.len(), data.close.len());
        assert_eq!(data.high.len(), data.close.len());
        assert_eq!(data.low.len(), data.close.len());
        assert_eq!(data.volume.len(), data.close.len());
        assert_eq!(data.timestamps.len(), data.close.len());

        // Check OHLC relationships
        for i in 0..data.len() {
            assert!(data.high[i] >= data.low[i], "High < Low at {}", i);
            assert!(data.high[i] >= data.open[i], "High < Open at {}", i);
            assert!(data.high[i] >= data.close[i], "High < Close at {}", i);
            assert!(data.low[i] <= data.open[i], "Low > Open at {}", i);
            assert!(data.low[i] <= data.close[i], "Low > Close at {}", i);
        }
    }
}
