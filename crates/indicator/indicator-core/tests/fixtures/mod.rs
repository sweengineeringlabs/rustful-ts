//! Test Fixtures - Real Market Data
//!
//! This module provides real market data for indicator testing.
//! Data is sourced from Yahoo Finance and stored as JSON fixtures.
//!
//! Available fixtures:
//!
//! ## Index ETFs (Full History)
//! - SPY (S&P 500 ETF) - Daily since 1993, Hourly/4H ~1yr
//! - NAS100/QQQ (Nasdaq 100 ETF) - Daily since 1999, Hourly/4H ~1yr
//! - GLD (Gold ETF) - Daily since 2004, Hourly/4H ~1yr
//!
//! ## FAANG Stocks (Full History)
//! - META (Facebook/Meta) - Daily since 2012, Hourly/4H ~1yr
//! - AAPL (Apple) - Daily since 1980, Hourly/4H ~1yr
//! - AMZN (Amazon) - Daily since 1997, Hourly/4H ~1yr
//! - NFLX (Netflix) - Daily since 2002, Hourly/4H ~1yr
//! - GOOGL (Google/Alphabet) - Daily since 2004, Hourly/4H ~1yr
//!
//! ## Crypto
//! - BTC-USD (Bitcoin) - Daily since 2014, Hourly/4H ~1yr
//!
//! ## Forex (Full History ~2003+)
//! - EURUSD - Daily + Hourly/4H
//! - GBPUSD - Daily + Hourly/4H
//! - USDJPY - Daily + Hourly/4H
//! - GBPJPY - Daily + Hourly/4H
//! - EURJPY - Daily + Hourly/4H
//!
//! Note: 4H timeframe is not available on Yahoo Finance.
//! 4H data is aggregated from 1H bars in code.

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

    /// Aggregate bars to higher timeframe (e.g., 1H -> 4H)
    /// `factor` is the number of bars to aggregate (e.g., 4 for 1H->4H)
    pub fn aggregate(&self, factor: usize) -> OHLCVData {
        let mut timestamps = Vec::new();
        let mut open = Vec::new();
        let mut high = Vec::new();
        let mut low = Vec::new();
        let mut close = Vec::new();
        let mut volume = Vec::new();

        for chunk in self.timestamps.chunks(factor) {
            if chunk.len() < factor {
                break; // Skip incomplete final chunk
            }
            let start_idx = self.timestamps.iter().position(|&t| t == chunk[0]).unwrap();
            let end_idx = start_idx + factor;

            timestamps.push(chunk[0]);
            open.push(self.open[start_idx]);
            high.push(
                self.high[start_idx..end_idx]
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max),
            );
            low.push(
                self.low[start_idx..end_idx]
                    .iter()
                    .cloned()
                    .fold(f64::INFINITY, f64::min),
            );
            close.push(self.close[end_idx - 1]);
            volume.push(self.volume[start_idx..end_idx].iter().sum());
        }

        OHLCVData {
            symbol: self.symbol.clone(),
            timestamps,
            open,
            high,
            low,
            close,
            volume,
        }
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
    // Volume might be None for forex pairs
    let volumes = quote.volume.unwrap_or_else(|| vec![Some(0); timestamps.len()]);

    // Filter out None values and build vectors
    let mut open = Vec::new();
    let mut high = Vec::new();
    let mut low = Vec::new();
    let mut close = Vec::new();
    let mut volume = Vec::new();
    let mut valid_timestamps = Vec::new();

    for i in 0..timestamps.len() {
        if let (Some(o), Some(h), Some(l), Some(c)) = (
            opens.get(i).and_then(|x| *x),
            highs.get(i).and_then(|x| *x),
            lows.get(i).and_then(|x| *x),
            closes.get(i).and_then(|x| *x),
        ) {
            valid_timestamps.push(timestamps[i]);
            open.push(o);
            high.push(h);
            low.push(l);
            close.push(c);
            // Volume might be None for some bars (forex)
            volume.push(volumes.get(i).and_then(|x| *x).unwrap_or(0) as f64);
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

// ============================================================================
// Equity Fixtures
// ============================================================================

/// Load SPY daily data (S&P 500 ETF) - Full history since 1993
pub fn spy_daily() -> OHLCVData {
    load_fixture("spy_1d.json").expect("Failed to load SPY daily fixture")
}

/// Load SPY hourly data (~1 year)
pub fn spy_hourly() -> OHLCVData {
    load_fixture("spy_1h.json").expect("Failed to load SPY hourly fixture")
}

/// Load SPY 4H data (aggregated from 1H)
pub fn spy_4h() -> OHLCVData {
    spy_hourly().aggregate(4)
}

/// Load NAS100/QQQ daily data (Nasdaq 100 ETF) - Full history since 1999
pub fn nas100_daily() -> OHLCVData {
    load_fixture("nas100_1d.json").expect("Failed to load NAS100 daily fixture")
}

/// Load NAS100/QQQ hourly data (~1 year)
pub fn nas100_hourly() -> OHLCVData {
    load_fixture("nas100_1h.json").expect("Failed to load NAS100 hourly fixture")
}

/// Load NAS100 4H data (aggregated from 1H)
pub fn nas100_4h() -> OHLCVData {
    nas100_hourly().aggregate(4)
}

/// Load BTC-USD daily data (Bitcoin) - Full history since 2014
pub fn btc_daily() -> OHLCVData {
    load_fixture("btc-usd_1d.json").expect("Failed to load BTC-USD daily fixture")
}

/// Load BTC-USD hourly data (~1 year)
pub fn btc_hourly() -> OHLCVData {
    load_fixture("btc-usd_1h.json").expect("Failed to load BTC-USD hourly fixture")
}

/// Load BTC 4H data (aggregated from 1H)
pub fn btc_4h() -> OHLCVData {
    btc_hourly().aggregate(4)
}

/// Load GLD daily data (Gold ETF) - Full history since 2004
pub fn gld_daily() -> OHLCVData {
    load_fixture("gld_1d.json").expect("Failed to load GLD daily fixture")
}

/// Load GLD hourly data (~1 year)
pub fn gld_hourly() -> OHLCVData {
    load_fixture("gld_1h.json").expect("Failed to load GLD hourly fixture")
}

/// Load GLD 4H data (aggregated from 1H)
pub fn gld_4h() -> OHLCVData {
    gld_hourly().aggregate(4)
}

// ============================================================================
// FAANG Fixtures
// ============================================================================

/// Load META daily data (Facebook/Meta) - Full history since 2012
pub fn meta_daily() -> OHLCVData {
    load_fixture("meta_1d.json").expect("Failed to load META daily fixture")
}

/// Load META hourly data (~1 year)
pub fn meta_hourly() -> OHLCVData {
    load_fixture("meta_1h.json").expect("Failed to load META hourly fixture")
}

/// Load META 4H data (aggregated from 1H)
pub fn meta_4h() -> OHLCVData {
    meta_hourly().aggregate(4)
}

/// Load AAPL daily data (Apple) - Full history since 1980
pub fn aapl_daily() -> OHLCVData {
    load_fixture("aapl_1d.json").expect("Failed to load AAPL daily fixture")
}

/// Load AAPL hourly data (~1 year)
pub fn aapl_hourly() -> OHLCVData {
    load_fixture("aapl_1h.json").expect("Failed to load AAPL hourly fixture")
}

/// Load AAPL 4H data (aggregated from 1H)
pub fn aapl_4h() -> OHLCVData {
    aapl_hourly().aggregate(4)
}

/// Load AMZN daily data (Amazon) - Full history since 1997
pub fn amzn_daily() -> OHLCVData {
    load_fixture("amzn_1d.json").expect("Failed to load AMZN daily fixture")
}

/// Load AMZN hourly data (~1 year)
pub fn amzn_hourly() -> OHLCVData {
    load_fixture("amzn_1h.json").expect("Failed to load AMZN hourly fixture")
}

/// Load AMZN 4H data (aggregated from 1H)
pub fn amzn_4h() -> OHLCVData {
    amzn_hourly().aggregate(4)
}

/// Load NFLX daily data (Netflix) - Full history since 2002
pub fn nflx_daily() -> OHLCVData {
    load_fixture("nflx_1d.json").expect("Failed to load NFLX daily fixture")
}

/// Load NFLX hourly data (~1 year)
pub fn nflx_hourly() -> OHLCVData {
    load_fixture("nflx_1h.json").expect("Failed to load NFLX hourly fixture")
}

/// Load NFLX 4H data (aggregated from 1H)
pub fn nflx_4h() -> OHLCVData {
    nflx_hourly().aggregate(4)
}

/// Load GOOGL daily data (Google/Alphabet) - Full history since 2004
pub fn googl_daily() -> OHLCVData {
    load_fixture("googl_1d.json").expect("Failed to load GOOGL daily fixture")
}

/// Load GOOGL hourly data (~1 year)
pub fn googl_hourly() -> OHLCVData {
    load_fixture("googl_1h.json").expect("Failed to load GOOGL hourly fixture")
}

/// Load GOOGL 4H data (aggregated from 1H)
pub fn googl_4h() -> OHLCVData {
    googl_hourly().aggregate(4)
}

// ============================================================================
// Forex Fixtures
// ============================================================================

/// Load EURUSD daily data - Full history since ~2003
pub fn eurusd_daily() -> OHLCVData {
    load_fixture("eurusd_1d.json").expect("Failed to load EURUSD daily fixture")
}

/// Load EURUSD hourly data (~1 year)
pub fn eurusd_hourly() -> OHLCVData {
    load_fixture("eurusd_1h.json").expect("Failed to load EURUSD hourly fixture")
}

/// Load EURUSD 4H data (aggregated from 1H)
pub fn eurusd_4h() -> OHLCVData {
    eurusd_hourly().aggregate(4)
}

/// Load GBPUSD daily data - Full history since ~2003
pub fn gbpusd_daily() -> OHLCVData {
    load_fixture("gbpusd_1d.json").expect("Failed to load GBPUSD daily fixture")
}

/// Load GBPUSD hourly data (~1 year)
pub fn gbpusd_hourly() -> OHLCVData {
    load_fixture("gbpusd_1h.json").expect("Failed to load GBPUSD hourly fixture")
}

/// Load GBPUSD 4H data (aggregated from 1H)
pub fn gbpusd_4h() -> OHLCVData {
    gbpusd_hourly().aggregate(4)
}

/// Load USDJPY daily data - Full history since ~2003
pub fn usdjpy_daily() -> OHLCVData {
    load_fixture("usdjpy_1d.json").expect("Failed to load USDJPY daily fixture")
}

/// Load USDJPY hourly data (~1 year)
pub fn usdjpy_hourly() -> OHLCVData {
    load_fixture("usdjpy_1h.json").expect("Failed to load USDJPY hourly fixture")
}

/// Load USDJPY 4H data (aggregated from 1H)
pub fn usdjpy_4h() -> OHLCVData {
    usdjpy_hourly().aggregate(4)
}

/// Load GBPJPY daily data - Full history since ~2003
pub fn gbpjpy_daily() -> OHLCVData {
    load_fixture("gbpjpy_1d.json").expect("Failed to load GBPJPY daily fixture")
}

/// Load GBPJPY hourly data (~1 year)
pub fn gbpjpy_hourly() -> OHLCVData {
    load_fixture("gbpjpy_1h.json").expect("Failed to load GBPJPY hourly fixture")
}

/// Load GBPJPY 4H data (aggregated from 1H)
pub fn gbpjpy_4h() -> OHLCVData {
    gbpjpy_hourly().aggregate(4)
}

/// Load EURJPY daily data - Full history since ~2003
pub fn eurjpy_daily() -> OHLCVData {
    load_fixture("eurjpy_1d.json").expect("Failed to load EURJPY daily fixture")
}

/// Load EURJPY hourly data (~1 year)
pub fn eurjpy_hourly() -> OHLCVData {
    load_fixture("eurjpy_1h.json").expect("Failed to load EURJPY hourly fixture")
}

/// Load EURJPY 4H data (aggregated from 1H)
pub fn eurjpy_4h() -> OHLCVData {
    eurjpy_hourly().aggregate(4)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Equity tests
    #[test]
    fn test_load_spy_daily() {
        let data = spy_daily();
        assert!(!data.is_empty());
        assert!(data.len() > 8000, "SPY should have full history since 1993");
        println!("SPY daily: {} bars", data.len());
    }

    #[test]
    fn test_load_spy_hourly() {
        let data = spy_hourly();
        assert!(!data.is_empty());
        assert!(data.len() > 1000, "SPY hourly should have ~1yr of data");
        println!("SPY hourly: {} bars", data.len());
    }

    #[test]
    fn test_load_spy_4h() {
        let data = spy_4h();
        assert!(!data.is_empty());
        println!("SPY 4H (aggregated): {} bars", data.len());
    }

    #[test]
    fn test_load_nas100_daily() {
        let data = nas100_daily();
        assert!(!data.is_empty());
        assert!(data.len() > 5000, "NAS100 should have history since 1999");
        println!("NAS100 daily: {} bars", data.len());
    }

    #[test]
    fn test_load_nas100_hourly() {
        let data = nas100_hourly();
        assert!(!data.is_empty());
        println!("NAS100 hourly: {} bars", data.len());
    }

    #[test]
    fn test_load_btc_daily() {
        let data = btc_daily();
        assert!(!data.is_empty());
        assert!(data.len() > 3000, "BTC should have history since 2014");
        println!("BTC daily: {} bars", data.len());
    }

    #[test]
    fn test_load_btc_hourly() {
        let data = btc_hourly();
        assert!(!data.is_empty());
        println!("BTC hourly: {} bars", data.len());
    }

    #[test]
    fn test_load_aapl_daily() {
        let data = aapl_daily();
        assert!(!data.is_empty());
        assert!(data.len() > 10000, "AAPL should have history since 1980");
        println!("AAPL daily: {} bars", data.len());
    }

    #[test]
    fn test_load_gld_daily() {
        let data = gld_daily();
        assert!(!data.is_empty());
        assert!(data.len() > 4000, "GLD should have history since 2004");
        println!("GLD daily: {} bars", data.len());
    }

    #[test]
    fn test_load_gld_hourly() {
        let data = gld_hourly();
        assert!(!data.is_empty());
        println!("GLD hourly: {} bars", data.len());
    }

    // FAANG tests
    #[test]
    fn test_load_meta_daily() {
        let data = meta_daily();
        assert!(!data.is_empty());
        assert!(data.len() > 2500, "META should have history since 2012");
        println!("META daily: {} bars", data.len());
    }

    #[test]
    fn test_load_meta_hourly() {
        let data = meta_hourly();
        assert!(!data.is_empty());
        println!("META hourly: {} bars", data.len());
    }

    #[test]
    fn test_load_aapl_hourly() {
        let data = aapl_hourly();
        assert!(!data.is_empty());
        println!("AAPL hourly: {} bars", data.len());
    }

    #[test]
    fn test_load_amzn_daily() {
        let data = amzn_daily();
        assert!(!data.is_empty());
        assert!(data.len() > 6000, "AMZN should have history since 1997");
        println!("AMZN daily: {} bars", data.len());
    }

    #[test]
    fn test_load_amzn_hourly() {
        let data = amzn_hourly();
        assert!(!data.is_empty());
        println!("AMZN hourly: {} bars", data.len());
    }

    #[test]
    fn test_load_nflx_daily() {
        let data = nflx_daily();
        assert!(!data.is_empty());
        assert!(data.len() > 5000, "NFLX should have history since 2002");
        println!("NFLX daily: {} bars", data.len());
    }

    #[test]
    fn test_load_nflx_hourly() {
        let data = nflx_hourly();
        assert!(!data.is_empty());
        println!("NFLX hourly: {} bars", data.len());
    }

    #[test]
    fn test_load_googl_daily() {
        let data = googl_daily();
        assert!(!data.is_empty());
        assert!(data.len() > 4000, "GOOGL should have history since 2004");
        println!("GOOGL daily: {} bars", data.len());
    }

    #[test]
    fn test_load_googl_hourly() {
        let data = googl_hourly();
        assert!(!data.is_empty());
        println!("GOOGL hourly: {} bars", data.len());
    }

    // Forex tests
    #[test]
    fn test_load_eurusd_daily() {
        let data = eurusd_daily();
        assert!(!data.is_empty());
        assert!(data.len() > 5000, "EURUSD should have history since 2003");
        println!("EURUSD daily: {} bars", data.len());
    }

    #[test]
    fn test_load_eurusd_hourly() {
        let data = eurusd_hourly();
        assert!(!data.is_empty());
        println!("EURUSD hourly: {} bars", data.len());
    }

    #[test]
    fn test_load_gbpusd_daily() {
        let data = gbpusd_daily();
        assert!(!data.is_empty());
        println!("GBPUSD daily: {} bars", data.len());
    }

    #[test]
    fn test_load_gbpusd_hourly() {
        let data = gbpusd_hourly();
        assert!(!data.is_empty());
        println!("GBPUSD hourly: {} bars", data.len());
    }

    #[test]
    fn test_load_usdjpy_daily() {
        let data = usdjpy_daily();
        assert!(!data.is_empty());
        println!("USDJPY daily: {} bars", data.len());
    }

    #[test]
    fn test_load_usdjpy_hourly() {
        let data = usdjpy_hourly();
        assert!(!data.is_empty());
        println!("USDJPY hourly: {} bars", data.len());
    }

    #[test]
    fn test_load_gbpjpy_daily() {
        let data = gbpjpy_daily();
        assert!(!data.is_empty());
        println!("GBPJPY daily: {} bars", data.len());
    }

    #[test]
    fn test_load_gbpjpy_hourly() {
        let data = gbpjpy_hourly();
        assert!(!data.is_empty());
        println!("GBPJPY hourly: {} bars", data.len());
    }

    #[test]
    fn test_load_eurjpy_daily() {
        let data = eurjpy_daily();
        assert!(!data.is_empty());
        println!("EURJPY daily: {} bars", data.len());
    }

    #[test]
    fn test_load_eurjpy_hourly() {
        let data = eurjpy_hourly();
        assert!(!data.is_empty());
        println!("EURJPY hourly: {} bars", data.len());
    }

    // Data integrity tests
    #[test]
    fn test_data_integrity_spy() {
        let data = spy_daily();
        verify_data_integrity(&data, "SPY");
    }

    #[test]
    fn test_data_integrity_forex() {
        let data = eurusd_daily();
        // Forex data from Yahoo can have minor OHLC inconsistencies due to rounding
        // Use relaxed verification
        verify_data_integrity_relaxed(&data, "EURUSD");
    }

    #[test]
    fn test_aggregation() {
        let h1 = spy_hourly();
        let h4 = spy_4h();

        // 4H should have roughly 1/4 the bars
        let expected_ratio = h1.len() / 4;
        assert!(
            (h4.len() as i32 - expected_ratio as i32).abs() <= 1,
            "4H aggregation: expected ~{}, got {}",
            expected_ratio,
            h4.len()
        );

        // Verify 4H high >= all constituent 1H highs
        // (spot check first 4H bar)
        if h1.len() >= 4 {
            let h1_max = h1.high[0..4].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            assert_eq!(h4.high[0], h1_max, "4H high should be max of 1H highs");
        }
    }

    fn verify_data_integrity(data: &OHLCVData, name: &str) {
        // Check all arrays have same length
        assert_eq!(data.open.len(), data.close.len(), "{} open/close mismatch", name);
        assert_eq!(data.high.len(), data.close.len(), "{} high/close mismatch", name);
        assert_eq!(data.low.len(), data.close.len(), "{} low/close mismatch", name);
        assert_eq!(data.volume.len(), data.close.len(), "{} volume/close mismatch", name);
        assert_eq!(data.timestamps.len(), data.close.len(), "{} timestamp/close mismatch", name);

        // Check OHLC relationships
        for i in 0..data.len() {
            assert!(data.high[i] >= data.low[i], "{} High < Low at {}", name, i);
            assert!(data.high[i] >= data.open[i], "{} High < Open at {}", name, i);
            assert!(data.high[i] >= data.close[i], "{} High < Close at {}", name, i);
            assert!(data.low[i] <= data.open[i], "{} Low > Open at {}", name, i);
            assert!(data.low[i] <= data.close[i], "{} Low > Close at {}", name, i);
        }
    }

    fn verify_data_integrity_relaxed(data: &OHLCVData, name: &str) {
        // Check all arrays have same length
        assert_eq!(data.open.len(), data.close.len(), "{} open/close mismatch", name);
        assert_eq!(data.high.len(), data.close.len(), "{} high/close mismatch", name);
        assert_eq!(data.low.len(), data.close.len(), "{} low/close mismatch", name);
        assert_eq!(data.volume.len(), data.close.len(), "{} volume/close mismatch", name);
        assert_eq!(data.timestamps.len(), data.close.len(), "{} timestamp/close mismatch", name);

        // Relaxed OHLC check - allow tiny tolerance for forex rounding issues
        let tolerance = 0.0001; // 0.01% tolerance for forex pip rounding
        let mut violations = 0;
        for i in 0..data.len() {
            let h = data.high[i];
            let l = data.low[i];
            let o = data.open[i];
            let c = data.close[i];

            // Check with tolerance
            if h < l - tolerance * l { violations += 1; }
            if h < o - tolerance * o { violations += 1; }
            if h < c - tolerance * c { violations += 1; }
            if l > o + tolerance * o { violations += 1; }
            if l > c + tolerance * c { violations += 1; }
        }

        // Allow up to 2% of bars to have minor violations (Yahoo forex data quality)
        let max_violations = data.len() / 50 + 1;
        assert!(
            violations <= max_violations,
            "{} has too many OHLC violations: {} (max allowed: {})",
            name, violations, max_violations
        );
    }
}
