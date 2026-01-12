//! Fixture-based data source loading from indicator-core test fixtures.

use optimizer_spi::{DataSource, MarketData, Timeframe, OptimizerError, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

/// Data source loading from indicator-core test fixtures.
pub struct FixtureDataSource {
    fixtures_dir: PathBuf,
    cache: HashMap<(String, Timeframe), MarketData>,
}

impl FixtureDataSource {
    /// Create new FixtureDataSource with default fixtures path.
    pub fn new() -> Self {
        // CARGO_MANIFEST_DIR is optimizer/optimizer-core
        // We need crates/indicator/indicator-core/tests/fixtures
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let fixtures_dir = manifest_dir
            .parent()  // optimizer
            .and_then(|p| p.parent())  // crates
            .map(|p| p.join("indicator/indicator-core/tests/fixtures"))
            .unwrap_or_else(|| PathBuf::from("tests/fixtures"));

        Self {
            fixtures_dir,
            cache: HashMap::new(),
        }
    }

    /// Create with custom fixtures directory.
    pub fn with_dir(fixtures_dir: PathBuf) -> Self {
        Self {
            fixtures_dir,
            cache: HashMap::new(),
        }
    }

    /// Map symbol to fixture filename.
    fn symbol_to_filename(&self, symbol: &str, timeframe: Timeframe) -> String {
        let base = match symbol.to_uppercase().as_str() {
            "SPY" => "spy",
            "NAS100" | "QQQ" => "nas100",
            "GLD" => "gld",
            "META" => "meta",
            "AAPL" => "aapl",
            "AMZN" => "amzn",
            "NFLX" => "nflx",
            "GOOGL" => "googl",
            "BTC" | "BTCUSD" | "BTC-USD" => "btc-usd",
            "EURUSD" => "eurusd",
            "GBPUSD" => "gbpusd",
            "USDJPY" => "usdjpy",
            "GBPJPY" => "gbpjpy",
            "EURJPY" => "eurjpy",
            _ => return format!("{}.json", symbol.to_lowercase()),
        };

        let tf_suffix = match timeframe {
            Timeframe::D1 => "1d",
            Timeframe::H1 => "1h",
            Timeframe::H4 => "1h", // H4 is aggregated from H1
            _ => "1d",
        };

        format!("{}_{}.json", base, tf_suffix)
    }

    /// Load fixture from JSON file.
    fn load_fixture(&self, filename: &str) -> Result<(String, Vec<i64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
        let path = self.fixtures_dir.join(filename);
        let content = fs::read_to_string(&path)
            .map_err(|e| OptimizerError::InvalidConfig(format!("Failed to read {}: {}", path.display(), e)))?;

        let response: YahooResponse = serde_json::from_str(&content)
            .map_err(|e| OptimizerError::InvalidConfig(format!("Failed to parse JSON: {}", e)))?;

        let result = response.chart.result
            .ok_or_else(|| OptimizerError::InvalidConfig("No chart result".to_string()))?
            .into_iter()
            .next()
            .ok_or_else(|| OptimizerError::InvalidConfig("Empty chart result".to_string()))?;

        let timestamps = result.timestamp
            .ok_or_else(|| OptimizerError::InvalidConfig("No timestamps".to_string()))?;

        let quote = result.indicators.quote.into_iter()
            .next()
            .ok_or_else(|| OptimizerError::InvalidConfig("No quote data".to_string()))?;

        let opens = quote.open.ok_or_else(|| OptimizerError::InvalidConfig("No open data".to_string()))?;
        let highs = quote.high.ok_or_else(|| OptimizerError::InvalidConfig("No high data".to_string()))?;
        let lows = quote.low.ok_or_else(|| OptimizerError::InvalidConfig("No low data".to_string()))?;
        let closes = quote.close.ok_or_else(|| OptimizerError::InvalidConfig("No close data".to_string()))?;
        let volumes = quote.volume.unwrap_or_else(|| vec![Some(0); timestamps.len()]);

        // Filter out None values
        let mut open = Vec::new();
        let mut high = Vec::new();
        let mut low = Vec::new();
        let mut close = Vec::new();
        let mut volume = Vec::new();
        let mut valid_ts = Vec::new();

        for i in 0..timestamps.len() {
            if let (Some(o), Some(h), Some(l), Some(c)) = (
                opens.get(i).and_then(|x| *x),
                highs.get(i).and_then(|x| *x),
                lows.get(i).and_then(|x| *x),
                closes.get(i).and_then(|x| *x),
            ) {
                valid_ts.push(timestamps[i]);
                open.push(o);
                high.push(h);
                low.push(l);
                close.push(c);
                volume.push(volumes.get(i).and_then(|x| *x).unwrap_or(0) as f64);
            }
        }

        Ok((result.meta.symbol, valid_ts, open, high, low, close, volume))
    }
}

impl Default for FixtureDataSource {
    fn default() -> Self {
        Self::new()
    }
}

impl DataSource for FixtureDataSource {
    fn load(&self, symbol: &str, timeframe: Timeframe) -> Result<MarketData> {
        let filename = self.symbol_to_filename(symbol, timeframe);
        let (sym, timestamps, open, high, low, close, volume) = self.load_fixture(&filename)?;

        let mut data = MarketData {
            symbol: sym,
            timeframe,
            timestamps,
            open,
            high,
            low,
            close,
            volume,
        };

        // Aggregate if H4 requested
        if timeframe == Timeframe::H4 {
            data = data.aggregate(4);
            data.timeframe = Timeframe::H4;
        }

        Ok(data)
    }

    fn symbols(&self) -> Vec<String> {
        vec![
            "SPY".to_string(),
            "NAS100".to_string(),
            "GLD".to_string(),
            "META".to_string(),
            "AAPL".to_string(),
            "AMZN".to_string(),
            "NFLX".to_string(),
            "GOOGL".to_string(),
            "BTC-USD".to_string(),
            "EURUSD".to_string(),
            "GBPUSD".to_string(),
            "USDJPY".to_string(),
            "GBPJPY".to_string(),
            "EURJPY".to_string(),
        ]
    }

    fn timeframes(&self, _symbol: &str) -> Vec<Timeframe> {
        vec![Timeframe::D1, Timeframe::H4, Timeframe::H1]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixture_datasource_load() {
        let ds = FixtureDataSource::new();

        // Try to load SPY daily
        if let Ok(data) = ds.load("SPY", Timeframe::D1) {
            assert!(!data.is_empty());
            println!("Loaded SPY D1: {} bars", data.len());
        }
    }

    #[test]
    fn test_fixture_datasource_h4() {
        let ds = FixtureDataSource::new();

        // Try to load SPY H4 (aggregated from H1)
        if let Ok(data) = ds.load("SPY", Timeframe::H4) {
            assert!(!data.is_empty());
            assert_eq!(data.timeframe, Timeframe::H4);
            println!("Loaded SPY H4: {} bars", data.len());
        }
    }

    #[test]
    fn test_symbols_list() {
        let ds = FixtureDataSource::new();
        let symbols = ds.symbols();
        assert!(symbols.contains(&"SPY".to_string()));
        assert!(symbols.contains(&"AAPL".to_string()));
        assert!(symbols.contains(&"EURUSD".to_string()));
    }
}
