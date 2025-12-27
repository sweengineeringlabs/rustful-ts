//! # rustful-cli
//!
//! Command-line interface for the rustful-ts time series library.

use algorithm::smoothing::{
    DoubleExponentialSmoothing, HoltWinters, SeasonalType, SimpleExponentialSmoothing,
};
use algorithm::regression::Arima;
use algorithm::Predictor;
use clap::{Parser, Subcommand};
use anomaly::{AnomalyDetector, IQRDetector, ZScoreDetector};
use financial::{BacktestResult, Signal, SignalGenerator, Trade};
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

type CliResult<T> = std::result::Result<T, String>;

#[derive(Parser)]
#[command(name = "rustful")]
#[command(about = "Time series forecasting CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate forecasts from input data
    Forecast {
        /// Input file (CSV or JSON)
        #[arg(short, long)]
        input: PathBuf,

        /// Number of steps to forecast
        #[arg(short, long)]
        steps: usize,

        /// Model type (arima, ses, holt, holtwinters, auto)
        #[arg(short, long, default_value = "auto")]
        model: String,

        /// Output file (optional)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Column name or index for time series values (default: first numeric column)
        #[arg(short, long)]
        column: Option<String>,
    },

    /// Detect anomalies in time series data
    Detect {
        /// Input file (CSV or JSON)
        #[arg(short, long)]
        input: PathBuf,

        /// Detection method (zscore, iqr)
        #[arg(short, long, default_value = "zscore")]
        method: String,

        /// Detection threshold
        #[arg(short, long, default_value = "3.0")]
        threshold: f64,

        /// Column name or index for time series values
        #[arg(short, long)]
        column: Option<String>,

        /// Output file (optional)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Run backtest on historical data
    Backtest {
        /// Input file with price data
        #[arg(short, long)]
        input: PathBuf,

        /// Trading strategy (sma_crossover, momentum, mean_reversion)
        #[arg(short, long)]
        strategy: String,

        /// Initial capital
        #[arg(long, default_value = "10000.0")]
        capital: f64,

        /// Short window for moving average strategies
        #[arg(long, default_value = "10")]
        short_window: usize,

        /// Long window for moving average strategies
        #[arg(long, default_value = "30")]
        long_window: usize,

        /// Column name for price data
        #[arg(short, long)]
        column: Option<String>,

        /// Output file (optional)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Start the REST API server
    Serve {
        /// Port to listen on
        #[arg(short, long, default_value = "8080")]
        port: u16,

        /// Host to bind to
        #[arg(long, default_value = "0.0.0.0")]
        host: String,
    },
}

/// Load time series data from a CSV file
fn load_csv_data(path: &PathBuf, column: Option<&str>) -> CliResult<Vec<f64>> {
    let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
    let mut reader = csv::Reader::from_reader(BufReader::new(file));

    let headers = reader
        .headers()
        .map_err(|e| format!("Failed to read headers: {}", e))?
        .clone();

    // Find the column index
    let col_idx = if let Some(col) = column {
        // Try to parse as index first
        if let Ok(idx) = col.parse::<usize>() {
            idx
        } else {
            // Find by name
            headers
                .iter()
                .position(|h| h == col)
                .ok_or_else(|| format!("Column '{}' not found", col))?
        }
    } else {
        // Find first numeric column by trying to parse values
        0
    };

    let mut data = Vec::new();
    for result in reader.records() {
        let record = result.map_err(|e| format!("Failed to read record: {}", e))?;
        if let Some(value) = record.get(col_idx) {
            if let Ok(num) = value.trim().parse::<f64>() {
                data.push(num);
            }
        }
    }

    if data.is_empty() {
        return Err("No numeric data found in the specified column".to_string());
    }

    Ok(data)
}

/// Load time series data from a JSON file
fn load_json_data(path: &PathBuf, column: Option<&str>) -> CliResult<Vec<f64>> {
    let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
    let reader = BufReader::new(file);
    let json: serde_json::Value =
        serde_json::from_reader(reader).map_err(|e| format!("Failed to parse JSON: {}", e))?;

    // Handle array of numbers
    if let Some(arr) = json.as_array() {
        if arr.iter().all(|v| v.is_number()) {
            return Ok(arr.iter().filter_map(|v| v.as_f64()).collect());
        }

        // Handle array of objects
        if let Some(col) = column {
            let data: Vec<f64> = arr
                .iter()
                .filter_map(|obj| obj.get(col).and_then(|v| v.as_f64()))
                .collect();
            if !data.is_empty() {
                return Ok(data);
            }
        }

        // Try "value" or "values" keys
        for key in &["value", "values", "data", "y"] {
            let data: Vec<f64> = arr
                .iter()
                .filter_map(|obj| obj.get(*key).and_then(|v| v.as_f64()))
                .collect();
            if !data.is_empty() {
                return Ok(data);
            }
        }
    }

    // Handle object with data array
    if let Some(obj) = json.as_object() {
        for key in &["data", "values", "series", "y"] {
            if let Some(arr) = obj.get(*key).and_then(|v| v.as_array()) {
                let data: Vec<f64> = arr.iter().filter_map(|v| v.as_f64()).collect();
                if !data.is_empty() {
                    return Ok(data);
                }
            }
        }
    }

    Err("Could not extract numeric data from JSON".to_string())
}

/// Load data from file (auto-detect format)
fn load_data(path: &PathBuf, column: Option<&str>) -> CliResult<Vec<f64>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "csv" => load_csv_data(path, column),
        "json" => load_json_data(path, column),
        _ => {
            // Try CSV first, then JSON
            load_csv_data(path, column).or_else(|_| load_json_data(path, column))
        }
    }
}

/// Write forecast results to file or stdout
fn write_forecast_results(
    forecasts: &[f64],
    output: Option<&PathBuf>,
    model_name: &str,
) -> CliResult<()> {
    let json = serde_json::json!({
        "model": model_name,
        "forecasts": forecasts,
        "steps": forecasts.len()
    });

    if let Some(path) = output {
        let mut file = File::create(path).map_err(|e| format!("Failed to create output: {}", e))?;
        serde_json::to_writer_pretty(&mut file, &json)
            .map_err(|e| format!("Failed to write JSON: {}", e))?;
        println!("Forecasts written to {:?}", path);
    } else {
        println!("{}", serde_json::to_string_pretty(&json).unwrap());
    }

    Ok(())
}

/// Run forecast command
fn run_forecast(
    input: PathBuf,
    steps: usize,
    model: String,
    output: Option<PathBuf>,
    column: Option<String>,
) -> CliResult<()> {
    let data = load_data(&input, column.as_deref())?;
    println!(
        "Loaded {} data points from {:?}",
        data.len(),
        input.file_name().unwrap_or_default()
    );

    let (forecasts, model_name) = match model.to_lowercase().as_str() {
        "arima" => {
            let mut m = Arima::new(1, 1, 1).map_err(|e| e.to_string())?;
            m.fit(&data).map_err(|e| e.to_string())?;
            let f = m.predict(steps).map_err(|e| e.to_string())?;
            (f, "ARIMA(1,1,1)")
        }
        "ses" => {
            let m = SimpleExponentialSmoothing::auto(&data).map_err(|e| e.to_string())?;
            let f = m.predict(steps).map_err(|e| e.to_string())?;
            (f, "SES (auto)")
        }
        "holt" => {
            let mut m = DoubleExponentialSmoothing::new(0.3, 0.1).map_err(|e| e.to_string())?;
            m.fit(&data).map_err(|e| e.to_string())?;
            let f = m.predict(steps).map_err(|e| e.to_string())?;
            (f, "Holt")
        }
        "holtwinters" => {
            // Detect period - try common values
            let period = detect_period(&data);
            let mut m = HoltWinters::new(0.3, 0.1, 0.2, period, SeasonalType::Additive)
                .map_err(|e| e.to_string())?;
            m.fit(&data).map_err(|e| e.to_string())?;
            let f = m.predict(steps).map_err(|e| e.to_string())?;
            (f, "Holt-Winters")
        }
        "auto" | _ => {
            // Try multiple models and pick the best
            auto_forecast(&data, steps)?
        }
    };

    println!("Model: {}", model_name);
    println!("Forecast {} steps:", steps);
    for (i, val) in forecasts.iter().enumerate() {
        println!("  Step {}: {:.4}", i + 1, val);
    }

    write_forecast_results(&forecasts, output.as_ref(), model_name)?;

    Ok(())
}

/// Detect seasonality period using autocorrelation
fn detect_period(data: &[f64]) -> usize {
    let n = data.len();
    if n < 24 {
        return 4; // Default for small datasets
    }

    let mean: f64 = data.iter().sum::<f64>() / n as f64;
    let var: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

    if var < 1e-10 {
        return 12; // Default
    }

    let max_lag = (n / 3).min(52);
    let mut best_lag = 12;
    let mut best_acf = 0.0;

    for lag in 2..max_lag {
        let mut sum = 0.0;
        for i in lag..n {
            sum += (data[i] - mean) * (data[i - lag] - mean);
        }
        let acf = sum / (n as f64 * var);

        if acf > best_acf {
            best_acf = acf;
            best_lag = lag;
        }
    }

    best_lag.max(2)
}

/// Automatically select and fit the best model
fn auto_forecast(data: &[f64], steps: usize) -> CliResult<(Vec<f64>, &'static str)> {
    let n = data.len();
    let train_size = (n as f64 * 0.8) as usize;
    let train = &data[..train_size];
    let test = &data[train_size..];

    if test.is_empty() {
        // Not enough data for validation, use ARIMA
        let mut m = Arima::new(1, 1, 0).map_err(|e| e.to_string())?;
        m.fit(data).map_err(|e| e.to_string())?;
        return Ok((m.predict(steps).map_err(|e| e.to_string())?, "ARIMA(1,1,0)"));
    }

    let mut best_model = "ARIMA";
    let mut best_mse = f64::MAX;
    let mut best_forecasts = Vec::new();

    // Try ARIMA
    if let Ok(mut m) = Arima::new(1, 1, 0) {
        if m.fit(train).is_ok() {
            if let Ok(preds) = m.predict(test.len()) {
                let mse = mse(&preds, test);
                if mse < best_mse {
                    best_mse = mse;
                    best_model = "ARIMA(1,1,0)";
                    m.fit(data).ok();
                    best_forecasts = m.predict(steps).unwrap_or_default();
                }
            }
        }
    }

    // Try SES
    if let Ok(mut m) = SimpleExponentialSmoothing::new(0.3) {
        if m.fit(train).is_ok() {
            if let Ok(preds) = m.predict(test.len()) {
                let mse = mse(&preds, test);
                if mse < best_mse {
                    best_mse = mse;
                    best_model = "SES";
                    m.fit(data).ok();
                    best_forecasts = m.predict(steps).unwrap_or_default();
                }
            }
        }
    }

    // Try Holt
    if let Ok(mut m) = DoubleExponentialSmoothing::new(0.3, 0.1) {
        if m.fit(train).is_ok() {
            if let Ok(preds) = m.predict(test.len()) {
                let mse = mse(&preds, test);
                if mse < best_mse {
                    best_mse = mse;
                    best_model = "Holt";
                    m.fit(data).ok();
                    best_forecasts = m.predict(steps).unwrap_or_default();
                }
            }
        }
    }

    // Try Holt-Winters if enough data
    let period = detect_period(data);
    if train.len() >= period * 2 {
        if let Ok(mut m) = HoltWinters::new(0.3, 0.1, 0.2, period, SeasonalType::Additive) {
            if m.fit(train).is_ok() {
                if let Ok(preds) = m.predict(test.len()) {
                    let mse = mse(&preds, test);
                    if mse < best_mse {
                        best_model = "Holt-Winters";
                        m.fit(data).ok();
                        best_forecasts = m.predict(steps).unwrap_or_default();
                    }
                }
            }
        }
    }

    if best_forecasts.is_empty() {
        return Err("Failed to fit any model".to_string());
    }

    Ok((best_forecasts, best_model))
}

fn mse(predictions: &[f64], actual: &[f64]) -> f64 {
    let n = predictions.len().min(actual.len());
    if n == 0 {
        return f64::MAX;
    }
    predictions
        .iter()
        .zip(actual.iter())
        .take(n)
        .map(|(p, a)| (p - a).powi(2))
        .sum::<f64>()
        / n as f64
}

/// Run anomaly detection command
fn run_detect(
    input: PathBuf,
    method: String,
    threshold: f64,
    column: Option<String>,
    output: Option<PathBuf>,
) -> CliResult<()> {
    let data = load_data(&input, column.as_deref())?;
    println!(
        "Loaded {} data points from {:?}",
        data.len(),
        input.file_name().unwrap_or_default()
    );

    let result = match method.to_lowercase().as_str() {
        "zscore" | "z-score" => {
            let mut detector = ZScoreDetector::new(threshold);
            detector.fit(&data).map_err(|e| e.to_string())?;
            detector.detect(&data).map_err(|e| e.to_string())?
        }
        "iqr" => {
            let mut detector = IQRDetector::new(threshold);
            detector.fit(&data).map_err(|e| e.to_string())?;
            detector.detect(&data).map_err(|e| e.to_string())?
        }
        _ => return Err(format!("Unknown method: {}. Use 'zscore' or 'iqr'", method)),
    };

    let anomaly_indices: Vec<usize> = result
        .is_anomaly
        .iter()
        .enumerate()
        .filter_map(|(i, &is_anom)| if is_anom { Some(i) } else { None })
        .collect();

    println!("Detection method: {}", method);
    println!("Threshold: {}", threshold);
    println!("Anomalies found: {}", anomaly_indices.len());

    if !anomaly_indices.is_empty() {
        println!("\nAnomaly details:");
        for &idx in &anomaly_indices {
            println!(
                "  Index {}: value={:.4}, score={:.4}",
                idx, data[idx], result.scores[idx]
            );
        }
    }

    let json = serde_json::json!({
        "method": method,
        "threshold": threshold,
        "total_points": data.len(),
        "anomaly_count": anomaly_indices.len(),
        "anomaly_indices": anomaly_indices,
        "anomalies": anomaly_indices.iter().map(|&i| {
            serde_json::json!({
                "index": i,
                "value": data[i],
                "score": result.scores[i]
            })
        }).collect::<Vec<_>>()
    });

    if let Some(path) = output {
        let mut file = File::create(&path).map_err(|e| format!("Failed to create output: {}", e))?;
        serde_json::to_writer_pretty(&mut file, &json)
            .map_err(|e| format!("Failed to write JSON: {}", e))?;
        println!("\nResults written to {:?}", path);
    }

    Ok(())
}

/// Simple Moving Average Crossover Strategy
struct SMACrossover {
    short_window: usize,
    long_window: usize,
}

impl SMACrossover {
    fn new(short_window: usize, long_window: usize) -> Self {
        Self {
            short_window,
            long_window,
        }
    }

    fn compute_sma(data: &[f64], window: usize) -> Vec<f64> {
        if data.len() < window {
            return vec![f64::NAN; data.len()];
        }
        let mut result = vec![f64::NAN; window - 1];
        for i in (window - 1)..data.len() {
            let sum: f64 = data[(i + 1 - window)..=i].iter().sum();
            result.push(sum / window as f64);
        }
        result
    }
}

impl SignalGenerator for SMACrossover {
    fn generate(&self, data: &[f64]) -> Signal {
        if data.len() < self.long_window {
            return Signal::Hold;
        }

        let short_sma = Self::compute_sma(data, self.short_window);
        let long_sma = Self::compute_sma(data, self.long_window);

        let n = data.len();
        if short_sma[n - 1].is_nan() || long_sma[n - 1].is_nan() {
            return Signal::Hold;
        }

        // Check for crossover
        if n >= 2 && !short_sma[n - 2].is_nan() && !long_sma[n - 2].is_nan() {
            let prev_diff = short_sma[n - 2] - long_sma[n - 2];
            let curr_diff = short_sma[n - 1] - long_sma[n - 1];

            if prev_diff <= 0.0 && curr_diff > 0.0 {
                return Signal::Buy;
            } else if prev_diff >= 0.0 && curr_diff < 0.0 {
                return Signal::Sell;
            }
        }

        Signal::Hold
    }
}

/// Momentum Strategy
struct MomentumStrategy {
    lookback: usize,
    threshold: f64,
}

impl MomentumStrategy {
    fn new(lookback: usize) -> Self {
        Self {
            lookback,
            threshold: 0.0,
        }
    }
}

impl SignalGenerator for MomentumStrategy {
    fn generate(&self, data: &[f64]) -> Signal {
        if data.len() < self.lookback + 1 {
            return Signal::Hold;
        }

        let n = data.len();
        let momentum = (data[n - 1] - data[n - 1 - self.lookback]) / data[n - 1 - self.lookback];

        if momentum > self.threshold {
            Signal::Buy
        } else if momentum < -self.threshold {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }
}

/// Mean Reversion Strategy
struct MeanReversionStrategy {
    window: usize,
    std_threshold: f64,
}

impl MeanReversionStrategy {
    fn new(window: usize, std_threshold: f64) -> Self {
        Self {
            window,
            std_threshold,
        }
    }
}

impl SignalGenerator for MeanReversionStrategy {
    fn generate(&self, data: &[f64]) -> Signal {
        if data.len() < self.window {
            return Signal::Hold;
        }

        let n = data.len();
        let window_data = &data[(n - self.window)..];
        let mean: f64 = window_data.iter().sum::<f64>() / self.window as f64;
        let std: f64 = (window_data.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / self.window as f64)
            .sqrt();

        if std < 1e-10 {
            return Signal::Hold;
        }

        let z_score = (data[n - 1] - mean) / std;

        if z_score < -self.std_threshold {
            Signal::Buy // Price below mean, expect reversion up
        } else if z_score > self.std_threshold {
            Signal::Sell // Price above mean, expect reversion down
        } else {
            Signal::Hold
        }
    }
}

/// Run backtest
fn run_backtest(
    input: PathBuf,
    strategy: String,
    capital: f64,
    short_window: usize,
    long_window: usize,
    column: Option<String>,
    output: Option<PathBuf>,
) -> CliResult<()> {
    let data = load_data(&input, column.as_deref())?;
    println!(
        "Loaded {} data points from {:?}",
        data.len(),
        input.file_name().unwrap_or_default()
    );

    let signal_gen: Box<dyn SignalGenerator> = match strategy.to_lowercase().as_str() {
        "sma_crossover" | "sma" => Box::new(SMACrossover::new(short_window, long_window)),
        "momentum" => Box::new(MomentumStrategy::new(short_window)),
        "mean_reversion" | "meanrev" => Box::new(MeanReversionStrategy::new(long_window, 2.0)),
        _ => {
            return Err(format!(
                "Unknown strategy: {}. Use 'sma_crossover', 'momentum', or 'mean_reversion'",
                strategy
            ))
        }
    };

    // Generate signals
    let signals = signal_gen.generate_series(&data);

    // Simulate trading
    let mut cash = capital;
    let mut position = 0.0;
    let mut trades: Vec<Trade> = Vec::new();
    let mut equity_curve = Vec::with_capacity(data.len());
    let mut entry_price = 0.0;
    let mut entry_time = 0i64;

    for (i, (&price, &signal)) in data.iter().zip(signals.iter()).enumerate() {
        match signal {
            Signal::Buy if position == 0.0 => {
                // Enter long position
                position = cash / price;
                entry_price = price;
                entry_time = i as i64;
                cash = 0.0;
            }
            Signal::Sell if position > 0.0 => {
                // Exit position
                let exit_value = position * price;
                let pnl = exit_value - (position * entry_price);
                trades.push(Trade {
                    entry_price,
                    exit_price: price,
                    quantity: position,
                    pnl,
                    entry_time,
                    exit_time: i as i64,
                });
                cash = exit_value;
                position = 0.0;
            }
            _ => {}
        }

        let equity = cash + position * price;
        equity_curve.push(equity);
    }

    // Close any open position at end
    if position > 0.0 {
        let final_price = *data.last().unwrap();
        let exit_value = position * final_price;
        let pnl = exit_value - (position * entry_price);
        trades.push(Trade {
            entry_price,
            exit_price: final_price,
            quantity: position,
            pnl,
            entry_time,
            exit_time: (data.len() - 1) as i64,
        });
        cash = exit_value;
    }

    // Calculate metrics
    let final_equity = cash;
    let total_return = (final_equity - capital) / capital;

    let winning_trades = trades.iter().filter(|t| t.pnl > 0.0).count();
    let win_rate = if trades.is_empty() {
        0.0
    } else {
        winning_trades as f64 / trades.len() as f64
    };

    // Calculate max drawdown
    let mut peak = capital;
    let mut max_drawdown = 0.0;
    for &equity in &equity_curve {
        if equity > peak {
            peak = equity;
        }
        let drawdown = (peak - equity) / peak;
        if drawdown > max_drawdown {
            max_drawdown = drawdown;
        }
    }

    // Calculate Sharpe ratio (simplified, assuming daily returns)
    let returns: Vec<f64> = equity_curve
        .windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect();
    let mean_return = if returns.is_empty() {
        0.0
    } else {
        returns.iter().sum::<f64>() / returns.len() as f64
    };
    let std_return = if returns.len() < 2 {
        1.0
    } else {
        (returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>() / returns.len() as f64)
            .sqrt()
    };
    let sharpe_ratio = if std_return > 1e-10 {
        mean_return / std_return * (252.0_f64).sqrt() // Annualized
    } else {
        0.0
    };

    let _result = BacktestResult {
        total_return,
        sharpe_ratio,
        max_drawdown,
        win_rate,
        num_trades: trades.len(),
        equity_curve: equity_curve.clone(),
    };

    println!("\n=== Backtest Results ===");
    println!("Strategy: {}", strategy);
    println!("Initial Capital: ${:.2}", capital);
    println!("Final Equity: ${:.2}", final_equity);
    println!("Total Return: {:.2}%", total_return * 100.0);
    println!("Sharpe Ratio: {:.3}", sharpe_ratio);
    println!("Max Drawdown: {:.2}%", max_drawdown * 100.0);
    println!("Number of Trades: {}", trades.len());
    println!("Win Rate: {:.2}%", win_rate * 100.0);

    if !trades.is_empty() {
        println!("\nTrade Summary:");
        let total_pnl: f64 = trades.iter().map(|t| t.pnl).sum();
        let avg_pnl = total_pnl / trades.len() as f64;
        println!("  Total P&L: ${:.2}", total_pnl);
        println!("  Average P&L per trade: ${:.2}", avg_pnl);
    }

    let json = serde_json::json!({
        "strategy": strategy,
        "initial_capital": capital,
        "final_equity": final_equity,
        "total_return": total_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "num_trades": trades.len(),
        "trades": trades.iter().map(|t| serde_json::json!({
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "quantity": t.quantity,
            "pnl": t.pnl,
            "entry_time": t.entry_time,
            "exit_time": t.exit_time
        })).collect::<Vec<_>>()
    });

    if let Some(path) = output {
        let mut file = File::create(&path).map_err(|e| format!("Failed to create output: {}", e))?;
        serde_json::to_writer_pretty(&mut file, &json)
            .map_err(|e| format!("Failed to write JSON: {}", e))?;
        println!("\nResults written to {:?}", path);
    }

    Ok(())
}

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Forecast {
            input,
            steps,
            model,
            output,
            column,
        } => run_forecast(input, steps, model, output, column),

        Commands::Detect {
            input,
            method,
            threshold,
            column,
            output,
        } => run_detect(input, method, threshold, column, output),

        Commands::Backtest {
            input,
            strategy,
            capital,
            short_window,
            long_window,
            column,
            output,
        } => run_backtest(
            input,
            strategy,
            capital,
            short_window,
            long_window,
            column,
            output,
        ),

        Commands::Serve { port, host } => {
            println!("Starting server on {}:{}", host, port);
            println!("Use rustful-server binary for full server functionality");
            Ok(())
        }
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
