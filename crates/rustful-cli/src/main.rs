//! # rustful-cli
//!
//! Command-line interface for the rustful-ts time series library.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

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
    },

    /// Run backtest on historical data
    Backtest {
        /// Input file with price data
        #[arg(short, long)]
        input: PathBuf,

        /// Trading strategy
        #[arg(short, long)]
        strategy: String,

        /// Initial capital
        #[arg(long, default_value = "10000.0")]
        capital: f64,
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

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Forecast { input, steps, model, output } => {
            println!("Forecasting {} steps using {} model from {:?}", steps, model, input);
            if let Some(out) = output {
                println!("Output will be written to {:?}", out);
            }
            // TODO: Implement forecast command
        }
        Commands::Detect { input, method, threshold } => {
            println!("Detecting anomalies in {:?} using {} method (threshold: {})", input, method, threshold);
            // TODO: Implement detect command
        }
        Commands::Backtest { input, strategy, capital } => {
            println!("Running backtest on {:?} with {} strategy (capital: {})", input, strategy, capital);
            // TODO: Implement backtest command
        }
        Commands::Serve { port, host } => {
            println!("Starting server on {}:{}", host, port);
            println!("Use rustful-server binary for full server functionality");
        }
    }
}
