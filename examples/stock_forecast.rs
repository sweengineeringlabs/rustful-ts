//! Example: Fetch stock data and forecast prices
//!
//! Run with:
//! ```bash
//! cargo run --example stock_forecast --features fetch
//! ```

use rustful_ts::algorithms::{
    arima::Arima,
    exponential_smoothing::DoubleExponentialSmoothing,
    linear_regression::LinearRegression,
    Predictor,
};
use rustful_ts::data::{fetch_stock_sync, closing_prices, daily_returns, Interval};
use rustful_ts::utils::metrics::{mae, rmse, mape};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Stock Price Forecasting Demo ===\n");

    // Fetch Apple stock data
    let symbol = "AAPL";
    let start_date = "2024-01-01";
    let end_date = "2024-12-01";

    println!("Fetching {} data from {} to {}...", symbol, start_date, end_date);

    let quotes = fetch_stock_sync(symbol, start_date, end_date, Interval::Daily)?;
    println!("Retrieved {} data points\n", quotes.len());

    // Extract closing prices
    let prices = closing_prices(&quotes);

    // Split into train/test (80/20)
    let split_idx = (prices.len() as f64 * 0.8) as usize;
    let train = &prices[..split_idx];
    let test = &prices[split_idx..];
    let forecast_horizon = test.len();

    println!("Training set: {} points", train.len());
    println!("Test set: {} points\n", test.len());

    // Show price range
    let min_price = prices.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_price = prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!("Price range: ${:.2} - ${:.2}\n", min_price, max_price);

    // Test different models
    println!("=== Model Comparison ===\n");

    // 1. Linear Regression
    {
        let mut model = LinearRegression::new();
        model.fit(train)?;
        let forecast = model.predict(forecast_horizon)?;

        println!("Linear Regression:");
        println!("  Slope: {:.4} ($ per day)", model.slope());
        println!("  R²: {:.4}", model.r_squared());
        print_metrics(test, &forecast);
    }

    // 2. Holt (Double Exponential Smoothing)
    {
        let mut model = DoubleExponentialSmoothing::new(0.3, 0.1)?;
        model.fit(train)?;
        let forecast = model.predict(forecast_horizon)?;

        println!("Holt (Double Exp. Smoothing):");
        print_metrics(test, &forecast);
    }

    // 3. ARIMA
    {
        let mut model = Arima::new(1, 1, 0)?;
        model.fit(train)?;
        let forecast = model.predict(forecast_horizon)?;

        println!("ARIMA(1,1,0):");
        print_metrics(test, &forecast);
    }

    // 4. ARIMA with higher order
    {
        let mut model = Arima::new(2, 1, 1)?;
        model.fit(train)?;
        let forecast = model.predict(forecast_horizon)?;

        println!("ARIMA(2,1,1):");
        print_metrics(test, &forecast);
    }

    // Show sample predictions
    println!("\n=== Sample Predictions (last 5 days) ===\n");
    println!("{:<12} {:>12} {:>12}", "Date", "Actual", "ARIMA");

    let mut model = Arima::new(1, 1, 0)?;
    model.fit(train)?;
    let forecast = model.predict(forecast_horizon)?;

    for i in (test.len().saturating_sub(5))..test.len() {
        println!(
            "{:<12} ${:>10.2} ${:>10.2}",
            format!("Day {}", split_idx + i + 1),
            test[i],
            forecast[i]
        );
    }

    // Daily returns analysis
    println!("\n=== Returns Analysis ===\n");
    let returns = daily_returns(&prices);
    let avg_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let volatility = (returns.iter().map(|r| (r - avg_return).powi(2)).sum::<f64>()
        / returns.len() as f64)
        .sqrt();

    println!("Average daily return: {:.4}%", avg_return * 100.0);
    println!("Daily volatility: {:.4}%", volatility * 100.0);
    println!(
        "Annualized volatility: {:.2}%",
        volatility * (252.0_f64).sqrt() * 100.0
    );

    Ok(())
}

fn print_metrics(actual: &[f64], predicted: &[f64]) {
    println!("  MAE:  ${:.2}", mae(actual, predicted));
    println!("  RMSE: ${:.2}", rmse(actual, predicted));
    println!("  MAPE: {:.2}%\n", mape(actual, predicted) * 100.0);
}
