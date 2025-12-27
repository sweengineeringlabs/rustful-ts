//! Basic example demonstrating core time series algorithms
//!
//! Run with: cargo run --example basic -p algorithm

use algorithm::{
    ml::{DistanceMetric, TimeSeriesKNN},
    regression::{Arima, LinearRegression},
    smoothing::{HoltWinters, SeasonalType, SimpleExponentialSmoothing, SimpleMovingAverage},
    Predictor,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Sample time series data
    let data = vec![
        10.0, 12.0, 13.0, 15.0, 14.0, 16.0, 18.0, 17.0, 19.0, 21.0, 20.0, 22.0, 24.0, 23.0, 25.0,
        27.0, 26.0, 28.0, 30.0, 29.0,
    ];

    println!("=== Algorithm Crate Examples ===\n");
    println!("Input data: {:?}\n", &data[..10]);

    // 1. Simple Exponential Smoothing
    println!("1. Simple Exponential Smoothing (alpha=0.3)");
    let mut ses = SimpleExponentialSmoothing::new(0.3)?;
    ses.fit(&data)?;
    let ses_forecast = ses.predict(5)?;
    println!("   Forecast: {:?}\n", ses_forecast);

    // 2. ARIMA
    println!("2. ARIMA(1,1,1)");
    let mut arima = Arima::new(1, 1, 1)?;
    arima.fit(&data)?;
    let arima_forecast = arima.predict(5)?;
    println!("   Forecast: {:?}\n", arima_forecast);

    // 3. Simple Moving Average
    println!("3. Simple Moving Average (window=3)");
    let mut sma = SimpleMovingAverage::new(3)?;
    sma.fit(&data)?;
    let sma_forecast = sma.predict(5)?;
    println!("   Forecast: {:?}\n", sma_forecast);

    // 4. Linear Regression
    println!("4. Linear Regression");
    let mut lr = LinearRegression::new();
    lr.fit(&data)?;
    let lr_forecast = lr.predict(5)?;
    println!(
        "   Slope: {:.4}, Intercept: {:.4}",
        lr.slope(),
        lr.intercept()
    );
    println!("   Forecast: {:?}\n", lr_forecast);

    // 5. K-Nearest Neighbors
    println!("5. Time Series KNN (k=3, window=3)");
    let mut knn = TimeSeriesKNN::new(3, 3, DistanceMetric::Euclidean)?;
    knn.fit(&data)?;
    let knn_forecast = knn.predict(5)?;
    println!("   Forecast: {:?}\n", knn_forecast);

    // 6. Holt-Winters (for seasonal data)
    let seasonal_data: Vec<f64> = (0..40)
        .map(|i| 10.0 + (i as f64) * 0.5 + 5.0 * ((i as f64) * std::f64::consts::PI / 6.0).sin())
        .collect();

    println!("6. Holt-Winters (seasonal data, period=12)");
    let mut hw = HoltWinters::new(0.3, 0.1, 0.1, 12, SeasonalType::Additive)?;
    hw.fit(&seasonal_data)?;
    let hw_forecast = hw.predict(5)?;
    println!("   Forecast: {:?}\n", hw_forecast);

    println!("=== Examples Complete ===");
    Ok(())
}
