//! Basic example demonstrating financial analytics
//!
//! Run with: cargo run --example basic -p rustful-financial

use financial::risk::{var_historical, sharpe_ratio, max_drawdown, sortino_ratio};

fn main() {
    println!("=== rustful-financial Basic Examples ===\n");

    // Sample daily returns (as decimals, e.g., 0.01 = 1%)
    let returns = vec![
        0.01, -0.005, 0.02, -0.01, 0.015,
        0.008, -0.02, 0.025, -0.008, 0.012,
        0.003, -0.015, 0.018, -0.003, 0.01,
    ];

    // Sample equity curve
    let equity_curve = vec![
        100.0, 101.0, 100.5, 102.5, 101.5, 103.0,
        103.8, 101.7, 104.2, 103.4, 104.6,
    ];

    println!("Sample returns: {:?}\n", &returns[..5]);

    // 1. Value at Risk
    let var_95 = var_historical(&returns, 0.95);
    println!("1. Value at Risk (95% confidence): {:.4}", var_95);

    // 2. Sharpe Ratio
    let risk_free_rate = 0.0001; // Daily risk-free rate (~2.5% annual)
    let sharpe = sharpe_ratio(&returns, risk_free_rate);
    println!("2. Sharpe Ratio: {:.4}", sharpe);

    // 3. Sortino Ratio
    let sortino = sortino_ratio(&returns, risk_free_rate);
    println!("3. Sortino Ratio: {:.4}", sortino);

    // 4. Maximum Drawdown
    let mdd = max_drawdown(&equity_curve);
    println!("4. Maximum Drawdown: {:.2}%", mdd * 100.0);

    println!("\n=== Examples Complete ===");
}
