//! Integration tests for rustful-financial

use financial::risk::{var_historical, sharpe_ratio, max_drawdown, sortino_ratio};

fn sample_returns() -> Vec<f64> {
    vec![0.01, -0.005, 0.02, -0.01, 0.015, 0.008, -0.02, 0.025, -0.008, 0.012]
}

fn sample_equity_curve() -> Vec<f64> {
    vec![100.0, 101.0, 100.5, 102.5, 101.5, 103.0, 103.8, 101.7, 104.2, 103.4, 104.6]
}

#[test]
fn test_var_historical() {
    let returns = sample_returns();

    let var_95 = var_historical(&returns, 0.95);
    assert!(var_95 < 0.0); // VaR should be negative (loss)

    let var_99 = var_historical(&returns, 0.99);
    assert!(var_99 <= var_95); // Higher confidence = larger loss
}

#[test]
fn test_var_empty_returns() {
    let empty: Vec<f64> = vec![];
    let var = var_historical(&empty, 0.95);
    assert_eq!(var, 0.0);
}

#[test]
fn test_sharpe_ratio() {
    let returns = sample_returns();
    let risk_free = 0.0001;

    let sharpe = sharpe_ratio(&returns, risk_free);
    // With positive mean returns, Sharpe should be positive
    assert!(sharpe > 0.0);
}

#[test]
fn test_sharpe_ratio_zero_std() {
    let constant_returns = vec![0.01, 0.01, 0.01, 0.01];
    let sharpe = sharpe_ratio(&constant_returns, 0.0);
    assert_eq!(sharpe, 0.0); // Zero std dev => Sharpe is 0
}

#[test]
fn test_sortino_ratio() {
    let returns = sample_returns();
    let risk_free = 0.0001;

    let sortino = sortino_ratio(&returns, risk_free);
    // Should be >= Sharpe (only considers downside)
    let sharpe = sharpe_ratio(&returns, risk_free);
    assert!(sortino >= sharpe);
}

#[test]
fn test_max_drawdown() {
    let equity = sample_equity_curve();

    let mdd = max_drawdown(&equity);
    assert!(mdd >= 0.0 && mdd <= 1.0);

    // Calculate expected max drawdown manually
    // Peak at 103.8, trough at 101.7 => (103.8-101.7)/103.8 = 0.0202
    assert!((mdd - 0.0202).abs() < 0.01);
}

#[test]
fn test_max_drawdown_increasing() {
    // Monotonically increasing equity = 0% drawdown
    let increasing = vec![100.0, 101.0, 102.0, 103.0, 104.0];
    let mdd = max_drawdown(&increasing);
    assert_eq!(mdd, 0.0);
}

#[test]
fn test_max_drawdown_empty() {
    let empty: Vec<f64> = vec![];
    let mdd = max_drawdown(&empty);
    assert_eq!(mdd, 0.0);
}
