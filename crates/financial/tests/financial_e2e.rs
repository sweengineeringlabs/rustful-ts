//! End-to-end tests for rustful-financial crate
//!
//! Tests complete financial analysis workflows using only this crate's API.

use financial::risk::{var_historical, sharpe_ratio, max_drawdown, sortino_ratio};

fn sample_returns() -> Vec<f64> {
    vec![
        0.01, -0.005, 0.02, -0.01, 0.015, 0.008, -0.02, 0.025, -0.008, 0.012,
        0.005, -0.003, 0.018, -0.012, 0.009, 0.007, -0.015, 0.022, -0.006, 0.011,
    ]
}

fn sample_equity_curve() -> Vec<f64> {
    let returns = sample_returns();
    let mut equity = vec![10000.0];
    for r in &returns {
        let last = *equity.last().unwrap();
        equity.push(last * (1.0 + r));
    }
    equity
}

#[test]
fn e2e_var_calculation_workflow() {
    let returns = sample_returns();

    let var_90 = var_historical(&returns, 0.90);
    let var_95 = var_historical(&returns, 0.95);
    let var_99 = var_historical(&returns, 0.99);

    // VaR should be negative (representing loss)
    assert!(var_95 <= 0.0 || var_95.is_nan());

    // Higher confidence = more severe VaR
    if !var_90.is_nan() && !var_99.is_nan() {
        assert!(var_99 <= var_90, "99% VaR should be more severe than 90%");
    }
}

#[test]
fn e2e_sharpe_ratio_workflow() {
    let returns = sample_returns();
    let risk_free = 0.0001; // ~2.5% annual

    let sharpe = sharpe_ratio(&returns, risk_free);

    assert!(sharpe.is_finite());

    // With mostly positive returns, Sharpe should be positive
    let avg_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
    if avg_return > risk_free {
        assert!(sharpe > 0.0, "Sharpe should be positive with positive excess returns");
    }
}

#[test]
fn e2e_sortino_ratio_workflow() {
    let returns = sample_returns();
    let risk_free = 0.0001;

    let sortino = sortino_ratio(&returns, risk_free);
    let sharpe = sharpe_ratio(&returns, risk_free);

    assert!(sortino.is_finite());

    // Sortino only penalizes downside, so typically >= Sharpe
    if sharpe.is_finite() && sortino.is_finite() && sharpe > 0.0 {
        assert!(sortino >= sharpe * 0.9, "Sortino should be close to or higher than Sharpe");
    }
}

#[test]
fn e2e_max_drawdown_workflow() {
    let equity = sample_equity_curve();

    let mdd = max_drawdown(&equity);

    assert!(mdd >= 0.0 && mdd <= 1.0, "Drawdown should be between 0 and 1");

    // Should detect the actual drawdown
    let mut peak = equity[0];
    let mut actual_mdd = 0.0;
    for &e in &equity {
        if e > peak {
            peak = e;
        }
        let dd = (peak - e) / peak;
        if dd > actual_mdd {
            actual_mdd = dd;
        }
    }

    assert!((mdd - actual_mdd).abs() < 0.001,
        "Calculated {} vs expected {}", mdd, actual_mdd);
}

#[test]
fn e2e_no_drawdown_on_increasing_equity() {
    let equity: Vec<f64> = (0..20).map(|i| 10000.0 + i as f64 * 100.0).collect();

    let mdd = max_drawdown(&equity);

    assert_eq!(mdd, 0.0, "Monotonically increasing should have 0 drawdown");
}

#[test]
fn e2e_risk_metrics_edge_cases() {
    // Empty data
    let empty: Vec<f64> = vec![];
    assert_eq!(var_historical(&empty, 0.95), 0.0);
    assert_eq!(max_drawdown(&empty), 0.0);

    // Single value
    let single = vec![0.01];
    let _ = sharpe_ratio(&single, 0.0);

    // Constant returns (zero volatility)
    let constant = vec![0.01; 10];
    let sharpe = sharpe_ratio(&constant, 0.0);
    assert_eq!(sharpe, 0.0, "Zero volatility should give 0 Sharpe");
}

#[test]
fn e2e_full_risk_analysis() {
    let returns = sample_returns();
    let equity = sample_equity_curve();
    let rf = 0.0001;

    // Calculate all metrics
    let sharpe = sharpe_ratio(&returns, rf);
    let sortino = sortino_ratio(&returns, rf);
    let mdd = max_drawdown(&equity);
    let var_95 = var_historical(&returns, 0.95);

    // Print summary (for manual verification if needed)
    let _ = format!(
        "Risk Report: Sharpe={:.3}, Sortino={:.3}, MDD={:.2}%, VaR95={:.2}%",
        sharpe, sortino, mdd * 100.0, var_95 * 100.0
    );

    // All should be valid
    assert!(sharpe.is_finite());
    assert!(sortino.is_finite());
    assert!(mdd.is_finite());
}

#[test]
fn e2e_negative_returns_analysis() {
    // Portfolio with overall negative returns
    let bad_returns = vec![-0.02, -0.01, 0.005, -0.03, -0.015, 0.01, -0.02, -0.01];

    let sharpe = sharpe_ratio(&bad_returns, 0.0);
    let mdd = max_drawdown(&{
        let mut eq = vec![10000.0];
        for r in &bad_returns {
            eq.push(eq.last().unwrap() * (1.0 + r));
        }
        eq
    });

    // Sharpe should be negative
    assert!(sharpe < 0.0, "Negative returns should give negative Sharpe");

    // Should have significant drawdown
    assert!(mdd > 0.05, "Should have significant drawdown: {}", mdd);
}
