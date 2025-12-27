//! Risk metrics

/// Calculate Value at Risk (historical method)
pub fn var_historical(returns: &[f64], confidence: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let index = ((1.0 - confidence) * sorted.len() as f64).floor() as usize;
    sorted.get(index).copied().unwrap_or(0.0)
}

/// Calculate Sharpe ratio
pub fn sharpe_ratio(returns: &[f64], risk_free_rate: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
    let excess = mean - risk_free_rate;
    let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
    let std_dev = variance.sqrt();
    if std_dev == 0.0 {
        0.0
    } else {
        excess / std_dev
    }
}

/// Calculate maximum drawdown
pub fn max_drawdown(equity_curve: &[f64]) -> f64 {
    if equity_curve.is_empty() {
        return 0.0;
    }
    let mut max_dd = 0.0;
    let mut peak = equity_curve[0];
    for &value in equity_curve {
        if value > peak {
            peak = value;
        }
        let dd = (peak - value) / peak;
        if dd > max_dd {
            max_dd = dd;
        }
    }
    max_dd
}

/// Calculate Sortino ratio (downside deviation only)
pub fn sortino_ratio(returns: &[f64], risk_free_rate: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
    let excess = mean - risk_free_rate;
    let downside: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();
    if downside.is_empty() {
        return f64::INFINITY;
    }
    let downside_variance: f64 = downside.iter().map(|r| r.powi(2)).sum::<f64>() / downside.len() as f64;
    let downside_dev = downside_variance.sqrt();
    if downside_dev == 0.0 {
        f64::INFINITY
    } else {
        excess / downside_dev
    }
}
