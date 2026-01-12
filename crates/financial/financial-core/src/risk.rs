//! Risk metrics implementation.

use financial_spi::RiskCalculator;

/// Standard risk metrics calculator.
#[derive(Debug, Clone, Default)]
pub struct StandardRiskCalculator;

impl StandardRiskCalculator {
    /// Create a new risk calculator.
    pub fn new() -> Self {
        Self
    }
}

impl RiskCalculator for StandardRiskCalculator {
    fn var_historical(&self, returns: &[f64], confidence: f64) -> f64 {
        var_historical(returns, confidence)
    }

    fn sharpe_ratio(&self, returns: &[f64], risk_free_rate: f64) -> f64 {
        sharpe_ratio(returns, risk_free_rate)
    }

    fn max_drawdown(&self, equity_curve: &[f64]) -> f64 {
        max_drawdown(equity_curve)
    }

    fn sortino_ratio(&self, returns: &[f64], risk_free_rate: f64) -> f64 {
        sortino_ratio(returns, risk_free_rate)
    }
}

/// Calculate Value at Risk (historical method).
pub fn var_historical(returns: &[f64], confidence: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let index = ((1.0 - confidence) * sorted.len() as f64).floor() as usize;
    sorted.get(index).copied().unwrap_or(0.0)
}

/// Calculate Sharpe ratio.
pub fn sharpe_ratio(returns: &[f64], risk_free_rate: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
    let excess = mean - risk_free_rate;
    let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
    let std_dev = variance.sqrt();
    if std_dev < 1e-10 {
        0.0
    } else {
        excess / std_dev
    }
}

/// Calculate maximum drawdown.
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

/// Calculate Sortino ratio (downside deviation only).
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
    if downside_dev < 1e-10 {
        f64::INFINITY
    } else {
        excess / downside_dev
    }
}

/// Calculate Calmar ratio (annualized return / max drawdown).
pub fn calmar_ratio(returns: &[f64], equity_curve: &[f64], annualization_factor: f64) -> f64 {
    if returns.is_empty() || equity_curve.is_empty() {
        return 0.0;
    }
    let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
    let annualized_return = mean_return * annualization_factor;
    let max_dd = max_drawdown(equity_curve);
    if max_dd < 1e-10 {
        f64::INFINITY
    } else {
        annualized_return / max_dd
    }
}

/// Calculate information ratio.
pub fn information_ratio(returns: &[f64], benchmark_returns: &[f64]) -> f64 {
    if returns.len() != benchmark_returns.len() || returns.is_empty() {
        return 0.0;
    }

    let excess_returns: Vec<f64> = returns
        .iter()
        .zip(benchmark_returns.iter())
        .map(|(r, b)| r - b)
        .collect();

    let mean_excess: f64 = excess_returns.iter().sum::<f64>() / excess_returns.len() as f64;
    let variance: f64 = excess_returns
        .iter()
        .map(|r| (r - mean_excess).powi(2))
        .sum::<f64>() / excess_returns.len() as f64;
    let tracking_error = variance.sqrt();

    if tracking_error < 1e-10 {
        0.0
    } else {
        mean_excess / tracking_error
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sharpe_ratio() {
        let returns = vec![0.01, 0.02, -0.01, 0.03, 0.01];
        let sharpe = sharpe_ratio(&returns, 0.0);
        assert!(sharpe > 0.0);
    }

    #[test]
    fn test_max_drawdown() {
        let equity = vec![100.0, 110.0, 105.0, 115.0, 100.0];
        let dd = max_drawdown(&equity);
        // Peak was 115, lowest after peak was 100
        // DD = (115 - 100) / 115 = 0.1304...
        assert!((dd - 0.1304).abs() < 0.01);
    }

    #[test]
    fn test_var_historical() {
        let returns = vec![-0.05, -0.03, -0.01, 0.01, 0.02, 0.03, 0.05, 0.06, 0.08, 0.10];
        let var = var_historical(&returns, 0.95);
        // At 95% confidence, we expect the 5th percentile
        assert!(var < 0.0);
    }

    #[test]
    fn test_sortino_ratio() {
        let returns = vec![0.01, 0.02, -0.01, 0.03, 0.01, -0.005];
        let sortino = sortino_ratio(&returns, 0.0);
        assert!(sortino > 0.0);
    }
}
