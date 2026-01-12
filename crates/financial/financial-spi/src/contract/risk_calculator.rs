//! Risk calculator trait.

/// Risk metrics calculator trait.
pub trait RiskCalculator: Send + Sync {
    /// Calculate Value at Risk (historical method).
    fn var_historical(&self, returns: &[f64], confidence: f64) -> f64;

    /// Calculate Sharpe ratio.
    fn sharpe_ratio(&self, returns: &[f64], risk_free_rate: f64) -> f64;

    /// Calculate maximum drawdown.
    fn max_drawdown(&self, equity_curve: &[f64]) -> f64;

    /// Calculate Sortino ratio (downside deviation only).
    fn sortino_ratio(&self, returns: &[f64], risk_free_rate: f64) -> f64;
}
