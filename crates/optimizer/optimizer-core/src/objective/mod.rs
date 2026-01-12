//! Objective function implementations.

use optimizer_spi::{Objective, ObjectiveFunction};

/// Sharpe Ratio objective function.
#[derive(Debug, Clone, Default)]
pub struct SharpeRatio {
    pub risk_free_rate: f64,
    pub annualization: f64,
}

impl SharpeRatio {
    pub fn new() -> Self {
        Self {
            risk_free_rate: 0.0,
            annualization: 252.0,
        }
    }

    pub fn with_risk_free(risk_free_rate: f64) -> Self {
        Self {
            risk_free_rate,
            annualization: 252.0,
        }
    }
}

impl ObjectiveFunction for SharpeRatio {
    fn compute(&self, signals: &[f64], returns: &[f64]) -> f64 {
        if signals.len() != returns.len() || signals.is_empty() {
            return f64::NEG_INFINITY;
        }

        let strategy_returns: Vec<f64> = signals.iter()
            .zip(returns.iter())
            .map(|(s, r)| s * r)
            .collect();

        let n = strategy_returns.len() as f64;
        let mean: f64 = strategy_returns.iter().sum::<f64>() / n;
        let variance: f64 = strategy_returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / n;
        let std = variance.sqrt();

        if std < 1e-10 {
            return 0.0;
        }

        let excess_return = mean - self.risk_free_rate / self.annualization;
        (excess_return / std) * self.annualization.sqrt()
    }

    fn objective_type(&self) -> Objective {
        Objective::SharpeRatio
    }
}

/// Sortino Ratio objective function.
#[derive(Debug, Clone, Default)]
pub struct SortinoRatio {
    pub risk_free_rate: f64,
    pub annualization: f64,
}

impl SortinoRatio {
    pub fn new() -> Self {
        Self {
            risk_free_rate: 0.0,
            annualization: 252.0,
        }
    }
}

impl ObjectiveFunction for SortinoRatio {
    fn compute(&self, signals: &[f64], returns: &[f64]) -> f64 {
        if signals.len() != returns.len() || signals.is_empty() {
            return f64::NEG_INFINITY;
        }

        let strategy_returns: Vec<f64> = signals.iter()
            .zip(returns.iter())
            .map(|(s, r)| s * r)
            .collect();

        let n = strategy_returns.len() as f64;
        let mean: f64 = strategy_returns.iter().sum::<f64>() / n;

        let downside_returns: Vec<f64> = strategy_returns.iter()
            .filter(|&&r| r < 0.0)
            .copied()
            .collect();

        if downside_returns.is_empty() {
            return f64::INFINITY;
        }

        let downside_variance: f64 = downside_returns.iter()
            .map(|r| r.powi(2))
            .sum::<f64>() / downside_returns.len() as f64;
        let downside_std = downside_variance.sqrt();

        if downside_std < 1e-10 {
            return 0.0;
        }

        let excess_return = mean - self.risk_free_rate / self.annualization;
        (excess_return / downside_std) * self.annualization.sqrt()
    }

    fn objective_type(&self) -> Objective {
        Objective::SortinoRatio
    }
}

/// Directional accuracy objective function.
#[derive(Debug, Clone, Default)]
pub struct DirectionalAccuracy;

impl DirectionalAccuracy {
    pub fn new() -> Self {
        Self
    }
}

impl ObjectiveFunction for DirectionalAccuracy {
    fn compute(&self, signals: &[f64], returns: &[f64]) -> f64 {
        if signals.len() != returns.len() || signals.is_empty() {
            return 0.0;
        }

        let correct: usize = signals.iter()
            .zip(returns.iter())
            .filter(|(s, r)| ((**s > 0.0) == (**r > 0.0)) && **s != 0.0)
            .count();

        let total: usize = signals.iter().filter(|&&s| s != 0.0).count();

        if total == 0 {
            return 0.5;
        }

        correct as f64 / total as f64
    }

    fn objective_type(&self) -> Objective {
        Objective::DirectionalAccuracy
    }
}

/// Total return objective function.
#[derive(Debug, Clone, Default)]
pub struct TotalReturn;

impl TotalReturn {
    pub fn new() -> Self {
        Self
    }
}

impl ObjectiveFunction for TotalReturn {
    fn compute(&self, signals: &[f64], returns: &[f64]) -> f64 {
        if signals.len() != returns.len() || signals.is_empty() {
            return 0.0;
        }

        signals.iter()
            .zip(returns.iter())
            .fold(1.0, |acc, (s, r)| acc * (1.0 + s * r)) - 1.0
    }

    fn objective_type(&self) -> Objective {
        Objective::TotalReturn
    }
}

/// Maximum drawdown objective function.
#[derive(Debug, Clone, Default)]
pub struct MaxDrawdown;

impl MaxDrawdown {
    pub fn new() -> Self {
        Self
    }
}

impl ObjectiveFunction for MaxDrawdown {
    fn compute(&self, signals: &[f64], returns: &[f64]) -> f64 {
        if signals.len() != returns.len() || signals.is_empty() {
            return 0.0;
        }

        let mut equity: f64 = 1.0;
        let mut peak: f64 = 1.0;
        let mut max_dd: f64 = 0.0;

        for (s, r) in signals.iter().zip(returns.iter()) {
            equity *= 1.0 + s * r;
            peak = peak.max(equity);
            let dd = (peak - equity) / peak;
            max_dd = max_dd.max(dd);
        }

        -max_dd
    }

    fn objective_type(&self) -> Objective {
        Objective::MaxDrawdown
    }
}

/// Profit factor objective function.
#[derive(Debug, Clone, Default)]
pub struct ProfitFactor;

impl ProfitFactor {
    pub fn new() -> Self {
        Self
    }
}

impl ObjectiveFunction for ProfitFactor {
    fn compute(&self, signals: &[f64], returns: &[f64]) -> f64 {
        if signals.len() != returns.len() || signals.is_empty() {
            return 0.0;
        }

        let mut gross_profit = 0.0;
        let mut gross_loss = 0.0;

        for (s, r) in signals.iter().zip(returns.iter()) {
            let pnl = s * r;
            if pnl > 0.0 {
                gross_profit += pnl;
            } else {
                gross_loss += pnl.abs();
            }
        }

        if gross_loss < 1e-10 {
            return f64::INFINITY;
        }

        gross_profit / gross_loss
    }

    fn objective_type(&self) -> Objective {
        Objective::ProfitFactor
    }
}

/// Create objective function from type.
pub fn create_objective(objective: Objective) -> Box<dyn ObjectiveFunction> {
    match objective {
        Objective::SharpeRatio => Box::new(SharpeRatio::new()),
        Objective::SortinoRatio => Box::new(SortinoRatio::new()),
        Objective::DirectionalAccuracy => Box::new(DirectionalAccuracy::new()),
        Objective::TotalReturn => Box::new(TotalReturn::new()),
        Objective::MaxDrawdown => Box::new(MaxDrawdown::new()),
        Objective::ProfitFactor => Box::new(ProfitFactor::new()),
        Objective::InformationCoefficient => Box::new(DirectionalAccuracy::new()),
        Objective::WinRate => Box::new(DirectionalAccuracy::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sharpe_ratio() {
        let sharpe = SharpeRatio::new();
        let signals = vec![1.0, 1.0, -1.0, 1.0, -1.0];
        let returns = vec![0.01, 0.02, -0.01, 0.015, -0.005];
        let result = sharpe.compute(&signals, &returns);
        assert!(result.is_finite());
    }

    #[test]
    fn test_directional_accuracy() {
        let da = DirectionalAccuracy::new();
        let signals = vec![1.0, -1.0, 1.0];
        let returns = vec![0.01, -0.01, 0.02];
        let result = da.compute(&signals, &returns);
        assert!((result - 1.0).abs() < 1e-10);
    }
}
