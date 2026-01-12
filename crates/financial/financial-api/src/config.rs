//! Financial configuration types.

use serde::{Deserialize, Serialize};

// ============================================================================
// Backtest Configuration
// ============================================================================

/// Backtest configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// Initial capital for backtesting.
    pub initial_capital: f64,
    /// Commission per trade (as percentage).
    pub commission: f64,
    /// Slippage per trade (as percentage).
    pub slippage: f64,
}

impl BacktestConfig {
    pub fn new(initial_capital: f64) -> Self {
        Self {
            initial_capital,
            commission: 0.0,
            slippage: 0.0,
        }
    }

    pub fn with_costs(initial_capital: f64, commission: f64, slippage: f64) -> Self {
        Self {
            initial_capital,
            commission,
            slippage,
        }
    }
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            commission: 0.001,
            slippage: 0.0005,
        }
    }
}

// ============================================================================
// Risk Configuration
// ============================================================================

/// Risk calculation configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskConfig {
    /// Risk-free rate for Sharpe/Sortino calculations.
    pub risk_free_rate: f64,
    /// Confidence level for VaR calculations.
    pub var_confidence: f64,
    /// Annualization factor (252 for daily, 12 for monthly).
    pub annualization_factor: f64,
}

impl RiskConfig {
    pub fn new(risk_free_rate: f64) -> Self {
        Self {
            risk_free_rate,
            var_confidence: 0.95,
            annualization_factor: 252.0,
        }
    }

    pub fn daily() -> Self {
        Self {
            risk_free_rate: 0.0,
            var_confidence: 0.95,
            annualization_factor: 252.0,
        }
    }

    pub fn monthly() -> Self {
        Self {
            risk_free_rate: 0.0,
            var_confidence: 0.95,
            annualization_factor: 12.0,
        }
    }
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self::daily()
    }
}

// ============================================================================
// Portfolio Configuration
// ============================================================================

/// Portfolio configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioConfig {
    /// Initial cash balance.
    pub initial_cash: f64,
    /// Maximum position size (as percentage of portfolio).
    pub max_position_size: f64,
    /// Maximum number of positions.
    pub max_positions: usize,
}

impl PortfolioConfig {
    pub fn new(initial_cash: f64) -> Self {
        Self {
            initial_cash,
            max_position_size: 1.0,
            max_positions: usize::MAX,
        }
    }

    pub fn with_limits(initial_cash: f64, max_position_size: f64, max_positions: usize) -> Self {
        Self {
            initial_cash,
            max_position_size,
            max_positions,
        }
    }
}

impl Default for PortfolioConfig {
    fn default() -> Self {
        Self {
            initial_cash: 100_000.0,
            max_position_size: 0.1,
            max_positions: 10,
        }
    }
}
