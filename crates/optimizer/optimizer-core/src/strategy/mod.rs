//! Strategy builder for designing trading strategies.

use serde::{Deserialize, Serialize};

// ============================================================================
// Entry/Exit Conditions
// ============================================================================

/// Entry condition type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntryCondition {
    /// RSI below threshold (oversold)
    RsiOversold { threshold: f64 },
    /// RSI above threshold (overbought)
    RsiOverbought { threshold: f64 },
    /// MACD histogram crosses above zero
    MacdCrossUp,
    /// MACD histogram crosses below zero
    MacdCrossDown,
    /// Price crosses above SMA/EMA
    PriceAboveMA,
    /// Price crosses below SMA/EMA
    PriceBelowMA,
    /// Price touches lower Bollinger band
    BollingerLowerTouch,
    /// Price touches upper Bollinger band
    BollingerUpperTouch,
}

/// Exit condition type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExitCondition {
    /// Fixed take profit percentage
    TakeProfit { percent: f64 },
    /// Fixed stop loss percentage
    StopLoss { percent: f64 },
    /// Trailing stop percentage
    TrailingStop { percent: f64 },
    /// Exit after N bars
    TimeBased { bars: usize },
    /// Opposite signal
    SignalReversal,
}

/// Position sizing method.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PositionSizing {
    /// Fixed position size
    Fixed { size: f64 },
    /// Percentage of equity
    PercentEquity { percent: f64 },
    /// Risk-based (Kelly criterion)
    Kelly { fraction: f64 },
    /// Volatility-adjusted
    VolatilityAdjusted { target_risk: f64 },
}

// ============================================================================
// Strategy Rule
// ============================================================================

/// A complete strategy rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyRule {
    pub entry: Vec<EntryCondition>,
    pub exit: Vec<ExitCondition>,
    pub sizing: PositionSizing,
}

impl Default for StrategyRule {
    fn default() -> Self {
        Self {
            entry: vec![],
            exit: vec![ExitCondition::StopLoss { percent: 2.0 }],
            sizing: PositionSizing::PercentEquity { percent: 1.0 },
        }
    }
}

// ============================================================================
// Strategy Metrics
// ============================================================================

/// Strategy performance metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StrategyMetrics {
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub total_trades: usize,
}

/// Strategy result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyResult {
    pub metrics: StrategyMetrics,
    pub trades: Vec<StrategyTrade>,
}

/// Individual trade record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyTrade {
    pub entry_bar: usize,
    pub exit_bar: usize,
    pub entry_price: f64,
    pub exit_price: f64,
    pub pnl: f64,
    pub pnl_percent: f64,
}

// ============================================================================
// Strategy Builder
// ============================================================================

/// Builder for constructing trading strategies.
#[derive(Debug, Clone, Default)]
pub struct StrategyBuilder {
    rules: Vec<StrategyRule>,
}

impl StrategyBuilder {
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    pub fn add_rule(mut self, rule: StrategyRule) -> Self {
        self.rules.push(rule);
        self
    }

    pub fn rules(&self) -> &[StrategyRule] {
        &self.rules
    }
}
