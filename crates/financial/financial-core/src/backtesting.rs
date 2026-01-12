//! Backtesting engine implementation.

use financial_spi::{BacktestResult, Backtester, Signal, Trade, Result, FinancialError};

/// Simple backtesting engine.
#[derive(Debug, Clone, Default)]
pub struct SimpleBacktester {
    trades: Vec<Trade>,
    initial_capital: f64,
}

impl SimpleBacktester {
    /// Create a new backtester with initial capital.
    pub fn new(initial_capital: f64) -> Self {
        Self {
            trades: Vec::new(),
            initial_capital,
        }
    }
}

impl Backtester for SimpleBacktester {
    fn run(&self, signals: &[Signal], prices: &[f64]) -> Result<BacktestResult> {
        if signals.len() != prices.len() {
            return Err(FinancialError::BacktestError(
                format!("Signals length ({}) != prices length ({})", signals.len(), prices.len())
            ));
        }

        if prices.is_empty() {
            return Ok(BacktestResult {
                total_return: 0.0,
                sharpe_ratio: 0.0,
                max_drawdown: 0.0,
                win_rate: 0.0,
                num_trades: 0,
                equity_curve: vec![],
            });
        }

        let mut equity = self.initial_capital;
        let mut equity_curve = Vec::with_capacity(prices.len());
        let mut position: Option<(f64, i64)> = None; // (entry_price, entry_time)
        let mut trades: Vec<Trade> = Vec::new();
        let mut returns: Vec<f64> = Vec::new();
        let mut prev_equity = equity;

        for (i, (&signal, &price)) in signals.iter().zip(prices.iter()).enumerate() {
            match signal {
                Signal::Buy if position.is_none() => {
                    position = Some((price, i as i64));
                }
                Signal::Sell if position.is_some() => {
                    let (entry_price, entry_time) = position.take().unwrap();
                    let pnl = price - entry_price;
                    equity += pnl * (self.initial_capital / entry_price);
                    trades.push(Trade {
                        entry_price,
                        exit_price: price,
                        quantity: self.initial_capital / entry_price,
                        pnl,
                        entry_time,
                        exit_time: i as i64,
                    });
                }
                _ => {}
            }

            equity_curve.push(equity);
            if prev_equity > 0.0 {
                returns.push((equity - prev_equity) / prev_equity);
            }
            prev_equity = equity;
        }

        // Calculate metrics
        let total_return = if self.initial_capital > 0.0 {
            (equity - self.initial_capital) / self.initial_capital
        } else {
            0.0
        };

        let sharpe_ratio = calculate_sharpe(&returns, 0.0);
        let max_drawdown = calculate_max_drawdown(&equity_curve);
        let win_rate = if trades.is_empty() {
            0.0
        } else {
            trades.iter().filter(|t| t.pnl > 0.0).count() as f64 / trades.len() as f64
        };

        Ok(BacktestResult {
            total_return,
            sharpe_ratio,
            max_drawdown,
            win_rate,
            num_trades: trades.len(),
            equity_curve,
        })
    }

    fn trades(&self) -> &[Trade] {
        &self.trades
    }
}

fn calculate_sharpe(returns: &[f64], risk_free_rate: f64) -> f64 {
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

fn calculate_max_drawdown(equity_curve: &[f64]) -> f64 {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_backtest() {
        let backtester = SimpleBacktester::new(10000.0);
        let signals = vec![Signal::Buy, Signal::Hold, Signal::Hold, Signal::Sell];
        let prices = vec![100.0, 105.0, 110.0, 115.0];

        let result = backtester.run(&signals, &prices).unwrap();
        assert!(result.total_return > 0.0);
        assert_eq!(result.num_trades, 1);
    }
}
