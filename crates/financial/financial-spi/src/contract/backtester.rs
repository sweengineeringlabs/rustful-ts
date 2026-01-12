//! Backtester trait.

use crate::error::Result;
use crate::model::{BacktestResult, Signal, Trade};

/// Backtest engine trait.
pub trait Backtester: Send + Sync {
    /// Run backtest with given signals and price data.
    fn run(&self, signals: &[Signal], prices: &[f64]) -> Result<BacktestResult>;

    /// Get all trades from the last backtest.
    fn trades(&self) -> &[Trade];
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::FinancialError;
    use std::sync::Arc;

    /// Helper function to run a backtest simulation (not a trait impl, just for testing logic)
    fn run_backtest_simulation(
        signals: &[Signal],
        prices: &[f64],
        initial_capital: f64,
    ) -> Result<BacktestResult> {
        if signals.len() != prices.len() {
            return Err(FinancialError::BacktestError(
                "Signals and prices length mismatch".to_string(),
            ));
        }
        if signals.is_empty() {
            return Err(FinancialError::BacktestError(
                "No data provided".to_string(),
            ));
        }

        let mut trades = Vec::new();
        let mut equity_curve = vec![initial_capital];
        let mut capital = initial_capital;
        let mut position: Option<(f64, i64)> = None; // (entry_price, entry_time)

        for (i, (signal, &price)) in signals.iter().zip(prices.iter()).enumerate() {
            let time = i as i64;

            match (signal, &position) {
                (Signal::Buy, None) => {
                    position = Some((price, time));
                }
                (Signal::Sell, Some((entry_price, entry_time))) => {
                    let pnl = price - entry_price;
                    capital += pnl;
                    trades.push(Trade {
                        entry_price: *entry_price,
                        exit_price: price,
                        quantity: 1.0,
                        pnl,
                        entry_time: *entry_time,
                        exit_time: time,
                    });
                    position = None;
                }
                _ => {}
            }
            equity_curve.push(capital);
        }

        // Calculate metrics
        let total_return = (capital - initial_capital) / initial_capital;
        let num_trades = trades.len();
        let winning_trades = trades.iter().filter(|t| t.pnl > 0.0).count();
        let win_rate = if num_trades > 0 {
            winning_trades as f64 / num_trades as f64
        } else {
            0.0
        };

        // Simple max drawdown calculation
        let mut max_drawdown = 0.0;
        let mut peak = equity_curve[0];
        for &equity in &equity_curve {
            if equity > peak {
                peak = equity;
            }
            let drawdown = (peak - equity) / peak;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        // Simple Sharpe approximation (not realistic, just for testing)
        let sharpe_ratio = if total_return > 0.0 {
            total_return / (max_drawdown + 0.01)
        } else {
            total_return
        };

        Ok(BacktestResult {
            total_return,
            sharpe_ratio,
            max_drawdown,
            win_rate,
            num_trades,
            equity_curve,
        })
    }

    /// Simple backtester that always returns fixed results (for testing trait contract)
    struct FixedResultBacktester {
        result: BacktestResult,
        trades: Vec<Trade>,
    }

    impl FixedResultBacktester {
        fn new(result: BacktestResult, trades: Vec<Trade>) -> Self {
            Self { result, trades }
        }
    }

    impl Backtester for FixedResultBacktester {
        fn run(&self, _signals: &[Signal], _prices: &[f64]) -> Result<BacktestResult> {
            Ok(self.result.clone())
        }

        fn trades(&self) -> &[Trade] {
            &self.trades
        }
    }

    #[test]
    fn test_backtest_simulation_basic() {
        let signals = vec![Signal::Buy, Signal::Hold, Signal::Sell];
        let prices = vec![100.0, 105.0, 110.0];

        let result = run_backtest_simulation(&signals, &prices, 1000.0).unwrap();

        assert_eq!(result.num_trades, 1);
        assert!(result.total_return > 0.0); // Profit: bought at 100, sold at 110
    }

    #[test]
    fn test_backtest_simulation_multiple_trades() {
        let signals = vec![
            Signal::Buy,
            Signal::Sell,
            Signal::Buy,
            Signal::Sell,
        ];
        let prices = vec![100.0, 110.0, 105.0, 115.0];

        let result = run_backtest_simulation(&signals, &prices, 1000.0).unwrap();

        assert_eq!(result.num_trades, 2);
        // Trade 1: Buy 100, Sell 110 = +10
        // Trade 2: Buy 105, Sell 115 = +10
        // Total return: 20/1000 = 0.02
        assert!((result.total_return - 0.02).abs() < 0.001);
    }

    #[test]
    fn test_backtest_simulation_no_trades() {
        let signals = vec![Signal::Hold, Signal::Hold, Signal::Hold];
        let prices = vec![100.0, 105.0, 110.0];

        let result = run_backtest_simulation(&signals, &prices, 1000.0).unwrap();

        assert_eq!(result.num_trades, 0);
        assert_eq!(result.total_return, 0.0);
        assert_eq!(result.win_rate, 0.0);
    }

    #[test]
    fn test_backtest_simulation_losing_trade() {
        let signals = vec![Signal::Buy, Signal::Sell];
        let prices = vec![100.0, 90.0]; // Buy at 100, sell at 90

        let result = run_backtest_simulation(&signals, &prices, 1000.0).unwrap();

        assert_eq!(result.num_trades, 1);
        assert!(result.total_return < 0.0); // Loss
        assert_eq!(result.win_rate, 0.0);
    }

    #[test]
    fn test_backtest_simulation_mixed_results() {
        let signals = vec![
            Signal::Buy,
            Signal::Sell, // Win
            Signal::Buy,
            Signal::Sell, // Loss
        ];
        let prices = vec![100.0, 110.0, 115.0, 105.0];

        let result = run_backtest_simulation(&signals, &prices, 1000.0).unwrap();

        assert_eq!(result.num_trades, 2);
        assert_eq!(result.win_rate, 0.5); // 1 win, 1 loss
    }

    #[test]
    fn test_backtest_simulation_length_mismatch() {
        let signals = vec![Signal::Buy, Signal::Sell];
        let prices = vec![100.0, 110.0, 120.0];

        let result = run_backtest_simulation(&signals, &prices, 1000.0);
        assert!(result.is_err());
        match result {
            Err(FinancialError::BacktestError(msg)) => {
                assert!(msg.contains("mismatch"));
            }
            _ => panic!("Expected BacktestError"),
        }
    }

    #[test]
    fn test_backtest_simulation_empty_data() {
        let signals: Vec<Signal> = vec![];
        let prices: Vec<f64> = vec![];

        let result = run_backtest_simulation(&signals, &prices, 1000.0);
        assert!(result.is_err());
        match result {
            Err(FinancialError::BacktestError(msg)) => {
                assert!(msg.contains("No data"));
            }
            _ => panic!("Expected BacktestError"),
        }
    }

    #[test]
    fn test_backtest_simulation_equity_curve() {
        let signals = vec![Signal::Buy, Signal::Hold, Signal::Sell];
        let prices = vec![100.0, 105.0, 110.0];

        let result = run_backtest_simulation(&signals, &prices, 1000.0).unwrap();

        assert!(!result.equity_curve.is_empty());
        assert_eq!(result.equity_curve[0], 1000.0); // Initial capital
    }

    #[test]
    fn test_backtest_simulation_max_drawdown() {
        let signals = vec![Signal::Buy, Signal::Hold, Signal::Sell];
        let prices = vec![100.0, 105.0, 110.0];

        let result = run_backtest_simulation(&signals, &prices, 1000.0).unwrap();

        assert!(result.max_drawdown >= 0.0);
        assert!(result.max_drawdown <= 1.0);
    }

    #[test]
    fn test_fixed_result_backtester() {
        let fixed_result = BacktestResult {
            total_return: 0.25,
            sharpe_ratio: 2.0,
            max_drawdown: 0.05,
            win_rate: 0.65,
            num_trades: 50,
            equity_curve: vec![1000.0, 1100.0, 1250.0],
        };
        let trades = vec![
            Trade {
                entry_price: 100.0,
                exit_price: 110.0,
                quantity: 10.0,
                pnl: 100.0,
                entry_time: 1704067200,
                exit_time: 1704153600,
            },
        ];

        let backtester = FixedResultBacktester::new(fixed_result.clone(), trades);

        let result = backtester.run(&[], &[]).unwrap();
        assert_eq!(result, fixed_result);
    }

    #[test]
    fn test_backtester_trades_method() {
        let trades = vec![
            Trade {
                entry_price: 100.0,
                exit_price: 110.0,
                quantity: 10.0,
                pnl: 100.0,
                entry_time: 1704067200,
                exit_time: 1704153600,
            },
            Trade {
                entry_price: 110.0,
                exit_price: 120.0,
                quantity: 5.0,
                pnl: 50.0,
                entry_time: 1704153600,
                exit_time: 1704240000,
            },
        ];
        let result = BacktestResult {
            total_return: 0.15,
            sharpe_ratio: 1.5,
            max_drawdown: 0.05,
            win_rate: 1.0,
            num_trades: 2,
            equity_curve: vec![1000.0, 1100.0, 1150.0],
        };

        let backtester = FixedResultBacktester::new(result, trades.clone());

        let stored_trades = backtester.trades();
        assert_eq!(stored_trades.len(), 2);
        assert_eq!(stored_trades[0].entry_price, 100.0);
        assert_eq!(stored_trades[1].entry_price, 110.0);
    }

    #[test]
    fn test_backtester_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<FixedResultBacktester>();
    }

    #[test]
    fn test_backtester_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<FixedResultBacktester>();
    }

    #[test]
    fn test_backtester_trait_object() {
        let result = BacktestResult {
            total_return: 0.10,
            sharpe_ratio: 1.0,
            max_drawdown: 0.05,
            win_rate: 0.50,
            num_trades: 10,
            equity_curve: vec![1000.0, 1100.0],
        };

        let backtester: Box<dyn Backtester> =
            Box::new(FixedResultBacktester::new(result.clone(), vec![]));

        let run_result = backtester.run(&[], &[]).unwrap();
        assert_eq!(run_result.total_return, 0.10);
    }

    #[test]
    fn test_backtester_in_arc() {
        let result = BacktestResult {
            total_return: 0.20,
            sharpe_ratio: 1.5,
            max_drawdown: 0.08,
            win_rate: 0.60,
            num_trades: 25,
            equity_curve: vec![1000.0, 1200.0],
        };

        let backtester: Arc<dyn Backtester> =
            Arc::new(FixedResultBacktester::new(result.clone(), vec![]));

        let run_result = backtester.run(&[], &[]).unwrap();
        assert_eq!(run_result.total_return, 0.20);
    }

    #[test]
    fn test_backtest_simulation_unclosed_position() {
        // Buy but never sell
        let signals = vec![Signal::Buy, Signal::Hold, Signal::Hold];
        let prices = vec![100.0, 110.0, 120.0];

        let result = run_backtest_simulation(&signals, &prices, 1000.0).unwrap();

        // No trades completed since position wasn't closed
        assert_eq!(result.num_trades, 0);
    }

    #[test]
    fn test_backtest_simulation_sell_without_position() {
        // Sell without having a position
        let signals = vec![Signal::Sell, Signal::Buy, Signal::Sell];
        let prices = vec![100.0, 110.0, 120.0];

        let result = run_backtest_simulation(&signals, &prices, 1000.0).unwrap();

        // Only one trade: Buy at 110, Sell at 120
        assert_eq!(result.num_trades, 1);
    }

    #[test]
    fn test_backtest_simulation_long_sequence() {
        let n = 100;
        let signals: Vec<Signal> = (0..n)
            .map(|i| match i % 4 {
                0 => Signal::Buy,
                2 => Signal::Sell,
                _ => Signal::Hold,
            })
            .collect();
        let prices: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64) * 0.5).collect();

        let result = run_backtest_simulation(&signals, &prices, 10000.0).unwrap();

        assert!(result.num_trades > 0);
        assert_eq!(result.equity_curve.len(), n + 1);
    }
}
