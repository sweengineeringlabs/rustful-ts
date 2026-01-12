//! Backtesting result and trade models.

use serde::{Deserialize, Serialize};

/// Result of a backtest run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BacktestResult {
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub num_trades: usize,
    pub equity_curve: Vec<f64>,
}

/// A single trade.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Trade {
    pub entry_price: f64,
    pub exit_price: f64,
    pub quantity: f64,
    pub pnl: f64,
    pub entry_time: i64,
    pub exit_time: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_trade() -> Trade {
        Trade {
            entry_price: 100.0,
            exit_price: 110.0,
            quantity: 10.0,
            pnl: 100.0, // (110 - 100) * 10 = 100
            entry_time: 1704067200,
            exit_time: 1704153600,
        }
    }

    fn sample_backtest_result() -> BacktestResult {
        BacktestResult {
            total_return: 0.15,
            sharpe_ratio: 1.5,
            max_drawdown: 0.10,
            win_rate: 0.55,
            num_trades: 20,
            equity_curve: vec![1000.0, 1050.0, 1025.0, 1100.0, 1150.0],
        }
    }

    // Trade tests
    #[test]
    fn test_trade_creation() {
        let trade = sample_trade();
        assert_eq!(trade.entry_price, 100.0);
        assert_eq!(trade.exit_price, 110.0);
        assert_eq!(trade.quantity, 10.0);
        assert_eq!(trade.pnl, 100.0);
        assert_eq!(trade.entry_time, 1704067200);
        assert_eq!(trade.exit_time, 1704153600);
    }

    #[test]
    fn test_trade_winning() {
        let trade = Trade {
            entry_price: 100.0,
            exit_price: 120.0,
            quantity: 5.0,
            pnl: 100.0,
            entry_time: 1704067200,
            exit_time: 1704153600,
        };
        assert!(trade.pnl > 0.0);
    }

    #[test]
    fn test_trade_losing() {
        let trade = Trade {
            entry_price: 100.0,
            exit_price: 90.0,
            quantity: 5.0,
            pnl: -50.0,
            entry_time: 1704067200,
            exit_time: 1704153600,
        };
        assert!(trade.pnl < 0.0);
    }

    #[test]
    fn test_trade_breakeven() {
        let trade = Trade {
            entry_price: 100.0,
            exit_price: 100.0,
            quantity: 5.0,
            pnl: 0.0,
            entry_time: 1704067200,
            exit_time: 1704153600,
        };
        assert_eq!(trade.pnl, 0.0);
    }

    #[test]
    fn test_trade_short_position() {
        // Short: profit when exit_price < entry_price
        let trade = Trade {
            entry_price: 100.0,
            exit_price: 90.0,
            quantity: -5.0, // negative for short
            pnl: 50.0,      // (100 - 90) * 5 = 50 profit on short
            entry_time: 1704067200,
            exit_time: 1704153600,
        };
        assert!(trade.quantity < 0.0);
        assert!(trade.pnl > 0.0);
    }

    #[test]
    fn test_trade_clone() {
        let trade = sample_trade();
        let cloned = trade.clone();
        assert_eq!(trade, cloned);
    }

    #[test]
    fn test_trade_debug() {
        let trade = sample_trade();
        let debug_str = format!("{:?}", trade);
        assert!(debug_str.contains("Trade"));
        assert!(debug_str.contains("100"));
        assert!(debug_str.contains("110"));
    }

    #[test]
    fn test_trade_serialize() {
        let trade = sample_trade();
        let json = serde_json::to_string(&trade).unwrap();
        assert!(json.contains("\"entry_price\":100.0"));
        assert!(json.contains("\"exit_price\":110.0"));
        assert!(json.contains("\"quantity\":10.0"));
        assert!(json.contains("\"pnl\":100.0"));
    }

    #[test]
    fn test_trade_deserialize() {
        let json = r#"{
            "entry_price": 50.0,
            "exit_price": 55.0,
            "quantity": 20.0,
            "pnl": 100.0,
            "entry_time": 1704240000,
            "exit_time": 1704326400
        }"#;
        let trade: Trade = serde_json::from_str(json).unwrap();
        assert_eq!(trade.entry_price, 50.0);
        assert_eq!(trade.exit_price, 55.0);
        assert_eq!(trade.quantity, 20.0);
        assert_eq!(trade.pnl, 100.0);
    }

    #[test]
    fn test_trade_roundtrip_serialization() {
        let trade = sample_trade();
        let json = serde_json::to_string(&trade).unwrap();
        let deserialized: Trade = serde_json::from_str(&json).unwrap();
        assert_eq!(trade, deserialized);
    }

    #[test]
    fn test_trade_duration() {
        let trade = sample_trade();
        let duration = trade.exit_time - trade.entry_time;
        assert_eq!(duration, 86400); // 1 day in seconds
    }

    #[test]
    fn test_trade_return_percentage() {
        let trade = sample_trade();
        let return_pct = (trade.exit_price - trade.entry_price) / trade.entry_price;
        assert!((return_pct - 0.1).abs() < 0.0001); // 10% return
    }

    // BacktestResult tests
    #[test]
    fn test_backtest_result_creation() {
        let result = sample_backtest_result();
        assert_eq!(result.total_return, 0.15);
        assert_eq!(result.sharpe_ratio, 1.5);
        assert_eq!(result.max_drawdown, 0.10);
        assert_eq!(result.win_rate, 0.55);
        assert_eq!(result.num_trades, 20);
        assert_eq!(result.equity_curve.len(), 5);
    }

    #[test]
    fn test_backtest_result_profitable() {
        let result = sample_backtest_result();
        assert!(result.total_return > 0.0);
    }

    #[test]
    fn test_backtest_result_unprofitable() {
        let result = BacktestResult {
            total_return: -0.05,
            sharpe_ratio: -0.5,
            max_drawdown: 0.20,
            win_rate: 0.40,
            num_trades: 15,
            equity_curve: vec![1000.0, 980.0, 960.0, 950.0],
        };
        assert!(result.total_return < 0.0);
        assert!(result.sharpe_ratio < 0.0);
    }

    #[test]
    fn test_backtest_result_win_rate_valid() {
        let result = sample_backtest_result();
        assert!(result.win_rate >= 0.0 && result.win_rate <= 1.0);
    }

    #[test]
    fn test_backtest_result_max_drawdown_positive() {
        let result = sample_backtest_result();
        assert!(result.max_drawdown >= 0.0);
    }

    #[test]
    fn test_backtest_result_clone() {
        let result = sample_backtest_result();
        let cloned = result.clone();
        assert_eq!(result, cloned);
    }

    #[test]
    fn test_backtest_result_debug() {
        let result = sample_backtest_result();
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("BacktestResult"));
        assert!(debug_str.contains("total_return"));
        assert!(debug_str.contains("sharpe_ratio"));
    }

    #[test]
    fn test_backtest_result_serialize() {
        let result = sample_backtest_result();
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"total_return\":0.15"));
        assert!(json.contains("\"sharpe_ratio\":1.5"));
        assert!(json.contains("\"max_drawdown\":0.1"));
        assert!(json.contains("\"win_rate\":0.55"));
        assert!(json.contains("\"num_trades\":20"));
        assert!(json.contains("equity_curve"));
    }

    #[test]
    fn test_backtest_result_deserialize() {
        let json = r#"{
            "total_return": 0.25,
            "sharpe_ratio": 2.0,
            "max_drawdown": 0.08,
            "win_rate": 0.60,
            "num_trades": 30,
            "equity_curve": [1000.0, 1100.0, 1250.0]
        }"#;
        let result: BacktestResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.total_return, 0.25);
        assert_eq!(result.sharpe_ratio, 2.0);
        assert_eq!(result.max_drawdown, 0.08);
        assert_eq!(result.win_rate, 0.60);
        assert_eq!(result.num_trades, 30);
        assert_eq!(result.equity_curve, vec![1000.0, 1100.0, 1250.0]);
    }

    #[test]
    fn test_backtest_result_roundtrip_serialization() {
        let result = sample_backtest_result();
        let json = serde_json::to_string(&result).unwrap();
        let deserialized: BacktestResult = serde_json::from_str(&json).unwrap();
        assert_eq!(result, deserialized);
    }

    #[test]
    fn test_backtest_result_empty_equity_curve() {
        let result = BacktestResult {
            total_return: 0.0,
            sharpe_ratio: 0.0,
            max_drawdown: 0.0,
            win_rate: 0.0,
            num_trades: 0,
            equity_curve: vec![],
        };
        assert!(result.equity_curve.is_empty());
        assert_eq!(result.num_trades, 0);
    }

    #[test]
    fn test_backtest_result_single_point_equity() {
        let result = BacktestResult {
            total_return: 0.0,
            sharpe_ratio: 0.0,
            max_drawdown: 0.0,
            win_rate: 0.0,
            num_trades: 0,
            equity_curve: vec![10000.0],
        };
        assert_eq!(result.equity_curve.len(), 1);
    }

    #[test]
    fn test_backtest_result_large_equity_curve() {
        let equity_curve: Vec<f64> = (0..1000).map(|i| 1000.0 + i as f64).collect();
        let result = BacktestResult {
            total_return: 0.999,
            sharpe_ratio: 3.0,
            max_drawdown: 0.01,
            win_rate: 0.90,
            num_trades: 500,
            equity_curve,
        };
        assert_eq!(result.equity_curve.len(), 1000);
        assert_eq!(result.equity_curve[0], 1000.0);
        assert_eq!(result.equity_curve[999], 1999.0);
    }

    #[test]
    fn test_backtest_result_extreme_values() {
        let result = BacktestResult {
            total_return: 10.0, // 1000% return
            sharpe_ratio: 5.0,
            max_drawdown: 0.99, // 99% drawdown
            win_rate: 0.99,
            num_trades: 10000,
            equity_curve: vec![100.0, 1000.0, 10000.0],
        };
        assert_eq!(result.total_return, 10.0);
        assert_eq!(result.max_drawdown, 0.99);
    }

    #[test]
    fn test_trade_vec_operations() {
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
                exit_price: 105.0,
                quantity: 10.0,
                pnl: -50.0,
                entry_time: 1704153600,
                exit_time: 1704240000,
            },
            Trade {
                entry_price: 105.0,
                exit_price: 115.0,
                quantity: 10.0,
                pnl: 100.0,
                entry_time: 1704240000,
                exit_time: 1704326400,
            },
        ];

        let total_pnl: f64 = trades.iter().map(|t| t.pnl).sum();
        let winning_trades = trades.iter().filter(|t| t.pnl > 0.0).count();
        let win_rate = winning_trades as f64 / trades.len() as f64;

        assert_eq!(total_pnl, 150.0);
        assert_eq!(winning_trades, 2);
        assert!((win_rate - 0.6667).abs() < 0.01);
    }
}
