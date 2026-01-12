//! Portfolio manager trait.

use crate::error::Result;
use crate::model::Position;
use std::collections::HashMap;

/// Portfolio trait for position management.
pub trait PortfolioManager: Send + Sync {
    /// Get current cash balance.
    fn cash(&self) -> f64;

    /// Get all positions.
    fn positions(&self) -> &HashMap<String, Position>;

    /// Calculate total portfolio value given current prices.
    fn value(&self, prices: &HashMap<String, f64>) -> f64;

    /// Add a position.
    fn add_position(&mut self, position: Position) -> Result<()>;

    /// Remove a position.
    fn remove_position(&mut self, symbol: &str) -> Result<Option<Position>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::FinancialError;
    use std::sync::Arc;
    use std::sync::Mutex;

    /// Mock portfolio manager implementation for testing
    struct MockPortfolio {
        cash: f64,
        positions: HashMap<String, Position>,
    }

    impl MockPortfolio {
        fn new(initial_cash: f64) -> Self {
            Self {
                cash: initial_cash,
                positions: HashMap::new(),
            }
        }
    }

    impl PortfolioManager for MockPortfolio {
        fn cash(&self) -> f64 {
            self.cash
        }

        fn positions(&self) -> &HashMap<String, Position> {
            &self.positions
        }

        fn value(&self, prices: &HashMap<String, f64>) -> f64 {
            let positions_value: f64 = self
                .positions
                .iter()
                .map(|(symbol, pos)| {
                    prices.get(symbol).unwrap_or(&pos.entry_price) * pos.quantity
                })
                .sum();
            self.cash + positions_value
        }

        fn add_position(&mut self, position: Position) -> Result<()> {
            if self.positions.contains_key(&position.symbol) {
                return Err(FinancialError::PortfolioError(format!(
                    "Position already exists: {}",
                    position.symbol
                )));
            }
            let cost = position.entry_price * position.quantity;
            if cost > self.cash {
                return Err(FinancialError::PortfolioError(
                    "Insufficient cash".to_string(),
                ));
            }
            self.cash -= cost;
            self.positions.insert(position.symbol.clone(), position);
            Ok(())
        }

        fn remove_position(&mut self, symbol: &str) -> Result<Option<Position>> {
            match self.positions.remove(symbol) {
                Some(pos) => {
                    self.cash += pos.entry_price * pos.quantity;
                    Ok(Some(pos))
                }
                None => Ok(None),
            }
        }
    }

    #[test]
    fn test_portfolio_initial_cash() {
        let portfolio = MockPortfolio::new(10000.0);
        assert_eq!(portfolio.cash(), 10000.0);
    }

    #[test]
    fn test_portfolio_empty_positions() {
        let portfolio = MockPortfolio::new(10000.0);
        assert!(portfolio.positions().is_empty());
    }

    #[test]
    fn test_portfolio_add_position() {
        let mut portfolio = MockPortfolio::new(10000.0);
        let position = Position {
            symbol: "AAPL".to_string(),
            quantity: 10.0,
            entry_price: 150.0,
            entry_timestamp: 1704067200,
        };

        let result = portfolio.add_position(position);
        assert!(result.is_ok());
        assert_eq!(portfolio.positions().len(), 1);
        assert!(portfolio.positions().contains_key("AAPL"));
        assert_eq!(portfolio.cash(), 10000.0 - 1500.0); // 10 * 150 = 1500
    }

    #[test]
    fn test_portfolio_add_multiple_positions() {
        let mut portfolio = MockPortfolio::new(100000.0);

        let positions = vec![
            Position {
                symbol: "AAPL".to_string(),
                quantity: 10.0,
                entry_price: 150.0,
                entry_timestamp: 1704067200,
            },
            Position {
                symbol: "MSFT".to_string(),
                quantity: 5.0,
                entry_price: 300.0,
                entry_timestamp: 1704067200,
            },
            Position {
                symbol: "GOOG".to_string(),
                quantity: 3.0,
                entry_price: 140.0,
                entry_timestamp: 1704067200,
            },
        ];

        for pos in positions {
            assert!(portfolio.add_position(pos).is_ok());
        }

        assert_eq!(portfolio.positions().len(), 3);
        // Cash reduced by: 10*150 + 5*300 + 3*140 = 1500 + 1500 + 420 = 3420
        assert_eq!(portfolio.cash(), 100000.0 - 3420.0);
    }

    #[test]
    fn test_portfolio_add_duplicate_position() {
        let mut portfolio = MockPortfolio::new(10000.0);
        let position1 = Position {
            symbol: "AAPL".to_string(),
            quantity: 10.0,
            entry_price: 150.0,
            entry_timestamp: 1704067200,
        };
        let position2 = Position {
            symbol: "AAPL".to_string(),
            quantity: 5.0,
            entry_price: 155.0,
            entry_timestamp: 1704153600,
        };

        assert!(portfolio.add_position(position1).is_ok());
        let result = portfolio.add_position(position2);
        assert!(result.is_err());
        match result {
            Err(FinancialError::PortfolioError(msg)) => {
                assert!(msg.contains("already exists"));
            }
            _ => panic!("Expected PortfolioError"),
        }
    }

    #[test]
    fn test_portfolio_add_position_insufficient_cash() {
        let mut portfolio = MockPortfolio::new(1000.0);
        let position = Position {
            symbol: "AAPL".to_string(),
            quantity: 10.0,
            entry_price: 150.0, // 10 * 150 = 1500 > 1000
            entry_timestamp: 1704067200,
        };

        let result = portfolio.add_position(position);
        assert!(result.is_err());
        match result {
            Err(FinancialError::PortfolioError(msg)) => {
                assert!(msg.contains("Insufficient cash"));
            }
            _ => panic!("Expected PortfolioError"),
        }
    }

    #[test]
    fn test_portfolio_remove_position() {
        let mut portfolio = MockPortfolio::new(10000.0);
        let position = Position {
            symbol: "AAPL".to_string(),
            quantity: 10.0,
            entry_price: 150.0,
            entry_timestamp: 1704067200,
        };

        portfolio.add_position(position).unwrap();
        let initial_cash = portfolio.cash();

        let removed = portfolio.remove_position("AAPL").unwrap();
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().symbol, "AAPL");
        assert!(portfolio.positions().is_empty());
        assert_eq!(portfolio.cash(), initial_cash + 1500.0); // Cash restored
    }

    #[test]
    fn test_portfolio_remove_nonexistent_position() {
        let mut portfolio = MockPortfolio::new(10000.0);
        let removed = portfolio.remove_position("AAPL").unwrap();
        assert!(removed.is_none());
    }

    #[test]
    fn test_portfolio_value_cash_only() {
        let portfolio = MockPortfolio::new(10000.0);
        let prices = HashMap::new();
        assert_eq!(portfolio.value(&prices), 10000.0);
    }

    #[test]
    fn test_portfolio_value_with_positions() {
        let mut portfolio = MockPortfolio::new(10000.0);
        let position = Position {
            symbol: "AAPL".to_string(),
            quantity: 10.0,
            entry_price: 150.0,
            entry_timestamp: 1704067200,
        };
        portfolio.add_position(position).unwrap();

        let mut prices = HashMap::new();
        prices.insert("AAPL".to_string(), 160.0); // Price increased

        // Cash: 10000 - 1500 = 8500
        // Position value: 10 * 160 = 1600
        // Total: 8500 + 1600 = 10100
        assert_eq!(portfolio.value(&prices), 10100.0);
    }

    #[test]
    fn test_portfolio_value_price_decrease() {
        let mut portfolio = MockPortfolio::new(10000.0);
        let position = Position {
            symbol: "AAPL".to_string(),
            quantity: 10.0,
            entry_price: 150.0,
            entry_timestamp: 1704067200,
        };
        portfolio.add_position(position).unwrap();

        let mut prices = HashMap::new();
        prices.insert("AAPL".to_string(), 140.0); // Price decreased

        // Cash: 10000 - 1500 = 8500
        // Position value: 10 * 140 = 1400
        // Total: 8500 + 1400 = 9900
        assert_eq!(portfolio.value(&prices), 9900.0);
    }

    #[test]
    fn test_portfolio_value_missing_price_uses_entry() {
        let mut portfolio = MockPortfolio::new(10000.0);
        let position = Position {
            symbol: "AAPL".to_string(),
            quantity: 10.0,
            entry_price: 150.0,
            entry_timestamp: 1704067200,
        };
        portfolio.add_position(position).unwrap();

        let prices = HashMap::new(); // No prices provided

        // Cash: 10000 - 1500 = 8500
        // Position value: 10 * 150 (entry price) = 1500
        // Total: 8500 + 1500 = 10000
        assert_eq!(portfolio.value(&prices), 10000.0);
    }

    #[test]
    fn test_portfolio_value_multiple_positions() {
        let mut portfolio = MockPortfolio::new(100000.0);

        portfolio
            .add_position(Position {
                symbol: "AAPL".to_string(),
                quantity: 10.0,
                entry_price: 150.0,
                entry_timestamp: 1704067200,
            })
            .unwrap();

        portfolio
            .add_position(Position {
                symbol: "MSFT".to_string(),
                quantity: 5.0,
                entry_price: 300.0,
                entry_timestamp: 1704067200,
            })
            .unwrap();

        let mut prices = HashMap::new();
        prices.insert("AAPL".to_string(), 160.0);
        prices.insert("MSFT".to_string(), 320.0);

        // Cash: 100000 - 1500 - 1500 = 97000
        // AAPL value: 10 * 160 = 1600
        // MSFT value: 5 * 320 = 1600
        // Total: 97000 + 1600 + 1600 = 100200
        assert_eq!(portfolio.value(&prices), 100200.0);
    }

    #[test]
    fn test_portfolio_manager_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<MockPortfolio>();
    }

    #[test]
    fn test_portfolio_manager_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<MockPortfolio>();
    }

    #[test]
    fn test_portfolio_manager_in_arc_mutex() {
        let portfolio: Arc<Mutex<MockPortfolio>> = Arc::new(Mutex::new(MockPortfolio::new(10000.0)));

        {
            let mut p = portfolio.lock().unwrap();
            p.add_position(Position {
                symbol: "AAPL".to_string(),
                quantity: 5.0,
                entry_price: 100.0,
                entry_timestamp: 1704067200,
            })
            .unwrap();
        }

        let p = portfolio.lock().unwrap();
        assert_eq!(p.positions().len(), 1);
        assert_eq!(p.cash(), 9500.0);
    }

    #[test]
    fn test_portfolio_positions_reference() {
        let mut portfolio = MockPortfolio::new(10000.0);
        portfolio
            .add_position(Position {
                symbol: "AAPL".to_string(),
                quantity: 10.0,
                entry_price: 150.0,
                entry_timestamp: 1704067200,
            })
            .unwrap();

        let positions = portfolio.positions();
        assert!(positions.contains_key("AAPL"));

        let aapl_pos = positions.get("AAPL").unwrap();
        assert_eq!(aapl_pos.quantity, 10.0);
        assert_eq!(aapl_pos.entry_price, 150.0);
    }

    #[test]
    fn test_portfolio_full_lifecycle() {
        let mut portfolio = MockPortfolio::new(10000.0);
        let mut prices = HashMap::new();

        // Initial state
        assert_eq!(portfolio.cash(), 10000.0);
        assert!(portfolio.positions().is_empty());
        assert_eq!(portfolio.value(&prices), 10000.0);

        // Add position
        let position = Position {
            symbol: "AAPL".to_string(),
            quantity: 10.0,
            entry_price: 150.0,
            entry_timestamp: 1704067200,
        };
        portfolio.add_position(position).unwrap();
        assert_eq!(portfolio.cash(), 8500.0);
        assert_eq!(portfolio.positions().len(), 1);

        // Check value with price increase
        prices.insert("AAPL".to_string(), 170.0);
        assert_eq!(portfolio.value(&prices), 8500.0 + 1700.0); // 10200

        // Remove position
        let removed = portfolio.remove_position("AAPL").unwrap().unwrap();
        assert_eq!(removed.symbol, "AAPL");
        assert_eq!(portfolio.cash(), 10000.0); // Cash restored to original + entry value
        assert!(portfolio.positions().is_empty());
    }
}
