//! Portfolio management implementation.

use financial_spi::{PortfolioManager, Position, Result, FinancialError};
use std::collections::HashMap;

/// Portfolio of positions.
#[derive(Debug, Clone, Default)]
pub struct Portfolio {
    positions: HashMap<String, Position>,
    cash: f64,
}

impl Portfolio {
    /// Create a new portfolio with initial cash.
    pub fn new(initial_cash: f64) -> Self {
        Self {
            positions: HashMap::new(),
            cash: initial_cash,
        }
    }
}

impl PortfolioManager for Portfolio {
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
        let cost = position.entry_price * position.quantity;
        if cost > self.cash {
            return Err(FinancialError::PortfolioError(
                format!("Insufficient cash: need {}, have {}", cost, self.cash)
            ));
        }
        self.cash -= cost;
        self.positions.insert(position.symbol.clone(), position);
        Ok(())
    }

    fn remove_position(&mut self, symbol: &str) -> Result<Option<Position>> {
        Ok(self.positions.remove(symbol))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portfolio_value() {
        let mut portfolio = Portfolio::new(10000.0);
        let position = Position {
            symbol: "AAPL".to_string(),
            quantity: 10.0,
            entry_price: 150.0,
            entry_timestamp: 0,
        };
        portfolio.add_position(position).unwrap();

        let mut prices = HashMap::new();
        prices.insert("AAPL".to_string(), 160.0);

        // Initial cash was 10000, cost was 1500, so cash = 8500
        // Position value = 10 * 160 = 1600
        // Total = 8500 + 1600 = 10100
        assert_eq!(portfolio.value(&prices), 10100.0);
    }
}
