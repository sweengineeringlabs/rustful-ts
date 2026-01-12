//! Portfolio position model.

use serde::{Deserialize, Serialize};

/// A single position in the portfolio.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Position {
    pub symbol: String,
    pub quantity: f64,
    pub entry_price: f64,
    pub entry_timestamp: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_position() -> Position {
        Position {
            symbol: "AAPL".to_string(),
            quantity: 100.0,
            entry_price: 150.50,
            entry_timestamp: 1704067200, // 2024-01-01 00:00:00 UTC
        }
    }

    #[test]
    fn test_position_creation() {
        let position = sample_position();
        assert_eq!(position.symbol, "AAPL");
        assert_eq!(position.quantity, 100.0);
        assert_eq!(position.entry_price, 150.50);
        assert_eq!(position.entry_timestamp, 1704067200);
    }

    #[test]
    fn test_position_with_zero_quantity() {
        let position = Position {
            symbol: "MSFT".to_string(),
            quantity: 0.0,
            entry_price: 300.0,
            entry_timestamp: 1704153600,
        };
        assert_eq!(position.quantity, 0.0);
    }

    #[test]
    fn test_position_with_negative_quantity() {
        // Short position
        let position = Position {
            symbol: "TSLA".to_string(),
            quantity: -50.0,
            entry_price: 250.0,
            entry_timestamp: 1704240000,
        };
        assert_eq!(position.quantity, -50.0);
    }

    #[test]
    fn test_position_with_fractional_quantity() {
        let position = Position {
            symbol: "BTC".to_string(),
            quantity: 0.5,
            entry_price: 45000.0,
            entry_timestamp: 1704326400,
        };
        assert_eq!(position.quantity, 0.5);
    }

    #[test]
    fn test_position_clone() {
        let position = sample_position();
        let cloned = position.clone();
        assert_eq!(position.symbol, cloned.symbol);
        assert_eq!(position.quantity, cloned.quantity);
        assert_eq!(position.entry_price, cloned.entry_price);
        assert_eq!(position.entry_timestamp, cloned.entry_timestamp);
    }

    #[test]
    fn test_position_debug() {
        let position = sample_position();
        let debug_str = format!("{:?}", position);
        assert!(debug_str.contains("Position"));
        assert!(debug_str.contains("AAPL"));
        assert!(debug_str.contains("100"));
        assert!(debug_str.contains("150.5"));
    }

    #[test]
    fn test_position_serialize() {
        let position = sample_position();
        let json = serde_json::to_string(&position).unwrap();
        assert!(json.contains("\"symbol\":\"AAPL\""));
        assert!(json.contains("\"quantity\":100.0"));
        assert!(json.contains("\"entry_price\":150.5"));
        assert!(json.contains("\"entry_timestamp\":1704067200"));
    }

    #[test]
    fn test_position_deserialize() {
        let json = r#"{
            "symbol": "GOOG",
            "quantity": 25.0,
            "entry_price": 140.25,
            "entry_timestamp": 1704412800
        }"#;
        let position: Position = serde_json::from_str(json).unwrap();
        assert_eq!(position.symbol, "GOOG");
        assert_eq!(position.quantity, 25.0);
        assert_eq!(position.entry_price, 140.25);
        assert_eq!(position.entry_timestamp, 1704412800);
    }

    #[test]
    fn test_position_roundtrip_serialization() {
        let position = sample_position();
        let json = serde_json::to_string(&position).unwrap();
        let deserialized: Position = serde_json::from_str(&json).unwrap();
        assert_eq!(position, deserialized);
    }

    #[test]
    fn test_position_equality() {
        let pos1 = sample_position();
        let pos2 = sample_position();
        assert_eq!(pos1, pos2);
    }

    #[test]
    fn test_position_inequality_symbol() {
        let pos1 = sample_position();
        let pos2 = Position {
            symbol: "MSFT".to_string(),
            ..pos1.clone()
        };
        assert_ne!(pos1, pos2);
    }

    #[test]
    fn test_position_inequality_quantity() {
        let pos1 = sample_position();
        let pos2 = Position {
            quantity: 200.0,
            ..pos1.clone()
        };
        assert_ne!(pos1, pos2);
    }

    #[test]
    fn test_position_in_vec() {
        let positions = vec![
            Position {
                symbol: "AAPL".to_string(),
                quantity: 100.0,
                entry_price: 150.0,
                entry_timestamp: 1704067200,
            },
            Position {
                symbol: "MSFT".to_string(),
                quantity: 50.0,
                entry_price: 300.0,
                entry_timestamp: 1704153600,
            },
        ];
        assert_eq!(positions.len(), 2);
        assert_eq!(positions[0].symbol, "AAPL");
        assert_eq!(positions[1].symbol, "MSFT");
    }

    #[test]
    fn test_position_market_value_calculation() {
        let position = sample_position();
        let current_price = 160.0;
        let market_value = position.quantity * current_price;
        assert_eq!(market_value, 16000.0);
    }

    #[test]
    fn test_position_pnl_calculation() {
        let position = sample_position();
        let current_price = 160.0;
        let pnl = position.quantity * (current_price - position.entry_price);
        assert_eq!(pnl, 950.0); // 100 * (160 - 150.5) = 950
    }

    #[test]
    fn test_position_with_special_symbols() {
        let symbols = vec!["BRK.A", "SPY-USD", "ETH/USDT", "AAPL_2024"];
        for symbol in symbols {
            let position = Position {
                symbol: symbol.to_string(),
                quantity: 10.0,
                entry_price: 100.0,
                entry_timestamp: 1704067200,
            };
            assert_eq!(position.symbol, symbol);
        }
    }

    #[test]
    fn test_position_large_values() {
        let position = Position {
            symbol: "BRK.A".to_string(),
            quantity: 1.0,
            entry_price: 500000.0,
            entry_timestamp: i64::MAX - 1000,
        };
        assert_eq!(position.entry_price, 500000.0);
        assert!(position.entry_timestamp > 0);
    }
}
