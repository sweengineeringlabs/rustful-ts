//! Trading signal model.

use serde::{Deserialize, Serialize};

/// Trading signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Signal {
    Buy,
    Sell,
    Hold,
}

impl Signal {
    /// Convert to numeric signal: Buy = 1, Sell = -1, Hold = 0.
    pub fn to_numeric(&self) -> f64 {
        match self {
            Signal::Buy => 1.0,
            Signal::Sell => -1.0,
            Signal::Hold => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_buy_to_numeric() {
        assert_eq!(Signal::Buy.to_numeric(), 1.0);
    }

    #[test]
    fn test_signal_sell_to_numeric() {
        assert_eq!(Signal::Sell.to_numeric(), -1.0);
    }

    #[test]
    fn test_signal_hold_to_numeric() {
        assert_eq!(Signal::Hold.to_numeric(), 0.0);
    }

    #[test]
    fn test_signal_clone() {
        let signal = Signal::Buy;
        let cloned = signal.clone();
        assert_eq!(signal, cloned);
    }

    #[test]
    fn test_signal_copy() {
        let signal = Signal::Sell;
        let copied = signal;
        assert_eq!(signal, copied);
    }

    #[test]
    fn test_signal_equality() {
        assert_eq!(Signal::Buy, Signal::Buy);
        assert_eq!(Signal::Sell, Signal::Sell);
        assert_eq!(Signal::Hold, Signal::Hold);
    }

    #[test]
    fn test_signal_inequality() {
        assert_ne!(Signal::Buy, Signal::Sell);
        assert_ne!(Signal::Buy, Signal::Hold);
        assert_ne!(Signal::Sell, Signal::Hold);
    }

    #[test]
    fn test_signal_debug() {
        assert_eq!(format!("{:?}", Signal::Buy), "Buy");
        assert_eq!(format!("{:?}", Signal::Sell), "Sell");
        assert_eq!(format!("{:?}", Signal::Hold), "Hold");
    }

    #[test]
    fn test_signal_serialize() {
        let buy_json = serde_json::to_string(&Signal::Buy).unwrap();
        let sell_json = serde_json::to_string(&Signal::Sell).unwrap();
        let hold_json = serde_json::to_string(&Signal::Hold).unwrap();

        assert_eq!(buy_json, "\"Buy\"");
        assert_eq!(sell_json, "\"Sell\"");
        assert_eq!(hold_json, "\"Hold\"");
    }

    #[test]
    fn test_signal_deserialize() {
        let buy: Signal = serde_json::from_str("\"Buy\"").unwrap();
        let sell: Signal = serde_json::from_str("\"Sell\"").unwrap();
        let hold: Signal = serde_json::from_str("\"Hold\"").unwrap();

        assert_eq!(buy, Signal::Buy);
        assert_eq!(sell, Signal::Sell);
        assert_eq!(hold, Signal::Hold);
    }

    #[test]
    fn test_signal_roundtrip_serialization() {
        for signal in [Signal::Buy, Signal::Sell, Signal::Hold] {
            let json = serde_json::to_string(&signal).unwrap();
            let deserialized: Signal = serde_json::from_str(&json).unwrap();
            assert_eq!(signal, deserialized);
        }
    }

    #[test]
    fn test_signal_to_numeric_vec() {
        let signals = vec![Signal::Buy, Signal::Hold, Signal::Sell, Signal::Buy];
        let numeric: Vec<f64> = signals.iter().map(|s| s.to_numeric()).collect();
        assert_eq!(numeric, vec![1.0, 0.0, -1.0, 1.0]);
    }

    #[test]
    fn test_signal_numeric_sum() {
        let signals = vec![Signal::Buy, Signal::Buy, Signal::Sell];
        let sum: f64 = signals.iter().map(|s| s.to_numeric()).sum();
        assert_eq!(sum, 1.0); // 1 + 1 + (-1) = 1
    }

    #[test]
    fn test_signal_in_hashset() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Signal::Buy);
        set.insert(Signal::Sell);
        set.insert(Signal::Hold);
        set.insert(Signal::Buy); // duplicate

        assert_eq!(set.len(), 3);
        assert!(set.contains(&Signal::Buy));
        assert!(set.contains(&Signal::Sell));
        assert!(set.contains(&Signal::Hold));
    }
}
