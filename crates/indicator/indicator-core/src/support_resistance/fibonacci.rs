//! Fibonacci Retracement implementation.

use serde::{Deserialize, Serialize};

/// Fibonacci Retracement levels.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FibonacciLevels {
    pub level_0: f64,    // 0%
    pub level_236: f64,  // 23.6%
    pub level_382: f64,  // 38.2%
    pub level_500: f64,  // 50%
    pub level_618: f64,  // 61.8%
    pub level_786: f64,  // 78.6%
    pub level_1000: f64, // 100%
}

/// Fibonacci Retracement calculator.
///
/// Calculates key Fibonacci retracement levels between two price points.
#[derive(Debug, Clone, Default)]
pub struct Fibonacci;

impl Fibonacci {
    pub fn new() -> Self {
        Self
    }

    /// Calculate Fibonacci retracement levels.
    ///
    /// # Arguments
    /// * `price_high` - The high price point
    /// * `price_low` - The low price point
    /// * `is_uptrend` - If true, retracement from high back to low; if false, from low to high
    pub fn calculate(&self, price_high: f64, price_low: f64, is_uptrend: bool) -> FibonacciLevels {
        let range = price_high - price_low;

        if is_uptrend {
            // Retracement from high back towards low
            FibonacciLevels {
                level_0: price_high,
                level_236: price_high - range * 0.236,
                level_382: price_high - range * 0.382,
                level_500: price_high - range * 0.500,
                level_618: price_high - range * 0.618,
                level_786: price_high - range * 0.786,
                level_1000: price_low,
            }
        } else {
            // Retracement from low back towards high
            FibonacciLevels {
                level_0: price_low,
                level_236: price_low + range * 0.236,
                level_382: price_low + range * 0.382,
                level_500: price_low + range * 0.500,
                level_618: price_low + range * 0.618,
                level_786: price_low + range * 0.786,
                level_1000: price_high,
            }
        }
    }

    /// Calculate extension levels beyond the range.
    pub fn extensions(&self, price_high: f64, price_low: f64, is_uptrend: bool) -> Vec<f64> {
        let range = price_high - price_low;
        let extensions = [1.272, 1.414, 1.618, 2.0, 2.618];

        if is_uptrend {
            extensions.iter().map(|&e| price_low + range * e).collect()
        } else {
            extensions.iter().map(|&e| price_high - range * e).collect()
        }
    }

    /// Find which Fibonacci level is closest to a given price.
    pub fn nearest_level(&self, price: f64, levels: &FibonacciLevels) -> (f64, &'static str) {
        let all_levels = [
            (levels.level_0, "0%"),
            (levels.level_236, "23.6%"),
            (levels.level_382, "38.2%"),
            (levels.level_500, "50%"),
            (levels.level_618, "61.8%"),
            (levels.level_786, "78.6%"),
            (levels.level_1000, "100%"),
        ];

        all_levels
            .iter()
            .min_by(|a, b| {
                let diff_a = (a.0 - price).abs();
                let diff_b = (b.0 - price).abs();
                diff_a.partial_cmp(&diff_b).unwrap()
            })
            .map(|&(level, name)| (level, name))
            .unwrap_or((f64::NAN, "unknown"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fibonacci_uptrend() {
        let fib = Fibonacci::new();
        let levels = fib.calculate(100.0, 80.0, true);

        // In uptrend, level_0 is the high
        assert!((levels.level_0 - 100.0).abs() < 1e-10);
        assert!((levels.level_1000 - 80.0).abs() < 1e-10);

        // 50% retracement
        assert!((levels.level_500 - 90.0).abs() < 1e-10);

        // 61.8% is golden ratio
        assert!((levels.level_618 - (100.0 - 20.0 * 0.618)).abs() < 1e-10);
    }

    #[test]
    fn test_fibonacci_downtrend() {
        let fib = Fibonacci::new();
        let levels = fib.calculate(100.0, 80.0, false);

        // In downtrend, level_0 is the low
        assert!((levels.level_0 - 80.0).abs() < 1e-10);
        assert!((levels.level_1000 - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_extensions() {
        let fib = Fibonacci::new();
        let ext = fib.extensions(100.0, 80.0, true);

        // 161.8% extension
        assert!(ext.iter().any(|&e| (e - (80.0 + 20.0 * 1.618)).abs() < 1e-10));
    }
}
