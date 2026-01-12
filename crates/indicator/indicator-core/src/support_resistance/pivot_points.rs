//! Pivot Points implementation.

use indicator_api::PivotType;
use serde::{Deserialize, Serialize};

/// Pivot Points result.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PivotPointsResult {
    pub pivot: f64,
    pub r1: f64,
    pub r2: f64,
    pub r3: f64,
    pub s1: f64,
    pub s2: f64,
    pub s3: f64,
}

impl PivotPointsResult {
    pub fn null() -> Self {
        Self {
            pivot: f64::NAN,
            r1: f64::NAN,
            r2: f64::NAN,
            r3: f64::NAN,
            s1: f64::NAN,
            s2: f64::NAN,
            s3: f64::NAN,
        }
    }
}

/// Pivot Points calculator.
///
/// Calculates support and resistance levels based on previous period's HLC.
#[derive(Debug, Clone, Default)]
pub struct PivotPoints {
    pivot_type: PivotType,
}

impl PivotPoints {
    pub fn new(pivot_type: PivotType) -> Self {
        Self { pivot_type }
    }

    pub fn standard() -> Self {
        Self { pivot_type: PivotType::Standard }
    }

    pub fn fibonacci() -> Self {
        Self { pivot_type: PivotType::Fibonacci }
    }

    pub fn woodie() -> Self {
        Self { pivot_type: PivotType::Woodie }
    }

    pub fn camarilla() -> Self {
        Self { pivot_type: PivotType::Camarilla }
    }

    /// Calculate pivot points from high, low, close of previous period.
    pub fn calculate(&self, high: f64, low: f64, close: f64) -> PivotPointsResult {
        match self.pivot_type {
            PivotType::Standard => self.calculate_standard(high, low, close),
            PivotType::Fibonacci => self.calculate_fibonacci(high, low, close),
            PivotType::Woodie => self.calculate_woodie(high, low, close),
            PivotType::Camarilla => self.calculate_camarilla(high, low, close),
        }
    }

    /// Standard pivot points.
    fn calculate_standard(&self, high: f64, low: f64, close: f64) -> PivotPointsResult {
        let pivot = (high + low + close) / 3.0;
        let range = high - low;

        PivotPointsResult {
            pivot,
            r1: 2.0 * pivot - low,
            r2: pivot + range,
            r3: high + 2.0 * (pivot - low),
            s1: 2.0 * pivot - high,
            s2: pivot - range,
            s3: low - 2.0 * (high - pivot),
        }
    }

    /// Fibonacci pivot points.
    fn calculate_fibonacci(&self, high: f64, low: f64, close: f64) -> PivotPointsResult {
        let pivot = (high + low + close) / 3.0;
        let range = high - low;

        PivotPointsResult {
            pivot,
            r1: pivot + 0.382 * range,
            r2: pivot + 0.618 * range,
            r3: pivot + range,
            s1: pivot - 0.382 * range,
            s2: pivot - 0.618 * range,
            s3: pivot - range,
        }
    }

    /// Woodie pivot points.
    fn calculate_woodie(&self, high: f64, low: f64, close: f64) -> PivotPointsResult {
        let pivot = (high + low + 2.0 * close) / 4.0;

        PivotPointsResult {
            pivot,
            r1: 2.0 * pivot - low,
            r2: pivot + high - low,
            r3: high + 2.0 * (pivot - low),
            s1: 2.0 * pivot - high,
            s2: pivot - high + low,
            s3: low - 2.0 * (high - pivot),
        }
    }

    /// Camarilla pivot points.
    fn calculate_camarilla(&self, high: f64, low: f64, close: f64) -> PivotPointsResult {
        let range = high - low;

        PivotPointsResult {
            pivot: (high + low + close) / 3.0,
            r1: close + range * 1.1 / 12.0,
            r2: close + range * 1.1 / 6.0,
            r3: close + range * 1.1 / 4.0,
            s1: close - range * 1.1 / 12.0,
            s2: close - range * 1.1 / 6.0,
            s3: close - range * 1.1 / 4.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_pivot() {
        let pp = PivotPoints::standard();
        let result = pp.calculate(105.0, 95.0, 100.0);

        assert!((result.pivot - 100.0).abs() < 1e-10);
        assert!(result.r1 > result.pivot);
        assert!(result.r2 > result.r1);
        assert!(result.s1 < result.pivot);
        assert!(result.s2 < result.s1);
    }

    #[test]
    fn test_fibonacci_pivot() {
        let pp = PivotPoints::fibonacci();
        let result = pp.calculate(105.0, 95.0, 100.0);

        // Fibonacci levels should be proportional to range
        let range = 105.0 - 95.0;
        assert!((result.r1 - (result.pivot + 0.382 * range)).abs() < 1e-10);
    }

    #[test]
    fn test_camarilla_pivot() {
        let pp = PivotPoints::camarilla();
        let result = pp.calculate(105.0, 95.0, 100.0);

        // Camarilla levels are tighter than standard
        assert!(result.r1 < result.r2);
        assert!(result.r2 < result.r3);
    }
}
