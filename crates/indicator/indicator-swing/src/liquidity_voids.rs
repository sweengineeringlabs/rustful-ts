//! Liquidity Voids indicator implementation.
//!
//! Identifies unfilled price gaps and liquidity vacuums in price action.

use indicator_spi::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};
use serde::{Deserialize, Serialize};

/// Liquidity void type.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum LiquidityVoidType {
    /// Void above current price (potential resistance)
    Above,
    /// Void below current price (potential support)
    Below,
}

/// Represents an identified liquidity void.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidityVoid {
    /// Type of liquidity void
    pub void_type: LiquidityVoidType,
    /// Starting bar index
    pub start_index: usize,
    /// Ending bar index (where gap was created)
    pub end_index: usize,
    /// Upper boundary of the void
    pub upper: f64,
    /// Lower boundary of the void
    pub lower: f64,
    /// Size of the void as percentage
    pub size_percent: f64,
    /// Whether the void has been filled
    pub filled: bool,
}

impl LiquidityVoid {
    /// Calculate the absolute size of the void.
    pub fn size(&self) -> f64 {
        self.upper - self.lower
    }

    /// Check if price is within the void.
    pub fn contains(&self, price: f64) -> bool {
        price >= self.lower && price <= self.upper
    }

    /// Check if the void has been filled by price action.
    pub fn check_fill(&self, high: f64, low: f64) -> bool {
        match self.void_type {
            LiquidityVoidType::Above => high >= self.upper,
            LiquidityVoidType::Below => low <= self.lower,
        }
    }
}

/// Liquidity Voids indicator.
///
/// Identifies areas where price moved rapidly, creating liquidity voids
/// that often act as magnets for future price action. Unlike FVGs which
/// are specific three-candle patterns, liquidity voids are larger areas
/// where price traveled without significant consolidation.
///
/// Liquidity voids are created by:
/// - Large impulse moves
/// - Gap opens
/// - Fast momentum candles
///
/// Price tends to return to fill these voids as market makers seek liquidity.
///
/// Output:
/// - Primary: Void signal (1 = void above, -1 = void below, 0 = none)
/// - Secondary: Upper boundary
/// - Tertiary: Lower boundary
#[derive(Debug, Clone)]
pub struct LiquidityVoids {
    /// Minimum void size as percentage of price.
    min_void_percent: f64,
    /// Number of bars for void detection window.
    lookback: usize,
    /// Minimum body ratio for impulse candle detection.
    min_body_ratio: f64,
}

impl LiquidityVoids {
    /// Create a new Liquidity Voids indicator.
    ///
    /// # Arguments
    /// * `min_void_percent` - Minimum void size as percentage (e.g., 0.5 = 0.5%)
    /// * `lookback` - Number of bars to analyze for void detection
    pub fn new(min_void_percent: f64, lookback: usize) -> Self {
        Self {
            min_void_percent,
            lookback: lookback.max(2),
            min_body_ratio: 0.6,
        }
    }

    /// Create with default parameters.
    pub fn default_params() -> Self {
        Self::new(0.3, 5)
    }

    /// Set minimum body ratio for impulse candle.
    pub fn with_body_ratio(mut self, ratio: f64) -> Self {
        self.min_body_ratio = ratio.clamp(0.0, 1.0);
        self
    }

    /// Check if a candle is an impulse candle.
    fn is_impulse_candle(&self, open: f64, high: f64, low: f64, close: f64) -> bool {
        let range = high - low;
        if range <= 0.0 {
            return false;
        }

        let body = (close - open).abs();
        let body_ratio = body / range;

        body_ratio >= self.min_body_ratio
    }

    /// Calculate liquidity void values.
    pub fn calculate(
        &self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut void_signal = vec![0.0; n];
        let mut void_upper = vec![f64::NAN; n];
        let mut void_lower = vec![f64::NAN; n];

        if n < self.lookback {
            return (void_signal, void_upper, void_lower);
        }

        for i in self.lookback..n {
            // Check for impulse candle
            if !self.is_impulse_candle(open[i], high[i], low[i], close[i]) {
                continue;
            }

            let is_bullish = close[i] > open[i];
            let current_price = close[i];

            if is_bullish {
                // Bullish impulse - look for void below
                let window_high = high[i - self.lookback..i]
                    .iter()
                    .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

                // Void exists if there's a gap between window high and current candle's body
                if low[i] > window_high {
                    let void_size_pct = ((low[i] - window_high) / current_price) * 100.0;

                    if void_size_pct >= self.min_void_percent {
                        void_signal[i] = -1.0; // Void below
                        void_upper[i] = low[i];
                        void_lower[i] = window_high;
                    }
                }
            } else {
                // Bearish impulse - look for void above
                let window_low = low[i - self.lookback..i]
                    .iter()
                    .fold(f64::INFINITY, |a, &b| a.min(b));

                // Void exists if there's a gap between window low and current candle's body
                if high[i] < window_low {
                    let void_size_pct = ((window_low - high[i]) / current_price) * 100.0;

                    if void_size_pct >= self.min_void_percent {
                        void_signal[i] = 1.0; // Void above
                        void_upper[i] = window_low;
                        void_lower[i] = high[i];
                    }
                }
            }
        }

        (void_signal, void_upper, void_lower)
    }

    /// Detect liquidity voids and return structured data.
    pub fn detect_voids(
        &self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Vec<LiquidityVoid> {
        let n = close.len();
        let mut voids = Vec::new();

        if n < self.lookback {
            return voids;
        }

        for i in self.lookback..n {
            if !self.is_impulse_candle(open[i], high[i], low[i], close[i]) {
                continue;
            }

            let is_bullish = close[i] > open[i];
            let current_price = close[i];

            if is_bullish {
                let window_high = high[i - self.lookback..i]
                    .iter()
                    .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

                if low[i] > window_high {
                    let void_size_pct = ((low[i] - window_high) / current_price) * 100.0;

                    if void_size_pct >= self.min_void_percent {
                        let mut lv = LiquidityVoid {
                            void_type: LiquidityVoidType::Below,
                            start_index: i - self.lookback,
                            end_index: i,
                            upper: low[i],
                            lower: window_high,
                            size_percent: void_size_pct,
                            filled: false,
                        };

                        // Check if filled by subsequent price action
                        for j in (i + 1)..n {
                            if lv.check_fill(high[j], low[j]) {
                                lv.filled = true;
                                break;
                            }
                        }

                        voids.push(lv);
                    }
                }
            } else {
                let window_low = low[i - self.lookback..i]
                    .iter()
                    .fold(f64::INFINITY, |a, &b| a.min(b));

                if high[i] < window_low {
                    let void_size_pct = ((window_low - high[i]) / current_price) * 100.0;

                    if void_size_pct >= self.min_void_percent {
                        let mut lv = LiquidityVoid {
                            void_type: LiquidityVoidType::Above,
                            start_index: i - self.lookback,
                            end_index: i,
                            upper: window_low,
                            lower: high[i],
                            size_percent: void_size_pct,
                            filled: false,
                        };

                        for j in (i + 1)..n {
                            if lv.check_fill(high[j], low[j]) {
                                lv.filled = true;
                                break;
                            }
                        }

                        voids.push(lv);
                    }
                }
            }
        }

        voids
    }

    /// Get only unfilled voids.
    pub fn unfilled_voids(
        &self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Vec<LiquidityVoid> {
        self.detect_voids(open, high, low, close)
            .into_iter()
            .filter(|v| !v.filled)
            .collect()
    }
}

impl TechnicalIndicator for LiquidityVoids {
    fn name(&self) -> &str {
        "LiquidityVoids"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.lookback {
            return Err(IndicatorError::InsufficientData {
                required: self.lookback,
                got: data.close.len(),
            });
        }

        let (signal, upper, lower) = self.calculate(
            &data.open,
            &data.high,
            &data.low,
            &data.close,
        );
        Ok(IndicatorOutput::triple(signal, upper, lower))
    }

    fn min_periods(&self) -> usize {
        self.lookback
    }

    fn output_features(&self) -> usize {
        3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_liquidity_void_basic() {
        let lv = LiquidityVoids::new(0.1, 3);

        // Create data with potential void
        let open = vec![100.0, 100.5, 101.0, 101.5, 105.0, 106.0];
        let high = vec![101.0, 101.5, 102.0, 102.5, 108.0, 107.0];
        let low = vec![99.5, 100.0, 100.5, 101.0, 104.0, 105.0];
        let close = vec![100.5, 101.0, 101.5, 102.0, 107.0, 106.5];

        let (signal, upper, lower) = lv.calculate(&open, &high, &low, &close);

        assert_eq!(signal.len(), 6);
        assert_eq!(upper.len(), 6);
        assert_eq!(lower.len(), 6);
    }

    #[test]
    fn test_liquidity_void_struct() {
        let void = LiquidityVoid {
            void_type: LiquidityVoidType::Below,
            start_index: 0,
            end_index: 5,
            upper: 105.0,
            lower: 100.0,
            size_percent: 5.0,
            filled: false,
        };

        assert_eq!(void.size(), 5.0);
        assert!(void.contains(102.0));
        assert!(!void.contains(99.0));
    }

    #[test]
    fn test_liquidity_void_fill_check() {
        let void_above = LiquidityVoid {
            void_type: LiquidityVoidType::Above,
            start_index: 0,
            end_index: 5,
            upper: 105.0,
            lower: 100.0,
            size_percent: 5.0,
            filled: false,
        };

        let void_below = LiquidityVoid {
            void_type: LiquidityVoidType::Below,
            start_index: 0,
            end_index: 5,
            upper: 105.0,
            lower: 100.0,
            size_percent: 5.0,
            filled: false,
        };

        // Above void filled when price reaches upper
        assert!(!void_above.check_fill(104.0, 102.0));
        assert!(void_above.check_fill(106.0, 104.0));

        // Below void filled when price reaches lower
        assert!(!void_below.check_fill(104.0, 101.0));
        assert!(void_below.check_fill(102.0, 99.0));
    }

    #[test]
    fn test_impulse_candle_detection() {
        let lv = LiquidityVoids::new(0.1, 3);

        // Strong body candle (high body ratio)
        assert!(lv.is_impulse_candle(100.0, 102.0, 99.5, 101.8));

        // Doji-like candle (low body ratio)
        assert!(!lv.is_impulse_candle(100.0, 102.0, 98.0, 100.1));
    }

    #[test]
    fn test_liquidity_voids_technical_indicator() {
        let lv = LiquidityVoids::new(0.1, 3);

        let mut data = OHLCVSeries::new();
        for i in 0..15 {
            let base = 100.0 + i as f64 * 0.5;
            data.open.push(base);
            data.high.push(base + 1.0);
            data.low.push(base - 0.5);
            data.close.push(base + 0.8);
            data.volume.push(1000.0);
        }

        let output = lv.compute(&data).unwrap();
        assert_eq!(output.primary.len(), 15);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }
}
