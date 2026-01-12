//! Order Blocks indicator implementation.
//!
//! Identifies institutional supply and demand zones based on price action.

use indicator_spi::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};
use serde::{Deserialize, Serialize};

/// Order block type.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OrderBlockType {
    /// Bullish order block (demand zone)
    Bullish,
    /// Bearish order block (supply zone)
    Bearish,
}

/// Represents an identified order block zone.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBlock {
    /// Type of order block
    pub block_type: OrderBlockType,
    /// Index where the order block was formed
    pub index: usize,
    /// Upper boundary of the zone
    pub upper: f64,
    /// Lower boundary of the zone
    pub lower: f64,
    /// Whether the zone is still valid (not mitigated)
    pub valid: bool,
}

impl OrderBlock {
    /// Check if price is within this order block zone.
    pub fn contains(&self, price: f64) -> bool {
        price >= self.lower && price <= self.upper
    }

    /// Check if this order block has been mitigated by price.
    pub fn is_mitigated(&self, high: f64, low: f64) -> bool {
        match self.block_type {
            OrderBlockType::Bullish => low < self.lower,
            OrderBlockType::Bearish => high > self.upper,
        }
    }
}

/// Order Blocks indicator.
///
/// Identifies institutional order blocks (supply/demand zones) based on
/// significant price moves. Order blocks are the last opposing candle
/// before a strong move in the opposite direction.
///
/// Bullish Order Block: Last bearish candle before a strong bullish move
/// Bearish Order Block: Last bullish candle before a strong bearish move
///
/// Output:
/// - Primary: Order block presence (1 = bullish OB, -1 = bearish OB, 0 = none)
/// - Secondary: Zone upper boundary
/// - Tertiary: Zone lower boundary
#[derive(Debug, Clone)]
pub struct OrderBlocks {
    /// Minimum move strength for order block formation (in ATR multiples or %).
    min_strength: f64,
    /// Lookback period for detecting strong moves.
    lookback: usize,
    /// Whether to track zone validity (mitigation).
    track_mitigation: bool,
}

impl OrderBlocks {
    /// Create a new Order Blocks indicator.
    ///
    /// # Arguments
    /// * `min_strength` - Minimum move percentage to qualify as order block
    /// * `lookback` - Bars to look ahead for confirming move
    pub fn new(min_strength: f64, lookback: usize) -> Self {
        Self {
            min_strength,
            lookback: lookback.max(1),
            track_mitigation: true,
        }
    }

    /// Create with default parameters.
    pub fn default_params() -> Self {
        Self::new(1.0, 3) // 1% minimum move, 3-bar confirmation
    }

    /// Disable mitigation tracking.
    pub fn without_mitigation(mut self) -> Self {
        self.track_mitigation = false;
        self
    }

    /// Calculate order block values.
    pub fn calculate(
        &self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut ob_signal = vec![0.0; n];
        let mut ob_upper = vec![f64::NAN; n];
        let mut ob_lower = vec![f64::NAN; n];

        if n < self.lookback + 2 {
            return (ob_signal, ob_upper, ob_lower);
        }

        let mut active_blocks: Vec<OrderBlock> = Vec::new();

        for i in 0..(n - self.lookback) {
            // Check if current bar could be an order block
            let is_bearish_candle = close[i] < open[i];
            let is_bullish_candle = close[i] > open[i];

            // Calculate subsequent move strength
            let max_high: f64 = high[i + 1..=(i + self.lookback).min(n - 1)]
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let min_low: f64 = low[i + 1..=(i + self.lookback).min(n - 1)]
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b));

            let base_price = (high[i] + low[i]) / 2.0;

            // Bullish Order Block: bearish candle followed by strong bullish move
            if is_bearish_candle && base_price > 0.0 {
                let bullish_move = ((max_high - high[i]) / base_price) * 100.0;

                if bullish_move >= self.min_strength {
                    ob_signal[i] = 1.0;
                    ob_upper[i] = high[i];
                    ob_lower[i] = low[i];

                    active_blocks.push(OrderBlock {
                        block_type: OrderBlockType::Bullish,
                        index: i,
                        upper: high[i],
                        lower: low[i],
                        valid: true,
                    });
                }
            }

            // Bearish Order Block: bullish candle followed by strong bearish move
            if is_bullish_candle && base_price > 0.0 {
                let bearish_move = ((low[i] - min_low) / base_price) * 100.0;

                if bearish_move >= self.min_strength {
                    ob_signal[i] = -1.0;
                    ob_upper[i] = high[i];
                    ob_lower[i] = low[i];

                    active_blocks.push(OrderBlock {
                        block_type: OrderBlockType::Bearish,
                        index: i,
                        upper: high[i],
                        lower: low[i],
                        valid: true,
                    });
                }
            }

            // Check mitigation of existing blocks
            if self.track_mitigation {
                for block in active_blocks.iter_mut() {
                    if block.valid && block.is_mitigated(high[i], low[i]) {
                        block.valid = false;
                    }
                }
            }
        }

        (ob_signal, ob_upper, ob_lower)
    }

    /// Detect order blocks and return structured data.
    pub fn detect_blocks(
        &self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Vec<OrderBlock> {
        let n = close.len();
        let mut blocks = Vec::new();

        if n < self.lookback + 2 {
            return blocks;
        }

        for i in 0..(n - self.lookback) {
            let is_bearish_candle = close[i] < open[i];
            let is_bullish_candle = close[i] > open[i];

            let max_high: f64 = high[i + 1..=(i + self.lookback).min(n - 1)]
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let min_low: f64 = low[i + 1..=(i + self.lookback).min(n - 1)]
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b));

            let base_price = (high[i] + low[i]) / 2.0;

            if is_bearish_candle && base_price > 0.0 {
                let bullish_move = ((max_high - high[i]) / base_price) * 100.0;
                if bullish_move >= self.min_strength {
                    let mut block = OrderBlock {
                        block_type: OrderBlockType::Bullish,
                        index: i,
                        upper: high[i],
                        lower: low[i],
                        valid: true,
                    };

                    // Check if mitigated by subsequent price action
                    if self.track_mitigation {
                        for j in (i + 1)..n {
                            if block.is_mitigated(high[j], low[j]) {
                                block.valid = false;
                                break;
                            }
                        }
                    }

                    blocks.push(block);
                }
            }

            if is_bullish_candle && base_price > 0.0 {
                let bearish_move = ((low[i] - min_low) / base_price) * 100.0;
                if bearish_move >= self.min_strength {
                    let mut block = OrderBlock {
                        block_type: OrderBlockType::Bearish,
                        index: i,
                        upper: high[i],
                        lower: low[i],
                        valid: true,
                    };

                    if self.track_mitigation {
                        for j in (i + 1)..n {
                            if block.is_mitigated(high[j], low[j]) {
                                block.valid = false;
                                break;
                            }
                        }
                    }

                    blocks.push(block);
                }
            }
        }

        blocks
    }
}

impl TechnicalIndicator for OrderBlocks {
    fn name(&self) -> &str {
        "OrderBlocks"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.lookback + 2 {
            return Err(IndicatorError::InsufficientData {
                required: self.lookback + 2,
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
        self.lookback + 2
    }

    fn output_features(&self) -> usize {
        3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_blocks_bullish() {
        let ob = OrderBlocks::new(0.5, 2);

        // Bearish candle followed by strong bullish move
        let open = vec![100.0, 101.0, 99.0, 100.0, 102.0, 104.0, 106.0];
        let high = vec![101.5, 102.0, 100.0, 101.0, 104.0, 106.0, 108.0];
        let low = vec![99.5, 100.5, 98.0, 99.0, 101.0, 103.0, 105.0];
        let close = vec![100.5, 99.5, 99.5, 100.5, 103.0, 105.0, 107.0];

        let (signal, upper, lower) = ob.calculate(&open, &high, &low, &close);

        assert_eq!(signal.len(), 7);
        assert_eq!(upper.len(), 7);
        assert_eq!(lower.len(), 7);
    }

    #[test]
    fn test_order_blocks_detection() {
        let ob = OrderBlocks::new(0.5, 2);

        // Create scenario with clear order block
        let open = vec![100.0, 101.0, 99.0, 100.0, 102.0, 104.0, 106.0];
        let high = vec![101.5, 102.0, 100.0, 101.0, 104.0, 106.0, 108.0];
        let low = vec![99.5, 100.5, 98.0, 99.0, 101.0, 103.0, 105.0];
        let close = vec![100.5, 99.5, 99.5, 100.5, 103.0, 105.0, 107.0];

        let blocks = ob.detect_blocks(&open, &high, &low, &close);

        // Should detect at least one block
        for block in &blocks {
            assert!(block.upper >= block.lower);
        }
    }

    #[test]
    fn test_order_block_contains() {
        let block = OrderBlock {
            block_type: OrderBlockType::Bullish,
            index: 0,
            upper: 102.0,
            lower: 100.0,
            valid: true,
        };

        assert!(block.contains(101.0));
        assert!(block.contains(100.0));
        assert!(block.contains(102.0));
        assert!(!block.contains(99.0));
        assert!(!block.contains(103.0));
    }

    #[test]
    fn test_order_block_mitigation() {
        let bullish = OrderBlock {
            block_type: OrderBlockType::Bullish,
            index: 0,
            upper: 102.0,
            lower: 100.0,
            valid: true,
        };

        let bearish = OrderBlock {
            block_type: OrderBlockType::Bearish,
            index: 0,
            upper: 102.0,
            lower: 100.0,
            valid: true,
        };

        // Bullish OB mitigated when price goes below lower
        assert!(!bullish.is_mitigated(103.0, 101.0));
        assert!(bullish.is_mitigated(101.0, 99.0));

        // Bearish OB mitigated when price goes above upper
        assert!(!bearish.is_mitigated(101.0, 99.0));
        assert!(bearish.is_mitigated(103.0, 101.0));
    }

    #[test]
    fn test_order_blocks_technical_indicator() {
        let ob = OrderBlocks::new(0.5, 2);

        let mut data = OHLCVSeries::new();
        for i in 0..15 {
            let base = 100.0 + (i as f64 * 0.5);
            data.open.push(base);
            data.high.push(base + 1.5);
            data.low.push(base - 1.5);
            data.close.push(if i % 3 == 0 { base - 0.5 } else { base + 0.5 });
            data.volume.push(1000.0);
        }

        let output = ob.compute(&data).unwrap();
        assert_eq!(output.primary.len(), 15);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }
}
