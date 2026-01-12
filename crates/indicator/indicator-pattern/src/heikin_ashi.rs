//! Heikin Ashi Candlestick Indicator
//!
//! Transforms OHLCV data into Heikin Ashi candles for smoother trend visualization.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries, OHLCV,
};

/// Heikin Ashi output containing transformed OHLCV data.
#[derive(Debug, Clone)]
pub struct HeikinAshiOutput {
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
}

impl HeikinAshiOutput {
    /// Convert to OHLCVSeries for further analysis.
    pub fn to_series(&self) -> OHLCVSeries {
        OHLCVSeries {
            open: self.open.clone(),
            high: self.high.clone(),
            low: self.low.clone(),
            close: self.close.clone(),
            volume: vec![0.0; self.close.len()],
        }
    }

    /// Get a specific candle at index.
    pub fn candle(&self, idx: usize) -> Option<OHLCV> {
        if idx >= self.close.len() {
            return None;
        }
        Some(OHLCV::new(
            self.open[idx],
            self.high[idx],
            self.low[idx],
            self.close[idx],
            0.0,
        ))
    }
}

/// Heikin Ashi candlestick indicator.
///
/// Transforms regular OHLCV candles into Heikin Ashi candles which provide
/// a smoother representation of price action and clearer trend identification.
///
/// Heikin Ashi formulas:
/// - Close = (Open + High + Low + Close) / 4
/// - Open = (Previous HA Open + Previous HA Close) / 2
/// - High = max(High, HA Open, HA Close)
/// - Low = min(Low, HA Open, HA Close)
#[derive(Debug, Clone, Default)]
pub struct HeikinAshi;

impl HeikinAshi {
    /// Create a new Heikin Ashi indicator.
    pub fn new() -> Self {
        Self
    }

    /// Calculate Heikin Ashi candles from OHLCV data.
    pub fn calculate(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> HeikinAshiOutput {
        let n = close.len();
        if n == 0 {
            return HeikinAshiOutput {
                open: vec![],
                high: vec![],
                low: vec![],
                close: vec![],
            };
        }

        let mut ha_open = Vec::with_capacity(n);
        let mut ha_high = Vec::with_capacity(n);
        let mut ha_low = Vec::with_capacity(n);
        let mut ha_close = Vec::with_capacity(n);

        // First candle
        let first_close = (open[0] + high[0] + low[0] + close[0]) / 4.0;
        let first_open = (open[0] + close[0]) / 2.0;
        let first_high = high[0].max(first_open).max(first_close);
        let first_low = low[0].min(first_open).min(first_close);

        ha_open.push(first_open);
        ha_high.push(first_high);
        ha_low.push(first_low);
        ha_close.push(first_close);

        // Subsequent candles
        for i in 1..n {
            let ha_c = (open[i] + high[i] + low[i] + close[i]) / 4.0;
            let ha_o = (ha_open[i - 1] + ha_close[i - 1]) / 2.0;
            let ha_h = high[i].max(ha_o).max(ha_c);
            let ha_l = low[i].min(ha_o).min(ha_c);

            ha_open.push(ha_o);
            ha_high.push(ha_h);
            ha_low.push(ha_l);
            ha_close.push(ha_c);
        }

        HeikinAshiOutput {
            open: ha_open,
            high: ha_high,
            low: ha_low,
            close: ha_close,
        }
    }

    /// Determine if a Heikin Ashi candle is bullish (no lower wick).
    fn is_bullish_candle(open: f64, _high: f64, low: f64, close: f64) -> bool {
        close > open && (low - open.min(close)).abs() < f64::EPSILON
    }

    /// Determine if a Heikin Ashi candle is bearish (no upper wick).
    fn is_bearish_candle(open: f64, high: f64, _low: f64, close: f64) -> bool {
        close < open && (high - open.max(close)).abs() < f64::EPSILON
    }
}

impl TechnicalIndicator for HeikinAshi {
    fn name(&self) -> &str {
        "HeikinAshi"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.is_empty() {
            return Err(IndicatorError::InsufficientData {
                required: 1,
                got: 0,
            });
        }

        let ha = self.calculate(&data.open, &data.high, &data.low, &data.close);
        // Return close as primary, open as secondary for trend analysis
        Ok(IndicatorOutput::dual(ha.close, ha.open))
    }

    fn min_periods(&self) -> usize {
        1
    }

    fn output_features(&self) -> usize {
        4 // open, high, low, close
    }
}

impl SignalIndicator for HeikinAshi {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        if data.close.is_empty() {
            return Ok(IndicatorSignal::Neutral);
        }

        let ha = self.calculate(&data.open, &data.high, &data.low, &data.close);
        let n = ha.close.len();

        if n < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let last_open = ha.open[n - 1];
        let last_high = ha.high[n - 1];
        let last_low = ha.low[n - 1];
        let last_close = ha.close[n - 1];

        // Strong bullish: no lower wick
        if Self::is_bullish_candle(last_open, last_high, last_low, last_close) {
            return Ok(IndicatorSignal::Bullish);
        }

        // Strong bearish: no upper wick
        if Self::is_bearish_candle(last_open, last_high, last_low, last_close) {
            return Ok(IndicatorSignal::Bearish);
        }

        // Weak signal based on direction
        if last_close > last_open {
            Ok(IndicatorSignal::Bullish)
        } else if last_close < last_open {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        if data.close.is_empty() {
            return Ok(vec![]);
        }

        let ha = self.calculate(&data.open, &data.high, &data.low, &data.close);
        let n = ha.close.len();

        let signals = (0..n).map(|i| {
            let o = ha.open[i];
            let h = ha.high[i];
            let l = ha.low[i];
            let c = ha.close[i];

            if Self::is_bullish_candle(o, h, l, c) {
                IndicatorSignal::Bullish
            } else if Self::is_bearish_candle(o, h, l, c) {
                IndicatorSignal::Bearish
            } else if c > o {
                IndicatorSignal::Bullish
            } else if c < o {
                IndicatorSignal::Bearish
            } else {
                IndicatorSignal::Neutral
            }
        }).collect();

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heikin_ashi_basic() {
        let ha = HeikinAshi::new();
        let open = vec![100.0, 102.0, 104.0, 103.0, 105.0];
        let high = vec![103.0, 105.0, 107.0, 106.0, 108.0];
        let low = vec![99.0, 101.0, 103.0, 101.0, 104.0];
        let close = vec![102.0, 104.0, 105.0, 104.0, 107.0];

        let result = ha.calculate(&open, &high, &low, &close);
        assert_eq!(result.close.len(), 5);
        assert_eq!(result.open.len(), 5);
        assert_eq!(result.high.len(), 5);
        assert_eq!(result.low.len(), 5);
    }

    #[test]
    fn test_heikin_ashi_trend() {
        let ha = HeikinAshi::new();
        // Uptrend data
        let open = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let high = vec![102.0, 103.0, 104.0, 105.0, 106.0];
        let low = vec![99.0, 100.0, 101.0, 102.0, 103.0];
        let close = vec![101.0, 102.0, 103.0, 104.0, 105.0];

        let result = ha.calculate(&open, &high, &low, &close);

        // In uptrend, HA close should be above HA open
        for i in 1..result.close.len() {
            assert!(result.close[i] >= result.open[i]);
        }
    }
}
