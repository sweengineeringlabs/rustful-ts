//! Choppiness Index implementation.
//!
//! Measures whether the market is choppy (trading sideways) or trending.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Choppiness Index.
///
/// The Choppiness Index determines if the market is choppy (trading sideways)
/// or trending. Values near 100 indicate choppiness, while values near 0
/// indicate trending conditions.
///
/// Formula:
/// CHOP = 100 * log10(sum(ATR, n) / (highest(high, n) - lowest(low, n))) / log10(n)
///
/// Where:
/// - ATR = True Range for each bar
/// - n = lookback period
#[derive(Debug, Clone)]
pub struct ChoppinessIndex {
    /// Lookback period (commonly 14).
    period: usize,
    /// Upper threshold for choppy market (typically 61.8).
    choppy_threshold: f64,
    /// Lower threshold for trending market (typically 38.2).
    trending_threshold: f64,
}

impl ChoppinessIndex {
    /// Create a new Choppiness Index indicator.
    ///
    /// # Arguments
    /// * `period` - Lookback period (commonly 14)
    pub fn new(period: usize) -> Self {
        Self {
            period,
            choppy_threshold: 61.8,
            trending_threshold: 38.2,
        }
    }

    /// Create with custom thresholds.
    pub fn with_thresholds(period: usize, choppy: f64, trending: f64) -> Self {
        Self {
            period,
            choppy_threshold: choppy,
            trending_threshold: trending,
        }
    }

    /// Calculate True Range for each bar.
    fn true_range(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = high.len();
        if n == 0 {
            return vec![];
        }

        let mut tr = Vec::with_capacity(n);
        tr.push(high[0] - low[0]);

        for i in 1..n {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            tr.push(hl.max(hc).max(lc));
        }

        tr
    }

    /// Calculate Choppiness Index values.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = high.len();
        if n < self.period || self.period < 2 {
            return vec![f64::NAN; n];
        }

        let tr = Self::true_range(high, low, close);
        let log_period = (self.period as f64).log10();

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;

            // Sum of True Range over period
            let tr_sum: f64 = tr[start..=i].iter().sum();

            // Highest high and lowest low over period
            let highest_high = high[start..=i].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let lowest_low = low[start..=i].iter().fold(f64::INFINITY, |a, &b| a.min(b));

            let range = highest_high - lowest_low;

            if range <= 0.0 || tr_sum <= 0.0 {
                result.push(f64::NAN);
                continue;
            }

            // Choppiness Index formula
            let chop = 100.0 * (tr_sum / range).log10() / log_period;

            // Clamp to valid range [0, 100]
            let chop_clamped = chop.clamp(0.0, 100.0);

            result.push(chop_clamped);
        }

        result
    }
}

impl TechnicalIndicator for ChoppinessIndex {
    fn name(&self) -> &str {
        "ChoppinessIndex"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.high.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.high.len(),
            });
        }

        let values = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.period
    }
}

impl SignalIndicator for ChoppinessIndex {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.high, &data.low, &data.close);
        let last = values.last().copied().unwrap_or(f64::NAN);

        if last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // High choppiness = sideways market (neutral/wait)
        // Low choppiness = trending market (can trade)
        if last >= self.choppy_threshold {
            Ok(IndicatorSignal::Neutral) // Choppy - stay out
        } else if last <= self.trending_threshold {
            Ok(IndicatorSignal::Bullish) // Trending - can enter
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.high, &data.low, &data.close);

        let signals = values.iter().map(|&chop| {
            if chop.is_nan() {
                IndicatorSignal::Neutral
            } else if chop >= self.choppy_threshold {
                IndicatorSignal::Neutral
            } else if chop <= self.trending_threshold {
                IndicatorSignal::Bullish
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
    fn test_choppiness_index() {
        let chop = ChoppinessIndex::new(14);

        // Generate sample OHLC data
        let high: Vec<f64> = (0..50)
            .map(|i| 102.0 + (i as f64 * 0.2).sin() * 3.0)
            .collect();
        let low: Vec<f64> = (0..50)
            .map(|i| 98.0 + (i as f64 * 0.2).sin() * 3.0)
            .collect();
        let close: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64 * 0.2).sin() * 3.0)
            .collect();

        let result = chop.calculate(&high, &low, &close);

        assert_eq!(result.len(), 50);

        // First 13 values should be NaN
        for i in 0..13 {
            assert!(result[i].is_nan());
        }

        // Choppiness should be between 0 and 100
        for i in 13..50 {
            assert!(result[i] >= 0.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_trending_market() {
        let chop = ChoppinessIndex::new(14);

        // Strong trending data (consistent upward movement)
        let high: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 2.0 + 2.0).collect();
        let low: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 2.0 - 2.0).collect();
        let close: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 2.0).collect();

        let result = chop.calculate(&high, &low, &close);

        // Trending market should have lower choppiness
        for i in 14..50 {
            assert!(result[i] < 50.0, "Trending market should have low choppiness");
        }
    }

    #[test]
    fn test_choppy_market() {
        let chop = ChoppinessIndex::new(14);

        // Sideways choppy data (oscillating around same level)
        let high: Vec<f64> = (0..50)
            .map(|i| 102.0 + (i as f64 * 0.5).sin() * 2.0)
            .collect();
        let low: Vec<f64> = (0..50)
            .map(|i| 98.0 + (i as f64 * 0.5).sin() * 2.0)
            .collect();
        let close: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64 * 0.5).sin() * 2.0)
            .collect();

        let result = chop.calculate(&high, &low, &close);

        // Choppy market should have higher choppiness
        for i in 20..50 {
            assert!(result[i] > 40.0, "Choppy market should have moderate to high choppiness");
        }
    }
}
