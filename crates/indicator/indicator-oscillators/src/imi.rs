//! Intraday Momentum Index (IMI).

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Intraday Momentum Index (IMI) - IND-156
///
/// RSI variant comparing close-open relationship.
/// IMI = (Sum of up days gains) / (Sum of up + down days) * 100
#[derive(Debug, Clone)]
pub struct IntradayMomentumIndex {
    period: usize,
    overbought: f64,
    oversold: f64,
}

impl IntradayMomentumIndex {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            overbought: 70.0,
            oversold: 30.0,
        }
    }

    pub fn with_thresholds(period: usize, overbought: f64, oversold: f64) -> Self {
        Self { period, overbought, oversold }
    }

    pub fn calculate(&self, open: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < self.period {
            return vec![f64::NAN; n];
        }

        // Calculate gains and losses based on close vs open
        let mut gains = vec![0.0; n];
        let mut losses = vec![0.0; n];

        for i in 0..n {
            if close[i] > open[i] {
                gains[i] = close[i] - open[i];
            } else if close[i] < open[i] {
                losses[i] = open[i] - close[i];
            }
        }

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let sum_gains: f64 = gains[(i - self.period + 1)..=i].iter().sum();
            let sum_losses: f64 = losses[(i - self.period + 1)..=i].iter().sum();
            let total = sum_gains + sum_losses;

            if total == 0.0 {
                result.push(50.0);
            } else {
                result.push((sum_gains / total) * 100.0);
            }
        }

        result
    }
}

impl Default for IntradayMomentumIndex {
    fn default() -> Self {
        Self::new(14)
    }
}

impl TechnicalIndicator for IntradayMomentumIndex {
    fn name(&self) -> &str {
        "IMI"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.open, &data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.period
    }
}

impl SignalIndicator for IntradayMomentumIndex {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.open, &data.close);
        let last = values.last().copied().unwrap_or(f64::NAN);

        if last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        if last >= self.overbought {
            Ok(IndicatorSignal::Bearish)
        } else if last <= self.oversold {
            Ok(IndicatorSignal::Bullish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.open, &data.close);
        let signals = values.iter().map(|&val| {
            if val.is_nan() {
                IndicatorSignal::Neutral
            } else if val >= self.overbought {
                IndicatorSignal::Bearish
            } else if val <= self.oversold {
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
    fn test_imi_range() {
        let imi = IntradayMomentumIndex::new(14);
        let open: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let close: Vec<f64> = (0..50).map(|i| 102.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let result = imi.calculate(&open, &close);

        // IMI should be in range [0, 100]
        for val in result.iter() {
            if !val.is_nan() {
                assert!(*val >= 0.0 && *val <= 100.0, "IMI value {} out of range", val);
            }
        }
    }
}
