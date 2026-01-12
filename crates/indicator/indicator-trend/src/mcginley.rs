//! McGinley Dynamic indicator.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// McGinley Dynamic - IND-026
///
/// Self-adjusting moving average that automatically adjusts to market speed.
/// MD = MD[-1] + (Close - MD[-1]) / (N * (Close / MD[-1])^4)
#[derive(Debug, Clone)]
pub struct McGinleyDynamic {
    period: usize,
}

impl McGinleyDynamic {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period {
            return vec![f64::NAN; n];
        }

        let mut result = Vec::with_capacity(n);

        // Initialize with first value
        result.push(data[0]);

        for i in 1..n {
            let prev_md = result[i - 1];
            let close = data[i];

            if prev_md <= 0.0 || close <= 0.0 {
                result.push(close);
                continue;
            }

            // McGinley Dynamic formula
            let ratio = close / prev_md;
            let k = self.period as f64 * ratio.powi(4);

            if k == 0.0 {
                result.push(close);
            } else {
                let md = prev_md + (close - prev_md) / k;
                result.push(md);
            }
        }

        // Mark warmup period as NaN
        for i in 0..(self.period - 1) {
            result[i] = f64::NAN;
        }

        result
    }
}

impl Default for McGinleyDynamic {
    fn default() -> Self {
        Self::new(14)
    }
}

impl TechnicalIndicator for McGinleyDynamic {
    fn name(&self) -> &str {
        "McGinley"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.period
    }
}

impl SignalIndicator for McGinleyDynamic {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.close);

        if values.len() < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let md = values[values.len() - 1];
        let close = data.close[data.close.len() - 1];

        if md.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Price above MD = bullish
        if close > md {
            Ok(IndicatorSignal::Bullish)
        } else if close < md {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.close);
        let n = values.len().min(data.close.len());
        let mut signals = Vec::with_capacity(n);

        for i in 0..n {
            if values[i].is_nan() {
                signals.push(IndicatorSignal::Neutral);
            } else if data.close[i] > values[i] {
                signals.push(IndicatorSignal::Bullish);
            } else if data.close[i] < values[i] {
                signals.push(IndicatorSignal::Bearish);
            } else {
                signals.push(IndicatorSignal::Neutral);
            }
        }

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcginley_basic() {
        let md = McGinleyDynamic::new(14);
        let data: Vec<f64> = (0..50).map(|i| 100.0 + i as f64).collect();
        let result = md.calculate(&data);

        assert_eq!(result.len(), 50);

        // In uptrend, MD should track below price (lagging)
        let last_md = result.last().unwrap();
        let last_price = data.last().unwrap();

        assert!(!last_md.is_nan());
        assert!(*last_md < *last_price);
    }
}
