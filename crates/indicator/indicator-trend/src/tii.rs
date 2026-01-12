//! Trend Intensity Index (TII).

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Trend Intensity Index (TII) - IND-063
///
/// Measures trend strength using deviation from moving average.
/// TII = 100 * (Positive deviations / (Positive + Negative deviations))
#[derive(Debug, Clone)]
pub struct TrendIntensityIndex {
    period: usize,
}

impl TrendIntensityIndex {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    fn sma(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        if n < period {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; period - 1];
        let mut sum: f64 = data[..period].iter().sum();
        result.push(sum / period as f64);

        for i in period..n {
            sum = sum - data[i - period] + data[i];
            result.push(sum / period as f64);
        }

        result
    }

    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period * 2 {
            return vec![f64::NAN; n];
        }

        let ma = Self::sma(data, self.period);

        let mut result = vec![f64::NAN; self.period * 2 - 1];

        for i in (self.period * 2 - 1)..n {
            let mut pos_dev = 0.0;
            let mut neg_dev = 0.0;

            for j in 0..self.period {
                let idx = i - self.period + 1 + j;
                let dev = data[idx] - ma[idx];

                if dev > 0.0 {
                    pos_dev += dev;
                } else {
                    neg_dev += (-dev);
                }
            }

            let total = pos_dev + neg_dev;
            if total > 0.0 {
                result.push((pos_dev / total) * 100.0);
            } else {
                result.push(50.0);
            }
        }

        result
    }
}

impl Default for TrendIntensityIndex {
    fn default() -> Self {
        Self::new(30)
    }
}

impl TechnicalIndicator for TrendIntensityIndex {
    fn name(&self) -> &str {
        "TII"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period * 2 {
            return Err(IndicatorError::InsufficientData {
                required: self.period * 2,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.period * 2
    }
}

impl SignalIndicator for TrendIntensityIndex {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.close);
        let last = values.last().copied().unwrap_or(f64::NAN);

        if last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Standard thresholds: > 80 bullish, < 20 bearish
        if last > 80.0 {
            Ok(IndicatorSignal::Bullish)
        } else if last < 20.0 {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.close);
        let signals = values.iter().map(|&val| {
            if val.is_nan() {
                IndicatorSignal::Neutral
            } else if val > 80.0 {
                IndicatorSignal::Bullish
            } else if val < 20.0 {
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
    fn test_tii_range() {
        let tii = TrendIntensityIndex::new(14);
        let data: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.1).sin() * 10.0).collect();
        let result = tii.calculate(&data);

        // TII should be in range [0, 100]
        for val in result.iter() {
            if !val.is_nan() {
                assert!(*val >= 0.0 && *val <= 100.0, "TII value {} out of range", val);
            }
        }
    }
}
