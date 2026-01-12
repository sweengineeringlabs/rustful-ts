//! Rate of Change (ROC) indicator.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Rate of Change (ROC) - IND-001
///
/// Measures the percentage change in price over N periods.
/// ROC = ((Current Price - Price N periods ago) / Price N periods ago) * 100
#[derive(Debug, Clone)]
pub struct ROC {
    period: usize,
}

impl ROC {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n <= self.period {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period];

        for i in self.period..n {
            let prev = data[i - self.period];
            if prev != 0.0 {
                result.push(((data[i] - prev) / prev) * 100.0);
            } else {
                result.push(f64::NAN);
            }
        }

        result
    }
}

impl TechnicalIndicator for ROC {
    fn name(&self) -> &str {
        "ROC"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() <= self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }
}

impl SignalIndicator for ROC {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.close);
        let last = values.last().copied().unwrap_or(f64::NAN);

        if last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        if last > 0.0 {
            Ok(IndicatorSignal::Bullish)
        } else if last < 0.0 {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.close);
        let signals = values.iter().map(|&roc| {
            if roc.is_nan() {
                IndicatorSignal::Neutral
            } else if roc > 0.0 {
                IndicatorSignal::Bullish
            } else if roc < 0.0 {
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
    fn test_roc_basic() {
        let roc = ROC::new(1);
        let data = vec![100.0, 110.0, 105.0, 115.0];
        let result = roc.calculate(&data);

        assert!(result[0].is_nan());
        assert!((result[1] - 10.0).abs() < 1e-10);
        assert!((result[2] - (-4.545454545454546)).abs() < 1e-10);
        assert!((result[3] - 9.523809523809524).abs() < 1e-10);
    }

    #[test]
    fn test_roc_period() {
        let roc = ROC::new(3);
        let data = vec![100.0, 102.0, 104.0, 106.0, 108.0];
        let result = roc.calculate(&data);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!(result[2].is_nan());
        assert!((result[3] - 6.0).abs() < 1e-10);
    }
}
