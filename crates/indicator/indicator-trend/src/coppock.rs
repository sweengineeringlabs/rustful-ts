//! Coppock Curve.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Coppock Curve - IND-030
///
/// Long-term momentum indicator.
/// Coppock = WMA(ROC(14) + ROC(11), 10)
#[derive(Debug, Clone)]
pub struct CoppockCurve {
    wma_period: usize,
    roc_long: usize,
    roc_short: usize,
}

impl CoppockCurve {
    pub fn new(wma_period: usize, roc_long: usize, roc_short: usize) -> Self {
        Self { wma_period, roc_long, roc_short }
    }

    fn roc(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        if n <= period {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; period];
        for i in period..n {
            if data[i - period] != 0.0 {
                result.push(((data[i] - data[i - period]) / data[i - period]) * 100.0);
            } else {
                result.push(f64::NAN);
            }
        }
        result
    }

    fn wma(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        if n < period {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; period - 1];
        let weight_sum: f64 = (1..=period).map(|x| x as f64).sum();

        for i in (period - 1)..n {
            let window = &data[(i - period + 1)..=i];
            let valid: Vec<(usize, f64)> = window.iter()
                .enumerate()
                .filter(|(_, x)| !x.is_nan())
                .map(|(j, &x)| (j, x))
                .collect();

            if valid.is_empty() {
                result.push(f64::NAN);
            } else {
                let weighted_sum: f64 = valid.iter()
                    .map(|(j, x)| (j + 1) as f64 * x)
                    .sum();
                result.push(weighted_sum / weight_sum);
            }
        }

        result
    }

    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.roc_long + self.wma_period {
            return vec![f64::NAN; n];
        }

        let roc_l = Self::roc(data, self.roc_long);
        let roc_s = Self::roc(data, self.roc_short);

        // Sum of ROCs
        let sum: Vec<f64> = roc_l.iter()
            .zip(roc_s.iter())
            .map(|(l, s)| {
                if l.is_nan() || s.is_nan() {
                    f64::NAN
                } else {
                    l + s
                }
            })
            .collect();

        // WMA of sum
        Self::wma(&sum, self.wma_period)
    }
}

impl Default for CoppockCurve {
    fn default() -> Self {
        Self::new(10, 14, 11)
    }
}

impl TechnicalIndicator for CoppockCurve {
    fn name(&self) -> &str {
        "Coppock"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.roc_long + self.wma_period {
            return Err(IndicatorError::InsufficientData {
                required: self.roc_long + self.wma_period,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.roc_long + self.wma_period
    }
}

impl SignalIndicator for CoppockCurve {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.close);

        if values.len() < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let last = values[values.len() - 1];
        let prev = values[values.len() - 2];

        if last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Buy signal when crossing above zero from below
        if last > 0.0 && prev <= 0.0 {
            Ok(IndicatorSignal::Bullish)
        } else if last < 0.0 && prev >= 0.0 {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.close);
        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..values.len() {
            if values[i].is_nan() || values[i-1].is_nan() {
                signals.push(IndicatorSignal::Neutral);
            } else if values[i] > 0.0 && values[i-1] <= 0.0 {
                signals.push(IndicatorSignal::Bullish);
            } else if values[i] < 0.0 && values[i-1] >= 0.0 {
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
    fn test_coppock_basic() {
        let coppock = CoppockCurve::default();
        let data: Vec<f64> = (0..50).map(|i| 100.0 + i as f64).collect();
        let result = coppock.calculate(&data);

        assert_eq!(result.len(), 50);

        // In uptrend, Coppock should be positive
        let last = result.last().unwrap();
        assert!(!last.is_nan());
        assert!(*last > 0.0);
    }
}
