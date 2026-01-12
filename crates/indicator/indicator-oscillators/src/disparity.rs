//! Disparity Index.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Disparity Index - IND-151
///
/// Distance from MA as percentage.
/// Disparity = ((Close - MA) / MA) * 100
#[derive(Debug, Clone)]
pub struct DisparityIndex {
    period: usize,
}

impl DisparityIndex {
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
        if n < self.period {
            return vec![f64::NAN; n];
        }

        let ma = Self::sma(data, self.period);

        data.iter()
            .zip(ma.iter())
            .map(|(close, avg)| {
                if avg.is_nan() || *avg == 0.0 {
                    f64::NAN
                } else {
                    ((close - avg) / avg) * 100.0
                }
            })
            .collect()
    }
}

impl Default for DisparityIndex {
    fn default() -> Self {
        Self::new(14)
    }
}

impl TechnicalIndicator for DisparityIndex {
    fn name(&self) -> &str {
        "DisparityIndex"
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

impl SignalIndicator for DisparityIndex {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.close);
        let last = values.last().copied().unwrap_or(f64::NAN);

        if last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Positive disparity = bullish (price above MA)
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
        let signals = values.iter().map(|&val| {
            if val.is_nan() {
                IndicatorSignal::Neutral
            } else if val > 0.0 {
                IndicatorSignal::Bullish
            } else if val < 0.0 {
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
    fn test_disparity_uptrend() {
        let di = DisparityIndex::new(5);
        // Uptrend: close above MA
        let data: Vec<f64> = (0..20).map(|i| 100.0 + i as f64 * 2.0).collect();
        let result = di.calculate(&data);

        // In an uptrend, recent prices should be above MA
        let last = result.last().unwrap();
        assert!(*last > 0.0);
    }
}
