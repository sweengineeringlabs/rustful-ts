//! Momentum indicator.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Momentum - IND-002
///
/// Simple price change over N periods (absolute difference).
/// Momentum = Current Price - Price N periods ago
#[derive(Debug, Clone)]
pub struct Momentum {
    period: usize,
}

impl Momentum {
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
            result.push(data[i] - data[i - self.period]);
        }

        result
    }
}

impl TechnicalIndicator for Momentum {
    fn name(&self) -> &str {
        "Momentum"
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

impl SignalIndicator for Momentum {
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
        let signals = values.iter().map(|&mom| {
            if mom.is_nan() {
                IndicatorSignal::Neutral
            } else if mom > 0.0 {
                IndicatorSignal::Bullish
            } else if mom < 0.0 {
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
    fn test_momentum_basic() {
        let mom = Momentum::new(1);
        let data = vec![100.0, 110.0, 105.0, 115.0];
        let result = mom.calculate(&data);

        assert!(result[0].is_nan());
        assert!((result[1] - 10.0).abs() < 1e-10);
        assert!((result[2] - (-5.0)).abs() < 1e-10);
        assert!((result[3] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_momentum_period() {
        let mom = Momentum::new(3);
        let data = vec![100.0, 102.0, 104.0, 106.0, 108.0];
        let result = mom.calculate(&data);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!(result[2].is_nan());
        assert!((result[3] - 6.0).abs() < 1e-10);
        assert!((result[4] - 6.0).abs() < 1e-10);
    }
}
