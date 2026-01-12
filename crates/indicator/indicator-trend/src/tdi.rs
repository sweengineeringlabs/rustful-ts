//! Trend Detection Index (TDI).

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Trend Detection Index (TDI) - IND-064
///
/// Composite trend detector comparing price momentum and direction.
#[derive(Debug, Clone)]
pub struct TrendDetectionIndex {
    period: usize,
}

impl TrendDetectionIndex {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    pub fn calculate(&self, data: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = data.len();
        if n < self.period * 2 {
            return (vec![f64::NAN; n], vec![f64::NAN; n]);
        }

        // Calculate momentum (price direction)
        let mut mom = vec![f64::NAN; self.period];
        for i in self.period..n {
            mom.push(data[i] - data[i - self.period]);
        }

        // Calculate absolute momentum sum
        let mut abs_mom_sum = vec![f64::NAN; self.period * 2 - 1];
        for i in (self.period * 2 - 1)..n {
            let sum: f64 = (0..self.period)
                .map(|j| (data[i - j] - data[i - j - 1]).abs())
                .sum();
            abs_mom_sum.push(sum);
        }

        // TDI = Momentum - Abs Momentum Sum (directional)
        let tdi: Vec<f64> = mom.iter()
            .zip(abs_mom_sum.iter())
            .map(|(m, a)| {
                if m.is_nan() || a.is_nan() {
                    f64::NAN
                } else {
                    m.abs() - *a
                }
            })
            .collect();

        // Direction indicator
        let direction: Vec<f64> = mom.iter()
            .map(|&m| {
                if m.is_nan() {
                    f64::NAN
                } else if m > 0.0 {
                    1.0
                } else if m < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            })
            .collect();

        (tdi, direction)
    }
}

impl Default for TrendDetectionIndex {
    fn default() -> Self {
        Self::new(20)
    }
}

impl TechnicalIndicator for TrendDetectionIndex {
    fn name(&self) -> &str {
        "TDI"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period * 2 {
            return Err(IndicatorError::InsufficientData {
                required: self.period * 2,
                got: data.close.len(),
            });
        }

        let (tdi, direction) = self.calculate(&data.close);
        Ok(IndicatorOutput::dual(tdi, direction))
    }

    fn min_periods(&self) -> usize {
        self.period * 2
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for TrendDetectionIndex {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (tdi, direction) = self.calculate(&data.close);

        let tdi_last = tdi.last().copied().unwrap_or(f64::NAN);
        let dir_last = direction.last().copied().unwrap_or(f64::NAN);

        if tdi_last.is_nan() || dir_last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Positive TDI with positive direction = strong uptrend
        if tdi_last > 0.0 && dir_last > 0.0 {
            Ok(IndicatorSignal::Bullish)
        } else if tdi_last > 0.0 && dir_last < 0.0 {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (tdi, direction) = self.calculate(&data.close);
        let n = tdi.len().min(direction.len());
        let mut signals = Vec::with_capacity(n);

        for i in 0..n {
            if tdi[i].is_nan() || direction[i].is_nan() {
                signals.push(IndicatorSignal::Neutral);
            } else if tdi[i] > 0.0 && direction[i] > 0.0 {
                signals.push(IndicatorSignal::Bullish);
            } else if tdi[i] > 0.0 && direction[i] < 0.0 {
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
    fn test_tdi_basic() {
        let tdi = TrendDetectionIndex::new(10);
        let data: Vec<f64> = (0..50).map(|i| 100.0 + i as f64).collect();
        let (tdi_line, direction) = tdi.calculate(&data);

        assert_eq!(tdi_line.len(), 50);
        assert_eq!(direction.len(), 50);

        // In uptrend, direction should be positive
        let dir_last = direction.last().unwrap();
        assert!(!dir_last.is_nan());
        assert!(*dir_last > 0.0);
    }
}
