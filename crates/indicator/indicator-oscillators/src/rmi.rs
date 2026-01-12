//! Relative Momentum Index (RMI).

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Relative Momentum Index (RMI) - IND-041
///
/// RSI variant that uses momentum over N periods instead of 1.
/// RMI = RSI calculated with momentum lookback instead of single-period change.
#[derive(Debug, Clone)]
pub struct RMI {
    period: usize,
    momentum: usize,
    overbought: f64,
    oversold: f64,
}

impl RMI {
    pub fn new(period: usize, momentum: usize) -> Self {
        Self {
            period,
            momentum,
            overbought: 70.0,
            oversold: 30.0,
        }
    }

    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period + self.momentum {
            return vec![f64::NAN; n];
        }

        // Calculate momentum changes (N-period changes)
        let mut gains = vec![0.0; n];
        let mut losses = vec![0.0; n];

        for i in self.momentum..n {
            let change = data[i] - data[i - self.momentum];
            if change > 0.0 {
                gains[i] = change;
            } else if change < 0.0 {
                losses[i] = -change;
            }
        }

        let mut result = vec![f64::NAN; self.momentum + self.period - 1];

        // Initial average gain/loss
        let start_idx = self.momentum;
        let mut avg_gain: f64 = gains[start_idx..(start_idx + self.period)].iter().sum::<f64>() / self.period as f64;
        let mut avg_loss: f64 = losses[start_idx..(start_idx + self.period)].iter().sum::<f64>() / self.period as f64;

        let rmi = if avg_loss == 0.0 { 100.0 } else { 100.0 - 100.0 / (1.0 + avg_gain / avg_loss) };
        result.push(rmi);

        // Wilder smoothing for subsequent values
        for i in (start_idx + self.period)..n {
            avg_gain = (avg_gain * (self.period - 1) as f64 + gains[i]) / self.period as f64;
            avg_loss = (avg_loss * (self.period - 1) as f64 + losses[i]) / self.period as f64;

            let rmi = if avg_loss == 0.0 { 100.0 } else { 100.0 - 100.0 / (1.0 + avg_gain / avg_loss) };
            result.push(rmi);
        }

        result
    }
}

impl Default for RMI {
    fn default() -> Self {
        Self::new(14, 5)
    }
}

impl TechnicalIndicator for RMI {
    fn name(&self) -> &str {
        "RMI"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + self.momentum {
            return Err(IndicatorError::InsufficientData {
                required: self.period + self.momentum,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.period + self.momentum
    }
}

impl SignalIndicator for RMI {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.close);
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
        let values = self.calculate(&data.close);
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
    fn test_rmi_range() {
        let rmi = RMI::default();
        let data: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.1).sin() * 10.0).collect();
        let result = rmi.calculate(&data);

        // RMI should be in range [0, 100]
        for val in result.iter() {
            if !val.is_nan() {
                assert!(*val >= 0.0 && *val <= 100.0, "RMI value {} out of range", val);
            }
        }
    }

    #[test]
    fn test_rmi_uptrend() {
        let rmi = RMI::new(5, 3);
        let data: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 * 2.0).collect();
        let result = rmi.calculate(&data);

        let last = result.last().unwrap();
        assert!(*last > 70.0, "RMI should be high in uptrend");
    }
}
