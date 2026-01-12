//! Aroon indicator.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Aroon Output containing Up, Down, and Oscillator values.
#[derive(Debug, Clone)]
pub struct AroonOutput {
    pub up: Vec<f64>,
    pub down: Vec<f64>,
    pub oscillator: Vec<f64>,
}

/// Aroon - IND-003
///
/// Measures time since highest high / lowest low.
/// Aroon Up = ((Period - Days Since High) / Period) * 100
/// Aroon Down = ((Period - Days Since Low) / Period) * 100
/// Oscillator = Aroon Up - Aroon Down
#[derive(Debug, Clone)]
pub struct Aroon {
    period: usize,
}

impl Aroon {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    pub fn calculate(&self, high: &[f64], low: &[f64]) -> AroonOutput {
        let n = high.len();
        if n < self.period {
            return AroonOutput {
                up: vec![f64::NAN; n],
                down: vec![f64::NAN; n],
                oscillator: vec![f64::NAN; n],
            };
        }

        let mut up = vec![f64::NAN; self.period - 1];
        let mut down = vec![f64::NAN; self.period - 1];
        let mut oscillator = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let window_high = &high[(i - self.period + 1)..=i];
            let window_low = &low[(i - self.period + 1)..=i];

            // Find position of highest high in window
            let mut max_idx = 0;
            let mut max_val = window_high[0];
            for (j, &val) in window_high.iter().enumerate() {
                if val >= max_val {
                    max_val = val;
                    max_idx = j;
                }
            }

            // Find position of lowest low in window
            let mut min_idx = 0;
            let mut min_val = window_low[0];
            for (j, &val) in window_low.iter().enumerate() {
                if val <= min_val {
                    min_val = val;
                    min_idx = j;
                }
            }

            let days_since_high = (self.period - 1) - max_idx;
            let days_since_low = (self.period - 1) - min_idx;

            let aroon_up = ((self.period - days_since_high) as f64 / self.period as f64) * 100.0;
            let aroon_down = ((self.period - days_since_low) as f64 / self.period as f64) * 100.0;

            up.push(aroon_up);
            down.push(aroon_down);
            oscillator.push(aroon_up - aroon_down);
        }

        AroonOutput { up, down, oscillator }
    }
}

impl Default for Aroon {
    fn default() -> Self {
        Self::new(25)
    }
}

impl TechnicalIndicator for Aroon {
    fn name(&self) -> &str {
        "Aroon"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.high.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.high.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low);
        Ok(IndicatorOutput::triple(result.up, result.down, result.oscillator))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        3
    }
}

impl SignalIndicator for Aroon {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let result = self.calculate(&data.high, &data.low);
        let osc = result.oscillator.last().copied().unwrap_or(f64::NAN);

        if osc.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Strong trend signals
        if osc > 50.0 {
            Ok(IndicatorSignal::Bullish)
        } else if osc < -50.0 {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let result = self.calculate(&data.high, &data.low);
        let signals = result.oscillator.iter().map(|&osc| {
            if osc.is_nan() {
                IndicatorSignal::Neutral
            } else if osc > 50.0 {
                IndicatorSignal::Bullish
            } else if osc < -50.0 {
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
    fn test_aroon_basic() {
        let aroon = Aroon::new(5);
        // Uptrend: progressively higher highs
        let high = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0];
        let low = vec![95.0, 96.0, 97.0, 98.0, 99.0, 100.0];
        let result = aroon.calculate(&high, &low);

        // Most recent is highest, so Aroon Up should be 100
        let last_up = result.up.last().unwrap();
        assert!((*last_up - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_aroon_range() {
        let aroon = Aroon::new(14);
        let high: Vec<f64> = (0..30).map(|i| 110.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let low: Vec<f64> = (0..30).map(|i| 90.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let result = aroon.calculate(&high, &low);

        // Aroon values should be in [0, 100]
        for val in result.up.iter().chain(result.down.iter()) {
            if !val.is_nan() {
                assert!(*val >= 0.0 && *val <= 100.0);
            }
        }

        // Oscillator should be in [-100, 100]
        for val in result.oscillator.iter() {
            if !val.is_nan() {
                assert!(*val >= -100.0 && *val <= 100.0);
            }
        }
    }
}
