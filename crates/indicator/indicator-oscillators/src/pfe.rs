//! Polarized Fractal Efficiency (PFE).

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Polarized Fractal Efficiency (PFE) - IND-155
///
/// Measures trend efficiency using fractal geometry.
/// PFE = (Distance / Length) * 100 * sign(direction)
#[derive(Debug, Clone)]
pub struct PolarizedFractalEfficiency {
    period: usize,
    smooth_period: usize,
}

impl PolarizedFractalEfficiency {
    pub fn new(period: usize, smooth_period: usize) -> Self {
        Self { period, smooth_period }
    }

    fn ema(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        if n == 0 {
            return vec![];
        }

        let alpha = 2.0 / (period as f64 + 1.0);
        let mut result = Vec::with_capacity(n);

        let first_valid = data.iter().position(|x| !x.is_nan());
        if first_valid.is_none() {
            return vec![f64::NAN; n];
        }

        let start = first_valid.unwrap();
        for _ in 0..start {
            result.push(f64::NAN);
        }

        let mut prev = data[start];
        result.push(prev);

        for i in (start + 1)..n {
            if data[i].is_nan() {
                result.push(prev);
            } else {
                let ema = alpha * data[i] + (1.0 - alpha) * prev;
                result.push(ema);
                prev = ema;
            }
        }

        result
    }

    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period {
            return vec![f64::NAN; n];
        }

        let mut raw_pfe = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            // Calculate total distance (sum of point-to-point distances)
            let mut length = 0.0;
            for j in 1..self.period {
                let idx = i - self.period + 1 + j;
                let diff = data[idx] - data[idx - 1];
                length += (1.0 + diff * diff).sqrt();
            }

            // Calculate direct distance
            let start_price = data[i - self.period + 1];
            let end_price = data[i];
            let distance = ((self.period as f64 - 1.0).powi(2) + (end_price - start_price).powi(2)).sqrt();

            // PFE = (distance / length) * 100 * sign
            if length > 0.0 {
                let efficiency = (distance / length) * 100.0;
                let sign = if end_price >= start_price { 1.0 } else { -1.0 };
                raw_pfe.push(efficiency * sign);
            } else {
                raw_pfe.push(0.0);
            }
        }

        // Smooth with EMA
        Self::ema(&raw_pfe, self.smooth_period)
    }
}

impl Default for PolarizedFractalEfficiency {
    fn default() -> Self {
        Self::new(10, 5)
    }
}

impl TechnicalIndicator for PolarizedFractalEfficiency {
    fn name(&self) -> &str {
        "PFE"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + self.smooth_period {
            return Err(IndicatorError::InsufficientData {
                required: self.period + self.smooth_period,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.period + self.smooth_period
    }
}

impl SignalIndicator for PolarizedFractalEfficiency {
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

        // Zero-line crossover
        if last > 0.0 && prev <= 0.0 {
            Ok(IndicatorSignal::Bullish)
        } else if last < 0.0 && prev >= 0.0 {
            Ok(IndicatorSignal::Bearish)
        } else if last > 50.0 {
            Ok(IndicatorSignal::Bullish)
        } else if last < -50.0 {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.close);
        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..values.len() {
            if values[i].is_nan() {
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
    fn test_pfe_range() {
        let pfe = PolarizedFractalEfficiency::default();
        let data: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.1).sin() * 10.0).collect();
        let result = pfe.calculate(&data);

        // PFE should be in range [-100, 100]
        for val in result.iter() {
            if !val.is_nan() {
                assert!(*val >= -100.0 && *val <= 100.0, "PFE value {} out of range", val);
            }
        }
    }
}
