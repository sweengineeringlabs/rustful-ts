//! Exponential Moving Average implementation.

use indicator_spi::{TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries};
use indicator_api::EMAConfig;

/// Exponential Moving Average (EMA).
///
/// Gives more weight to recent prices using an exponential decay.
#[derive(Debug, Clone)]
pub struct EMA {
    period: usize,
    alpha: f64,
}

impl EMA {
    pub fn new(period: usize) -> Self {
        let alpha = 2.0 / (period as f64 + 1.0);
        Self { period, alpha }
    }

    pub fn with_alpha(period: usize, alpha: f64) -> Self {
        Self { period, alpha }
    }

    pub fn from_config(config: EMAConfig) -> Self {
        Self {
            period: config.period,
            alpha: config.smoothing_factor(),
        }
    }

    /// Calculate EMA values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        if data.len() < self.period || self.period == 0 {
            return vec![f64::NAN; data.len()];
        }

        let mut result = vec![f64::NAN; self.period - 1];

        // Find first valid starting position (skip NaN values)
        let mut start_idx = None;
        for i in 0..=(data.len().saturating_sub(self.period)) {
            let window = &data[i..(i + self.period)];
            if window.iter().all(|x| !x.is_nan()) {
                start_idx = Some(i);
                break;
            }
        }

        let start_idx = match start_idx {
            Some(idx) => idx,
            None => return vec![f64::NAN; data.len()],
        };

        // Prepend NaN for skipped values
        while result.len() < start_idx + self.period - 1 {
            result.push(f64::NAN);
        }

        // Initial SMA as seed
        let initial_sma: f64 = data[start_idx..(start_idx + self.period)]
            .iter()
            .sum::<f64>() / self.period as f64;
        result.push(initial_sma);

        // EMA calculation
        let mut ema = initial_sma;
        for i in (start_idx + self.period)..data.len() {
            if data[i].is_nan() {
                result.push(f64::NAN);
            } else {
                ema = self.alpha * data[i] + (1.0 - self.alpha) * ema;
                result.push(ema);
            }
        }

        result
    }
}

impl TechnicalIndicator for EMA {
    fn name(&self) -> &str {
        "EMA"
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ema() {
        let ema = EMA::new(3);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ema.calculate(&data);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        // First EMA value is SMA
        assert!((result[2] - 2.0).abs() < 1e-10);
        // Subsequent values use EMA formula
        assert!(result[3] > 2.0);
        assert!(result[4] > result[3]);
    }
}
