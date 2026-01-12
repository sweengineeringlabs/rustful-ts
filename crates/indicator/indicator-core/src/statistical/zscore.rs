//! Z-Score implementation.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Z-Score.
///
/// Measures how many standard deviations a value is from the mean.
/// Useful for detecting outliers and mean reversion signals.
///
/// Z-Score = (Value - Mean) / StdDev
#[derive(Debug, Clone)]
pub struct ZScore {
    period: usize,
    /// Threshold for overbought signal
    upper_threshold: f64,
    /// Threshold for oversold signal
    lower_threshold: f64,
}

impl ZScore {
    /// Create a new ZScore indicator with default thresholds (+/- 2.0).
    pub fn new(period: usize) -> Self {
        Self {
            period,
            upper_threshold: 2.0,
            lower_threshold: -2.0,
        }
    }

    /// Create with custom thresholds.
    pub fn with_thresholds(period: usize, upper: f64, lower: f64) -> Self {
        Self {
            period,
            upper_threshold: upper,
            lower_threshold: lower,
        }
    }

    /// Calculate Z-Score values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period || self.period == 0 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &data[start..=i];

            // Calculate mean
            let mean: f64 = window.iter().sum::<f64>() / self.period as f64;

            // Calculate standard deviation (population)
            let sum_sq: f64 = window.iter().map(|x| (x - mean).powi(2)).sum();
            let std_dev = (sum_sq / self.period as f64).sqrt();

            // Calculate Z-Score
            let zscore = if std_dev.abs() < 1e-10 {
                0.0 // No variation, z-score is 0
            } else {
                (data[i] - mean) / std_dev
            };
            result.push(zscore);
        }

        result
    }
}

impl TechnicalIndicator for ZScore {
    fn name(&self) -> &str {
        "ZScore"
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

impl SignalIndicator for ZScore {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.close);
        let last = values.last().copied().unwrap_or(f64::NAN);

        if last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Z-Score above upper threshold = overbought (bearish)
        // Z-Score below lower threshold = oversold (bullish)
        if last >= self.upper_threshold {
            Ok(IndicatorSignal::Bearish)
        } else if last <= self.lower_threshold {
            Ok(IndicatorSignal::Bullish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.close);
        let signals = values
            .iter()
            .map(|&z| {
                if z.is_nan() {
                    IndicatorSignal::Neutral
                } else if z >= self.upper_threshold {
                    IndicatorSignal::Bearish
                } else if z <= self.lower_threshold {
                    IndicatorSignal::Bullish
                } else {
                    IndicatorSignal::Neutral
                }
            })
            .collect();
        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zscore() {
        let zscore = ZScore::new(20);
        let data: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64).sin() * 5.0).collect();
        let result = zscore.calculate(&data);

        // Check that Z-scores are calculated after warmup
        for i in 19..50 {
            assert!(!result[i].is_nan());
            // Z-scores should typically be within -3 to 3 for normal data
            assert!(result[i] > -5.0 && result[i] < 5.0);
        }
    }

    #[test]
    fn test_zscore_constant() {
        let zscore = ZScore::new(5);
        let data = vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0];
        let result = zscore.calculate(&data);

        // Z-score of constant values should be 0
        for i in 4..data.len() {
            assert!((result[i] - 0.0).abs() < 1e-10);
        }
    }
}
