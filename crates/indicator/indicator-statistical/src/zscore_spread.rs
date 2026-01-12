//! Z-Score Spread implementation.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Z-Score Spread.
///
/// Calculates the z-score of the spread between two series.
/// Particularly useful for pairs trading and mean reversion strategies.
///
/// Z-Score Spread = (Spread - Mean(Spread)) / StdDev(Spread)
#[derive(Debug, Clone)]
pub struct ZScoreSpread {
    period: usize,
    /// Threshold for overbought signal (long spread)
    upper_threshold: f64,
    /// Threshold for oversold signal (short spread)
    lower_threshold: f64,
}

impl ZScoreSpread {
    /// Create a new ZScoreSpread indicator with default thresholds (+/- 2.0).
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

    /// Calculate z-score of spread between price and its MA.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period || self.period == 0 {
            return vec![f64::NAN; n];
        }

        // First calculate the spread (price - SMA)
        let mut spreads = vec![f64::NAN; self.period - 1];
        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &data[start..=i];
            let sma: f64 = window.iter().sum::<f64>() / self.period as f64;
            spreads.push(data[i] - sma);
        }

        // Then calculate z-score of the spreads
        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;

            // Get valid spreads in window
            let window: Vec<f64> = spreads[start..=i]
                .iter()
                .copied()
                .filter(|x| !x.is_nan())
                .collect();

            if window.is_empty() {
                result.push(f64::NAN);
                continue;
            }

            let mean: f64 = window.iter().sum::<f64>() / window.len() as f64;
            let variance: f64 = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / window.len() as f64;
            let std_dev = variance.sqrt();

            let zscore = if std_dev.abs() < 1e-10 {
                0.0
            } else {
                (spreads[i] - mean) / std_dev
            };
            result.push(zscore);
        }

        result
    }

    /// Calculate z-score spread between two series.
    pub fn calculate_between(&self, series1: &[f64], series2: &[f64]) -> Vec<f64> {
        let n = series1.len().min(series2.len());
        if n < self.period || self.period == 0 {
            return vec![f64::NAN; n];
        }

        // Calculate raw spread
        let spreads: Vec<f64> = series1[..n]
            .iter()
            .zip(series2[..n].iter())
            .map(|(a, b)| a - b)
            .collect();

        // Calculate z-score of spreads
        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &spreads[start..=i];

            let mean: f64 = window.iter().sum::<f64>() / self.period as f64;
            let variance: f64 = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / self.period as f64;
            let std_dev = variance.sqrt();

            let zscore = if std_dev.abs() < 1e-10 {
                0.0
            } else {
                (spreads[i] - mean) / std_dev
            };
            result.push(zscore);
        }

        result
    }
}

impl TechnicalIndicator for ZScoreSpread {
    fn name(&self) -> &str {
        "ZScoreSpread"
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

impl SignalIndicator for ZScoreSpread {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.close);
        let last = values.last().copied().unwrap_or(f64::NAN);

        if last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // For mean reversion:
        // High z-score = spread is high = expect it to decrease = bearish
        // Low z-score = spread is low = expect it to increase = bullish
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
    fn test_zscore_spread() {
        let zs = ZScoreSpread::new(20);
        let data: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64).sin() * 5.0).collect();
        let result = zs.calculate(&data);

        // Check that z-scores are calculated after warmup
        for i in 19..50 {
            assert!(!result[i].is_nan());
            // Z-scores should typically be within reasonable range
            assert!(result[i] > -5.0 && result[i] < 5.0);
        }
    }

    #[test]
    fn test_zscore_spread_between() {
        let zs = ZScoreSpread::new(10);
        // Two series with stable spread
        let series1: Vec<f64> = (0..30).map(|i| 100.0 + i as f64).collect();
        let series2: Vec<f64> = (0..30).map(|i| 90.0 + i as f64).collect();
        let result = zs.calculate_between(&series1, &series2);

        // Constant spread should have z-score near 0
        for i in 9..30 {
            assert!((result[i] - 0.0).abs() < 1e-10);
        }
    }
}
