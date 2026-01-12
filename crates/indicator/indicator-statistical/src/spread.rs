//! Spread implementation.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Spread.
///
/// Calculates the difference between two series or between
/// the current price and a moving average.
///
/// Useful for pairs trading and mean reversion strategies.
#[derive(Debug, Clone)]
pub struct Spread {
    period: usize,
}

impl Spread {
    /// Create a new Spread indicator.
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Calculate spread between price and its moving average.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period || self.period == 0 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &data[start..=i];

            // Calculate SMA
            let sma: f64 = window.iter().sum::<f64>() / self.period as f64;

            // Spread = Price - SMA
            let spread = data[i] - sma;
            result.push(spread);
        }

        result
    }

    /// Calculate spread between two series.
    pub fn calculate_between(&self, series1: &[f64], series2: &[f64]) -> Vec<f64> {
        let n = series1.len().min(series2.len());
        series1[..n]
            .iter()
            .zip(series2[..n].iter())
            .map(|(a, b)| a - b)
            .collect()
    }

    /// Calculate percentage spread between price and its moving average.
    pub fn calculate_percent(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period || self.period == 0 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &data[start..=i];

            // Calculate SMA
            let sma: f64 = window.iter().sum::<f64>() / self.period as f64;

            // Percentage spread = (Price - SMA) / SMA * 100
            let spread = if sma.abs() < 1e-10 {
                f64::NAN
            } else {
                (data[i] - sma) / sma * 100.0
            };
            result.push(spread);
        }

        result
    }
}

impl TechnicalIndicator for Spread {
    fn name(&self) -> &str {
        "Spread"
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

impl SignalIndicator for Spread {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.close);
        let last = values.last().copied().unwrap_or(f64::NAN);

        if last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Price above MA = bullish, below MA = bearish
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
        let signals = values
            .iter()
            .map(|&spread| {
                if spread.is_nan() {
                    IndicatorSignal::Neutral
                } else if spread > 0.0 {
                    IndicatorSignal::Bullish
                } else if spread < 0.0 {
                    IndicatorSignal::Bearish
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
    fn test_spread() {
        let spread = Spread::new(3);
        let data = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let result = spread.calculate(&data);

        // First two values should be NaN
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());

        // Spread at index 2: price=12, SMA=(10+11+12)/3=11, spread=1
        assert!((result[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_spread_between() {
        let spread = Spread::new(3);
        let series1 = vec![100.0, 110.0, 120.0];
        let series2 = vec![90.0, 100.0, 110.0];
        let result = spread.calculate_between(&series1, &series2);

        assert_eq!(result.len(), 3);
        assert!((result[0] - 10.0).abs() < 1e-10);
        assert!((result[1] - 10.0).abs() < 1e-10);
        assert!((result[2] - 10.0).abs() < 1e-10);
    }
}
