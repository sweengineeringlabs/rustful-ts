//! Ratio implementation.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Ratio.
///
/// Calculates the ratio between two series or between
/// the current price and a moving average.
///
/// Useful for pairs trading, relative strength, and momentum analysis.
#[derive(Debug, Clone)]
pub struct Ratio {
    period: usize,
    /// Threshold above which ratio is considered high
    upper_threshold: f64,
    /// Threshold below which ratio is considered low
    lower_threshold: f64,
}

impl Ratio {
    /// Create a new Ratio indicator with default thresholds.
    pub fn new(period: usize) -> Self {
        Self {
            period,
            upper_threshold: 1.02, // 2% above MA
            lower_threshold: 0.98, // 2% below MA
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

    /// Calculate ratio between price and its moving average.
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

            // Ratio = Price / SMA
            let ratio = if sma.abs() < 1e-10 {
                f64::NAN
            } else {
                data[i] / sma
            };
            result.push(ratio);
        }

        result
    }

    /// Calculate ratio between two series.
    pub fn calculate_between(&self, series1: &[f64], series2: &[f64]) -> Vec<f64> {
        let n = series1.len().min(series2.len());
        series1[..n]
            .iter()
            .zip(series2[..n].iter())
            .map(|(a, b)| {
                if b.abs() < 1e-10 {
                    f64::NAN
                } else {
                    a / b
                }
            })
            .collect()
    }

    /// Calculate log ratio (log(price/SMA)).
    pub fn calculate_log(&self, data: &[f64]) -> Vec<f64> {
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

            // Log ratio = ln(Price / SMA)
            let ratio = if sma <= 0.0 || data[i] <= 0.0 {
                f64::NAN
            } else {
                (data[i] / sma).ln()
            };
            result.push(ratio);
        }

        result
    }

    /// Calculate percentage over moving average.
    pub fn calculate_percent_over(&self, data: &[f64]) -> Vec<f64> {
        self.calculate(data)
            .iter()
            .map(|&r| {
                if r.is_nan() {
                    f64::NAN
                } else {
                    (r - 1.0) * 100.0
                }
            })
            .collect()
    }
}

impl TechnicalIndicator for Ratio {
    fn name(&self) -> &str {
        "Ratio"
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

impl SignalIndicator for Ratio {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.close);
        let last = values.last().copied().unwrap_or(f64::NAN);

        if last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Ratio above upper threshold = overbought (bearish mean reversion)
        // Ratio below lower threshold = oversold (bullish mean reversion)
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
            .map(|&ratio| {
                if ratio.is_nan() {
                    IndicatorSignal::Neutral
                } else if ratio >= self.upper_threshold {
                    IndicatorSignal::Bearish
                } else if ratio <= self.lower_threshold {
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
    fn test_ratio() {
        let ratio = Ratio::new(3);
        let data = vec![10.0, 10.0, 10.0, 12.0, 12.0]; // Price jumps from 10 to 12
        let result = ratio.calculate(&data);

        // First two values should be NaN
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());

        // At index 2: price=10, SMA=10, ratio=1.0
        assert!((result[2] - 1.0).abs() < 1e-10);

        // At index 3: price=12, SMA=(10+10+12)/3=10.67, ratio=1.125
        assert!(result[3] > 1.0);
    }

    #[test]
    fn test_ratio_between() {
        let ratio = Ratio::new(3);
        let series1 = vec![100.0, 110.0, 120.0];
        let series2 = vec![50.0, 55.0, 60.0];
        let result = ratio.calculate_between(&series1, &series2);

        assert_eq!(result.len(), 3);
        assert!((result[0] - 2.0).abs() < 1e-10);
        assert!((result[1] - 2.0).abs() < 1e-10);
        assert!((result[2] - 2.0).abs() < 1e-10);
    }
}
