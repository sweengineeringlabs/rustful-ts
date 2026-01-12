//! Skewness implementation.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Skewness.
///
/// Measures the asymmetry of the return distribution.
///
/// - Positive skewness: distribution has a longer right tail (more extreme positive returns)
/// - Negative skewness: distribution has a longer left tail (more extreme negative returns)
/// - Zero: symmetric distribution
///
/// Useful for risk assessment and understanding tail risk.
#[derive(Debug, Clone)]
pub struct Skewness {
    period: usize,
    /// Use returns instead of raw prices
    use_returns: bool,
}

impl Skewness {
    /// Create a new Skewness indicator using returns.
    pub fn new(period: usize) -> Self {
        Self {
            period,
            use_returns: true,
        }
    }

    /// Create using raw price values.
    pub fn from_prices(period: usize) -> Self {
        Self {
            period,
            use_returns: false,
        }
    }

    /// Create using returns.
    pub fn from_returns(period: usize) -> Self {
        Self {
            period,
            use_returns: true,
        }
    }

    /// Calculate sample skewness for a window.
    fn calculate_skewness(values: &[f64]) -> f64 {
        let n = values.len() as f64;
        if n < 3.0 {
            return f64::NAN;
        }

        let mean: f64 = values.iter().sum::<f64>() / n;

        let m2: f64 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let m3: f64 = values.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / n;

        let std_dev = m2.sqrt();
        if std_dev.abs() < 1e-10 {
            return 0.0; // No variation
        }

        // Sample skewness with bias correction
        let skew = m3 / std_dev.powi(3);

        // Apply bias correction for sample skewness
        let correction = ((n * (n - 1.0)).sqrt()) / (n - 2.0);
        skew * correction
    }

    /// Calculate skewness values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();

        if self.use_returns {
            // Need one extra data point for returns calculation
            if n < self.period + 1 || self.period < 3 {
                return vec![f64::NAN; n];
            }

            // Calculate returns
            let returns: Vec<f64> = data
                .windows(2)
                .map(|w| {
                    if w[0].abs() < 1e-10 {
                        0.0
                    } else {
                        (w[1] - w[0]) / w[0]
                    }
                })
                .collect();

            let mut result = vec![f64::NAN; self.period];

            for i in (self.period - 1)..returns.len() {
                let start = i + 1 - self.period;
                let window = &returns[start..=i];
                let skew = Self::calculate_skewness(window);
                result.push(skew);
            }

            result
        } else {
            if n < self.period || self.period < 3 {
                return vec![f64::NAN; n];
            }

            let mut result = vec![f64::NAN; self.period - 1];

            for i in (self.period - 1)..n {
                let start = i + 1 - self.period;
                let window = &data[start..=i];
                let skew = Self::calculate_skewness(window);
                result.push(skew);
            }

            result
        }
    }
}

impl TechnicalIndicator for Skewness {
    fn name(&self) -> &str {
        "Skewness"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let required = if self.use_returns {
            self.period + 1
        } else {
            self.period
        };

        if data.close.len() < required {
            return Err(IndicatorError::InsufficientData {
                required,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        if self.use_returns {
            self.period + 1
        } else {
            self.period
        }
    }
}

impl SignalIndicator for Skewness {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.close);
        let last = values.last().copied().unwrap_or(f64::NAN);

        if last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Positive skewness = potential for positive outliers = cautiously bullish
        // Negative skewness = potential for negative outliers = cautiously bearish
        // This is a risk-based interpretation
        if last > 0.5 {
            Ok(IndicatorSignal::Bullish)
        } else if last < -0.5 {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.close);
        let signals = values
            .iter()
            .map(|&skew| {
                if skew.is_nan() {
                    IndicatorSignal::Neutral
                } else if skew > 0.5 {
                    IndicatorSignal::Bullish
                } else if skew < -0.5 {
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
    fn test_skewness_symmetric() {
        let skew = Skewness::from_prices(5);
        // Symmetric data around mean should have skewness near 0
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0];
        let result = skew.calculate(&data);

        // Check that values are calculated after warmup
        for i in 4..result.len() {
            assert!(!result[i].is_nan());
        }
    }

    #[test]
    fn test_skewness_positive() {
        let skew = Skewness::from_prices(5);
        // Right-skewed data
        let data = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0];
        let result = skew.calculate(&data);

        // Last value should show positive skewness
        let last = result.last().unwrap();
        assert!(*last > 0.0);
    }

    #[test]
    fn test_skewness_negative() {
        let skew = Skewness::from_prices(10);
        // Left-skewed data: most values high, one low outlier in window
        let data = vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 1.0];
        let result = skew.calculate(&data);

        // Last value should show negative skewness (tail on the left)
        let last = result.last().unwrap();
        assert!(*last < 0.0, "Expected negative skewness, got {}", last);
    }
}
