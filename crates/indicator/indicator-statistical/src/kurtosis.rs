//! Kurtosis implementation.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Kurtosis.
///
/// Measures the "tailedness" of the return distribution.
///
/// - Excess kurtosis > 0 (leptokurtic): heavier tails, more outliers
/// - Excess kurtosis < 0 (platykurtic): lighter tails, fewer outliers
/// - Excess kurtosis = 0 (mesokurtic): similar to normal distribution
///
/// Useful for risk assessment, particularly for understanding tail risk
/// and the likelihood of extreme returns.
#[derive(Debug, Clone)]
pub struct Kurtosis {
    period: usize,
    /// Use returns instead of raw prices
    use_returns: bool,
    /// Return excess kurtosis (subtract 3) vs raw kurtosis
    excess: bool,
}

impl Kurtosis {
    /// Create a new Kurtosis indicator using returns and excess kurtosis.
    pub fn new(period: usize) -> Self {
        Self {
            period,
            use_returns: true,
            excess: true,
        }
    }

    /// Create using raw price values.
    pub fn from_prices(period: usize) -> Self {
        Self {
            period,
            use_returns: false,
            excess: true,
        }
    }

    /// Create using returns.
    pub fn from_returns(period: usize) -> Self {
        Self {
            period,
            use_returns: true,
            excess: true,
        }
    }

    /// Create with raw kurtosis (not excess).
    pub fn raw(period: usize) -> Self {
        Self {
            period,
            use_returns: true,
            excess: false,
        }
    }

    /// Calculate sample kurtosis for a window.
    fn calculate_kurtosis(values: &[f64], excess: bool) -> f64 {
        let n = values.len() as f64;
        if n < 4.0 {
            return f64::NAN;
        }

        let mean: f64 = values.iter().sum::<f64>() / n;

        let m2: f64 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let m4: f64 = values.iter().map(|x| (x - mean).powi(4)).sum::<f64>() / n;

        if m2.abs() < 1e-10 {
            return 0.0; // No variation
        }

        // Raw kurtosis
        let kurt = m4 / m2.powi(2);

        // Apply bias correction for sample kurtosis
        let n1 = n - 1.0;
        let n2 = n - 2.0;
        let n3 = n - 3.0;

        let corrected = ((n + 1.0) * n * kurt - 3.0 * n1 * n1) / (n1 * n2 * n3);
        let corrected = corrected + 3.0; // Add back 3 for raw kurtosis

        if excess {
            corrected - 3.0 // Excess kurtosis
        } else {
            corrected // Raw kurtosis
        }
    }

    /// Calculate kurtosis values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();

        if self.use_returns {
            // Need one extra data point for returns calculation
            if n < self.period + 1 || self.period < 4 {
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
                let kurt = Self::calculate_kurtosis(window, self.excess);
                result.push(kurt);
            }

            result
        } else {
            if n < self.period || self.period < 4 {
                return vec![f64::NAN; n];
            }

            let mut result = vec![f64::NAN; self.period - 1];

            for i in (self.period - 1)..n {
                let start = i + 1 - self.period;
                let window = &data[start..=i];
                let kurt = Self::calculate_kurtosis(window, self.excess);
                result.push(kurt);
            }

            result
        }
    }
}

impl TechnicalIndicator for Kurtosis {
    fn name(&self) -> &str {
        "Kurtosis"
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

impl SignalIndicator for Kurtosis {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.close);
        let last = values.last().copied().unwrap_or(f64::NAN);

        if last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // High kurtosis = fat tails = higher risk of extreme moves
        // This is a risk signal, not directional
        // We interpret high kurtosis as bearish (risky) for conservative strategies
        if last > 3.0 {
            Ok(IndicatorSignal::Bearish) // High tail risk
        } else if last < -1.0 {
            Ok(IndicatorSignal::Bullish) // Low tail risk
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.close);
        let signals = values
            .iter()
            .map(|&kurt| {
                if kurt.is_nan() {
                    IndicatorSignal::Neutral
                } else if kurt > 3.0 {
                    IndicatorSignal::Bearish
                } else if kurt < -1.0 {
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
    fn test_kurtosis_normal() {
        let kurt = Kurtosis::from_prices(20);
        // Roughly normal-ish data should have excess kurtosis near 0
        let data: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0)
            .collect();
        let result = kurt.calculate(&data);

        // Check that values are calculated after warmup
        for i in 19..result.len() {
            assert!(!result[i].is_nan());
        }
    }

    #[test]
    fn test_kurtosis_fat_tails() {
        let kurt = Kurtosis::from_prices(10);
        // Data with outliers should have positive excess kurtosis
        let mut data = vec![1.0; 9];
        data.push(100.0); // Extreme outlier
        data.extend(vec![1.0; 5]);

        let result = kurt.calculate(&data);

        // Should have high kurtosis where outlier is in window
        let max_kurt = result.iter().filter(|x| !x.is_nan()).fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        assert!(max_kurt > 0.0, "Expected positive kurtosis with outlier");
    }

    #[test]
    fn test_kurtosis_uniform() {
        let kurt = Kurtosis::from_prices(5);
        // Uniform-ish data should have negative excess kurtosis
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let result = kurt.calculate(&data);

        // Check values exist
        for i in 4..result.len() {
            assert!(!result[i].is_nan());
        }
    }
}
