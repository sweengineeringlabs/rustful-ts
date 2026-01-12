//! Autocorrelation implementation.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Autocorrelation.
///
/// Measures the correlation of a time series with a lagged version of itself.
/// Useful for detecting trends, cycles, and mean reversion characteristics.
///
/// - High positive autocorrelation: trending/momentum
/// - Negative autocorrelation: mean reverting
/// - Near zero: random walk
#[derive(Debug, Clone)]
pub struct Autocorrelation {
    period: usize,
    lag: usize,
}

impl Autocorrelation {
    /// Create a new Autocorrelation indicator with lag 1.
    pub fn new(period: usize) -> Self {
        Self { period, lag: 1 }
    }

    /// Create with custom lag.
    pub fn with_lag(period: usize, lag: usize) -> Self {
        Self { period, lag: lag.max(1) }
    }

    /// Calculate Pearson correlation between two slices.
    fn pearson(x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return f64::NAN;
        }

        let n = x.len() as f64;

        let mean_x: f64 = x.iter().sum::<f64>() / n;
        let mean_y: f64 = y.iter().sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;

        for (xi, yi) in x.iter().zip(y.iter()) {
            let dx = xi - mean_x;
            let dy = yi - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        let denominator = (var_x * var_y).sqrt();
        if denominator.abs() < 1e-10 {
            0.0
        } else {
            cov / denominator
        }
    }

    /// Calculate autocorrelation values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let required = self.period + self.lag;

        if n < required || self.period == 0 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; required - 1];

        for i in (required - 1)..n {
            let start = i + 1 - self.period;
            let series_current = &data[start..=i];
            let series_lagged = &data[(start - self.lag)..=(i - self.lag)];

            let autocorr = Self::pearson(series_current, series_lagged);
            result.push(autocorr);
        }

        result
    }

    /// Calculate autocorrelation for multiple lags.
    pub fn calculate_multi_lag(&self, data: &[f64], max_lag: usize) -> Vec<Vec<f64>> {
        (1..=max_lag)
            .map(|lag| {
                let autocorr = Autocorrelation::with_lag(self.period, lag);
                autocorr.calculate(data)
            })
            .collect()
    }
}

impl TechnicalIndicator for Autocorrelation {
    fn name(&self) -> &str {
        "Autocorrelation"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let required = self.period + self.lag;
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
        self.period + self.lag
    }
}

impl SignalIndicator for Autocorrelation {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.close);
        let last = values.last().copied().unwrap_or(f64::NAN);

        if last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // High positive autocorrelation = trending = follow trend
        // This is a simplification; real trading would need price direction
        if last > 0.5 {
            // Strong positive autocorrelation (momentum)
            // Check if price is trending up or down
            let n = data.close.len();
            if n >= 2 && data.close[n - 1] > data.close[n - 2] {
                Ok(IndicatorSignal::Bullish)
            } else if n >= 2 && data.close[n - 1] < data.close[n - 2] {
                Ok(IndicatorSignal::Bearish)
            } else {
                Ok(IndicatorSignal::Neutral)
            }
        } else if last < -0.3 {
            // Negative autocorrelation = mean reverting
            // Opposite of current direction
            let n = data.close.len();
            if n >= 2 && data.close[n - 1] > data.close[n - 2] {
                Ok(IndicatorSignal::Bearish) // Expect reversal
            } else if n >= 2 && data.close[n - 1] < data.close[n - 2] {
                Ok(IndicatorSignal::Bullish) // Expect reversal
            } else {
                Ok(IndicatorSignal::Neutral)
            }
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.close);

        let signals = values
            .iter()
            .enumerate()
            .map(|(i, &autocorr)| {
                if autocorr.is_nan() || i == 0 {
                    IndicatorSignal::Neutral
                } else if autocorr > 0.5 {
                    // Momentum regime
                    if data.close[i] > data.close[i - 1] {
                        IndicatorSignal::Bullish
                    } else if data.close[i] < data.close[i - 1] {
                        IndicatorSignal::Bearish
                    } else {
                        IndicatorSignal::Neutral
                    }
                } else if autocorr < -0.3 {
                    // Mean reversion regime
                    if data.close[i] > data.close[i - 1] {
                        IndicatorSignal::Bearish
                    } else if data.close[i] < data.close[i - 1] {
                        IndicatorSignal::Bullish
                    } else {
                        IndicatorSignal::Neutral
                    }
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
    fn test_autocorrelation_trending() {
        let autocorr = Autocorrelation::new(10);
        // Strong trend should have high positive autocorrelation
        let data: Vec<f64> = (0..30).map(|i| 100.0 + i as f64).collect();
        let result = autocorr.calculate(&data);

        for i in 10..30 {
            assert!(result[i] > 0.9, "Expected high autocorrelation for trending data");
        }
    }

    #[test]
    fn test_autocorrelation_oscillating() {
        let autocorr = Autocorrelation::new(10);
        // Oscillating data should have negative autocorrelation
        let data: Vec<f64> = (0..30).map(|i| if i % 2 == 0 { 100.0 } else { 110.0 }).collect();
        let result = autocorr.calculate(&data);

        for i in 10..30 {
            assert!(result[i] < 0.0, "Expected negative autocorrelation for oscillating data");
        }
    }

    #[test]
    fn test_autocorrelation_with_lag() {
        let autocorr = Autocorrelation::with_lag(10, 2);
        let data: Vec<f64> = (0..30).map(|i| 100.0 + i as f64).collect();
        let result = autocorr.calculate(&data);

        // Should still have high autocorrelation for trending data
        for i in 11..30 {
            assert!(result[i] > 0.8);
        }
    }
}
