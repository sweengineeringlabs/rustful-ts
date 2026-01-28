//! Extended Filter Indicators
//!
//! Additional filtering and noise reduction indicators.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Exponential Smoothing Filter
#[derive(Debug, Clone)]
pub struct ExponentialSmoothingFilter {
    alpha: f64,
}

impl ExponentialSmoothingFilter {
    pub fn new(alpha: f64) -> Result<Self> {
        if alpha <= 0.0 || alpha > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "alpha".to_string(),
                reason: "must be between 0 and 1".to_string(),
            });
        }
        Ok(Self { alpha })
    }

    /// Calculate exponential smoothing
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n == 0 {
            return vec![];
        }

        let mut result = vec![0.0; n];
        result[0] = data[0];

        for i in 1..n {
            result[i] = self.alpha * data[i] + (1.0 - self.alpha) * result[i - 1];
        }
        result
    }
}

impl TechnicalIndicator for ExponentialSmoothingFilter {
    fn name(&self) -> &str {
        "Exponential Smoothing Filter"
    }

    fn min_periods(&self) -> usize {
        1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Butterworth Low-Pass Filter
#[derive(Debug, Clone)]
pub struct ButterworthFilter {
    period: usize,
}

impl ButterworthFilter {
    pub fn new(period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate 2-pole Butterworth filter
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < 3 {
            return data.to_vec();
        }

        let mut result = vec![0.0; n];

        // Butterworth coefficients
        let pi = std::f64::consts::PI;
        let a = (-2.0_f64).exp().sqrt() * pi / self.period as f64;
        let b = 2.0 * a.cos();
        let c2 = b;
        let c3 = -a.exp().powi(-2);
        let c1 = 1.0 - c2 - c3;

        // Initialize
        result[0] = data[0];
        result[1] = c1 * data[1] + c2 * result[0];

        for i in 2..n {
            result[i] = c1 * data[i] + c2 * result[i - 1] + c3 * result[i - 2];
        }
        result
    }
}

impl TechnicalIndicator for ButterworthFilter {
    fn name(&self) -> &str {
        "Butterworth Filter"
    }

    fn min_periods(&self) -> usize {
        3
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// High-Pass Filter - Removes trend, keeps cycles
#[derive(Debug, Clone)]
pub struct HighPassFilter {
    period: usize,
}

impl HighPassFilter {
    pub fn new(period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate high-pass filter
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < 2 {
            return vec![0.0; n];
        }

        let mut result = vec![0.0; n];

        // High-pass filter coefficients
        let pi = std::f64::consts::PI;
        let alpha = (1.0 + (2.0 * pi / self.period as f64).cos()) /
                    ((2.0 * pi / self.period as f64).sin());
        let alpha = alpha - (alpha.powi(2) - 1.0).sqrt();

        result[0] = 0.0;
        for i in 1..n {
            result[i] = (1.0 - alpha / 2.0) * (data[i] - data[i - 1]) +
                       (1.0 - alpha) * result[i - 1];
        }
        result
    }
}

impl TechnicalIndicator for HighPassFilter {
    fn name(&self) -> &str {
        "High Pass Filter"
    }

    fn min_periods(&self) -> usize {
        2
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Band-Pass Filter - Keeps specific frequency band
#[derive(Debug, Clone)]
pub struct BandPassFilter {
    period: usize,
    bandwidth: f64,
}

impl BandPassFilter {
    pub fn new(period: usize, bandwidth: f64) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if bandwidth <= 0.0 || bandwidth > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "bandwidth".to_string(),
                reason: "must be between 0 and 1".to_string(),
            });
        }
        Ok(Self { period, bandwidth })
    }

    /// Calculate band-pass filter
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < 3 {
            return vec![0.0; n];
        }

        let mut result = vec![0.0; n];

        // Band-pass coefficients
        let pi = std::f64::consts::PI;
        let delta = (2.0 * pi / self.period as f64 * self.bandwidth).cos();
        let beta = (1.0 - delta) / (1.0 + delta);
        let gamma = (1.0 / delta).cos();
        let alpha = (1.0 - beta) / 2.0;

        result[0] = 0.0;
        result[1] = 0.0;
        for i in 2..n {
            result[i] = alpha * (data[i] - data[i - 2]) +
                       gamma * (1.0 + beta) * result[i - 1] -
                       beta * result[i - 2];
        }
        result
    }
}

impl TechnicalIndicator for BandPassFilter {
    fn name(&self) -> &str {
        "Band Pass Filter"
    }

    fn min_periods(&self) -> usize {
        3
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Adaptive Noise Filter - Adapts to market conditions
#[derive(Debug, Clone)]
pub struct AdaptiveNoiseFilter {
    period: usize,
}

impl AdaptiveNoiseFilter {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate adaptive noise filter
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n == 0 {
            return vec![];
        }

        let mut result = vec![0.0; n];
        result[0] = data[0];

        for i in 1..n {
            let start = i.saturating_sub(self.period);

            // Calculate signal and noise
            let mean: f64 = data[start..=i].iter().sum::<f64>() / (i - start + 1) as f64;
            let variance: f64 = data[start..=i].iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / (i - start + 1) as f64;
            let std_dev = variance.sqrt();

            // Calculate noise ratio
            let signal_change = (data[i] - result[i - 1]).abs();
            let noise_ratio = if std_dev > 1e-10 {
                (signal_change / std_dev).min(1.0)
            } else {
                0.5
            };

            // Adaptive alpha: more smoothing when noisy
            let alpha = 0.1 + 0.8 * (1.0 - noise_ratio);
            result[i] = alpha * data[i] + (1.0 - alpha) * result[i - 1];
        }
        result
    }
}

impl TechnicalIndicator for AdaptiveNoiseFilter {
    fn name(&self) -> &str {
        "Adaptive Noise Filter"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Trend Filter - Smooths while following trend
#[derive(Debug, Clone)]
pub struct TrendFilter {
    period: usize,
}

impl TrendFilter {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate trend-following filter
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n == 0 {
            return vec![];
        }

        let mut result = vec![0.0; n];
        result[0] = data[0];

        for i in 1..n {
            let start = i.saturating_sub(self.period);

            // Calculate trend direction
            let trend = if i >= self.period {
                let old_avg = data[start..start + self.period / 2].iter().sum::<f64>() / (self.period / 2) as f64;
                let new_avg = data[i - self.period / 2..=i].iter().sum::<f64>() / (self.period / 2 + 1) as f64;
                (new_avg - old_avg).signum()
            } else {
                0.0
            };

            // Alpha depends on whether price is following trend
            let price_direction = (data[i] - result[i - 1]).signum();
            let alpha = if trend != 0.0 && price_direction == trend {
                0.3 // Faster response when following trend
            } else {
                0.1 // Slower response when counter-trend
            };

            result[i] = alpha * data[i] + (1.0 - alpha) * result[i - 1];
        }
        result
    }
}

impl TechnicalIndicator for TrendFilter {
    fn name(&self) -> &str {
        "Trend Filter"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> Vec<f64> {
        vec![100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0, 106.0, 105.5,
             107.0, 108.0, 107.5, 109.0, 110.0, 109.5, 111.0, 112.0, 111.5, 113.0,
             114.0, 113.5, 115.0, 116.0, 115.5, 117.0, 118.0, 117.5, 119.0, 120.0]
    }

    #[test]
    fn test_exponential_smoothing_filter() {
        let data = make_test_data();
        let esf = ExponentialSmoothingFilter::new(0.3).unwrap();
        let result = esf.calculate(&data);

        assert_eq!(result.len(), data.len());
        assert!(result[15] > 0.0);
    }

    #[test]
    fn test_butterworth_filter() {
        let data = make_test_data();
        let bf = ButterworthFilter::new(10).unwrap();
        let result = bf.calculate(&data);

        assert_eq!(result.len(), data.len());
        assert!(result[15] > 0.0);
    }

    #[test]
    fn test_high_pass_filter() {
        let data = make_test_data();
        let hpf = HighPassFilter::new(10).unwrap();
        let result = hpf.calculate(&data);

        assert_eq!(result.len(), data.len());
    }

    #[test]
    fn test_band_pass_filter() {
        let data = make_test_data();
        let bpf = BandPassFilter::new(10, 0.5).unwrap();
        let result = bpf.calculate(&data);

        assert_eq!(result.len(), data.len());
    }

    #[test]
    fn test_adaptive_noise_filter() {
        let data = make_test_data();
        let anf = AdaptiveNoiseFilter::new(10).unwrap();
        let result = anf.calculate(&data);

        assert_eq!(result.len(), data.len());
        assert!(result[15] > 0.0);
    }

    #[test]
    fn test_trend_filter() {
        let data = make_test_data();
        let tf = TrendFilter::new(10).unwrap();
        let result = tf.calculate(&data);

        assert_eq!(result.len(), data.len());
        assert!(result[15] > 0.0);
    }
}
