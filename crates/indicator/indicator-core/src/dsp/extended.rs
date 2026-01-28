//! Extended DSP Indicators
//!
//! Additional digital signal processing indicators.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Spectral Density - Power spectral density estimate
#[derive(Debug, Clone)]
pub struct SpectralDensity {
    period: usize,
}

impl SpectralDensity {
    pub fn new(period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate spectral density (simple approximation)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate variance of returns as spectral density proxy
            let returns: Vec<f64> = (start + 1..=i)
                .map(|j| close[j] / close[j - 1] - 1.0)
                .collect();

            let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
            let var: f64 = returns.iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>() / returns.len() as f64;

            result[i] = var * 10000.0; // Scale for readability
        }
        result
    }
}

impl TechnicalIndicator for SpectralDensity {
    fn name(&self) -> &str {
        "Spectral Density"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Phase Indicator - Phase of dominant cycle
#[derive(Debug, Clone)]
pub struct PhaseIndicator {
    period: usize,
}

impl PhaseIndicator {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate phase (0-360 degrees)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Simple phase calculation based on position in cycle
            let slice = &close[start..=i];
            let max_val = slice.iter().cloned().fold(f64::MIN, f64::max);
            let min_val = slice.iter().cloned().fold(f64::MAX, f64::min);
            let range = max_val - min_val;

            if range > 1e-10 {
                // Position in range as phase proxy
                let position = (close[i] - min_val) / range;
                // Convert to degrees (0-360)
                result[i] = position * 360.0;
            }
        }
        result
    }
}

impl TechnicalIndicator for PhaseIndicator {
    fn name(&self) -> &str {
        "Phase Indicator"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Instantaneous Frequency - Rate of phase change
#[derive(Debug, Clone)]
pub struct InstantaneousFrequency {
    period: usize,
}

impl InstantaneousFrequency {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate instantaneous frequency
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // Calculate rate of change as frequency proxy
        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Count zero crossings of detrended data
            let mean: f64 = close[start..=i].iter().sum::<f64>() / self.period as f64;
            let mut crossings = 0;

            for j in (start + 1)..=i {
                let prev_dev = close[j - 1] - mean;
                let curr_dev = close[j] - mean;
                if (prev_dev > 0.0 && curr_dev < 0.0) || (prev_dev < 0.0 && curr_dev > 0.0) {
                    crossings += 1;
                }
            }

            // Frequency = half the number of zero crossings per period
            result[i] = crossings as f64 / 2.0;
        }
        result
    }
}

impl TechnicalIndicator for InstantaneousFrequency {
    fn name(&self) -> &str {
        "Instantaneous Frequency"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Adaptive Bandwidth Filter - Bandwidth-adaptive filter
#[derive(Debug, Clone)]
pub struct AdaptiveBandwidthFilter {
    period: usize,
    bandwidth: f64,
}

impl AdaptiveBandwidthFilter {
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

    /// Calculate adaptive bandwidth filter
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // Calculate volatility for adaptive alpha
        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate volatility
            let returns: Vec<f64> = (start + 1..=i)
                .map(|j| (close[j] / close[j - 1]).ln())
                .collect();

            let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
            let var: f64 = returns.iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>() / returns.len() as f64;
            let vol = var.sqrt();

            // Adaptive alpha based on volatility
            let adaptive_alpha = self.bandwidth * (1.0 + vol * 10.0).min(1.0);

            // Apply filter
            if i == self.period {
                result[i] = close[i];
            } else {
                result[i] = adaptive_alpha * close[i] + (1.0 - adaptive_alpha) * result[i - 1];
            }
        }
        result
    }
}

impl TechnicalIndicator for AdaptiveBandwidthFilter {
    fn name(&self) -> &str {
        "Adaptive Bandwidth Filter"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Zero Lag Indicator - Reduced lag smoothing
#[derive(Debug, Clone)]
pub struct ZeroLagIndicator {
    period: usize,
}

impl ZeroLagIndicator {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate zero lag moving average
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        let lag = (self.period - 1) / 2;

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate EMA
            let alpha = 2.0 / (self.period as f64 + 1.0);
            let mut ema = close[start];
            for j in (start + 1)..=i {
                ema = alpha * close[j] + (1.0 - alpha) * ema;
            }

            // Apply lag compensation
            let lag_idx = i.saturating_sub(lag);
            let compensation = if lag_idx > start {
                close[i] - close[lag_idx]
            } else {
                0.0
            };

            result[i] = ema + compensation * alpha;
        }
        result
    }
}

impl TechnicalIndicator for ZeroLagIndicator {
    fn name(&self) -> &str {
        "Zero Lag Indicator"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Signal to Noise Ratio - Measures trend vs noise
#[derive(Debug, Clone)]
pub struct SignalToNoiseRatio {
    period: usize,
}

impl SignalToNoiseRatio {
    pub fn new(period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate SNR in dB
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Signal: trend component (linear regression)
            let x_vals: Vec<f64> = (0..=self.period).map(|x| x as f64).collect();
            let y_vals: Vec<f64> = close[start..=i].to_vec();

            let n_points = x_vals.len() as f64;
            let sum_x: f64 = x_vals.iter().sum();
            let sum_y: f64 = y_vals.iter().sum();
            let sum_xy: f64 = x_vals.iter().zip(y_vals.iter()).map(|(x, y)| x * y).sum();
            let sum_xx: f64 = x_vals.iter().map(|x| x * x).sum();

            let slope = (n_points * sum_xy - sum_x * sum_y) / (n_points * sum_xx - sum_x * sum_x);
            let intercept = (sum_y - slope * sum_x) / n_points;

            // Signal power (trend variance)
            let mut signal_var = 0.0;
            let mut noise_var = 0.0;
            let mean_y = sum_y / n_points;

            for (j, &y) in y_vals.iter().enumerate() {
                let pred = intercept + slope * j as f64;
                let residual = y - pred;
                signal_var += (pred - mean_y).powi(2);
                noise_var += residual.powi(2);
            }

            signal_var /= n_points;
            noise_var /= n_points;

            // SNR in dB
            if noise_var > 1e-10 {
                result[i] = 10.0 * (signal_var / noise_var).log10();
            }
        }
        result
    }
}

impl TechnicalIndicator for SignalToNoiseRatio {
    fn name(&self) -> &str {
        "Signal to Noise Ratio"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
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
    fn test_spectral_density() {
        let close = make_test_data();
        let sd = SpectralDensity::new(10).unwrap();
        let result = sd.calculate(&close);

        assert_eq!(result.len(), close.len());
        assert!(result[15] >= 0.0);
    }

    #[test]
    fn test_phase_indicator() {
        let close = make_test_data();
        let pi = PhaseIndicator::new(10).unwrap();
        let result = pi.calculate(&close);

        assert_eq!(result.len(), close.len());
        assert!(result[15] >= 0.0 && result[15] <= 360.0);
    }

    #[test]
    fn test_instantaneous_frequency() {
        let close = make_test_data();
        let if_ind = InstantaneousFrequency::new(10).unwrap();
        let result = if_ind.calculate(&close);

        assert_eq!(result.len(), close.len());
        assert!(result[15] >= 0.0);
    }

    #[test]
    fn test_adaptive_bandwidth_filter() {
        let close = make_test_data();
        let abf = AdaptiveBandwidthFilter::new(10, 0.5).unwrap();
        let result = abf.calculate(&close);

        assert_eq!(result.len(), close.len());
        assert!(result[15] > 0.0);
    }

    #[test]
    fn test_zero_lag_indicator() {
        let close = make_test_data();
        let zl = ZeroLagIndicator::new(10).unwrap();
        let result = zl.calculate(&close);

        assert_eq!(result.len(), close.len());
        assert!(result[15] > 0.0);
    }

    #[test]
    fn test_signal_to_noise_ratio() {
        let close = make_test_data();
        let snr = SignalToNoiseRatio::new(10).unwrap();
        let result = snr.calculate(&close);

        assert_eq!(result.len(), close.len());
    }
}
