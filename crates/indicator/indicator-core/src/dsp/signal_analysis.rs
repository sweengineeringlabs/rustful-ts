//! DSP Signal Analysis Indicators
//!
//! Signal processing based analysis indicators.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Auto Correlation Period - Estimates dominant cycle length via autocorrelation
#[derive(Debug, Clone)]
pub struct AutoCorrelationPeriod {
    min_period: usize,
    max_period: usize,
}

impl AutoCorrelationPeriod {
    pub fn new(min_period: usize, max_period: usize) -> Result<Self> {
        if min_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "min_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if max_period <= min_period {
            return Err(IndicatorError::InvalidParameter {
                name: "max_period".to_string(),
                reason: "must be greater than min_period".to_string(),
            });
        }
        Ok(Self { min_period, max_period })
    }

    /// Estimate dominant cycle period using autocorrelation
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.max_period..n {
            let start = i.saturating_sub(self.max_period);
            let slice = &close[start..=i];
            let len = slice.len();

            // Calculate mean
            let mean: f64 = slice.iter().sum::<f64>() / len as f64;

            // Find period with highest autocorrelation
            let mut best_period = self.min_period;
            let mut best_corr = f64::NEG_INFINITY;

            for period in self.min_period..=self.max_period.min(len / 2) {
                let mut num = 0.0;
                let mut denom1 = 0.0;
                let mut denom2 = 0.0;

                for j in 0..(len - period) {
                    let x = slice[j] - mean;
                    let y = slice[j + period] - mean;
                    num += x * y;
                    denom1 += x * x;
                    denom2 += y * y;
                }

                let denom = (denom1 * denom2).sqrt();
                let corr = if denom > 1e-10 { num / denom } else { 0.0 };

                if corr > best_corr {
                    best_corr = corr;
                    best_period = period;
                }
            }

            result[i] = best_period as f64;
        }

        result
    }
}

impl TechnicalIndicator for AutoCorrelationPeriod {
    fn name(&self) -> &str {
        "Dominant Cycle Period"
    }

    fn min_periods(&self) -> usize {
        self.max_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Trend Strength via FFT approximation - Frequency domain analysis
#[derive(Debug, Clone)]
pub struct TrendStrengthFFT {
    period: usize,
}

impl TrendStrengthFFT {
    pub fn new(period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate trend strength using spectral analysis approximation
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let slice = &close[start..=i];

            // Calculate variance (total power)
            let mean: f64 = slice.iter().sum::<f64>() / slice.len() as f64;
            let total_variance: f64 = slice.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / slice.len() as f64;

            // Calculate trend component (low frequency power approximation)
            // Using linear regression residuals
            let len = slice.len() as f64;
            let x_mean = (len - 1.0) / 2.0;
            let y_mean = mean;

            let mut cov = 0.0;
            let mut var_x = 0.0;
            for (j, &val) in slice.iter().enumerate() {
                let x = j as f64 - x_mean;
                let y = val - y_mean;
                cov += x * y;
                var_x += x * x;
            }

            let slope = if var_x > 1e-10 { cov / var_x } else { 0.0 };
            let intercept = y_mean - slope * x_mean;

            // Trend variance (explained by linear trend)
            let mut trend_variance = 0.0;
            for j in 0..slice.len() {
                let predicted = intercept + slope * j as f64;
                trend_variance += (predicted - mean).powi(2);
            }
            trend_variance /= slice.len() as f64;

            // Trend strength: ratio of trend power to total power
            if total_variance > 1e-10 {
                result[i] = (trend_variance / total_variance).sqrt() * 100.0;
            }
        }

        result
    }
}

impl TechnicalIndicator for TrendStrengthFFT {
    fn name(&self) -> &str {
        "Trend Strength FFT"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Cycle Amplitude - Measures the amplitude of price cycles
#[derive(Debug, Clone)]
pub struct CycleDeviationAmplitude {
    period: usize,
}

impl CycleDeviationAmplitude {
    pub fn new(period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate cycle amplitude (deviation from trend)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate linear trend
            let len = (i - start + 1) as f64;
            let x_mean = (len - 1.0) / 2.0;
            let y_mean: f64 = close[start..=i].iter().sum::<f64>() / len;

            let mut cov = 0.0;
            let mut var_x = 0.0;
            for (j, &val) in close[start..=i].iter().enumerate() {
                let x = j as f64 - x_mean;
                let y = val - y_mean;
                cov += x * y;
                var_x += x * x;
            }

            let slope = if var_x > 1e-10 { cov / var_x } else { 0.0 };
            let intercept = y_mean - slope * x_mean;

            // Calculate amplitude (max deviation from trend)
            let mut max_dev = 0.0;
            for (j, &val) in close[start..=i].iter().enumerate() {
                let trend = intercept + slope * j as f64;
                let dev = (val - trend).abs();
                if dev > max_dev {
                    max_dev = dev;
                }
            }

            // Normalize by average price
            result[i] = max_dev / y_mean * 100.0;
        }

        result
    }
}

impl TechnicalIndicator for CycleDeviationAmplitude {
    fn name(&self) -> &str {
        "Cycle Amplitude"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Phase Accumulator - Tracks cumulative phase angle
#[derive(Debug, Clone)]
pub struct PhaseAccumulator {
    period: usize,
}

impl PhaseAccumulator {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate phase accumulator (cycle position)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];
        let mut phase = 0.0;

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Estimate instantaneous frequency from zero crossings
            let mean: f64 = close[start..=i].iter().sum::<f64>() / self.period as f64;
            let mut crossings = 0;

            for j in (start + 1)..=i {
                let prev_sign = (close[j - 1] - mean).signum();
                let curr_sign = (close[j] - mean).signum();
                if prev_sign != curr_sign && prev_sign != 0.0 && curr_sign != 0.0 {
                    crossings += 1;
                }
            }

            // Each full cycle has 2 crossings
            let frequency = crossings as f64 / (2.0 * self.period as f64);
            let delta_phase = 2.0 * std::f64::consts::PI * frequency;

            phase += delta_phase;
            // Normalize to 0-360 degrees
            result[i] = (phase % (2.0 * std::f64::consts::PI)) * 180.0 / std::f64::consts::PI;
        }

        result
    }
}

impl TechnicalIndicator for PhaseAccumulator {
    fn name(&self) -> &str {
        "Phase Accumulator"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Spectral Noise Ratio - Signal quality measurement
#[derive(Debug, Clone)]
pub struct SpectralNoiseRatio {
    period: usize,
}

impl SpectralNoiseRatio {
    pub fn new(period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate spectral noise ratio (signal vs noise)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let slice = &close[start..=i];

            // Signal power: variance of smoothed series
            let mut smoothed = vec![0.0; slice.len()];
            let smooth_period = 3.min(slice.len());
            for j in smooth_period..slice.len() {
                smoothed[j] = slice[(j - smooth_period + 1)..=j].iter().sum::<f64>() / smooth_period as f64;
            }

            let smooth_mean: f64 = smoothed.iter().sum::<f64>() / smoothed.len() as f64;
            let signal_power: f64 = smoothed.iter()
                .map(|x| (x - smooth_mean).powi(2))
                .sum::<f64>() / smoothed.len() as f64;

            // Noise power: variance of residuals (original - smoothed)
            let mut noise_power = 0.0;
            for j in smooth_period..slice.len() {
                noise_power += (slice[j] - smoothed[j]).powi(2);
            }
            noise_power /= (slice.len() - smooth_period) as f64;

            // SNR in decibels (simplified)
            if noise_power > 1e-10 {
                result[i] = (signal_power / noise_power).sqrt() * 10.0;
            }
        }

        result
    }
}

impl TechnicalIndicator for SpectralNoiseRatio {
    fn name(&self) -> &str {
        "Spectral Noise Ratio"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Adaptive Cycle Filter - Filters based on detected cycle
#[derive(Debug, Clone)]
pub struct AdaptiveCycleFilter {
    min_period: usize,
    max_period: usize,
}

impl AdaptiveCycleFilter {
    pub fn new(min_period: usize, max_period: usize) -> Result<Self> {
        if min_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "min_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if max_period <= min_period {
            return Err(IndicatorError::InvalidParameter {
                name: "max_period".to_string(),
                reason: "must be greater than min_period".to_string(),
            });
        }
        Ok(Self { min_period, max_period })
    }

    /// Apply adaptive cycle-based filtering
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.max_period..n {
            let start = i.saturating_sub(self.max_period);
            let slice = &close[start..=i];
            let len = slice.len();

            // Detect dominant cycle period (simplified)
            let mean: f64 = slice.iter().sum::<f64>() / len as f64;
            let mut best_period = self.min_period;
            let mut best_corr = 0.0;

            for period in self.min_period..=self.max_period.min(len / 2) {
                let mut corr = 0.0;
                for j in 0..(len - period) {
                    corr += (slice[j] - mean) * (slice[j + period] - mean);
                }
                if corr > best_corr {
                    best_corr = corr;
                    best_period = period;
                }
            }

            // Apply SMA with detected period
            let filter_period = best_period.min(i - start + 1);
            let filter_start = (i - filter_period + 1).max(start);
            result[i] = close[filter_start..=i].iter().sum::<f64>() / (i - filter_start + 1) as f64;
        }

        result
    }
}

impl TechnicalIndicator for AdaptiveCycleFilter {
    fn name(&self) -> &str {
        "Adaptive Cycle Filter"
    }

    fn min_periods(&self) -> usize {
        self.max_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> OHLCVSeries {
        let close: Vec<f64> = (0..60).map(|i| 100.0 + (i as f64) * 0.2 + (i as f64 * 0.3).sin() * 3.0).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
        let open: Vec<f64> = close.clone();
        let volume: Vec<f64> = vec![1000.0; 60];

        OHLCVSeries { open, high, low, close, volume }
    }

    #[test]
    fn test_dominant_cycle_period() {
        let data = make_test_data();
        let dcp = AutoCorrelationPeriod::new(5, 30).unwrap();
        let result = dcp.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // Period should be within range
        for i in 35..result.len() {
            if result[i] > 0.0 {
                assert!(result[i] >= 5.0 && result[i] <= 30.0);
            }
        }
    }

    #[test]
    fn test_trend_strength_fft() {
        let data = make_test_data();
        let tsf = TrendStrengthFFT::new(20).unwrap();
        let result = tsf.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // Strength should be 0-100
        for i in 25..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_cycle_amplitude() {
        let data = make_test_data();
        let ca = CycleDeviationAmplitude::new(20).unwrap();
        let result = ca.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // Amplitude should be non-negative
        for i in 25..result.len() {
            assert!(result[i] >= 0.0);
        }
    }

    #[test]
    fn test_phase_accumulator() {
        let data = make_test_data();
        let pa = PhaseAccumulator::new(10).unwrap();
        let result = pa.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // Phase should be 0-360
        for i in 15..result.len() {
            assert!(result[i] >= 0.0 && result[i] < 360.0);
        }
    }

    #[test]
    fn test_spectral_noise_ratio() {
        let data = make_test_data();
        let snr = SpectralNoiseRatio::new(20).unwrap();
        let result = snr.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // SNR should be non-negative
        for i in 25..result.len() {
            assert!(result[i] >= 0.0);
        }
    }

    #[test]
    fn test_adaptive_cycle_filter() {
        let data = make_test_data();
        let acf = AdaptiveCycleFilter::new(5, 20).unwrap();
        let result = acf.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
    }

    #[test]
    fn test_validation() {
        assert!(AutoCorrelationPeriod::new(2, 20).is_err());
        assert!(AutoCorrelationPeriod::new(10, 5).is_err()); // max <= min
        assert!(TrendStrengthFFT::new(5).is_err());
        assert!(CycleDeviationAmplitude::new(5).is_err());
        assert!(PhaseAccumulator::new(2).is_err());
        assert!(SpectralNoiseRatio::new(5).is_err());
        assert!(AdaptiveCycleFilter::new(2, 20).is_err());
    }
}
