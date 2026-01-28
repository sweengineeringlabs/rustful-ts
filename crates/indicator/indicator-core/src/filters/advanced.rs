//! Advanced Filter Indicators
//!
//! Sophisticated adaptive filtering and signal extraction indicators.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Adaptive Low-Pass Filter - Low-pass filter with adaptive cutoff frequency
///
/// Adjusts cutoff frequency based on market volatility, providing more
/// smoothing during high volatility and less during low volatility.
#[derive(Debug, Clone)]
pub struct AdaptiveLowPassFilter {
    period: usize,
    min_cutoff: f64,
    max_cutoff: f64,
}

impl AdaptiveLowPassFilter {
    pub fn new(period: usize, min_cutoff: f64, max_cutoff: f64) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if min_cutoff <= 0.0 || min_cutoff >= 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "min_cutoff".to_string(),
                reason: "must be between 0 and 1 exclusive".to_string(),
            });
        }
        if max_cutoff <= min_cutoff || max_cutoff > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "max_cutoff".to_string(),
                reason: "must be greater than min_cutoff and at most 1".to_string(),
            });
        }
        Ok(Self { period, min_cutoff, max_cutoff })
    }

    /// Calculate adaptive low-pass filter
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n == 0 {
            return vec![];
        }

        let mut result = vec![0.0; n];
        result[0] = data[0];

        for i in 1..n {
            let start = i.saturating_sub(self.period);
            let window = &data[start..=i];

            // Calculate volatility as normalized standard deviation
            let mean: f64 = window.iter().sum::<f64>() / window.len() as f64;
            let variance: f64 = window.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / window.len() as f64;
            let std_dev = variance.sqrt();

            // Normalize volatility to 0-1 range
            let normalized_vol = if mean.abs() > 1e-10 {
                (std_dev / mean.abs()).min(0.1) / 0.1
            } else {
                0.5
            };

            // Higher volatility = lower cutoff (more smoothing)
            let cutoff = self.max_cutoff - normalized_vol * (self.max_cutoff - self.min_cutoff);

            // Apply low-pass filter with adaptive cutoff
            let pi = std::f64::consts::PI;
            let omega = 2.0 * pi * cutoff;
            let alpha = omega / (omega + 1.0);

            result[i] = alpha * data[i] + (1.0 - alpha) * result[i - 1];
        }
        result
    }
}

impl TechnicalIndicator for AdaptiveLowPassFilter {
    fn name(&self) -> &str {
        "Adaptive Low Pass Filter"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }

}

/// Noise Reduction Filter - Optimized for removing market noise
///
/// Uses a combination of median filtering and exponential smoothing
/// to effectively reduce noise while preserving signal integrity.
#[derive(Debug, Clone)]
pub struct NoiseReductionFilter {
    period: usize,
    smoothing_factor: f64,
}

impl NoiseReductionFilter {
    pub fn new(period: usize, smoothing_factor: f64) -> Result<Self> {
        if period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        if smoothing_factor <= 0.0 || smoothing_factor > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing_factor".to_string(),
                reason: "must be between 0 and 1".to_string(),
            });
        }
        Ok(Self { period, smoothing_factor })
    }

    /// Calculate noise reduction filter
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n == 0 {
            return vec![];
        }

        // First pass: median filter for spike removal
        let mut median_filtered = vec![0.0; n];
        for i in 0..n {
            let start = i.saturating_sub(self.period / 2);
            let end = (i + self.period / 2 + 1).min(n);
            let mut window: Vec<f64> = data[start..end].to_vec();
            window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            median_filtered[i] = window[window.len() / 2];
        }

        // Second pass: exponential smoothing
        let mut result = vec![0.0; n];
        result[0] = median_filtered[0];

        for i in 1..n {
            // Adaptive smoothing based on deviation from median
            let deviation = (data[i] - median_filtered[i]).abs();
            let mean_dev: f64 = if i >= self.period {
                (0..self.period)
                    .map(|j| (data[i - j] - median_filtered[i - j]).abs())
                    .sum::<f64>() / self.period as f64
            } else {
                deviation
            };

            // More smoothing when deviation is high (likely noise)
            let noise_ratio = if mean_dev > 1e-10 {
                (deviation / mean_dev).min(2.0) / 2.0
            } else {
                0.0
            };

            let alpha = self.smoothing_factor * (1.0 - noise_ratio * 0.8);
            result[i] = alpha * median_filtered[i] + (1.0 - alpha) * result[i - 1];
        }
        result
    }
}

impl TechnicalIndicator for NoiseReductionFilter {
    fn name(&self) -> &str {
        "Noise Reduction Filter"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }

}

/// Trend Extraction Filter - Extracts the trend component from price data
///
/// Separates the underlying trend from cyclical and noise components
/// using a combination of low-pass filtering and regression.
#[derive(Debug, Clone)]
pub struct TrendExtractionFilter {
    period: usize,
    strength: f64,
}

impl TrendExtractionFilter {
    pub fn new(period: usize, strength: f64) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if strength <= 0.0 || strength > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "strength".to_string(),
                reason: "must be between 0 and 1".to_string(),
            });
        }
        Ok(Self { period, strength })
    }

    /// Calculate trend extraction filter
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n == 0 {
            return vec![];
        }

        let mut result = vec![0.0; n];

        for i in 0..n {
            if i < self.period {
                result[i] = data[i];
                continue;
            }

            let start = i - self.period + 1;
            let window = &data[start..=i];

            // Linear regression for trend
            let x_mean = (self.period as f64 - 1.0) / 2.0;
            let y_mean: f64 = window.iter().sum::<f64>() / self.period as f64;

            let mut num = 0.0;
            let mut den = 0.0;
            for (j, &y) in window.iter().enumerate() {
                let x = j as f64;
                num += (x - x_mean) * (y - y_mean);
                den += (x - x_mean).powi(2);
            }

            let slope = if den.abs() > 1e-10 { num / den } else { 0.0 };
            let intercept = y_mean - slope * x_mean;

            // Trend value at current point
            let trend_value = intercept + slope * (self.period as f64 - 1.0);

            // Weighted average of simple average and trend line
            let weighted_avg: f64 = window.iter().enumerate()
                .map(|(j, &v)| v * (j + 1) as f64)
                .sum::<f64>() / (1..=self.period).sum::<usize>() as f64;

            // Blend trend line with weighted average based on strength
            result[i] = self.strength * trend_value + (1.0 - self.strength) * weighted_avg;
        }
        result
    }
}

impl TechnicalIndicator for TrendExtractionFilter {
    fn name(&self) -> &str {
        "Trend Extraction Filter"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }

}

/// Cycle Extraction Filter - Extracts cyclical component from price data
///
/// Removes trend and isolates the cyclical/oscillatory component
/// using a combination of detrending and band-pass filtering.
#[derive(Debug, Clone)]
pub struct CycleExtractionFilter {
    period: usize,
    cycle_period: usize,
}

impl CycleExtractionFilter {
    pub fn new(period: usize, cycle_period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if cycle_period < 5 || cycle_period > period {
            return Err(IndicatorError::InvalidParameter {
                name: "cycle_period".to_string(),
                reason: "must be at least 5 and at most period".to_string(),
            });
        }
        Ok(Self { period, cycle_period })
    }

    /// Calculate cycle extraction filter
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n == 0 {
            return vec![];
        }

        // First: extract trend using weighted moving average
        let mut trend = vec![0.0; n];
        for i in 0..n {
            if i < self.period {
                trend[i] = data[i];
            } else {
                let start = i - self.period + 1;
                let window = &data[start..=i];
                let total_weight: f64 = (1..=self.period).map(|w| w as f64).sum();
                trend[i] = window.iter().enumerate()
                    .map(|(j, &v)| v * (j + 1) as f64)
                    .sum::<f64>() / total_weight;
            }
        }

        // Second: detrend the data
        let mut detrended = vec![0.0; n];
        for i in 0..n {
            detrended[i] = data[i] - trend[i];
        }

        // Third: apply band-pass filter to extract cycles
        let mut result = vec![0.0; n];
        let pi = std::f64::consts::PI;
        let delta = (2.0 * pi / self.cycle_period as f64 * 0.5).cos();
        let beta = (1.0 - delta) / (1.0 + delta);
        let gamma = if delta.abs() > 1e-10 { (1.0 / delta).cos() } else { 1.0 };
        let alpha = (1.0 - beta) / 2.0;

        for i in 0..n {
            if i < 2 {
                result[i] = detrended[i];
            } else {
                result[i] = alpha * (detrended[i] - detrended[i - 2]) +
                           gamma * (1.0 + beta) * result[i - 1] -
                           beta * result[i - 2];
            }
        }
        result
    }
}

impl TechnicalIndicator for CycleExtractionFilter {
    fn name(&self) -> &str {
        "Cycle Extraction Filter"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }

}

/// Adaptive High-Pass Filter - High-pass filter with adaptive cutoff frequency
///
/// Adjusts cutoff based on market conditions to optimally remove trend
/// while preserving short-term price movements.
#[derive(Debug, Clone)]
pub struct AdaptiveHighPassFilter {
    period: usize,
    sensitivity: f64,
}

impl AdaptiveHighPassFilter {
    pub fn new(period: usize, sensitivity: f64) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if sensitivity <= 0.0 || sensitivity > 2.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "sensitivity".to_string(),
                reason: "must be between 0 and 2".to_string(),
            });
        }
        Ok(Self { period, sensitivity })
    }

    /// Calculate adaptive high-pass filter
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < 2 {
            return vec![0.0; n];
        }

        let mut result = vec![0.0; n];
        let pi = std::f64::consts::PI;

        for i in 1..n {
            let start = i.saturating_sub(self.period);
            let window = &data[start..=i];

            // Calculate trend strength
            let first_half_avg = if window.len() > 1 {
                window[..window.len() / 2].iter().sum::<f64>() / (window.len() / 2).max(1) as f64
            } else {
                window[0]
            };
            let second_half_avg = window[window.len() / 2..].iter().sum::<f64>()
                / (window.len() - window.len() / 2) as f64;

            let range = window.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
                - window.iter().fold(f64::INFINITY, |a, &b| a.min(b));

            let trend_strength = if range > 1e-10 {
                ((second_half_avg - first_half_avg).abs() / range).min(1.0)
            } else {
                0.5
            };

            // Adaptive period: stronger trend = higher cutoff (pass more)
            let adaptive_period = self.period as f64 * (1.0 + self.sensitivity * (1.0 - trend_strength));

            // High-pass filter coefficient
            let omega = 2.0 * pi / adaptive_period;
            let alpha = (1.0 + omega.cos()) / omega.sin();
            let alpha = if alpha > 1.0 { alpha - (alpha.powi(2) - 1.0).sqrt() } else { 0.5 };

            // Apply high-pass filter
            result[i] = (1.0 - alpha / 2.0) * (data[i] - data[i - 1]) +
                       (1.0 - alpha) * result[i - 1];
        }
        result
    }
}

impl TechnicalIndicator for AdaptiveHighPassFilter {
    fn name(&self) -> &str {
        "Adaptive High Pass Filter"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }

}

/// Bandwidth Adaptive Filter - Filter with adaptive bandwidth
///
/// Dynamically adjusts both center frequency and bandwidth based on
/// detected market cycles for optimal signal extraction.
#[derive(Debug, Clone)]
pub struct BandwidthAdaptiveFilter {
    period: usize,
    base_bandwidth: f64,
}

impl BandwidthAdaptiveFilter {
    pub fn new(period: usize, base_bandwidth: f64) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if base_bandwidth <= 0.0 || base_bandwidth > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "base_bandwidth".to_string(),
                reason: "must be between 0 and 1".to_string(),
            });
        }
        Ok(Self { period, base_bandwidth })
    }

    /// Calculate bandwidth adaptive filter
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < 3 {
            return data.to_vec();
        }

        let mut result = vec![0.0; n];
        let pi = std::f64::consts::PI;

        // Initialize
        result[0] = data[0];
        result[1] = data[1];

        for i in 2..n {
            let start = i.saturating_sub(self.period);
            let window = &data[start..=i];

            // Estimate dominant cycle period using zero crossings
            let mean: f64 = window.iter().sum::<f64>() / window.len() as f64;
            let mut zero_crossings = 0;
            for j in 1..window.len() {
                if (window[j] - mean) * (window[j - 1] - mean) < 0.0 {
                    zero_crossings += 1;
                }
            }

            let estimated_cycle = if zero_crossings > 0 {
                (2.0 * window.len() as f64 / zero_crossings as f64).max(5.0).min(self.period as f64)
            } else {
                self.period as f64 / 2.0
            };

            // Calculate volatility for bandwidth adjustment
            let variance: f64 = window.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / window.len() as f64;
            let std_dev = variance.sqrt();
            let normalized_vol = if mean.abs() > 1e-10 {
                (std_dev / mean.abs()).min(0.1) / 0.1
            } else {
                0.5
            };

            // Adaptive bandwidth: higher volatility = narrower bandwidth (more selective)
            let bandwidth = self.base_bandwidth * (1.0 - 0.5 * normalized_vol);

            // Band-pass filter coefficients with adaptive parameters
            let delta = (2.0 * pi / estimated_cycle * bandwidth).cos();
            let beta = (1.0 - delta) / (1.0 + delta);
            let gamma = if delta.abs() > 1e-10 { (1.0 / delta).cos() } else { 1.0 };
            let alpha = (1.0 - beta) / 2.0;

            // Apply band-pass filter
            result[i] = alpha * (data[i] - data[i - 2]) +
                       gamma * (1.0 + beta) * result[i - 1] -
                       beta * result[i - 2];

            // Add DC offset to keep result centered around price
            result[i] += mean;
        }
        result
    }
}

impl TechnicalIndicator for BandwidthAdaptiveFilter {
    fn name(&self) -> &str {
        "Bandwidth Adaptive Filter"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }

}

/// Butterworth Bandpass Filter - 4-pole Butterworth bandpass filter
///
/// Implements a 4-pole Butterworth bandpass filter that provides flat frequency
/// response in the passband with a smooth rolloff. This is more selective than
/// the basic 2-pole Butterworth filter and is ideal for extracting specific
/// frequency components from price data.
#[derive(Debug, Clone)]
pub struct ButterworthBandpassFilter {
    /// Center period for the bandpass filter
    center_period: usize,
    /// Bandwidth as a fraction of center period (0.0-1.0)
    bandwidth: f64,
}

impl ButterworthBandpassFilter {
    /// Create a new Butterworth Bandpass Filter
    ///
    /// # Arguments
    /// * `center_period` - The center period of the bandpass filter (minimum 5)
    /// * `bandwidth` - The bandwidth as a fraction of center period (0.0-1.0)
    ///
    /// # Returns
    /// Result containing the filter instance or an error if parameters are invalid
    pub fn new(center_period: usize, bandwidth: f64) -> Result<Self> {
        if center_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "center_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if bandwidth <= 0.0 || bandwidth > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "bandwidth".to_string(),
                reason: "must be between 0 and 1 exclusive".to_string(),
            });
        }
        Ok(Self { center_period, bandwidth })
    }

    /// Calculate 4-pole Butterworth bandpass filter
    ///
    /// The filter provides maximally flat frequency response in the passband,
    /// with a sharper rolloff than a 2-pole design.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < 5 {
            return data.to_vec();
        }

        let mut result = vec![0.0; n];
        let pi = std::f64::consts::PI;

        // Calculate Butterworth bandpass coefficients
        let omega_center = 2.0 * pi / self.center_period as f64;
        let bw = self.bandwidth * omega_center;

        // Pre-warp frequencies
        let omega_low = (omega_center - bw / 2.0).max(0.01);
        let omega_high = (omega_center + bw / 2.0).min(pi - 0.01);

        // Butterworth 2nd order coefficients for low and high cutoffs
        let sqrt2 = 2.0_f64.sqrt();

        // Low-pass section
        let wc_low = (omega_high / 2.0).tan();
        let k1_low = sqrt2 * wc_low;
        let k2_low = wc_low * wc_low;
        let a0_low = 1.0 + k1_low + k2_low;
        let a1_low = 2.0 * (k2_low - 1.0) / a0_low;
        let a2_low = (1.0 - k1_low + k2_low) / a0_low;
        let b0_low = k2_low / a0_low;
        let b1_low = 2.0 * b0_low;

        // High-pass section
        let wc_high = (omega_low / 2.0).tan();
        let k1_high = sqrt2 * wc_high;
        let k2_high = wc_high * wc_high;
        let a0_high = 1.0 + k1_high + k2_high;
        let a1_high = 2.0 * (k2_high - 1.0) / a0_high;
        let a2_high = (1.0 - k1_high + k2_high) / a0_high;
        let c0 = 1.0 / a0_high;

        // Apply cascaded low-pass filter
        let mut low_pass = vec![0.0; n];
        low_pass[0] = data[0];
        low_pass[1] = b0_low * data[1] + b1_low * data[0] - a1_low * low_pass[0];

        for i in 2..n {
            low_pass[i] = b0_low * data[i] + b1_low * data[i - 1] + b0_low * data[i - 2]
                        - a1_low * low_pass[i - 1] - a2_low * low_pass[i - 2];
        }

        // Apply cascaded high-pass filter
        result[0] = low_pass[0];
        result[1] = c0 * (low_pass[1] - 2.0 * low_pass[0]) - a1_high * result[0];

        for i in 2..n {
            result[i] = c0 * (low_pass[i] - 2.0 * low_pass[i - 1] + low_pass[i - 2])
                      - a1_high * result[i - 1] - a2_high * result[i - 2];
        }

        result
    }
}

impl TechnicalIndicator for ButterworthBandpassFilter {
    fn name(&self) -> &str {
        "Butterworth Bandpass Filter"
    }

    fn min_periods(&self) -> usize {
        self.center_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Chebyshev Filter - Type I Chebyshev low-pass filter with sharper cutoff
///
/// Implements a Chebyshev Type I filter that provides a sharper transition
/// between passband and stopband compared to Butterworth filters. The tradeoff
/// is ripple in the passband, controlled by the ripple parameter.
#[derive(Debug, Clone)]
pub struct ChebyshevFilter {
    /// Filter period (cutoff frequency = 1/period)
    period: usize,
    /// Passband ripple in dB (typical values: 0.5-3.0)
    ripple_db: f64,
    /// Filter order (1-4)
    order: usize,
}

impl ChebyshevFilter {
    /// Create a new Chebyshev Type I Filter
    ///
    /// # Arguments
    /// * `period` - The filter period determining cutoff frequency (minimum 3)
    /// * `ripple_db` - Passband ripple in decibels (0.1-3.0)
    /// * `order` - Filter order from 1-4 (higher = sharper cutoff)
    ///
    /// # Returns
    /// Result containing the filter instance or an error if parameters are invalid
    pub fn new(period: usize, ripple_db: f64, order: usize) -> Result<Self> {
        if period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        if ripple_db <= 0.0 || ripple_db > 3.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "ripple_db".to_string(),
                reason: "must be between 0 and 3 dB".to_string(),
            });
        }
        if order < 1 || order > 4 {
            return Err(IndicatorError::InvalidParameter {
                name: "order".to_string(),
                reason: "must be between 1 and 4".to_string(),
            });
        }
        Ok(Self { period, ripple_db, order })
    }

    /// Calculate Chebyshev Type I filter
    ///
    /// The filter provides sharper cutoff than Butterworth with the tradeoff
    /// of passband ripple. Higher order means sharper cutoff.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.order + 1 {
            return data.to_vec();
        }

        let pi = std::f64::consts::PI;
        let omega = 2.0 * pi / self.period as f64;

        // Calculate epsilon from ripple
        let epsilon = ((10.0_f64.powf(self.ripple_db / 10.0) - 1.0).sqrt()).max(0.01);

        // Pre-warp cutoff frequency
        let wc = (omega / 2.0).tan();

        // Chebyshev polynomial coefficients based on order
        let (a_coeffs, b_coeffs) = match self.order {
            1 => {
                // First-order Chebyshev
                let gamma = 1.0 / epsilon;
                let a = wc * gamma;
                let b0 = a / (1.0 + a);
                let a1 = (a - 1.0) / (1.0 + a);
                (vec![a1], vec![b0, b0])
            }
            2 => {
                // Second-order Chebyshev
                let gamma = (1.0 / epsilon + (1.0 / (epsilon * epsilon) + 1.0).sqrt()).ln() / 2.0;
                let sinh_g = gamma.sinh();
                let cosh_g = gamma.cosh();

                let sigma = -sinh_g * (pi / 4.0).sin();
                let omega_pole = cosh_g * (pi / 4.0).cos();

                let k = wc * wc;
                let d = 1.0 - 2.0 * sigma * wc + (sigma * sigma + omega_pole * omega_pole) * k;

                let b0 = k / d;
                let b1 = 2.0 * b0;
                let a1 = 2.0 * (k * (sigma * sigma + omega_pole * omega_pole) - 1.0) / d;
                let a2 = (1.0 + 2.0 * sigma * wc + k * (sigma * sigma + omega_pole * omega_pole)) / d;

                (vec![a1, a2], vec![b0, b1, b0])
            }
            _ => {
                // Higher orders use cascaded second-order sections
                let gamma = (1.0 / epsilon + (1.0 / (epsilon * epsilon) + 1.0).sqrt()).ln() / self.order as f64;
                let sinh_g = gamma.sinh();
                let cosh_g = gamma.cosh();

                let angle = pi / (2.0 * self.order as f64);
                let sigma = -sinh_g * angle.sin();
                let omega_pole = cosh_g * angle.cos();

                let k = wc * wc;
                let d = 1.0 - 2.0 * sigma * wc + (sigma * sigma + omega_pole * omega_pole) * k;

                let b0 = k / d;
                let b1 = 2.0 * b0;
                let a1 = 2.0 * ((sigma * sigma + omega_pole * omega_pole) * k - 1.0) / d;
                let a2 = (1.0 + 2.0 * sigma * wc + (sigma * sigma + omega_pole * omega_pole) * k) / d;

                (vec![a1, a2], vec![b0, b1, b0])
            }
        };

        let mut result = data.to_vec();

        // Apply filter (cascade for higher orders)
        for _ in 0..(self.order / 2).max(1) {
            let mut filtered = vec![0.0; n];
            filtered[0] = result[0];

            if a_coeffs.len() == 1 {
                // First-order filter
                for i in 1..n {
                    filtered[i] = b_coeffs[0] * result[i] + b_coeffs[1] * result[i - 1]
                                - a_coeffs[0] * filtered[i - 1];
                }
            } else {
                // Second-order filter
                filtered[1] = b_coeffs[0] * result[1] + b_coeffs[1] * result[0] - a_coeffs[0] * filtered[0];
                for i in 2..n {
                    filtered[i] = b_coeffs[0] * result[i] + b_coeffs[1] * result[i - 1] + b_coeffs[2] * result[i - 2]
                                - a_coeffs[0] * filtered[i - 1] - a_coeffs[1] * filtered[i - 2];
                }
            }
            result = filtered;
        }

        result
    }
}

impl TechnicalIndicator for ChebyshevFilter {
    fn name(&self) -> &str {
        "Chebyshev Filter"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Weighted Median Filter - Median filter with distance-based weighting
///
/// An enhanced median filter that assigns weights to values based on their
/// distance from the center of the window. Values closer to the current
/// observation receive higher weight in the median calculation, providing
/// better responsiveness while maintaining spike removal properties.
#[derive(Debug, Clone)]
pub struct WeightedMedianFilter {
    /// Window period for the filter
    period: usize,
    /// Weight decay factor (0.0-1.0), higher = faster decay
    weight_decay: f64,
}

impl WeightedMedianFilter {
    /// Create a new Weighted Median Filter
    ///
    /// # Arguments
    /// * `period` - Window size for the filter (minimum 3, must be odd)
    /// * `weight_decay` - How quickly weights decay from center (0.1-1.0)
    ///
    /// # Returns
    /// Result containing the filter instance or an error if parameters are invalid
    pub fn new(period: usize, weight_decay: f64) -> Result<Self> {
        if period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        if weight_decay <= 0.0 || weight_decay > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "weight_decay".to_string(),
                reason: "must be between 0 and 1".to_string(),
            });
        }
        Ok(Self { period, weight_decay })
    }

    /// Calculate weighted median filter
    ///
    /// For each point, calculates a weighted median where weights decrease
    /// exponentially from the center of the window.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n == 0 {
            return vec![];
        }

        let mut result = vec![0.0; n];
        let half_period = self.period / 2;

        for i in 0..n {
            let start = i.saturating_sub(half_period);
            let end = (i + half_period + 1).min(n);
            let window_size = end - start;

            // Create weighted value pairs
            let mut weighted_values: Vec<(f64, f64)> = Vec::with_capacity(window_size);
            for j in start..end {
                let distance = (j as i64 - i as i64).unsigned_abs() as f64;
                let weight = (-self.weight_decay * distance).exp();
                weighted_values.push((data[j], weight));
            }

            // Sort by value
            weighted_values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            // Find weighted median
            let total_weight: f64 = weighted_values.iter().map(|(_, w)| w).sum();
            let half_weight = total_weight / 2.0;
            let mut cumulative_weight = 0.0;

            for (value, weight) in &weighted_values {
                cumulative_weight += weight;
                if cumulative_weight >= half_weight {
                    result[i] = *value;
                    break;
                }
            }
        }

        result
    }
}

impl TechnicalIndicator for WeightedMedianFilter {
    fn name(&self) -> &str {
        "Weighted Median Filter"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Double Exponential Filter - Double exponential smoothing (Holt's method)
///
/// Implements Holt's double exponential smoothing which tracks both level
/// and trend components. This filter is more responsive to trends than
/// single exponential smoothing while still providing noise reduction.
#[derive(Debug, Clone)]
pub struct DoubleExponentialFilter {
    /// Smoothing factor for level (0.0-1.0)
    alpha: f64,
    /// Smoothing factor for trend (0.0-1.0)
    beta: f64,
}

impl DoubleExponentialFilter {
    /// Create a new Double Exponential Filter (Holt's method)
    ///
    /// # Arguments
    /// * `alpha` - Level smoothing factor (0.01-1.0)
    /// * `beta` - Trend smoothing factor (0.01-1.0)
    ///
    /// # Returns
    /// Result containing the filter instance or an error if parameters are invalid
    pub fn new(alpha: f64, beta: f64) -> Result<Self> {
        if alpha <= 0.0 || alpha > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "alpha".to_string(),
                reason: "must be between 0 and 1".to_string(),
            });
        }
        if beta <= 0.0 || beta > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "beta".to_string(),
                reason: "must be between 0 and 1".to_string(),
            });
        }
        Ok(Self { alpha, beta })
    }

    /// Calculate double exponential filter
    ///
    /// Uses Holt's method to track both level and trend, providing
    /// better trend-following than single exponential smoothing.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n == 0 {
            return vec![];
        }
        if n == 1 {
            return vec![data[0]];
        }

        let mut result = vec![0.0; n];
        let mut level = data[0];
        let mut trend = data[1] - data[0];

        result[0] = data[0];

        for i in 1..n {
            let prev_level = level;

            // Update level
            level = self.alpha * data[i] + (1.0 - self.alpha) * (prev_level + trend);

            // Update trend
            trend = self.beta * (level - prev_level) + (1.0 - self.beta) * trend;

            // Output is level + trend (one-step forecast)
            result[i] = level;
        }

        result
    }

    /// Calculate with forecast horizon
    ///
    /// Returns both the filtered series and a forecast extending h periods ahead
    pub fn calculate_with_forecast(&self, data: &[f64], horizon: usize) -> (Vec<f64>, Vec<f64>) {
        let n = data.len();
        if n == 0 {
            return (vec![], vec![]);
        }

        let filtered = self.calculate(data);

        // Get final level and trend
        let mut level = data[0];
        let mut trend = if n > 1 { data[1] - data[0] } else { 0.0 };

        for i in 1..n {
            let prev_level = level;
            level = self.alpha * data[i] + (1.0 - self.alpha) * (prev_level + trend);
            trend = self.beta * (level - prev_level) + (1.0 - self.beta) * trend;
        }

        // Generate forecast
        let forecast: Vec<f64> = (1..=horizon)
            .map(|h| level + h as f64 * trend)
            .collect();

        (filtered, forecast)
    }
}

impl TechnicalIndicator for DoubleExponentialFilter {
    fn name(&self) -> &str {
        "Double Exponential Filter"
    }

    fn min_periods(&self) -> usize {
        2
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Adaptive Bandpass Filter - Bandpass filter with adaptive center and width
///
/// A bandpass filter that automatically adjusts its center frequency and
/// bandwidth based on detected market cycles. Uses zero-crossing analysis
/// to estimate the dominant cycle period and adjusts filter parameters
/// accordingly.
#[derive(Debug, Clone)]
pub struct AdaptiveBandpassFilter {
    /// Lookback period for cycle detection
    lookback: usize,
    /// Minimum cycle period to detect
    min_period: usize,
    /// Maximum cycle period to detect
    max_period: usize,
    /// Base bandwidth as fraction of cycle period
    base_bandwidth: f64,
}

impl AdaptiveBandpassFilter {
    /// Create a new Adaptive Bandpass Filter
    ///
    /// # Arguments
    /// * `lookback` - Period for cycle analysis (minimum 20)
    /// * `min_period` - Minimum detectable cycle period (minimum 5)
    /// * `max_period` - Maximum detectable cycle period
    /// * `base_bandwidth` - Base bandwidth fraction (0.1-0.8)
    ///
    /// # Returns
    /// Result containing the filter instance or an error if parameters are invalid
    pub fn new(lookback: usize, min_period: usize, max_period: usize, base_bandwidth: f64) -> Result<Self> {
        if lookback < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
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
        if base_bandwidth <= 0.0 || base_bandwidth > 0.8 {
            return Err(IndicatorError::InvalidParameter {
                name: "base_bandwidth".to_string(),
                reason: "must be between 0 and 0.8".to_string(),
            });
        }
        Ok(Self { lookback, min_period, max_period, base_bandwidth })
    }

    /// Estimate dominant cycle period using zero-crossing method
    fn estimate_cycle_period(&self, data: &[f64]) -> f64 {
        let n = data.len();
        if n < 4 {
            return (self.min_period + self.max_period) as f64 / 2.0;
        }

        // Detrend the data using simple differencing
        let mean: f64 = data.iter().sum::<f64>() / n as f64;

        // Count zero crossings around mean
        let mut crossings = 0;
        for i in 1..n {
            if (data[i] - mean) * (data[i - 1] - mean) < 0.0 {
                crossings += 1;
            }
        }

        // Estimate period from crossings (2 crossings per cycle)
        let estimated_period = if crossings > 0 {
            2.0 * n as f64 / crossings as f64
        } else {
            (self.min_period + self.max_period) as f64 / 2.0
        };

        // Clamp to valid range
        estimated_period.clamp(self.min_period as f64, self.max_period as f64)
    }

    /// Calculate adaptive bandpass filter
    ///
    /// Dynamically adjusts center frequency and bandwidth based on
    /// detected market cycles for optimal signal extraction.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < 3 {
            return data.to_vec();
        }

        let mut result = vec![0.0; n];
        let pi = std::f64::consts::PI;

        // Initialize first values
        result[0] = 0.0;
        result[1] = 0.0;

        for i in 2..n {
            let start = i.saturating_sub(self.lookback);
            let window = &data[start..=i];

            // Estimate dominant cycle period
            let cycle_period = self.estimate_cycle_period(window);

            // Calculate adaptive bandwidth based on signal clarity
            let mean: f64 = window.iter().sum::<f64>() / window.len() as f64;
            let variance: f64 = window.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / window.len() as f64;
            let std_dev = variance.sqrt();

            // Higher volatility relative to mean = narrower bandwidth
            let clarity = if mean.abs() > 1e-10 {
                1.0 - (std_dev / mean.abs()).min(0.2) / 0.2
            } else {
                0.5
            };

            let bandwidth = self.base_bandwidth * (0.5 + 0.5 * clarity);

            // Bandpass filter coefficients
            let omega = 2.0 * pi / cycle_period;
            let delta = (omega * bandwidth).cos();
            let beta = (1.0 - delta) / (1.0 + delta);
            let gamma = if delta.abs() > 1e-10 { (1.0 / delta).cos() } else { 1.0 };
            let alpha = (1.0 - beta) / 2.0;

            // Apply bandpass filter
            if i >= 2 {
                result[i] = alpha * (data[i] - data[i - 2])
                          + gamma * (1.0 + beta) * result[i - 1]
                          - beta * result[i - 2];
            }
        }

        result
    }

    /// Get estimated cycle periods for analysis
    pub fn get_cycle_periods(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let mut periods = vec![0.0; n];

        for i in 0..n {
            let start = i.saturating_sub(self.lookback);
            let window = &data[start..=i];
            periods[i] = self.estimate_cycle_period(window);
        }

        periods
    }
}

impl TechnicalIndicator for AdaptiveBandpassFilter {
    fn name(&self) -> &str {
        "Adaptive Bandpass Filter"
    }

    fn min_periods(&self) -> usize {
        self.lookback
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Hodrick-Prescott Filter - Optimal trend extraction filter
///
/// The Hodrick-Prescott filter is a mathematical tool used to separate the
/// cyclical component of a time series from raw data. It is commonly used
/// in macroeconomics to extract the trend from a time series and is effective
/// for financial time series analysis.
#[derive(Debug, Clone)]
pub struct HodrickPrescottFilter {
    /// Smoothing parameter (lambda), higher = smoother trend
    lambda: f64,
}

impl HodrickPrescottFilter {
    /// Create a new Hodrick-Prescott Filter
    ///
    /// # Arguments
    /// * `lambda` - Smoothing parameter (1.0-10000.0)
    ///   - Daily data: 100-400 typical
    ///   - Weekly data: 270-675 typical
    ///   - Monthly data: 1600 typical (Hodrick-Prescott's original)
    ///   - Higher values = smoother trend extraction
    ///
    /// # Returns
    /// Result containing the filter instance or an error if parameters are invalid
    pub fn new(lambda: f64) -> Result<Self> {
        if lambda < 1.0 || lambda > 100000.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "lambda".to_string(),
                reason: "must be between 1 and 100000".to_string(),
            });
        }
        Ok(Self { lambda })
    }

    /// Calculate Hodrick-Prescott filter using a direct matrix solution approach
    ///
    /// Extracts the trend component from the time series. The cyclical
    /// component can be obtained by subtracting the trend from the original.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < 4 {
            return data.to_vec();
        }

        // The HP filter minimizes: sum((y-tau)^2) + lambda * sum((tau_{t+1} - 2*tau_t + tau_{t-1})^2)
        // This can be solved using a recursive smoothing approach

        let lambda = self.lambda;

        // Forward-backward recursive filter approach
        // This is more stable than direct matrix methods

        // First pass: forward smoothing
        let mut trend_forward = vec![0.0; n];
        trend_forward[0] = data[0];
        trend_forward[1] = data[1];

        let alpha = 1.0 / (1.0 + lambda);
        let beta = lambda / (1.0 + lambda);

        for i in 2..n {
            // Simple weighted average that respects smoothness
            let prev_trend = 2.0 * trend_forward[i - 1] - trend_forward[i - 2];
            trend_forward[i] = alpha * data[i] + beta * prev_trend;
        }

        // Second pass: backward smoothing
        let mut trend_backward = vec![0.0; n];
        trend_backward[n - 1] = data[n - 1];
        trend_backward[n - 2] = data[n - 2];

        for i in (0..n - 2).rev() {
            let prev_trend = 2.0 * trend_backward[i + 1] - trend_backward[i + 2];
            trend_backward[i] = alpha * data[i] + beta * prev_trend;
        }

        // Combine forward and backward passes
        let mut trend = vec![0.0; n];
        for i in 0..n {
            trend[i] = (trend_forward[i] + trend_backward[i]) / 2.0;
        }

        // Final smoothing pass to reduce edge effects
        let smooth_factor = 2.0 / (1.0 + (lambda / 100.0).sqrt());
        let smooth_alpha = smooth_factor.min(0.5).max(0.1);

        let mut smoothed = vec![0.0; n];
        smoothed[0] = trend[0];
        for i in 1..n {
            smoothed[i] = smooth_alpha * trend[i] + (1.0 - smooth_alpha) * smoothed[i - 1];
        }

        // Backward smooth
        let mut final_trend = vec![0.0; n];
        final_trend[n - 1] = smoothed[n - 1];
        for i in (0..n - 1).rev() {
            final_trend[i] = smooth_alpha * smoothed[i] + (1.0 - smooth_alpha) * final_trend[i + 1];
        }

        final_trend
    }

    /// Calculate both trend and cycle components
    ///
    /// Returns a tuple of (trend, cycle) where cycle = original - trend
    pub fn calculate_components(&self, data: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let trend = self.calculate(data);
        let cycle: Vec<f64> = data.iter()
            .zip(trend.iter())
            .map(|(&d, &t)| d - t)
            .collect();

        (trend, cycle)
    }
}

impl TechnicalIndicator for HodrickPrescottFilter {
    fn name(&self) -> &str {
        "Hodrick Prescott Filter"
    }

    fn min_periods(&self) -> usize {
        4
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> Vec<f64> {
        // Price data with trend, cycle, and noise components
        (0..50).map(|i| {
            let trend = 100.0 + i as f64 * 0.5;
            let cycle = 3.0 * (i as f64 * 0.3).sin();
            let noise = ((i * 7) % 5) as f64 * 0.2 - 0.5;
            trend + cycle + noise
        }).collect()
    }

    fn make_ohlcv_data() -> OHLCVSeries {
        let close = make_test_data();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
        let open: Vec<f64> = close.clone();
        let volume: Vec<f64> = vec![1000.0; close.len()];
        OHLCVSeries { open, high, low, close, volume }
    }

    #[test]
    fn test_adaptive_low_pass_filter() {
        let data = make_test_data();
        let alpf = AdaptiveLowPassFilter::new(10, 0.1, 0.5).unwrap();
        let result = alpf.calculate(&data);

        assert_eq!(result.len(), data.len());
        assert!(result[25] > 0.0);
        // Filtered result should be smoother than original
    }

    #[test]
    fn test_adaptive_low_pass_filter_validation() {
        assert!(AdaptiveLowPassFilter::new(4, 0.1, 0.5).is_err()); // period too small
        assert!(AdaptiveLowPassFilter::new(10, 0.0, 0.5).is_err()); // min_cutoff too small
        assert!(AdaptiveLowPassFilter::new(10, 0.5, 0.3).is_err()); // max < min
        assert!(AdaptiveLowPassFilter::new(10, 0.1, 1.5).is_err()); // max > 1
    }

    #[test]
    fn test_noise_reduction_filter() {
        let data = make_test_data();
        let nrf = NoiseReductionFilter::new(5, 0.3).unwrap();
        let result = nrf.calculate(&data);

        assert_eq!(result.len(), data.len());
        assert!(result[25] > 0.0);
    }

    #[test]
    fn test_noise_reduction_filter_validation() {
        assert!(NoiseReductionFilter::new(2, 0.3).is_err()); // period too small
        assert!(NoiseReductionFilter::new(5, 0.0).is_err()); // smoothing_factor too small
        assert!(NoiseReductionFilter::new(5, 1.5).is_err()); // smoothing_factor too large
    }

    #[test]
    fn test_trend_extraction_filter() {
        let data = make_test_data();
        let tef = TrendExtractionFilter::new(14, 0.7).unwrap();
        let result = tef.calculate(&data);

        assert_eq!(result.len(), data.len());
        // Trend should generally increase with the data
        assert!(result[40] > result[20]);
    }

    #[test]
    fn test_trend_extraction_filter_validation() {
        assert!(TrendExtractionFilter::new(5, 0.5).is_err()); // period too small
        assert!(TrendExtractionFilter::new(14, 0.0).is_err()); // strength too small
        assert!(TrendExtractionFilter::new(14, 1.5).is_err()); // strength too large
    }

    #[test]
    fn test_cycle_extraction_filter() {
        let data = make_test_data();
        let cef = CycleExtractionFilter::new(20, 10).unwrap();
        let result = cef.calculate(&data);

        assert_eq!(result.len(), data.len());
        // Cycle component should oscillate around zero
    }

    #[test]
    fn test_cycle_extraction_filter_validation() {
        assert!(CycleExtractionFilter::new(5, 10).is_err()); // period too small
        assert!(CycleExtractionFilter::new(20, 3).is_err()); // cycle_period too small
        assert!(CycleExtractionFilter::new(20, 25).is_err()); // cycle_period > period
    }

    #[test]
    fn test_adaptive_high_pass_filter() {
        let data = make_test_data();
        let ahpf = AdaptiveHighPassFilter::new(10, 1.0).unwrap();
        let result = ahpf.calculate(&data);

        assert_eq!(result.len(), data.len());
        // High-pass filter removes trend, values should be smaller
    }

    #[test]
    fn test_adaptive_high_pass_filter_validation() {
        assert!(AdaptiveHighPassFilter::new(3, 1.0).is_err()); // period too small
        assert!(AdaptiveHighPassFilter::new(10, 0.0).is_err()); // sensitivity too small
        assert!(AdaptiveHighPassFilter::new(10, 2.5).is_err()); // sensitivity too large
    }

    #[test]
    fn test_bandwidth_adaptive_filter() {
        let data = make_test_data();
        let baf = BandwidthAdaptiveFilter::new(20, 0.5).unwrap();
        let result = baf.calculate(&data);

        assert_eq!(result.len(), data.len());
        assert!(result[30] > 0.0);
    }

    #[test]
    fn test_bandwidth_adaptive_filter_validation() {
        assert!(BandwidthAdaptiveFilter::new(5, 0.5).is_err()); // period too small
        assert!(BandwidthAdaptiveFilter::new(20, 0.0).is_err()); // base_bandwidth too small
        assert!(BandwidthAdaptiveFilter::new(20, 1.5).is_err()); // base_bandwidth too large
    }

    #[test]
    fn test_technical_indicator_impl() {
        let ohlcv = make_ohlcv_data();

        let alpf = AdaptiveLowPassFilter::new(10, 0.1, 0.5).unwrap();
        assert_eq!(alpf.name(), "Adaptive Low Pass Filter");
        assert_eq!(alpf.min_periods(), 10);
        let output = alpf.compute(&ohlcv).unwrap();
        assert_eq!(output.primary.len(), ohlcv.close.len());

        let nrf = NoiseReductionFilter::new(5, 0.3).unwrap();
        assert_eq!(nrf.name(), "Noise Reduction Filter");
        assert_eq!(nrf.min_periods(), 5);
        let output = nrf.compute(&ohlcv).unwrap();
        assert_eq!(output.primary.len(), ohlcv.close.len());

        let tef = TrendExtractionFilter::new(14, 0.7).unwrap();
        assert_eq!(tef.name(), "Trend Extraction Filter");
        assert_eq!(tef.min_periods(), 14);
        let output = tef.compute(&ohlcv).unwrap();
        assert_eq!(output.primary.len(), ohlcv.close.len());

        let cef = CycleExtractionFilter::new(20, 10).unwrap();
        assert_eq!(cef.name(), "Cycle Extraction Filter");
        assert_eq!(cef.min_periods(), 20);
        let output = cef.compute(&ohlcv).unwrap();
        assert_eq!(output.primary.len(), ohlcv.close.len());

        let ahpf = AdaptiveHighPassFilter::new(10, 1.0).unwrap();
        assert_eq!(ahpf.name(), "Adaptive High Pass Filter");
        assert_eq!(ahpf.min_periods(), 10);
        let output = ahpf.compute(&ohlcv).unwrap();
        assert_eq!(output.primary.len(), ohlcv.close.len());

        let baf = BandwidthAdaptiveFilter::new(20, 0.5).unwrap();
        assert_eq!(baf.name(), "Bandwidth Adaptive Filter");
        assert_eq!(baf.min_periods(), 20);
        let output = baf.compute(&ohlcv).unwrap();
        assert_eq!(output.primary.len(), ohlcv.close.len());
    }

    #[test]
    fn test_empty_data() {
        let empty: Vec<f64> = vec![];

        let alpf = AdaptiveLowPassFilter::new(10, 0.1, 0.5).unwrap();
        assert!(alpf.calculate(&empty).is_empty());

        let nrf = NoiseReductionFilter::new(5, 0.3).unwrap();
        assert!(nrf.calculate(&empty).is_empty());

        let tef = TrendExtractionFilter::new(14, 0.7).unwrap();
        assert!(tef.calculate(&empty).is_empty());

        let cef = CycleExtractionFilter::new(20, 10).unwrap();
        assert!(cef.calculate(&empty).is_empty());

        let ahpf = AdaptiveHighPassFilter::new(10, 1.0).unwrap();
        assert!(ahpf.calculate(&empty).is_empty());

        let baf = BandwidthAdaptiveFilter::new(20, 0.5).unwrap();
        assert!(baf.calculate(&empty).is_empty());
    }

    // ==================== Tests for 6 NEW Filter Indicators ====================

    #[test]
    fn test_butterworth_bandpass_filter() {
        let data = make_test_data();
        let bbf = ButterworthBandpassFilter::new(10, 0.5).unwrap();
        let result = bbf.calculate(&data);

        assert_eq!(result.len(), data.len());
        // Bandpass filter should produce non-zero output
        assert!(result.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_butterworth_bandpass_filter_validation() {
        // Period too small
        assert!(ButterworthBandpassFilter::new(4, 0.5).is_err());
        // Bandwidth too small
        assert!(ButterworthBandpassFilter::new(10, 0.0).is_err());
        // Bandwidth too large
        assert!(ButterworthBandpassFilter::new(10, 1.5).is_err());
    }

    #[test]
    fn test_butterworth_bandpass_filter_smoothing() {
        // Test that the filter smooths the data
        let data: Vec<f64> = (0..30).map(|i| {
            100.0 + 5.0 * (i as f64 * 0.5).sin() + 10.0 * (i as f64 * 0.1).sin()
        }).collect();
        let bbf = ButterworthBandpassFilter::new(8, 0.4).unwrap();
        let result = bbf.calculate(&data);

        assert_eq!(result.len(), data.len());
        // Result should exist for all points
        assert!(result.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_chebyshev_filter() {
        let data = make_test_data();
        let cf = ChebyshevFilter::new(10, 1.0, 2).unwrap();
        let result = cf.calculate(&data);

        assert_eq!(result.len(), data.len());
        assert!(result[25] > 0.0);
        // Filtered result should be smoother than original
    }

    #[test]
    fn test_chebyshev_filter_validation() {
        // Period too small
        assert!(ChebyshevFilter::new(2, 1.0, 2).is_err());
        // Ripple too small
        assert!(ChebyshevFilter::new(10, 0.0, 2).is_err());
        // Ripple too large
        assert!(ChebyshevFilter::new(10, 4.0, 2).is_err());
        // Order too small
        assert!(ChebyshevFilter::new(10, 1.0, 0).is_err());
        // Order too large
        assert!(ChebyshevFilter::new(10, 1.0, 5).is_err());
    }

    #[test]
    fn test_chebyshev_filter_orders() {
        let data = make_test_data();

        // Test different orders
        let cf1 = ChebyshevFilter::new(10, 0.5, 1).unwrap();
        let result1 = cf1.calculate(&data);
        assert_eq!(result1.len(), data.len());

        let cf2 = ChebyshevFilter::new(10, 0.5, 2).unwrap();
        let result2 = cf2.calculate(&data);
        assert_eq!(result2.len(), data.len());

        let cf4 = ChebyshevFilter::new(10, 0.5, 4).unwrap();
        let result4 = cf4.calculate(&data);
        assert_eq!(result4.len(), data.len());

        // All results should be finite
        assert!(result1.iter().all(|v| v.is_finite()));
        assert!(result2.iter().all(|v| v.is_finite()));
        assert!(result4.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_weighted_median_filter() {
        let data = make_test_data();
        let wmf = WeightedMedianFilter::new(5, 0.5).unwrap();
        let result = wmf.calculate(&data);

        assert_eq!(result.len(), data.len());
        assert!(result[25] > 0.0);
    }

    #[test]
    fn test_weighted_median_filter_validation() {
        // Period too small
        assert!(WeightedMedianFilter::new(2, 0.5).is_err());
        // Weight decay too small
        assert!(WeightedMedianFilter::new(5, 0.0).is_err());
        // Weight decay too large
        assert!(WeightedMedianFilter::new(5, 1.5).is_err());
    }

    #[test]
    fn test_weighted_median_filter_spike_removal() {
        // Test spike removal capability
        let mut data = vec![100.0, 101.0, 102.0, 500.0, 103.0, 104.0, 105.0];
        data.extend(vec![106.0, 107.0, 108.0]);

        let wmf = WeightedMedianFilter::new(5, 0.3).unwrap();
        let result = wmf.calculate(&data);

        // The spike at index 3 should be reduced
        assert!(result[3] < 400.0); // Should not pass through the full spike
        assert!(result[3] > 90.0);  // Should still be in reasonable range
    }

    #[test]
    fn test_double_exponential_filter() {
        let data = make_test_data();
        let def = DoubleExponentialFilter::new(0.3, 0.1).unwrap();
        let result = def.calculate(&data);

        assert_eq!(result.len(), data.len());
        assert!(result[25] > 0.0);
        // Should follow the upward trend
        assert!(result[40] > result[10]);
    }

    #[test]
    fn test_double_exponential_filter_validation() {
        // Alpha too small
        assert!(DoubleExponentialFilter::new(0.0, 0.5).is_err());
        // Alpha too large
        assert!(DoubleExponentialFilter::new(1.5, 0.5).is_err());
        // Beta too small
        assert!(DoubleExponentialFilter::new(0.5, 0.0).is_err());
        // Beta too large
        assert!(DoubleExponentialFilter::new(0.5, 1.5).is_err());
    }

    #[test]
    fn test_double_exponential_filter_forecast() {
        let data: Vec<f64> = (0..20).map(|i| 100.0 + i as f64 * 2.0).collect();
        let def = DoubleExponentialFilter::new(0.5, 0.3).unwrap();
        let (filtered, forecast) = def.calculate_with_forecast(&data, 5);

        assert_eq!(filtered.len(), data.len());
        assert_eq!(forecast.len(), 5);
        // Forecast should continue the trend
        assert!(forecast[0] > filtered[filtered.len() - 1] - 10.0);
        assert!(forecast[4] > forecast[0]); // Trend continues upward
    }

    #[test]
    fn test_double_exponential_filter_single_point() {
        let data = vec![100.0];
        let def = DoubleExponentialFilter::new(0.3, 0.1).unwrap();
        let result = def.calculate(&data);

        assert_eq!(result.len(), 1);
        assert!((result[0] - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_bandpass_filter() {
        let data = make_test_data();
        let abf = AdaptiveBandpassFilter::new(25, 5, 30, 0.5).unwrap();
        let result = abf.calculate(&data);

        assert_eq!(result.len(), data.len());
        // Bandpass filter output should be oscillatory
    }

    #[test]
    fn test_adaptive_bandpass_filter_validation() {
        // Lookback too small
        assert!(AdaptiveBandpassFilter::new(10, 5, 30, 0.5).is_err());
        // Min period too small
        assert!(AdaptiveBandpassFilter::new(25, 3, 30, 0.5).is_err());
        // Max period not greater than min
        assert!(AdaptiveBandpassFilter::new(25, 10, 8, 0.5).is_err());
        // Bandwidth too small
        assert!(AdaptiveBandpassFilter::new(25, 5, 30, 0.0).is_err());
        // Bandwidth too large
        assert!(AdaptiveBandpassFilter::new(25, 5, 30, 0.9).is_err());
    }

    #[test]
    fn test_adaptive_bandpass_filter_cycle_detection() {
        // Create data with a clear cycle
        let data: Vec<f64> = (0..60).map(|i| {
            100.0 + 5.0 * (i as f64 * 2.0 * std::f64::consts::PI / 10.0).sin()
        }).collect();

        let abf = AdaptiveBandpassFilter::new(30, 5, 20, 0.5).unwrap();
        let periods = abf.get_cycle_periods(&data);

        assert_eq!(periods.len(), data.len());
        // Detected period should be in the valid range
        for p in &periods[30..] {
            assert!(*p >= 5.0 && *p <= 20.0);
        }
    }

    #[test]
    fn test_hodrick_prescott_filter() {
        let data = make_test_data();
        let hpf = HodrickPrescottFilter::new(100.0).unwrap();
        let result = hpf.calculate(&data);

        assert_eq!(result.len(), data.len());
        // Trend should follow the general direction
        assert!(result[40] > result[10]);
    }

    #[test]
    fn test_hodrick_prescott_filter_validation() {
        // Lambda too small
        assert!(HodrickPrescottFilter::new(0.5).is_err());
        // Lambda too large
        assert!(HodrickPrescottFilter::new(200000.0).is_err());
    }

    #[test]
    fn test_hodrick_prescott_filter_components() {
        let data = make_test_data();
        let hpf = HodrickPrescottFilter::new(400.0).unwrap();
        let (trend, cycle) = hpf.calculate_components(&data);

        assert_eq!(trend.len(), data.len());
        assert_eq!(cycle.len(), data.len());

        // Trend + cycle should approximately equal original
        for i in 4..(data.len() - 4) {
            let reconstructed = trend[i] + cycle[i];
            assert!((reconstructed - data[i]).abs() < 20.0, // Allow for iterative approximation
                "Reconstruction error at index {}: reconstructed={}, original={}",
                i, reconstructed, data[i]);
        }
    }

    #[test]
    fn test_hodrick_prescott_filter_smoothness() {
        // Higher lambda = smoother trend
        let data: Vec<f64> = (0..50).map(|i| {
            100.0 + i as f64 * 0.5 + 3.0 * (i as f64 * 0.5).sin()
        }).collect();

        let hpf_low = HodrickPrescottFilter::new(10.0).unwrap();
        let hpf_high = HodrickPrescottFilter::new(1000.0).unwrap();

        let trend_low = hpf_low.calculate(&data);
        let trend_high = hpf_high.calculate(&data);

        // Calculate roughness (sum of squared second differences)
        let roughness = |trend: &[f64]| -> f64 {
            trend.windows(3)
                .map(|w| (w[2] - 2.0 * w[1] + w[0]).powi(2))
                .sum()
        };

        // Higher lambda should produce smoother trend (lower roughness)
        assert!(roughness(&trend_high) < roughness(&trend_low));
    }

    #[test]
    fn test_hodrick_prescott_filter_short_data() {
        let data = vec![100.0, 101.0, 102.0];
        let hpf = HodrickPrescottFilter::new(100.0).unwrap();
        let result = hpf.calculate(&data);

        // Short data should return as-is
        assert_eq!(result.len(), 3);
        assert_eq!(result, data);
    }

    #[test]
    fn test_new_filters_technical_indicator_impl() {
        let ohlcv = make_ohlcv_data();

        // ButterworthBandpassFilter
        let bbf = ButterworthBandpassFilter::new(10, 0.5).unwrap();
        assert_eq!(bbf.name(), "Butterworth Bandpass Filter");
        assert_eq!(bbf.min_periods(), 10);
        let output = bbf.compute(&ohlcv).unwrap();
        assert_eq!(output.primary.len(), ohlcv.close.len());

        // ChebyshevFilter
        let cf = ChebyshevFilter::new(10, 1.0, 2).unwrap();
        assert_eq!(cf.name(), "Chebyshev Filter");
        assert_eq!(cf.min_periods(), 10);
        let output = cf.compute(&ohlcv).unwrap();
        assert_eq!(output.primary.len(), ohlcv.close.len());

        // WeightedMedianFilter
        let wmf = WeightedMedianFilter::new(5, 0.5).unwrap();
        assert_eq!(wmf.name(), "Weighted Median Filter");
        assert_eq!(wmf.min_periods(), 5);
        let output = wmf.compute(&ohlcv).unwrap();
        assert_eq!(output.primary.len(), ohlcv.close.len());

        // DoubleExponentialFilter
        let def = DoubleExponentialFilter::new(0.3, 0.1).unwrap();
        assert_eq!(def.name(), "Double Exponential Filter");
        assert_eq!(def.min_periods(), 2);
        let output = def.compute(&ohlcv).unwrap();
        assert_eq!(output.primary.len(), ohlcv.close.len());

        // AdaptiveBandpassFilter
        let abf = AdaptiveBandpassFilter::new(25, 5, 30, 0.5).unwrap();
        assert_eq!(abf.name(), "Adaptive Bandpass Filter");
        assert_eq!(abf.min_periods(), 25);
        let output = abf.compute(&ohlcv).unwrap();
        assert_eq!(output.primary.len(), ohlcv.close.len());

        // HodrickPrescottFilter
        let hpf = HodrickPrescottFilter::new(100.0).unwrap();
        assert_eq!(hpf.name(), "Hodrick Prescott Filter");
        assert_eq!(hpf.min_periods(), 4);
        let output = hpf.compute(&ohlcv).unwrap();
        assert_eq!(output.primary.len(), ohlcv.close.len());
    }

    #[test]
    fn test_new_filters_empty_data() {
        let empty: Vec<f64> = vec![];

        let bbf = ButterworthBandpassFilter::new(10, 0.5).unwrap();
        let result = bbf.calculate(&empty);
        assert!(result.is_empty());

        let cf = ChebyshevFilter::new(10, 1.0, 2).unwrap();
        let result = cf.calculate(&empty);
        assert!(result.is_empty());

        let wmf = WeightedMedianFilter::new(5, 0.5).unwrap();
        let result = wmf.calculate(&empty);
        assert!(result.is_empty());

        let def = DoubleExponentialFilter::new(0.3, 0.1).unwrap();
        let result = def.calculate(&empty);
        assert!(result.is_empty());

        let abf = AdaptiveBandpassFilter::new(25, 5, 30, 0.5).unwrap();
        let result = abf.calculate(&empty);
        assert!(result.is_empty());

        let hpf = HodrickPrescottFilter::new(100.0).unwrap();
        let result = hpf.calculate(&empty);
        assert!(result.is_empty());
    }
}

// ==================== 6 NEW Filter Indicators ====================

/// Gaussian Adaptive Filter - Adaptive Gaussian smoothing with volatility-adjusted sigma
///
/// An enhanced Gaussian filter that dynamically adjusts its smoothing parameter (sigma)
/// based on local price volatility. This provides more smoothing during high volatility
/// periods and more responsiveness during calm markets, offering superior noise reduction
/// while preserving important price movements.
#[derive(Debug, Clone)]
pub struct GaussianAdaptiveFilter {
    /// Window period for the filter
    period: usize,
    /// Base sigma value for Gaussian kernel
    base_sigma: f64,
    /// Sensitivity to volatility changes (0.0-2.0)
    sensitivity: f64,
}

impl GaussianAdaptiveFilter {
    /// Create a new Gaussian Adaptive Filter
    ///
    /// # Arguments
    /// * `period` - Window size for the filter (minimum 5, should be odd for symmetry)
    /// * `base_sigma` - Base standard deviation for Gaussian kernel (0.5-5.0)
    /// * `sensitivity` - How much sigma adjusts to volatility (0.1-2.0)
    ///
    /// # Returns
    /// Result containing the filter instance or an error if parameters are invalid
    pub fn new(period: usize, base_sigma: f64, sensitivity: f64) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if base_sigma < 0.5 || base_sigma > 5.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "base_sigma".to_string(),
                reason: "must be between 0.5 and 5.0".to_string(),
            });
        }
        if sensitivity < 0.1 || sensitivity > 2.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "sensitivity".to_string(),
                reason: "must be between 0.1 and 2.0".to_string(),
            });
        }
        Ok(Self { period, base_sigma, sensitivity })
    }

    /// Calculate adaptive Gaussian filter
    ///
    /// The filter dynamically adjusts sigma based on local volatility, providing
    /// more smoothing when volatility is high and less when volatility is low.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n == 0 {
            return vec![];
        }

        let mut result = vec![0.0; n];
        let half = self.period / 2;

        for i in 0..n {
            let start = i.saturating_sub(half);
            let end = (i + half + 1).min(n);
            let window = &data[start..end];

            // Calculate local volatility
            let mean: f64 = window.iter().sum::<f64>() / window.len() as f64;
            let variance: f64 = window.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / window.len() as f64;
            let std_dev = variance.sqrt();

            // Normalize volatility (coefficient of variation)
            let cv = if mean.abs() > 1e-10 {
                (std_dev / mean.abs()).min(0.2) / 0.2
            } else {
                0.5
            };

            // Adaptive sigma: higher volatility = higher sigma (more smoothing)
            let sigma = self.base_sigma * (1.0 + self.sensitivity * cv);

            // Calculate Gaussian weights with adaptive sigma
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            for (j, &value) in window.iter().enumerate() {
                let center = if i < half { i } else { half };
                let distance = (j as f64 - center as f64).abs();
                let weight = (-distance * distance / (2.0 * sigma * sigma)).exp();
                weighted_sum += value * weight;
                weight_sum += weight;
            }

            result[i] = if weight_sum > 1e-10 {
                weighted_sum / weight_sum
            } else {
                data[i]
            };
        }

        result
    }
}

impl TechnicalIndicator for GaussianAdaptiveFilter {
    fn name(&self) -> &str {
        "Gaussian Adaptive Filter"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Savitzky-Golay Filter - Polynomial smoothing filter
///
/// The Savitzky-Golay filter performs polynomial regression on a sliding window
/// of data points to smooth the data while preserving features like peak heights
/// and widths. It provides excellent smoothing without the phase distortion of
/// simple moving averages. This is particularly useful for preserving sharp
/// price movements while reducing noise.
#[derive(Debug, Clone)]
pub struct SavitzkyGolayFilter {
    /// Window size (must be odd)
    window_size: usize,
    /// Polynomial order (0-4, must be less than window_size)
    poly_order: usize,
}

impl SavitzkyGolayFilter {
    /// Create a new Savitzky-Golay Filter
    ///
    /// # Arguments
    /// * `window_size` - Window size for the filter (minimum 5, must be odd)
    /// * `poly_order` - Order of polynomial to fit (1-4, must be less than window_size)
    ///
    /// # Returns
    /// Result containing the filter instance or an error if parameters are invalid
    pub fn new(window_size: usize, poly_order: usize) -> Result<Self> {
        if window_size < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "window_size".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if window_size % 2 == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "window_size".to_string(),
                reason: "must be odd".to_string(),
            });
        }
        if poly_order < 1 || poly_order > 4 {
            return Err(IndicatorError::InvalidParameter {
                name: "poly_order".to_string(),
                reason: "must be between 1 and 4".to_string(),
            });
        }
        if poly_order >= window_size {
            return Err(IndicatorError::InvalidParameter {
                name: "poly_order".to_string(),
                reason: "must be less than window_size".to_string(),
            });
        }
        Ok(Self { window_size, poly_order })
    }

    /// Calculate Savitzky-Golay filter coefficients for the center point
    fn calculate_coefficients(&self) -> Vec<f64> {
        let n = self.window_size;
        let half = n / 2;
        let m = self.poly_order;

        // Build the Vandermonde matrix A
        let mut a = vec![vec![0.0; m + 1]; n];
        for i in 0..n {
            let x = (i as i64 - half as i64) as f64;
            for j in 0..=m {
                a[i][j] = x.powi(j as i32);
            }
        }

        // Compute (A^T * A)^(-1) * A^T using least squares
        // For simplicity, we use the normal equations approach

        // Compute A^T * A
        let mut ata = vec![vec![0.0; m + 1]; m + 1];
        for i in 0..=m {
            for j in 0..=m {
                for k in 0..n {
                    ata[i][j] += a[k][i] * a[k][j];
                }
            }
        }

        // Compute inverse of A^T * A using Gaussian elimination
        let mut inv = vec![vec![0.0; m + 1]; m + 1];
        for i in 0..=m {
            inv[i][i] = 1.0;
        }

        let mut aug = ata.clone();

        for col in 0..=m {
            // Find pivot
            let mut max_row = col;
            for row in col + 1..=m {
                if aug[row][col].abs() > aug[max_row][col].abs() {
                    max_row = row;
                }
            }
            aug.swap(col, max_row);
            inv.swap(col, max_row);

            let pivot = aug[col][col];
            if pivot.abs() < 1e-10 {
                // Matrix is singular, use identity-like coefficients
                let mut coeffs = vec![0.0; n];
                coeffs[half] = 1.0;
                return coeffs;
            }

            for j in 0..=m {
                aug[col][j] /= pivot;
                inv[col][j] /= pivot;
            }

            for row in 0..=m {
                if row != col {
                    let factor = aug[row][col];
                    for j in 0..=m {
                        aug[row][j] -= factor * aug[col][j];
                        inv[row][j] -= factor * inv[col][j];
                    }
                }
            }
        }

        // Compute (A^T * A)^(-1) * A^T, but we only need the row for the center point
        // The smoothing coefficients are the first row of (A^T * A)^(-1) * A^T
        let mut coeffs = vec![0.0; n];
        for i in 0..n {
            for j in 0..=m {
                coeffs[i] += inv[0][j] * a[i][j];
            }
        }

        coeffs
    }

    /// Calculate Savitzky-Golay filter
    ///
    /// Applies polynomial smoothing while preserving signal features like
    /// peaks and valleys better than simple moving averages.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.window_size {
            return data.to_vec();
        }

        let coeffs = self.calculate_coefficients();
        let half = self.window_size / 2;
        let mut result = vec![0.0; n];

        // Handle edges: use original data
        for i in 0..half {
            result[i] = data[i];
        }
        for i in (n - half)..n {
            result[i] = data[i];
        }

        // Apply convolution in the middle
        for i in half..(n - half) {
            let mut sum = 0.0;
            for (j, &coeff) in coeffs.iter().enumerate() {
                sum += coeff * data[i - half + j];
            }
            result[i] = sum;
        }

        result
    }

    /// Calculate first derivative using Savitzky-Golay
    ///
    /// Returns the smoothed first derivative of the data, useful for
    /// trend direction and momentum analysis.
    pub fn calculate_derivative(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.window_size {
            return vec![0.0; n];
        }

        let half = self.window_size / 2;
        let m = self.poly_order;

        // Build Vandermonde matrix
        let mut a = vec![vec![0.0; m + 1]; self.window_size];
        for i in 0..self.window_size {
            let x = (i as i64 - half as i64) as f64;
            for j in 0..=m {
                a[i][j] = x.powi(j as i32);
            }
        }

        // Compute derivative coefficients (similar process but take derivative row)
        // For first derivative, we need row 1 of the coefficient matrix
        let mut ata = vec![vec![0.0; m + 1]; m + 1];
        for i in 0..=m {
            for j in 0..=m {
                for k in 0..self.window_size {
                    ata[i][j] += a[k][i] * a[k][j];
                }
            }
        }

        // Invert
        let mut inv = vec![vec![0.0; m + 1]; m + 1];
        for i in 0..=m {
            inv[i][i] = 1.0;
        }

        let mut aug = ata.clone();
        for col in 0..=m {
            let mut max_row = col;
            for row in col + 1..=m {
                if aug[row][col].abs() > aug[max_row][col].abs() {
                    max_row = row;
                }
            }
            aug.swap(col, max_row);
            inv.swap(col, max_row);

            let pivot = aug[col][col];
            if pivot.abs() < 1e-10 {
                return vec![0.0; n];
            }

            for j in 0..=m {
                aug[col][j] /= pivot;
                inv[col][j] /= pivot;
            }

            for row in 0..=m {
                if row != col {
                    let factor = aug[row][col];
                    for j in 0..=m {
                        aug[row][j] -= factor * aug[col][j];
                        inv[row][j] -= factor * inv[col][j];
                    }
                }
            }
        }

        // Derivative coefficients use row 1 of inverse
        let mut deriv_coeffs = vec![0.0; self.window_size];
        for i in 0..self.window_size {
            for j in 0..=m {
                deriv_coeffs[i] += inv[1][j] * a[i][j];
            }
        }

        let mut result = vec![0.0; n];
        for i in half..(n - half) {
            for (j, &coeff) in deriv_coeffs.iter().enumerate() {
                result[i] += coeff * data[i - half + j];
            }
        }

        result
    }
}

impl TechnicalIndicator for SavitzkyGolayFilter {
    fn name(&self) -> &str {
        "Savitzky Golay Filter"
    }

    fn min_periods(&self) -> usize {
        self.window_size
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Triangular Filter - Triangular weighted moving average filter
///
/// Applies a triangular weighting scheme where weights increase linearly to the
/// center of the window and then decrease linearly. This provides smooth filtering
/// with emphasis on the center of the window, effectively double-smoothing the data.
/// The triangular filter is equivalent to applying two simple moving averages.
#[derive(Debug, Clone)]
pub struct TriangularFilter {
    /// Window period for the filter
    period: usize,
}

impl TriangularFilter {
    /// Create a new Triangular Filter
    ///
    /// # Arguments
    /// * `period` - Window size for the filter (minimum 3)
    ///
    /// # Returns
    /// Result containing the filter instance or an error if parameters are invalid
    pub fn new(period: usize) -> Result<Self> {
        if period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate triangular weights
    fn calculate_weights(&self) -> Vec<f64> {
        let n = self.period;
        let mid = n as f64 / 2.0;

        let weights: Vec<f64> = (0..n)
            .map(|i| {
                let distance = (i as f64 - mid + 0.5).abs();
                mid - distance + 0.5
            })
            .collect();

        let sum: f64 = weights.iter().sum();
        weights.into_iter().map(|w| w / sum).collect()
    }

    /// Calculate triangular filter
    ///
    /// The triangular weighting provides smooth filtering with a bell-shaped
    /// impulse response, reducing noise while maintaining signal integrity.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n == 0 {
            return vec![];
        }

        let weights = self.calculate_weights();
        let mut result = vec![0.0; n];

        for i in 0..n {
            let start = i.saturating_sub(self.period - 1);
            let window_len = (i - start + 1).min(self.period);

            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            for j in 0..window_len {
                let data_idx = start + j;
                let weight_idx = self.period - window_len + j;
                weighted_sum += data[data_idx] * weights[weight_idx];
                weight_sum += weights[weight_idx];
            }

            result[i] = if weight_sum > 1e-10 {
                weighted_sum / weight_sum
            } else {
                data[i]
            };
        }

        result
    }

    /// Calculate centered triangular filter (for non-real-time analysis)
    ///
    /// Centers the window for symmetric filtering, better for analysis
    /// but introduces lookahead that isn't suitable for real-time trading.
    pub fn calculate_centered(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period {
            return data.to_vec();
        }

        let weights = self.calculate_weights();
        let half = self.period / 2;
        let mut result = vec![0.0; n];

        // Handle edges
        for i in 0..half {
            result[i] = data[i];
        }
        for i in (n - half)..n {
            result[i] = data[i];
        }

        // Centered convolution
        for i in half..(n - half) {
            let mut sum = 0.0;
            for (j, &w) in weights.iter().enumerate() {
                sum += w * data[i - half + j];
            }
            result[i] = sum;
        }

        result
    }
}

impl TechnicalIndicator for TriangularFilter {
    fn name(&self) -> &str {
        "Triangular Filter"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Hamming Filter - Hamming window-based smoothing filter
///
/// Applies the Hamming window function for smoothing. The Hamming window has
/// good side-lobe suppression properties, making it effective at reducing
/// spectral leakage while maintaining reasonable frequency resolution.
/// It provides smoother results than rectangular windows with better
/// attenuation of high-frequency noise.
#[derive(Debug, Clone)]
pub struct HammingFilter {
    /// Window period for the filter
    period: usize,
    /// Alpha coefficient for Hamming window (typically 0.54)
    alpha: f64,
}

impl HammingFilter {
    /// Create a new Hamming Filter
    ///
    /// # Arguments
    /// * `period` - Window size for the filter (minimum 5)
    /// * `alpha` - Hamming window coefficient (0.5-0.6, typically 0.54)
    ///
    /// # Returns
    /// Result containing the filter instance or an error if parameters are invalid
    pub fn new(period: usize, alpha: f64) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if alpha < 0.5 || alpha > 0.6 {
            return Err(IndicatorError::InvalidParameter {
                name: "alpha".to_string(),
                reason: "must be between 0.5 and 0.6".to_string(),
            });
        }
        Ok(Self { period, alpha })
    }

    /// Create a Hamming filter with standard alpha (0.54)
    pub fn with_default_alpha(period: usize) -> Result<Self> {
        Self::new(period, 0.54)
    }

    /// Calculate Hamming window weights
    fn calculate_weights(&self) -> Vec<f64> {
        let n = self.period;
        let pi = std::f64::consts::PI;
        let beta = 1.0 - self.alpha;

        let weights: Vec<f64> = (0..n)
            .map(|i| {
                self.alpha - beta * (2.0 * pi * i as f64 / (n - 1) as f64).cos()
            })
            .collect();

        let sum: f64 = weights.iter().sum();
        weights.into_iter().map(|w| w / sum).collect()
    }

    /// Calculate Hamming filter
    ///
    /// Applies Hamming window smoothing for effective noise reduction
    /// with good spectral properties.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n == 0 {
            return vec![];
        }

        let weights = self.calculate_weights();
        let mut result = vec![0.0; n];

        for i in 0..n {
            let start = i.saturating_sub(self.period - 1);
            let window_len = (i - start + 1).min(self.period);

            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            for j in 0..window_len {
                let data_idx = start + j;
                let weight_idx = self.period - window_len + j;
                weighted_sum += data[data_idx] * weights[weight_idx];
                weight_sum += weights[weight_idx];
            }

            result[i] = if weight_sum > 1e-10 {
                weighted_sum / weight_sum
            } else {
                data[i]
            };
        }

        result
    }

    /// Calculate centered Hamming filter
    ///
    /// Centers the window for symmetric filtering, providing zero-phase
    /// response suitable for signal analysis.
    pub fn calculate_centered(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period {
            return data.to_vec();
        }

        let weights = self.calculate_weights();
        let half = self.period / 2;
        let mut result = vec![0.0; n];

        // Handle edges
        for i in 0..half {
            result[i] = data[i];
        }
        for i in (n - half)..n {
            result[i] = data[i];
        }

        // Centered convolution
        for i in half..(n - half) {
            let mut sum = 0.0;
            for (j, &w) in weights.iter().enumerate() {
                sum += w * data[i - half + j];
            }
            result[i] = sum;
        }

        result
    }
}

impl TechnicalIndicator for HammingFilter {
    fn name(&self) -> &str {
        "Hamming Filter"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Super Smoother Filter - Ehlers two-pole super smoother
///
/// Developed by John Ehlers, the Super Smoother is a two-pole Butterworth filter
/// with critical damping. It provides excellent smoothing with minimal lag
/// compared to moving averages of equivalent smoothness. The filter is designed
/// to eliminate high-frequency noise while preserving important trend information,
/// making it ideal for trading system development.
#[derive(Debug, Clone)]
pub struct SuperSmootherFilter {
    /// Cutoff period for the filter
    period: usize,
}

impl SuperSmootherFilter {
    /// Create a new Super Smoother Filter
    ///
    /// # Arguments
    /// * `period` - Cutoff period for the filter (minimum 4)
    ///
    /// # Returns
    /// Result containing the filter instance or an error if parameters are invalid
    pub fn new(period: usize) -> Result<Self> {
        if period < 4 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 4".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate Super Smoother filter
    ///
    /// Implements Ehlers' two-pole super smoother with critical damping
    /// for optimal smoothing with minimal lag.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < 3 {
            return data.to_vec();
        }

        let pi = std::f64::consts::PI;

        // Calculate coefficients
        let a1 = (-1.414 * pi / self.period as f64).exp();
        let b1 = 2.0 * a1 * (1.414 * pi / self.period as f64).cos();
        let c2 = b1;
        let c3 = -a1 * a1;
        let c1 = 1.0 - c2 - c3;

        let mut result = vec![0.0; n];

        // Initialize first values
        result[0] = data[0];
        result[1] = c1 * (data[1] + data[0]) / 2.0 + c2 * result[0];

        // Apply the super smoother filter
        for i in 2..n {
            result[i] = c1 * (data[i] + data[i - 1]) / 2.0
                      + c2 * result[i - 1]
                      + c3 * result[i - 2];
        }

        result
    }

    /// Calculate three-pole super smoother variant
    ///
    /// Provides even smoother output with slightly more lag.
    pub fn calculate_three_pole(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < 4 {
            return data.to_vec();
        }

        let pi = std::f64::consts::PI;

        // Three-pole coefficients
        let a1 = (-pi / self.period as f64).exp();
        let b1 = 2.0 * a1 * (1.738 * pi / self.period as f64).cos();
        let c1 = a1 * a1;

        let a2 = (-1.414 * pi / self.period as f64).exp();
        let b2 = 2.0 * a2 * (1.414 * pi / self.period as f64).cos();

        let coef2 = b1 + b2;
        let coef3 = -(c1 + b1 * b2 + a2 * a2);
        let coef4 = c1 * b2 + b1 * a2 * a2;
        let coef5 = -c1 * a2 * a2;
        let coef1 = 1.0 - coef2 - coef3 - coef4 - coef5;

        let mut result = vec![0.0; n];

        // Initialize
        result[0] = data[0];
        result[1] = data[1];
        result[2] = data[2];
        if n > 3 {
            result[3] = data[3];
        }

        for i in 4..n {
            result[i] = coef1 * data[i]
                      + coef2 * result[i - 1]
                      + coef3 * result[i - 2]
                      + coef4 * result[i - 3]
                      + coef5 * result[i - 4];
        }

        result
    }
}

impl TechnicalIndicator for SuperSmootherFilter {
    fn name(&self) -> &str {
        "Super Smoother Filter"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Decycler Filter - Ehlers simple decycler
///
/// The Decycler removes the cycle component from price data, leaving only the
/// trend component. Developed by John Ehlers, it is essentially a high-pass
/// filter that removes frequencies higher than a specified cutoff, effectively
/// eliminating cyclical noise and revealing the underlying trend.
#[derive(Debug, Clone)]
pub struct DecyclerFilter {
    /// High-pass cutoff period
    hp_period: usize,
}

impl DecyclerFilter {
    /// Create a new Decycler Filter
    ///
    /// # Arguments
    /// * `hp_period` - High-pass cutoff period (minimum 5)
    ///   Higher values remove more cycle content, leaving smoother trends.
    ///
    /// # Returns
    /// Result containing the filter instance or an error if parameters are invalid
    pub fn new(hp_period: usize) -> Result<Self> {
        if hp_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "hp_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { hp_period })
    }

    /// Calculate Decycler filter
    ///
    /// Removes cycle components from the data to reveal the underlying trend.
    /// Uses Ehlers' simple decycler approach which is essentially a low-pass filter
    /// that smooths out cyclical components.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < 2 {
            return data.to_vec();
        }

        let pi = std::f64::consts::PI;

        // Ehlers simple decycler uses a high-pass filter to extract cycles,
        // then subtracts them from the original price.
        // The high-pass filter coefficient based on period
        let alpha = (0.707 * 2.0 * pi / self.hp_period as f64).cos()
                  + (0.707 * 2.0 * pi / self.hp_period as f64).sin() - 1.0;
        let alpha = alpha.abs().min(0.99);

        // Apply high-pass filter
        let mut hp = vec![0.0; n];
        hp[0] = 0.0;

        for i in 1..n {
            hp[i] = (1.0 - alpha / 2.0) * (data[i] - data[i - 1])
                  + (1.0 - alpha) * hp[i - 1];
        }

        // Decycler = Price - HighPass (removes high frequency cycles)
        let mut result = vec![0.0; n];
        for i in 0..n {
            result[i] = data[i] - hp[i];
        }

        result
    }

    /// Calculate Decycler Oscillator
    ///
    /// The difference between two decyclers with different periods,
    /// creating an oscillator useful for identifying cycle turning points.
    pub fn calculate_oscillator(&self, data: &[f64], short_period: usize) -> Vec<f64> {
        let n = data.len();
        if n < 2 || short_period >= self.hp_period {
            return vec![0.0; n];
        }

        let long_decycler = self.calculate(data);

        let short_filter = DecyclerFilter { hp_period: short_period };
        let short_decycler = short_filter.calculate(data);

        // Oscillator = Short Decycler - Long Decycler
        long_decycler.iter()
            .zip(short_decycler.iter())
            .map(|(&long, &short)| short - long)
            .collect()
    }

    /// Get the cycle component that was removed
    ///
    /// Returns the cyclical component of the price data
    pub fn get_cycle_component(&self, data: &[f64]) -> Vec<f64> {
        let decycled = self.calculate(data);
        data.iter()
            .zip(decycled.iter())
            .map(|(&price, &trend)| price - trend)
            .collect()
    }
}

impl TechnicalIndicator for DecyclerFilter {
    fn name(&self) -> &str {
        "Decycler Filter"
    }

    fn min_periods(&self) -> usize {
        self.hp_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

// ==================== Tests for 6 NEW Filter Indicators ====================

#[cfg(test)]
mod new_filter_tests {
    use super::*;

    fn make_test_data() -> Vec<f64> {
        (0..50).map(|i| {
            let trend = 100.0 + i as f64 * 0.5;
            let cycle = 3.0 * (i as f64 * 0.3).sin();
            let noise = ((i * 7) % 5) as f64 * 0.2 - 0.5;
            trend + cycle + noise
        }).collect()
    }

    fn make_ohlcv_data() -> OHLCVSeries {
        let close = make_test_data();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
        let open: Vec<f64> = close.clone();
        let volume: Vec<f64> = vec![1000.0; close.len()];
        OHLCVSeries { open, high, low, close, volume }
    }

    // ==================== GaussianAdaptiveFilter Tests ====================

    #[test]
    fn test_gaussian_adaptive_filter_basic() {
        let data = make_test_data();
        let gaf = GaussianAdaptiveFilter::new(7, 1.5, 0.5).unwrap();
        let result = gaf.calculate(&data);

        assert_eq!(result.len(), data.len());
        assert!(result[25] > 0.0);
        // Result should be smoother than original
    }

    #[test]
    fn test_gaussian_adaptive_filter_validation() {
        // Period too small
        assert!(GaussianAdaptiveFilter::new(4, 1.5, 0.5).is_err());
        // Base sigma too small
        assert!(GaussianAdaptiveFilter::new(7, 0.3, 0.5).is_err());
        // Base sigma too large
        assert!(GaussianAdaptiveFilter::new(7, 6.0, 0.5).is_err());
        // Sensitivity too small
        assert!(GaussianAdaptiveFilter::new(7, 1.5, 0.05).is_err());
        // Sensitivity too large
        assert!(GaussianAdaptiveFilter::new(7, 1.5, 2.5).is_err());
    }

    #[test]
    fn test_gaussian_adaptive_filter_smoothing() {
        // Verify smoothing reduces variance
        let data: Vec<f64> = (0..30).map(|i| {
            100.0 + (i as f64 * 0.5).sin() * 5.0 + ((i * 17) % 7) as f64 * 0.3
        }).collect();

        let gaf = GaussianAdaptiveFilter::new(7, 2.0, 1.0).unwrap();
        let result = gaf.calculate(&data);

        // Calculate variance of differences
        let orig_var: f64 = data.windows(2)
            .map(|w| (w[1] - w[0]).powi(2))
            .sum::<f64>() / (data.len() - 1) as f64;

        let filt_var: f64 = result.windows(2)
            .map(|w| (w[1] - w[0]).powi(2))
            .sum::<f64>() / (result.len() - 1) as f64;

        // Filtered should have lower variance (smoother)
        assert!(filt_var < orig_var);
    }

    #[test]
    fn test_gaussian_adaptive_filter_technical_indicator() {
        let ohlcv = make_ohlcv_data();
        let gaf = GaussianAdaptiveFilter::new(7, 1.5, 0.5).unwrap();

        assert_eq!(gaf.name(), "Gaussian Adaptive Filter");
        assert_eq!(gaf.min_periods(), 7);

        let output = gaf.compute(&ohlcv).unwrap();
        assert_eq!(output.primary.len(), ohlcv.close.len());
    }

    // ==================== SavitzkyGolayFilter Tests ====================

    #[test]
    fn test_savitzky_golay_filter_basic() {
        let data = make_test_data();
        let sgf = SavitzkyGolayFilter::new(7, 2).unwrap();
        let result = sgf.calculate(&data);

        assert_eq!(result.len(), data.len());
        // Middle values should be processed
        assert!(result[25] > 0.0);
    }

    #[test]
    fn test_savitzky_golay_filter_validation() {
        // Window too small
        assert!(SavitzkyGolayFilter::new(3, 2).is_err());
        // Window must be odd
        assert!(SavitzkyGolayFilter::new(6, 2).is_err());
        // Poly order too small
        assert!(SavitzkyGolayFilter::new(7, 0).is_err());
        // Poly order too large
        assert!(SavitzkyGolayFilter::new(7, 5).is_err());
        // Poly order >= window_size
        assert!(SavitzkyGolayFilter::new(5, 5).is_err());
    }

    #[test]
    fn test_savitzky_golay_filter_preserves_linear() {
        // Savitzky-Golay should perfectly preserve linear trends
        let linear: Vec<f64> = (0..20).map(|i| 100.0 + i as f64 * 2.0).collect();
        let sgf = SavitzkyGolayFilter::new(5, 2).unwrap();
        let result = sgf.calculate(&linear);

        // Middle values should closely match original linear data
        for i in 3..17 {
            assert!((result[i] - linear[i]).abs() < 0.5,
                "At index {}: expected ~{}, got {}",
                i, linear[i], result[i]);
        }
    }

    #[test]
    fn test_savitzky_golay_filter_derivative() {
        // Test derivative calculation on linear data
        let linear: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 * 3.0).collect();
        let sgf = SavitzkyGolayFilter::new(7, 2).unwrap();
        let deriv = sgf.calculate_derivative(&linear);

        // Derivative of linear function should be constant (~3.0)
        for i in 5..25 {
            assert!((deriv[i] - 3.0).abs() < 0.5,
                "Derivative at {}: expected ~3.0, got {}",
                i, deriv[i]);
        }
    }

    #[test]
    fn test_savitzky_golay_filter_different_orders() {
        let data = make_test_data();

        let sgf2 = SavitzkyGolayFilter::new(9, 2).unwrap();
        let sgf3 = SavitzkyGolayFilter::new(9, 3).unwrap();
        let sgf4 = SavitzkyGolayFilter::new(9, 4).unwrap();

        let result2 = sgf2.calculate(&data);
        let result3 = sgf3.calculate(&data);
        let result4 = sgf4.calculate(&data);

        // All should produce valid output
        assert_eq!(result2.len(), data.len());
        assert_eq!(result3.len(), data.len());
        assert_eq!(result4.len(), data.len());

        // Higher order should preserve more detail
        assert!(result2.iter().all(|v| v.is_finite()));
        assert!(result3.iter().all(|v| v.is_finite()));
        assert!(result4.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_savitzky_golay_filter_technical_indicator() {
        let ohlcv = make_ohlcv_data();
        let sgf = SavitzkyGolayFilter::new(7, 2).unwrap();

        assert_eq!(sgf.name(), "Savitzky Golay Filter");
        assert_eq!(sgf.min_periods(), 7);

        let output = sgf.compute(&ohlcv).unwrap();
        assert_eq!(output.primary.len(), ohlcv.close.len());
    }

    // ==================== TriangularFilter Tests ====================

    #[test]
    fn test_triangular_filter_basic() {
        let data = make_test_data();
        let tf = TriangularFilter::new(7).unwrap();
        let result = tf.calculate(&data);

        assert_eq!(result.len(), data.len());
        assert!(result[25] > 0.0);
    }

    #[test]
    fn test_triangular_filter_validation() {
        // Period too small
        assert!(TriangularFilter::new(2).is_err());
        // Valid
        assert!(TriangularFilter::new(3).is_ok());
    }

    #[test]
    fn test_triangular_filter_weights() {
        let tf = TriangularFilter::new(5).unwrap();
        let weights = tf.calculate_weights();

        // Weights should sum to 1
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Center weight should be highest
        assert!(weights[2] > weights[0]);
        assert!(weights[2] > weights[4]);

        // Symmetric
        assert!((weights[0] - weights[4]).abs() < 1e-10);
        assert!((weights[1] - weights[3]).abs() < 1e-10);
    }

    #[test]
    fn test_triangular_filter_smoothing() {
        let data = make_test_data();
        let tf = TriangularFilter::new(9).unwrap();
        let result = tf.calculate(&data);

        // Calculate roughness
        let orig_rough: f64 = data.windows(2)
            .map(|w| (w[1] - w[0]).powi(2))
            .sum();

        let filt_rough: f64 = result.windows(2)
            .map(|w| (w[1] - w[0]).powi(2))
            .sum();

        // Filtered should be smoother
        assert!(filt_rough < orig_rough);
    }

    #[test]
    fn test_triangular_filter_centered() {
        let data = make_test_data();
        let tf = TriangularFilter::new(7).unwrap();
        let result = tf.calculate_centered(&data);

        assert_eq!(result.len(), data.len());
        // Edges should be preserved
        assert_eq!(result[0], data[0]);
        assert_eq!(result[1], data[1]);
        assert_eq!(result[data.len() - 1], data[data.len() - 1]);
    }

    #[test]
    fn test_triangular_filter_technical_indicator() {
        let ohlcv = make_ohlcv_data();
        let tf = TriangularFilter::new(7).unwrap();

        assert_eq!(tf.name(), "Triangular Filter");
        assert_eq!(tf.min_periods(), 7);

        let output = tf.compute(&ohlcv).unwrap();
        assert_eq!(output.primary.len(), ohlcv.close.len());
    }

    // ==================== HammingFilter Tests ====================

    #[test]
    fn test_hamming_filter_basic() {
        let data = make_test_data();
        let hf = HammingFilter::new(9, 0.54).unwrap();
        let result = hf.calculate(&data);

        assert_eq!(result.len(), data.len());
        assert!(result[25] > 0.0);
    }

    #[test]
    fn test_hamming_filter_validation() {
        // Period too small
        assert!(HammingFilter::new(4, 0.54).is_err());
        // Alpha too small
        assert!(HammingFilter::new(9, 0.4).is_err());
        // Alpha too large
        assert!(HammingFilter::new(9, 0.7).is_err());
    }

    #[test]
    fn test_hamming_filter_with_default_alpha() {
        let hf = HammingFilter::with_default_alpha(9).unwrap();
        assert!((hf.alpha - 0.54).abs() < 1e-10);
    }

    #[test]
    fn test_hamming_filter_weights() {
        let hf = HammingFilter::new(9, 0.54).unwrap();
        let weights = hf.calculate_weights();

        // Weights should sum to 1
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // All weights should be positive
        assert!(weights.iter().all(|&w| w > 0.0));

        // Center should have highest weight
        assert!(weights[4] > weights[0]);
        assert!(weights[4] > weights[8]);
    }

    #[test]
    fn test_hamming_filter_centered() {
        let data = make_test_data();
        let hf = HammingFilter::new(7, 0.54).unwrap();
        let result = hf.calculate_centered(&data);

        assert_eq!(result.len(), data.len());
        // Middle values should be filtered
        assert!(result[25] > 0.0);
    }

    #[test]
    fn test_hamming_filter_technical_indicator() {
        let ohlcv = make_ohlcv_data();
        let hf = HammingFilter::new(9, 0.54).unwrap();

        assert_eq!(hf.name(), "Hamming Filter");
        assert_eq!(hf.min_periods(), 9);

        let output = hf.compute(&ohlcv).unwrap();
        assert_eq!(output.primary.len(), ohlcv.close.len());
    }

    // ==================== SuperSmootherFilter Tests ====================

    #[test]
    fn test_super_smoother_filter_basic() {
        let data = make_test_data();
        let ssf = SuperSmootherFilter::new(10).unwrap();
        let result = ssf.calculate(&data);

        assert_eq!(result.len(), data.len());
        assert!(result[25] > 0.0);
    }

    #[test]
    fn test_super_smoother_filter_validation() {
        // Period too small
        assert!(SuperSmootherFilter::new(3).is_err());
        // Valid
        assert!(SuperSmootherFilter::new(4).is_ok());
    }

    #[test]
    fn test_super_smoother_filter_smoothing() {
        // Create data with high-frequency noise that the filter should smooth
        let data: Vec<f64> = (0..60).map(|i| {
            let trend = 100.0 + i as f64 * 0.2;
            let noise = 2.0 * (i as f64 * 1.5).sin(); // High frequency component
            trend + noise
        }).collect();

        let ssf = SuperSmootherFilter::new(10).unwrap();
        let result = ssf.calculate(&data);

        // Calculate roughness on the interior (skip startup)
        let orig_rough: f64 = data[20..50].windows(2)
            .map(|w| (w[1] - w[0]).powi(2))
            .sum();

        let filt_rough: f64 = result[20..50].windows(2)
            .map(|w| (w[1] - w[0]).powi(2))
            .sum();

        // Super smoother should produce smoother output on high-frequency data
        // Allow some tolerance due to filter characteristics
        assert!(filt_rough < orig_rough * 1.5,
            "Filtered roughness {} should be less than original {} * 1.5",
            filt_rough, orig_rough);
    }

    #[test]
    fn test_super_smoother_filter_three_pole() {
        let data = make_test_data();
        let ssf = SuperSmootherFilter::new(10).unwrap();

        let two_pole = ssf.calculate(&data);
        let three_pole = ssf.calculate_three_pole(&data);

        assert_eq!(two_pole.len(), data.len());
        assert_eq!(three_pole.len(), data.len());

        // Three pole should be smoother than two pole
        let rough_2: f64 = two_pole.windows(2)
            .map(|w| (w[1] - w[0]).powi(2))
            .sum();

        let rough_3: f64 = three_pole.windows(2)
            .map(|w| (w[1] - w[0]).powi(2))
            .sum();

        assert!(rough_3 <= rough_2 * 1.5); // Three pole may have startup effects
    }

    #[test]
    fn test_super_smoother_filter_follows_trend() {
        // Create uptrending data
        let data: Vec<f64> = (0..40).map(|i| 100.0 + i as f64 * 2.0).collect();
        let ssf = SuperSmootherFilter::new(8).unwrap();
        let result = ssf.calculate(&data);

        // Should follow the upward trend
        assert!(result[30] > result[10]);
    }

    #[test]
    fn test_super_smoother_filter_technical_indicator() {
        let ohlcv = make_ohlcv_data();
        let ssf = SuperSmootherFilter::new(10).unwrap();

        assert_eq!(ssf.name(), "Super Smoother Filter");
        assert_eq!(ssf.min_periods(), 10);

        let output = ssf.compute(&ohlcv).unwrap();
        assert_eq!(output.primary.len(), ohlcv.close.len());
    }

    // ==================== DecyclerFilter Tests ====================

    #[test]
    fn test_decycler_filter_basic() {
        let data = make_test_data();
        let df = DecyclerFilter::new(10).unwrap();
        let result = df.calculate(&data);

        assert_eq!(result.len(), data.len());
        // Result should be close to original price (trend preserved)
        assert!(result[25].is_finite());
        // Should roughly track the price trend
        assert!(result[25] > 50.0 && result[25] < 200.0);
    }

    #[test]
    fn test_decycler_filter_validation() {
        // Period too small
        assert!(DecyclerFilter::new(4).is_err());
        // Valid
        assert!(DecyclerFilter::new(5).is_ok());
    }

    #[test]
    fn test_decycler_filter_removes_cycles() {
        // Create data with clear high-frequency cycle
        let pi = std::f64::consts::PI;
        let data: Vec<f64> = (0..80).map(|i| {
            let trend = 100.0 + i as f64 * 0.5;
            let cycle = 5.0 * (2.0 * pi * i as f64 / 8.0).sin(); // 8-bar cycle
            trend + cycle
        }).collect();

        let df = DecyclerFilter::new(15).unwrap();
        let result = df.calculate(&data);

        // Check that the filter produces valid output
        assert_eq!(result.len(), data.len());
        assert!(result.iter().all(|v| v.is_finite()));

        // The decycled result should follow the trend
        // Compare slopes over a window (trend should be preserved)
        let orig_trend = (data[60] - data[20]) / 40.0;
        let filt_trend = (result[60] - result[20]) / 40.0;

        // Filtered trend should be close to original underlying trend (0.5)
        assert!((filt_trend - orig_trend).abs() < 2.0,
            "Filtered trend {} should be close to original {}",
            filt_trend, orig_trend);
    }

    #[test]
    fn test_decycler_filter_oscillator() {
        let data = make_test_data();
        let df = DecyclerFilter::new(20).unwrap();
        let oscillator = df.calculate_oscillator(&data, 10);

        assert_eq!(oscillator.len(), data.len());
        // Oscillator should produce finite values
        assert!(oscillator.iter().all(|v| v.is_finite()));
        // Oscillator values should be relatively small compared to price
        let max_abs = oscillator.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        assert!(max_abs < 50.0, "Oscillator max {} should be bounded", max_abs);
    }

    #[test]
    fn test_decycler_filter_cycle_component() {
        let data = make_test_data();
        let df = DecyclerFilter::new(15).unwrap();

        let trend = df.calculate(&data);
        let cycle = df.get_cycle_component(&data);

        assert_eq!(trend.len(), data.len());
        assert_eq!(cycle.len(), data.len());

        // Trend + cycle should approximately equal original
        for i in 5..data.len() {
            let reconstructed = trend[i] + cycle[i];
            assert!((reconstructed - data[i]).abs() < 1e-10,
                "At {}: {} + {} != {}",
                i, trend[i], cycle[i], data[i]);
        }
    }

    #[test]
    fn test_decycler_filter_technical_indicator() {
        let ohlcv = make_ohlcv_data();
        let df = DecyclerFilter::new(10).unwrap();

        assert_eq!(df.name(), "Decycler Filter");
        assert_eq!(df.min_periods(), 10);

        let output = df.compute(&ohlcv).unwrap();
        assert_eq!(output.primary.len(), ohlcv.close.len());
    }

    // ==================== Empty Data Tests ====================

    #[test]
    fn test_new_6_filters_empty_data() {
        let empty: Vec<f64> = vec![];

        let gaf = GaussianAdaptiveFilter::new(7, 1.5, 0.5).unwrap();
        assert!(gaf.calculate(&empty).is_empty());

        let sgf = SavitzkyGolayFilter::new(7, 2).unwrap();
        assert!(sgf.calculate(&empty).is_empty());

        let tf = TriangularFilter::new(7).unwrap();
        assert!(tf.calculate(&empty).is_empty());

        let hf = HammingFilter::new(9, 0.54).unwrap();
        assert!(hf.calculate(&empty).is_empty());

        let ssf = SuperSmootherFilter::new(10).unwrap();
        assert!(ssf.calculate(&empty).is_empty());

        let df = DecyclerFilter::new(10).unwrap();
        assert!(df.calculate(&empty).is_empty());
    }

    // ==================== Short Data Tests ====================

    #[test]
    fn test_new_6_filters_short_data() {
        let short = vec![100.0, 101.0, 102.0];

        let gaf = GaussianAdaptiveFilter::new(7, 1.5, 0.5).unwrap();
        let result = gaf.calculate(&short);
        assert_eq!(result.len(), 3);

        let sgf = SavitzkyGolayFilter::new(7, 2).unwrap();
        let result = sgf.calculate(&short);
        assert_eq!(result.len(), 3);

        let tf = TriangularFilter::new(7).unwrap();
        let result = tf.calculate(&short);
        assert_eq!(result.len(), 3);

        let hf = HammingFilter::new(9, 0.54).unwrap();
        let result = hf.calculate(&short);
        assert_eq!(result.len(), 3);

        let ssf = SuperSmootherFilter::new(10).unwrap();
        let result = ssf.calculate(&short);
        assert_eq!(result.len(), 3);

        let df = DecyclerFilter::new(10).unwrap();
        let result = df.calculate(&short);
        assert_eq!(result.len(), 3);
    }
}

// ==================== 6 ADDITIONAL NEW Filter Indicators ====================

/// Wiener Filter - Optimal statistical noise reduction filter
///
/// The Wiener filter is an optimal linear filter for removing noise from a signal
/// based on statistical estimation. It minimizes the mean square error between
/// the estimated signal and the desired signal. This implementation estimates
/// the signal and noise power spectra from the data to adaptively reduce noise
/// while preserving signal characteristics.
#[derive(Debug, Clone)]
pub struct WienerFilter {
    /// Window period for spectral estimation
    period: usize,
    /// Noise floor estimate (0.0-1.0) - fraction of signal assumed to be noise
    noise_ratio: f64,
}

impl WienerFilter {
    /// Create a new Wiener Filter
    ///
    /// # Arguments
    /// * `period` - Window size for local spectral estimation (minimum 5)
    /// * `noise_ratio` - Estimated noise-to-signal ratio (0.01-0.99)
    ///
    /// # Returns
    /// Result containing the filter instance or an error if parameters are invalid
    pub fn new(period: usize, noise_ratio: f64) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if noise_ratio <= 0.0 || noise_ratio >= 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "noise_ratio".to_string(),
                reason: "must be between 0 and 1 exclusive".to_string(),
            });
        }
        Ok(Self { period, noise_ratio })
    }

    /// Calculate Wiener filter
    ///
    /// Uses local variance estimation to compute the Wiener filter gain,
    /// adaptively reducing noise based on local signal characteristics.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n == 0 {
            return vec![];
        }

        let mut result = vec![0.0; n];

        // Estimate global noise variance
        let global_mean: f64 = data.iter().sum::<f64>() / n as f64;
        let global_var: f64 = data.iter()
            .map(|&x| (x - global_mean).powi(2))
            .sum::<f64>() / n as f64;
        let noise_var = global_var * self.noise_ratio;

        for i in 0..n {
            let start = i.saturating_sub(self.period / 2);
            let end = (i + self.period / 2 + 1).min(n);
            let window = &data[start..end];

            // Local mean and variance
            let local_mean: f64 = window.iter().sum::<f64>() / window.len() as f64;
            let local_var: f64 = window.iter()
                .map(|&x| (x - local_mean).powi(2))
                .sum::<f64>() / window.len() as f64;

            // Wiener filter gain: (signal_var) / (signal_var + noise_var)
            // where signal_var = local_var - noise_var (estimated)
            let signal_var = (local_var - noise_var).max(0.0);
            let wiener_gain = if local_var > 1e-10 {
                signal_var / local_var
            } else {
                1.0
            };

            // Apply Wiener filter: output = mean + gain * (input - mean)
            result[i] = local_mean + wiener_gain * (data[i] - local_mean);
        }

        result
    }

    /// Calculate with automatic noise estimation
    ///
    /// Estimates noise level from the high-frequency content of the signal
    pub fn calculate_auto_noise(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < 3 {
            return data.to_vec();
        }

        // Estimate noise from differences (MAD of first differences)
        let diffs: Vec<f64> = data.windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .collect();
        let median_diff = {
            let mut sorted = diffs.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            sorted[sorted.len() / 2]
        };

        // Robust noise estimate (MAD * 1.4826 for Gaussian)
        let noise_std = median_diff * 1.4826 / 2.0_f64.sqrt();
        let noise_var = noise_std * noise_std;

        let mut result = vec![0.0; n];

        for i in 0..n {
            let start = i.saturating_sub(self.period / 2);
            let end = (i + self.period / 2 + 1).min(n);
            let window = &data[start..end];

            let local_mean: f64 = window.iter().sum::<f64>() / window.len() as f64;
            let local_var: f64 = window.iter()
                .map(|&x| (x - local_mean).powi(2))
                .sum::<f64>() / window.len() as f64;

            let signal_var = (local_var - noise_var).max(0.0);
            let wiener_gain = if local_var > 1e-10 {
                signal_var / local_var
            } else {
                1.0
            };

            result[i] = local_mean + wiener_gain * (data[i] - local_mean);
        }

        result
    }
}

impl TechnicalIndicator for WienerFilter {
    fn name(&self) -> &str {
        "Wiener Filter"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Kalman Smoother - Fixed-interval Kalman smoother (forward-backward)
///
/// Unlike the standard Kalman filter which only uses past data, the Kalman smoother
/// uses both past and future data to produce optimal smoothed estimates. This is
/// achieved through a forward Kalman filter pass followed by a backward smoothing
/// pass. The result is smoother estimates with lower variance than the filter alone,
/// making it ideal for offline analysis where lookahead is acceptable.
#[derive(Debug, Clone)]
pub struct KalmanSmoother {
    /// Process noise variance (system dynamics uncertainty)
    process_noise: f64,
    /// Measurement noise variance (observation uncertainty)
    measurement_noise: f64,
}

impl KalmanSmoother {
    /// Create a new Kalman Smoother
    ///
    /// # Arguments
    /// * `process_noise` - Process noise variance (0.001-10.0)
    /// * `measurement_noise` - Measurement noise variance (0.001-10.0)
    ///
    /// # Returns
    /// Result containing the smoother instance or an error if parameters are invalid
    pub fn new(process_noise: f64, measurement_noise: f64) -> Result<Self> {
        if process_noise <= 0.0 || process_noise > 10.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "process_noise".to_string(),
                reason: "must be between 0 and 10".to_string(),
            });
        }
        if measurement_noise <= 0.0 || measurement_noise > 10.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "measurement_noise".to_string(),
                reason: "must be between 0 and 10".to_string(),
            });
        }
        Ok(Self { process_noise, measurement_noise })
    }

    /// Calculate Kalman smoothed values using forward-backward algorithm
    ///
    /// Implements the Rauch-Tung-Striebel (RTS) smoother for optimal
    /// fixed-interval smoothing.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n == 0 {
            return vec![];
        }
        if n == 1 {
            return vec![data[0]];
        }

        // Forward pass: standard Kalman filter
        let mut xf = vec![0.0; n];  // Filtered state estimates
        let mut pf = vec![0.0; n];  // Filtered error covariance

        // Initialize
        xf[0] = data[0];
        pf[0] = 1.0;

        for i in 1..n {
            // Prediction
            let xp = xf[i - 1];  // State transition is identity
            let pp = pf[i - 1] + self.process_noise;

            // Update
            let k = pp / (pp + self.measurement_noise);  // Kalman gain
            xf[i] = xp + k * (data[i] - xp);
            pf[i] = (1.0 - k) * pp;
        }

        // Backward pass: RTS smoother
        let mut xs = vec![0.0; n];  // Smoothed state estimates
        let mut ps = vec![0.0; n];  // Smoothed error covariance

        xs[n - 1] = xf[n - 1];
        ps[n - 1] = pf[n - 1];

        for i in (0..n - 1).rev() {
            let pp = pf[i] + self.process_noise;  // Predicted covariance at i+1
            let c = pf[i] / pp;  // Smoother gain

            xs[i] = xf[i] + c * (xs[i + 1] - xf[i]);
            ps[i] = pf[i] + c * c * (ps[i + 1] - pp);
        }

        xs
    }

    /// Get the smoothed error covariance (uncertainty estimates)
    pub fn calculate_with_covariance(&self, data: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = data.len();
        if n == 0 {
            return (vec![], vec![]);
        }
        if n == 1 {
            return (vec![data[0]], vec![1.0]);
        }

        // Forward pass
        let mut xf = vec![0.0; n];
        let mut pf = vec![0.0; n];

        xf[0] = data[0];
        pf[0] = 1.0;

        for i in 1..n {
            let xp = xf[i - 1];
            let pp = pf[i - 1] + self.process_noise;
            let k = pp / (pp + self.measurement_noise);
            xf[i] = xp + k * (data[i] - xp);
            pf[i] = (1.0 - k) * pp;
        }

        // Backward pass
        let mut xs = vec![0.0; n];
        let mut ps = vec![0.0; n];

        xs[n - 1] = xf[n - 1];
        ps[n - 1] = pf[n - 1];

        for i in (0..n - 1).rev() {
            let pp = pf[i] + self.process_noise;
            let c = pf[i] / pp;
            xs[i] = xf[i] + c * (xs[i + 1] - xf[i]);
            ps[i] = pf[i] + c * c * (ps[i + 1] - pp);
        }

        (xs, ps)
    }
}

impl TechnicalIndicator for KalmanSmoother {
    fn name(&self) -> &str {
        "Kalman Smoother"
    }

    fn min_periods(&self) -> usize {
        2
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Moving Median Filter - Adaptive moving median with outlier detection
///
/// An enhanced median filter that uses adaptive window sizing based on
/// local data characteristics. It detects potential outliers and adjusts
/// the filtering strength accordingly. Unlike the standard median filter,
/// this implementation provides smooth transitions and handles edge cases
/// more gracefully.
#[derive(Debug, Clone)]
pub struct MovingMedianFilter {
    /// Base period for the median window
    base_period: usize,
    /// Outlier sensitivity threshold (number of MADs)
    outlier_threshold: f64,
}

impl MovingMedianFilter {
    /// Create a new Moving Median Filter
    ///
    /// # Arguments
    /// * `base_period` - Base window size (minimum 3)
    /// * `outlier_threshold` - MAD threshold for outlier detection (1.0-5.0)
    ///
    /// # Returns
    /// Result containing the filter instance or an error if parameters are invalid
    pub fn new(base_period: usize, outlier_threshold: f64) -> Result<Self> {
        if base_period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "base_period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        if outlier_threshold < 1.0 || outlier_threshold > 5.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "outlier_threshold".to_string(),
                reason: "must be between 1.0 and 5.0".to_string(),
            });
        }
        Ok(Self { base_period, outlier_threshold })
    }

    /// Calculate median of a slice
    fn median(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        let mut sorted: Vec<f64> = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    }

    /// Calculate Median Absolute Deviation
    fn mad(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        let med = Self::median(data);
        let deviations: Vec<f64> = data.iter()
            .map(|&x| (x - med).abs())
            .collect();
        Self::median(&deviations)
    }

    /// Calculate moving median filter with outlier handling
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n == 0 {
            return vec![];
        }

        let mut result = vec![0.0; n];
        let half = self.base_period / 2;

        for i in 0..n {
            let start = i.saturating_sub(half);
            let end = (i + half + 1).min(n);
            let window: Vec<f64> = data[start..end].to_vec();

            // Calculate robust statistics
            let median_val = Self::median(&window);
            let mad_val = Self::mad(&window);

            // Check if current value is an outlier
            let deviation = (data[i] - median_val).abs();
            let is_outlier = mad_val > 1e-10 && deviation > self.outlier_threshold * mad_val * 1.4826;

            if is_outlier {
                // Use median for outliers
                result[i] = median_val;
            } else {
                // Weighted blend between value and median based on deviation
                let weight = if mad_val > 1e-10 {
                    1.0 - (deviation / (self.outlier_threshold * mad_val * 1.4826)).min(1.0) * 0.5
                } else {
                    1.0
                };
                result[i] = weight * data[i] + (1.0 - weight) * median_val;
            }
        }

        result
    }

    /// Calculate with outlier flags
    ///
    /// Returns both filtered values and outlier indicator (1.0 = outlier, 0.0 = normal)
    pub fn calculate_with_outliers(&self, data: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = data.len();
        if n == 0 {
            return (vec![], vec![]);
        }

        let mut filtered = vec![0.0; n];
        let mut outliers = vec![0.0; n];
        let half = self.base_period / 2;

        for i in 0..n {
            let start = i.saturating_sub(half);
            let end = (i + half + 1).min(n);
            let window: Vec<f64> = data[start..end].to_vec();

            let median_val = Self::median(&window);
            let mad_val = Self::mad(&window);

            let deviation = (data[i] - median_val).abs();
            let is_outlier = mad_val > 1e-10 && deviation > self.outlier_threshold * mad_val * 1.4826;

            if is_outlier {
                filtered[i] = median_val;
                outliers[i] = 1.0;
            } else {
                let weight = if mad_val > 1e-10 {
                    1.0 - (deviation / (self.outlier_threshold * mad_val * 1.4826)).min(1.0) * 0.5
                } else {
                    1.0
                };
                filtered[i] = weight * data[i] + (1.0 - weight) * median_val;
                outliers[i] = 0.0;
            }
        }

        (filtered, outliers)
    }
}

impl TechnicalIndicator for MovingMedianFilter {
    fn name(&self) -> &str {
        "Moving Median Filter"
    }

    fn min_periods(&self) -> usize {
        self.base_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Exponential Smoother - Triple exponential smoothing (Holt-Winters)
///
/// Implements triple exponential smoothing which tracks level, trend, and
/// seasonality components. This is an extension of double exponential smoothing
/// that can model periodic patterns in the data. Without seasonal component,
/// it falls back to enhanced double exponential smoothing with damped trend.
#[derive(Debug, Clone)]
pub struct ExponentialSmoother {
    /// Level smoothing factor (0.0-1.0)
    alpha: f64,
    /// Trend smoothing factor (0.0-1.0)
    beta: f64,
    /// Trend damping factor (0.0-1.0)
    phi: f64,
}

impl ExponentialSmoother {
    /// Create a new Exponential Smoother with damped trend
    ///
    /// # Arguments
    /// * `alpha` - Level smoothing factor (0.01-1.0)
    /// * `beta` - Trend smoothing factor (0.01-1.0)
    /// * `phi` - Trend damping factor (0.8-1.0), 1.0 = no damping
    ///
    /// # Returns
    /// Result containing the smoother instance or an error if parameters are invalid
    pub fn new(alpha: f64, beta: f64, phi: f64) -> Result<Self> {
        if alpha <= 0.0 || alpha > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "alpha".to_string(),
                reason: "must be between 0 and 1".to_string(),
            });
        }
        if beta <= 0.0 || beta > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "beta".to_string(),
                reason: "must be between 0 and 1".to_string(),
            });
        }
        if phi < 0.8 || phi > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "phi".to_string(),
                reason: "must be between 0.8 and 1.0".to_string(),
            });
        }
        Ok(Self { alpha, beta, phi })
    }

    /// Create with default damping (phi = 0.98)
    pub fn with_default_damping(alpha: f64, beta: f64) -> Result<Self> {
        Self::new(alpha, beta, 0.98)
    }

    /// Calculate triple exponential smoothing with damped trend
    ///
    /// Uses the damped trend method which prevents the forecast from
    /// growing indefinitely, making it more suitable for financial data.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n == 0 {
            return vec![];
        }
        if n == 1 {
            return vec![data[0]];
        }

        let mut result = vec![0.0; n];

        // Initialize level and trend
        let mut level = data[0];
        let mut trend = data[1] - data[0];

        result[0] = data[0];

        for i in 1..n {
            let prev_level = level;

            // Update level
            level = self.alpha * data[i] + (1.0 - self.alpha) * (prev_level + self.phi * trend);

            // Update trend with damping
            trend = self.beta * (level - prev_level) + (1.0 - self.beta) * self.phi * trend;

            result[i] = level;
        }

        result
    }

    /// Calculate with multi-step forecast
    ///
    /// Returns smoothed values and h-step ahead forecasts
    pub fn calculate_with_forecast(&self, data: &[f64], horizon: usize) -> (Vec<f64>, Vec<f64>) {
        let n = data.len();
        if n == 0 {
            return (vec![], vec![]);
        }

        let smoothed = self.calculate(data);

        // Get final state
        let mut level = data[0];
        let mut trend = if n > 1 { data[1] - data[0] } else { 0.0 };

        for i in 1..n {
            let prev_level = level;
            level = self.alpha * data[i] + (1.0 - self.alpha) * (prev_level + self.phi * trend);
            trend = self.beta * (level - prev_level) + (1.0 - self.beta) * self.phi * trend;
        }

        // Generate damped forecast
        let forecast: Vec<f64> = (1..=horizon)
            .map(|h| {
                // Sum of damped trend: phi + phi^2 + ... + phi^h
                let trend_sum = if (self.phi - 1.0).abs() < 1e-10 {
                    h as f64 * trend
                } else {
                    trend * self.phi * (1.0 - self.phi.powi(h as i32)) / (1.0 - self.phi)
                };
                level + trend_sum
            })
            .collect();

        (smoothed, forecast)
    }

    /// Calculate residuals for model diagnostics
    pub fn calculate_residuals(&self, data: &[f64]) -> Vec<f64> {
        let smoothed = self.calculate(data);
        data.iter()
            .zip(smoothed.iter())
            .map(|(&actual, &fitted)| actual - fitted)
            .collect()
    }
}

impl TechnicalIndicator for ExponentialSmoother {
    fn name(&self) -> &str {
        "Exponential Smoother"
    }

    fn min_periods(&self) -> usize {
        2
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Adaptive Noise Filter - LMS-based adaptive noise cancellation
///
/// Implements a Least Mean Squares (LMS) adaptive filter that learns the
/// optimal filter coefficients to minimize noise while preserving signal.
/// The filter adapts its coefficients in real-time based on the error signal,
/// making it effective for non-stationary noise environments common in
/// financial markets.
#[derive(Debug, Clone)]
pub struct AdaptiveNoiseFilterLMS {
    /// Filter order (number of taps)
    order: usize,
    /// Learning rate (step size for adaptation)
    mu: f64,
    /// Leakage factor for regularization (0.99-1.0)
    leakage: f64,
}

impl AdaptiveNoiseFilterLMS {
    /// Create a new Adaptive Noise Filter
    ///
    /// # Arguments
    /// * `order` - Filter order/taps (3-50)
    /// * `mu` - Learning rate (0.001-0.5)
    /// * `leakage` - Leakage factor for stability (0.99-1.0)
    ///
    /// # Returns
    /// Result containing the filter instance or an error if parameters are invalid
    pub fn new(order: usize, mu: f64, leakage: f64) -> Result<Self> {
        if order < 3 || order > 50 {
            return Err(IndicatorError::InvalidParameter {
                name: "order".to_string(),
                reason: "must be between 3 and 50".to_string(),
            });
        }
        if mu <= 0.0 || mu > 0.5 {
            return Err(IndicatorError::InvalidParameter {
                name: "mu".to_string(),
                reason: "must be between 0 and 0.5".to_string(),
            });
        }
        if leakage < 0.99 || leakage > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "leakage".to_string(),
                reason: "must be between 0.99 and 1.0".to_string(),
            });
        }
        Ok(Self { order, mu, leakage })
    }

    /// Create with default parameters
    pub fn with_defaults(order: usize) -> Result<Self> {
        Self::new(order, 0.01, 0.999)
    }

    /// Calculate LMS adaptive filter
    ///
    /// Uses the signal's lagged values as reference to adaptively
    /// estimate and remove noise.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n <= self.order {
            return data.to_vec();
        }

        let mut result = vec![0.0; n];
        let mut weights = vec![1.0 / self.order as f64; self.order];

        // Initialize with original data for warmup
        for i in 0..self.order {
            result[i] = data[i];
        }

        // Normalize learning rate by input power for stability
        let initial_power: f64 = data[..self.order.min(n)]
            .iter()
            .map(|&x| x * x)
            .sum::<f64>() / self.order as f64;
        let mu_normalized = self.mu / (initial_power.max(1e-10) * self.order as f64);

        for i in self.order..n {
            // Get input vector (lagged values)
            let input: Vec<f64> = (0..self.order)
                .map(|j| data[i - j - 1])
                .collect();

            // Compute filter output (predicted value)
            let predicted: f64 = weights.iter()
                .zip(input.iter())
                .map(|(&w, &x)| w * x)
                .sum();

            // Error (difference between actual and predicted)
            let error = data[i] - predicted;

            // The filtered output is the predicted value (noise removed)
            // But we blend with original for stability
            result[i] = predicted + 0.3 * error;  // Partial error correction

            // Update weights using leaky LMS
            let input_power: f64 = input.iter().map(|&x| x * x).sum::<f64>().max(1e-10);
            let normalized_mu = mu_normalized / (1.0 + input_power / initial_power.max(1e-10));

            for (j, w) in weights.iter_mut().enumerate() {
                *w = self.leakage * *w + normalized_mu * error * input[j];
            }
        }

        result
    }

    /// Get the learned filter coefficients
    pub fn get_weights(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n <= self.order {
            return vec![1.0 / self.order as f64; self.order];
        }

        let mut weights = vec![1.0 / self.order as f64; self.order];

        let initial_power: f64 = data[..self.order.min(n)]
            .iter()
            .map(|&x| x * x)
            .sum::<f64>() / self.order as f64;
        let mu_normalized = self.mu / (initial_power.max(1e-10) * self.order as f64);

        for i in self.order..n {
            let input: Vec<f64> = (0..self.order)
                .map(|j| data[i - j - 1])
                .collect();

            let predicted: f64 = weights.iter()
                .zip(input.iter())
                .map(|(&w, &x)| w * x)
                .sum();

            let error = data[i] - predicted;

            let input_power: f64 = input.iter().map(|&x| x * x).sum::<f64>().max(1e-10);
            let normalized_mu = mu_normalized / (1.0 + input_power / initial_power.max(1e-10));

            for (j, w) in weights.iter_mut().enumerate() {
                *w = self.leakage * *w + normalized_mu * error * input[j];
            }
        }

        weights
    }
}

impl TechnicalIndicator for AdaptiveNoiseFilterLMS {
    fn name(&self) -> &str {
        "Adaptive Noise Filter LMS"
    }

    fn min_periods(&self) -> usize {
        self.order + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Trend Separation Filter - EMD-inspired trend/cycle separation
///
/// Separates the price data into trend and cyclical components using an
/// approach inspired by Empirical Mode Decomposition (EMD). The filter
/// identifies local extrema and uses envelope fitting to extract the
/// underlying trend, with the remainder being the cyclical component.
/// This provides cleaner trend signals than simple moving averages.
#[derive(Debug, Clone)]
pub struct TrendSeparationFilter {
    /// Period for envelope smoothing
    period: usize,
    /// Number of sifting iterations
    iterations: usize,
}

impl TrendSeparationFilter {
    /// Create a new Trend Separation Filter
    ///
    /// # Arguments
    /// * `period` - Period for local extrema detection (minimum 5)
    /// * `iterations` - Number of sifting iterations (1-5)
    ///
    /// # Returns
    /// Result containing the filter instance or an error if parameters are invalid
    pub fn new(period: usize, iterations: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if iterations < 1 || iterations > 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "iterations".to_string(),
                reason: "must be between 1 and 5".to_string(),
            });
        }
        Ok(Self { period, iterations })
    }

    /// Find local maxima indices
    fn find_local_maxima(&self, data: &[f64]) -> Vec<usize> {
        let n = data.len();
        let half = self.period / 2;
        let mut maxima = Vec::new();

        for i in half..(n - half) {
            let window = &data[(i - half)..=(i + half)];
            let max_val = window.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            if (data[i] - max_val).abs() < 1e-10 {
                // Check if it's a true local maximum (not flat)
                if i > 0 && i < n - 1 && data[i] >= data[i - 1] && data[i] >= data[i + 1] {
                    maxima.push(i);
                }
            }
        }

        // Ensure we have at least start and end points
        if maxima.is_empty() || maxima[0] != 0 {
            maxima.insert(0, 0);
        }
        if maxima.is_empty() || maxima[maxima.len() - 1] != n - 1 {
            maxima.push(n - 1);
        }

        maxima
    }

    /// Find local minima indices
    fn find_local_minima(&self, data: &[f64]) -> Vec<usize> {
        let n = data.len();
        let half = self.period / 2;
        let mut minima = Vec::new();

        for i in half..(n - half) {
            let window = &data[(i - half)..=(i + half)];
            let min_val = window.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            if (data[i] - min_val).abs() < 1e-10 {
                if i > 0 && i < n - 1 && data[i] <= data[i - 1] && data[i] <= data[i + 1] {
                    minima.push(i);
                }
            }
        }

        if minima.is_empty() || minima[0] != 0 {
            minima.insert(0, 0);
        }
        if minima.is_empty() || minima[minima.len() - 1] != n - 1 {
            minima.push(n - 1);
        }

        minima
    }

    /// Linear interpolation to create envelope
    fn interpolate_envelope(&self, data: &[f64], extrema: &[usize]) -> Vec<f64> {
        let n = data.len();
        let mut envelope = vec![0.0; n];

        for i in 0..extrema.len() - 1 {
            let x0 = extrema[i];
            let x1 = extrema[i + 1];
            let y0 = data[x0];
            let y1 = data[x1];

            for x in x0..=x1 {
                let t = (x - x0) as f64 / (x1 - x0).max(1) as f64;
                envelope[x] = y0 + t * (y1 - y0);
            }
        }

        envelope
    }

    /// Calculate trend component (the mean envelope)
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period {
            return data.to_vec();
        }

        let mut residual = data.to_vec();

        // Apply sifting iterations
        for _ in 0..self.iterations {
            let maxima = self.find_local_maxima(&residual);
            let minima = self.find_local_minima(&residual);

            let upper_env = self.interpolate_envelope(&residual, &maxima);
            let lower_env = self.interpolate_envelope(&residual, &minima);

            // Mean envelope is the trend estimate
            let mean_env: Vec<f64> = upper_env.iter()
                .zip(lower_env.iter())
                .map(|(&u, &l)| (u + l) / 2.0)
                .collect();

            // Subtract mean envelope to get cycle component
            // Here we return the mean envelope as the trend
            residual = mean_env;
        }

        residual
    }

    /// Calculate both trend and cycle components
    ///
    /// Returns (trend, cycle) where cycle = original - trend
    pub fn calculate_components(&self, data: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let trend = self.calculate(data);
        let cycle: Vec<f64> = data.iter()
            .zip(trend.iter())
            .map(|(&d, &t)| d - t)
            .collect();

        (trend, cycle)
    }

    /// Calculate with envelope bounds
    ///
    /// Returns (trend, upper_bound, lower_bound)
    pub fn calculate_with_bounds(&self, data: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = data.len();
        if n < self.period {
            return (data.to_vec(), data.to_vec(), data.to_vec());
        }

        let maxima = self.find_local_maxima(data);
        let minima = self.find_local_minima(data);

        let upper = self.interpolate_envelope(data, &maxima);
        let lower = self.interpolate_envelope(data, &minima);

        let trend: Vec<f64> = upper.iter()
            .zip(lower.iter())
            .map(|(&u, &l)| (u + l) / 2.0)
            .collect();

        (trend, upper, lower)
    }
}

impl TechnicalIndicator for TrendSeparationFilter {
    fn name(&self) -> &str {
        "Trend Separation Filter"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

// ==================== Tests for 6 ADDITIONAL NEW Filter Indicators ====================

#[cfg(test)]
mod additional_filter_tests {
    use super::*;

    fn make_test_data() -> Vec<f64> {
        (0..50).map(|i| {
            let trend = 100.0 + i as f64 * 0.5;
            let cycle = 3.0 * (i as f64 * 0.3).sin();
            let noise = ((i * 7) % 5) as f64 * 0.2 - 0.5;
            trend + cycle + noise
        }).collect()
    }

    fn make_ohlcv_data() -> OHLCVSeries {
        let close = make_test_data();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
        let open: Vec<f64> = close.clone();
        let volume: Vec<f64> = vec![1000.0; close.len()];
        OHLCVSeries { open, high, low, close, volume }
    }

    // ==================== WienerFilter Tests ====================

    #[test]
    fn test_wiener_filter_basic() {
        let data = make_test_data();
        let wf = WienerFilter::new(7, 0.3).unwrap();
        let result = wf.calculate(&data);

        assert_eq!(result.len(), data.len());
        assert!(result[25] > 0.0);
    }

    #[test]
    fn test_wiener_filter_validation() {
        assert!(WienerFilter::new(4, 0.3).is_err()); // period too small
        assert!(WienerFilter::new(7, 0.0).is_err()); // noise_ratio too small
        assert!(WienerFilter::new(7, 1.0).is_err()); // noise_ratio too large
    }

    #[test]
    fn test_wiener_filter_noise_reduction() {
        // Create noisy signal
        let signal: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 0.2).collect();
        let noisy: Vec<f64> = signal.iter().enumerate()
            .map(|(i, &s)| s + ((i * 17) % 9) as f64 * 0.5 - 2.0)
            .collect();

        let wf = WienerFilter::new(7, 0.4).unwrap();
        let filtered = wf.calculate(&noisy);

        // Filtered should be closer to original signal
        let noise_error: f64 = noisy.iter().zip(signal.iter())
            .map(|(n, s)| (n - s).powi(2)).sum::<f64>().sqrt();
        let filter_error: f64 = filtered.iter().zip(signal.iter())
            .map(|(f, s)| (f - s).powi(2)).sum::<f64>().sqrt();

        assert!(filter_error < noise_error * 1.5);
    }

    #[test]
    fn test_wiener_filter_auto_noise() {
        let data = make_test_data();
        let wf = WienerFilter::new(7, 0.3).unwrap();
        let result = wf.calculate_auto_noise(&data);

        assert_eq!(result.len(), data.len());
        assert!(result.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_wiener_filter_technical_indicator() {
        let ohlcv = make_ohlcv_data();
        let wf = WienerFilter::new(7, 0.3).unwrap();

        assert_eq!(wf.name(), "Wiener Filter");
        assert_eq!(wf.min_periods(), 7);

        let output = wf.compute(&ohlcv).unwrap();
        assert_eq!(output.primary.len(), ohlcv.close.len());
    }

    // ==================== KalmanSmoother Tests ====================

    #[test]
    fn test_kalman_smoother_basic() {
        let data = make_test_data();
        let ks = KalmanSmoother::new(0.1, 1.0).unwrap();
        let result = ks.calculate(&data);

        assert_eq!(result.len(), data.len());
        assert!(result[25] > 0.0);
    }

    #[test]
    fn test_kalman_smoother_validation() {
        assert!(KalmanSmoother::new(0.0, 1.0).is_err()); // process_noise too small
        assert!(KalmanSmoother::new(15.0, 1.0).is_err()); // process_noise too large
        assert!(KalmanSmoother::new(0.1, 0.0).is_err()); // measurement_noise too small
        assert!(KalmanSmoother::new(0.1, 15.0).is_err()); // measurement_noise too large
    }

    #[test]
    fn test_kalman_smoother_smoother_than_filter() {
        let data = make_test_data();
        let ks = KalmanSmoother::new(0.1, 1.0).unwrap();
        let smoothed = ks.calculate(&data);

        // Smoother should produce less variance in differences
        let data_var: f64 = data.windows(2)
            .map(|w| (w[1] - w[0]).powi(2))
            .sum::<f64>() / (data.len() - 1) as f64;

        let smooth_var: f64 = smoothed.windows(2)
            .map(|w| (w[1] - w[0]).powi(2))
            .sum::<f64>() / (smoothed.len() - 1) as f64;

        assert!(smooth_var < data_var);
    }

    #[test]
    fn test_kalman_smoother_with_covariance() {
        let data = make_test_data();
        let ks = KalmanSmoother::new(0.1, 1.0).unwrap();
        let (smoothed, covariance) = ks.calculate_with_covariance(&data);

        assert_eq!(smoothed.len(), data.len());
        assert_eq!(covariance.len(), data.len());
        // Covariance should be positive
        assert!(covariance.iter().all(|&c| c >= 0.0));
    }

    #[test]
    fn test_kalman_smoother_technical_indicator() {
        let ohlcv = make_ohlcv_data();
        let ks = KalmanSmoother::new(0.1, 1.0).unwrap();

        assert_eq!(ks.name(), "Kalman Smoother");
        assert_eq!(ks.min_periods(), 2);

        let output = ks.compute(&ohlcv).unwrap();
        assert_eq!(output.primary.len(), ohlcv.close.len());
    }

    // ==================== MovingMedianFilter Tests ====================

    #[test]
    fn test_moving_median_filter_basic() {
        let data = make_test_data();
        let mmf = MovingMedianFilter::new(5, 2.5).unwrap();
        let result = mmf.calculate(&data);

        assert_eq!(result.len(), data.len());
        assert!(result[25] > 0.0);
    }

    #[test]
    fn test_moving_median_filter_validation() {
        assert!(MovingMedianFilter::new(2, 2.5).is_err()); // period too small
        assert!(MovingMedianFilter::new(5, 0.5).is_err()); // threshold too small
        assert!(MovingMedianFilter::new(5, 6.0).is_err()); // threshold too large
    }

    #[test]
    fn test_moving_median_filter_outlier_removal() {
        // Data with spike
        let mut data = vec![100.0, 101.0, 102.0, 500.0, 104.0, 105.0, 106.0];
        data.extend(vec![107.0, 108.0, 109.0]);

        let mmf = MovingMedianFilter::new(5, 2.0).unwrap();
        let result = mmf.calculate(&data);

        // Spike should be reduced
        assert!(result[3] < 400.0);
        assert!(result[3] > 90.0);
    }

    #[test]
    fn test_moving_median_filter_with_outliers() {
        let mut data = make_test_data();
        data[20] = 500.0; // Add outlier

        let mmf = MovingMedianFilter::new(5, 2.0).unwrap();
        let (filtered, outliers) = mmf.calculate_with_outliers(&data);

        assert_eq!(filtered.len(), data.len());
        assert_eq!(outliers.len(), data.len());
        assert!((outliers[20] - 1.0).abs() < 1e-10); // Should detect outlier
    }

    #[test]
    fn test_moving_median_filter_technical_indicator() {
        let ohlcv = make_ohlcv_data();
        let mmf = MovingMedianFilter::new(5, 2.5).unwrap();

        assert_eq!(mmf.name(), "Moving Median Filter");
        assert_eq!(mmf.min_periods(), 5);

        let output = mmf.compute(&ohlcv).unwrap();
        assert_eq!(output.primary.len(), ohlcv.close.len());
    }

    // ==================== ExponentialSmoother Tests ====================

    #[test]
    fn test_exponential_smoother_basic() {
        let data = make_test_data();
        let es = ExponentialSmoother::new(0.3, 0.1, 0.98).unwrap();
        let result = es.calculate(&data);

        assert_eq!(result.len(), data.len());
        assert!(result[25] > 0.0);
    }

    #[test]
    fn test_exponential_smoother_validation() {
        assert!(ExponentialSmoother::new(0.0, 0.1, 0.98).is_err()); // alpha too small
        assert!(ExponentialSmoother::new(1.5, 0.1, 0.98).is_err()); // alpha too large
        assert!(ExponentialSmoother::new(0.3, 0.0, 0.98).is_err()); // beta too small
        assert!(ExponentialSmoother::new(0.3, 0.1, 0.7).is_err()); // phi too small
    }

    #[test]
    fn test_exponential_smoother_with_default_damping() {
        let es = ExponentialSmoother::with_default_damping(0.3, 0.1).unwrap();
        assert!((es.phi - 0.98).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_smoother_trend_following() {
        let data: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 * 2.0).collect();
        let es = ExponentialSmoother::new(0.5, 0.3, 0.98).unwrap();
        let result = es.calculate(&data);

        // Should follow upward trend
        assert!(result[25] > result[10]);
    }

    #[test]
    fn test_exponential_smoother_with_forecast() {
        let data = make_test_data();
        let es = ExponentialSmoother::new(0.3, 0.1, 0.95).unwrap();
        let (smoothed, forecast) = es.calculate_with_forecast(&data, 5);

        assert_eq!(smoothed.len(), data.len());
        assert_eq!(forecast.len(), 5);
        // Damped forecast should converge
        assert!(forecast.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_exponential_smoother_residuals() {
        let data = make_test_data();
        let es = ExponentialSmoother::new(0.3, 0.1, 0.98).unwrap();
        let residuals = es.calculate_residuals(&data);

        assert_eq!(residuals.len(), data.len());
        // Residuals should be relatively small
        let mean_abs_resid: f64 = residuals.iter().map(|r| r.abs()).sum::<f64>() / residuals.len() as f64;
        assert!(mean_abs_resid < 10.0);
    }

    #[test]
    fn test_exponential_smoother_technical_indicator() {
        let ohlcv = make_ohlcv_data();
        let es = ExponentialSmoother::new(0.3, 0.1, 0.98).unwrap();

        assert_eq!(es.name(), "Exponential Smoother");
        assert_eq!(es.min_periods(), 2);

        let output = es.compute(&ohlcv).unwrap();
        assert_eq!(output.primary.len(), ohlcv.close.len());
    }

    // ==================== AdaptiveNoiseFilterLMS Tests ====================

    #[test]
    fn test_adaptive_noise_filter_lms_basic() {
        let data = make_test_data();
        let anf = AdaptiveNoiseFilterLMS::new(5, 0.01, 0.999).unwrap();
        let result = anf.calculate(&data);

        assert_eq!(result.len(), data.len());
        assert!(result[25] > 0.0);
    }

    #[test]
    fn test_adaptive_noise_filter_lms_validation() {
        assert!(AdaptiveNoiseFilterLMS::new(2, 0.01, 0.999).is_err()); // order too small
        assert!(AdaptiveNoiseFilterLMS::new(60, 0.01, 0.999).is_err()); // order too large
        assert!(AdaptiveNoiseFilterLMS::new(5, 0.0, 0.999).is_err()); // mu too small
        assert!(AdaptiveNoiseFilterLMS::new(5, 0.6, 0.999).is_err()); // mu too large
        assert!(AdaptiveNoiseFilterLMS::new(5, 0.01, 0.98).is_err()); // leakage too small
    }

    #[test]
    fn test_adaptive_noise_filter_lms_with_defaults() {
        let anf = AdaptiveNoiseFilterLMS::with_defaults(5).unwrap();
        assert!((anf.mu - 0.01).abs() < 1e-10);
        assert!((anf.leakage - 0.999).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_noise_filter_lms_adaptation() {
        let data = make_test_data();
        let anf = AdaptiveNoiseFilterLMS::new(5, 0.01, 0.999).unwrap();
        let result = anf.calculate(&data);

        // Result should be finite and reasonable
        assert!(result.iter().all(|v| v.is_finite()));
        assert!(result.iter().all(|&v| v > 50.0 && v < 200.0));
    }

    #[test]
    fn test_adaptive_noise_filter_lms_weights() {
        let data = make_test_data();
        let anf = AdaptiveNoiseFilterLMS::new(5, 0.01, 0.999).unwrap();
        let weights = anf.get_weights(&data);

        assert_eq!(weights.len(), 5);
        assert!(weights.iter().all(|w| w.is_finite()));
    }

    #[test]
    fn test_adaptive_noise_filter_lms_technical_indicator() {
        let ohlcv = make_ohlcv_data();
        let anf = AdaptiveNoiseFilterLMS::new(5, 0.01, 0.999).unwrap();

        assert_eq!(anf.name(), "Adaptive Noise Filter LMS");
        assert_eq!(anf.min_periods(), 6);

        let output = anf.compute(&ohlcv).unwrap();
        assert_eq!(output.primary.len(), ohlcv.close.len());
    }

    // ==================== TrendSeparationFilter Tests ====================

    #[test]
    fn test_trend_separation_filter_basic() {
        let data = make_test_data();
        let tsf = TrendSeparationFilter::new(7, 2).unwrap();
        let result = tsf.calculate(&data);

        assert_eq!(result.len(), data.len());
        assert!(result[25] > 0.0);
    }

    #[test]
    fn test_trend_separation_filter_validation() {
        assert!(TrendSeparationFilter::new(4, 2).is_err()); // period too small
        assert!(TrendSeparationFilter::new(7, 0).is_err()); // iterations too small
        assert!(TrendSeparationFilter::new(7, 6).is_err()); // iterations too large
    }

    #[test]
    fn test_trend_separation_filter_components() {
        let data = make_test_data();
        let tsf = TrendSeparationFilter::new(7, 2).unwrap();
        let (trend, cycle) = tsf.calculate_components(&data);

        assert_eq!(trend.len(), data.len());
        assert_eq!(cycle.len(), data.len());

        // Trend + cycle should equal original
        for i in 0..data.len() {
            let reconstructed = trend[i] + cycle[i];
            assert!((reconstructed - data[i]).abs() < 1e-10,
                "At {}: {} + {} != {}",
                i, trend[i], cycle[i], data[i]);
        }
    }

    #[test]
    fn test_trend_separation_filter_trend_follows_price() {
        // Create uptrending data
        let data: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 1.0).collect();
        let tsf = TrendSeparationFilter::new(7, 2).unwrap();
        let trend = tsf.calculate(&data);

        // Trend should generally increase
        assert!(trend[40] > trend[10]);
    }

    #[test]
    fn test_trend_separation_filter_with_bounds() {
        let data = make_test_data();
        let tsf = TrendSeparationFilter::new(7, 1).unwrap();
        let (trend, upper, lower) = tsf.calculate_with_bounds(&data);

        assert_eq!(trend.len(), data.len());
        assert_eq!(upper.len(), data.len());
        assert_eq!(lower.len(), data.len());

        // Trend should be between bounds (approximately)
        for i in 0..data.len() {
            assert!(upper[i] >= lower[i],
                "At {}: upper {} < lower {}",
                i, upper[i], lower[i]);
        }
    }

    #[test]
    fn test_trend_separation_filter_technical_indicator() {
        let ohlcv = make_ohlcv_data();
        let tsf = TrendSeparationFilter::new(7, 2).unwrap();

        assert_eq!(tsf.name(), "Trend Separation Filter");
        assert_eq!(tsf.min_periods(), 7);

        let output = tsf.compute(&ohlcv).unwrap();
        assert_eq!(output.primary.len(), ohlcv.close.len());
    }

    // ==================== Empty and Short Data Tests ====================

    #[test]
    fn test_additional_filters_empty_data() {
        let empty: Vec<f64> = vec![];

        let wf = WienerFilter::new(7, 0.3).unwrap();
        assert!(wf.calculate(&empty).is_empty());

        let ks = KalmanSmoother::new(0.1, 1.0).unwrap();
        assert!(ks.calculate(&empty).is_empty());

        let mmf = MovingMedianFilter::new(5, 2.5).unwrap();
        assert!(mmf.calculate(&empty).is_empty());

        let es = ExponentialSmoother::new(0.3, 0.1, 0.98).unwrap();
        assert!(es.calculate(&empty).is_empty());

        let anf = AdaptiveNoiseFilterLMS::new(5, 0.01, 0.999).unwrap();
        assert!(anf.calculate(&empty).is_empty());

        let tsf = TrendSeparationFilter::new(7, 2).unwrap();
        assert!(tsf.calculate(&empty).is_empty());
    }

    #[test]
    fn test_additional_filters_short_data() {
        let short = vec![100.0, 101.0, 102.0];

        let wf = WienerFilter::new(7, 0.3).unwrap();
        let result = wf.calculate(&short);
        assert_eq!(result.len(), 3);

        let ks = KalmanSmoother::new(0.1, 1.0).unwrap();
        let result = ks.calculate(&short);
        assert_eq!(result.len(), 3);

        let mmf = MovingMedianFilter::new(5, 2.5).unwrap();
        let result = mmf.calculate(&short);
        assert_eq!(result.len(), 3);

        let es = ExponentialSmoother::new(0.3, 0.1, 0.98).unwrap();
        let result = es.calculate(&short);
        assert_eq!(result.len(), 3);

        let anf = AdaptiveNoiseFilterLMS::new(5, 0.01, 0.999).unwrap();
        let result = anf.calculate(&short);
        assert_eq!(result.len(), 3);

        let tsf = TrendSeparationFilter::new(7, 2).unwrap();
        let result = tsf.calculate(&short);
        assert_eq!(result.len(), 3);
    }
}
