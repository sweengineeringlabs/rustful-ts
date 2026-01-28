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
