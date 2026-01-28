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
}
