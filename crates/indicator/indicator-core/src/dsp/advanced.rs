//! Advanced DSP Indicators
//!
//! Advanced digital signal processing indicators for frequency analysis,
//! phase synchronization, and signal strength measurement.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Adaptive Frequency Filter - Filter that adapts to dominant frequency
///
/// This indicator detects the dominant frequency in price data and applies
/// a filter that adapts its cutoff frequency based on the detected cycle.
#[derive(Debug, Clone)]
pub struct AdaptiveFrequencyFilter {
    min_period: usize,
    max_period: usize,
    smoothing_factor: f64,
}

impl AdaptiveFrequencyFilter {
    pub fn new(min_period: usize, max_period: usize, smoothing_factor: f64) -> Result<Self> {
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
        if smoothing_factor <= 0.0 || smoothing_factor > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing_factor".to_string(),
                reason: "must be between 0 (exclusive) and 1 (inclusive)".to_string(),
            });
        }
        Ok(Self { min_period, max_period, smoothing_factor })
    }

    /// Detect dominant period using autocorrelation
    fn detect_dominant_period(&self, slice: &[f64]) -> usize {
        let len = slice.len();
        let mean: f64 = slice.iter().sum::<f64>() / len as f64;

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

        best_period
    }

    /// Calculate adaptive frequency filter output
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];
        let mut prev_period = self.min_period as f64;

        for i in self.max_period..n {
            let start = i.saturating_sub(self.max_period);
            let slice = &close[start..=i];

            // Detect dominant period
            let detected = self.detect_dominant_period(slice) as f64;

            // Smooth the period transition
            let smooth_period = self.smoothing_factor * detected + (1.0 - self.smoothing_factor) * prev_period;
            prev_period = smooth_period;

            // Calculate adaptive alpha based on detected frequency
            let alpha = 2.0 / (smooth_period + 1.0);

            // Apply EMA with adaptive alpha
            if i == self.max_period {
                result[i] = close[i];
            } else {
                result[i] = alpha * close[i] + (1.0 - alpha) * result[i - 1];
            }
        }

        result
    }
}

impl TechnicalIndicator for AdaptiveFrequencyFilter {
    fn name(&self) -> &str {
        "Adaptive Frequency Filter"
    }

    fn min_periods(&self) -> usize {
        self.max_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Trend Phase Analysis - Analyzes the phase of the trend cycle
///
/// Determines where price is within its current trend cycle,
/// providing phase information in degrees (0-360).
#[derive(Debug, Clone)]
pub struct TrendPhaseAnalysis {
    period: usize,
}

impl TrendPhaseAnalysis {
    pub fn new(period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate trend phase using Hilbert transform approximation
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // First, detrend the data using linear regression
        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let slice = &close[start..=i];
            let len = slice.len() as f64;

            // Linear detrending
            let x_mean = (len - 1.0) / 2.0;
            let y_mean: f64 = slice.iter().sum::<f64>() / len;

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

            // Get detrended value at current position
            let detrended_val = close[i] - (intercept + slope * (len - 1.0));

            // Calculate in-phase (I) and quadrature (Q) components
            // Using simplified Hilbert transform via weighted moving average
            let half_period = self.period / 2;
            let mut in_phase = 0.0;
            let mut quadrature = 0.0;

            for j in 0..self.period.min(i - start + 1) {
                let idx = i - j;
                if idx >= start {
                    let trend_at_j = intercept + slope * ((len - 1.0) - j as f64);
                    let det_val = close[idx] - trend_at_j;

                    // Cosine and sine weights for I and Q
                    let angle = 2.0 * std::f64::consts::PI * j as f64 / self.period as f64;
                    in_phase += det_val * angle.cos();
                    quadrature += det_val * angle.sin();
                }
            }

            in_phase /= half_period as f64;
            quadrature /= half_period as f64;

            // Calculate phase angle
            let phase = quadrature.atan2(in_phase);
            // Convert to degrees (0-360)
            let phase_degrees = (phase * 180.0 / std::f64::consts::PI + 360.0) % 360.0;

            result[i] = phase_degrees;
        }

        result
    }
}

impl TechnicalIndicator for TrendPhaseAnalysis {
    fn name(&self) -> &str {
        "Trend Phase Analysis"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Cycle Mode Indicator - Detects dominant cycle mode (trending/cycling)
///
/// Returns a value between 0 and 100 where:
/// - 0-30: Strong cycling/mean-reverting behavior
/// - 30-70: Mixed/transition mode
/// - 70-100: Strong trending behavior
#[derive(Debug, Clone)]
pub struct CycleModeIndicator {
    period: usize,
}

impl CycleModeIndicator {
    pub fn new(period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate cycle mode indicator
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let slice = &close[start..=i];
            let len = slice.len() as f64;

            // Calculate trend strength via R-squared of linear regression
            let x_mean = (len - 1.0) / 2.0;
            let y_mean: f64 = slice.iter().sum::<f64>() / len;

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

            // Calculate R-squared
            let mut ss_res = 0.0;
            let mut ss_tot = 0.0;
            for (j, &val) in slice.iter().enumerate() {
                let predicted = intercept + slope * j as f64;
                ss_res += (val - predicted).powi(2);
                ss_tot += (val - y_mean).powi(2);
            }

            let r_squared = if ss_tot > 1e-10 { 1.0 - (ss_res / ss_tot) } else { 0.0 };

            // Calculate cycle strength via autocorrelation at half-period
            let half_period = self.period / 2;
            let mut auto_corr = 0.0;
            let mut norm1 = 0.0;
            let mut norm2 = 0.0;

            for j in 0..(slice.len() - half_period) {
                let x = slice[j] - y_mean;
                let y = slice[j + half_period] - y_mean;
                auto_corr += x * y;
                norm1 += x * x;
                norm2 += y * y;
            }

            let norm = (norm1 * norm2).sqrt();
            let cycle_corr = if norm > 1e-10 { auto_corr / norm } else { 0.0 };

            // Negative autocorrelation suggests cycling behavior
            // Combine R-squared (trend) with -autocorrelation (cycle)
            // R-squared high and autocorr positive = trending
            // R-squared low and autocorr negative = cycling

            let trend_score = r_squared.max(0.0).min(1.0);
            let cycle_score = (-cycle_corr).max(0.0).min(1.0);

            // Mode: 100 = pure trend, 0 = pure cycle
            let mode = trend_score * 100.0 - cycle_score * 30.0;
            result[i] = mode.max(0.0).min(100.0);
        }

        result
    }
}

impl TechnicalIndicator for CycleModeIndicator {
    fn name(&self) -> &str {
        "Cycle Mode Indicator"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Signal Strength Meter - Measures signal strength vs noise
///
/// Calculates the ratio of coherent price movement (signal) to
/// random fluctuations (noise), expressed as a percentage.
#[derive(Debug, Clone)]
pub struct SignalStrengthMeter {
    period: usize,
    noise_period: usize,
}

impl SignalStrengthMeter {
    pub fn new(period: usize, noise_period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if noise_period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "noise_period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        if noise_period >= period {
            return Err(IndicatorError::InvalidParameter {
                name: "noise_period".to_string(),
                reason: "must be less than period".to_string(),
            });
        }
        Ok(Self { period, noise_period })
    }

    /// Calculate signal strength meter
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let slice = &close[start..=i];

            // Calculate signal power: variance of smoothed (low-pass filtered) data
            let mut smoothed = Vec::with_capacity(slice.len());
            for j in 0..slice.len() {
                if j < self.noise_period {
                    smoothed.push(slice[j]);
                } else {
                    let sum: f64 = slice[(j - self.noise_period + 1)..=j].iter().sum();
                    smoothed.push(sum / self.noise_period as f64);
                }
            }

            let smooth_mean: f64 = smoothed.iter().sum::<f64>() / smoothed.len() as f64;
            let signal_power: f64 = smoothed.iter()
                .map(|x| (x - smooth_mean).powi(2))
                .sum::<f64>() / smoothed.len() as f64;

            // Calculate noise power: variance of high-frequency residuals
            let mut noise_power = 0.0;
            let mut noise_count = 0;
            for j in self.noise_period..slice.len() {
                let residual = slice[j] - smoothed[j];
                noise_power += residual.powi(2);
                noise_count += 1;
            }
            noise_power = if noise_count > 0 { noise_power / noise_count as f64 } else { 1e-10 };

            // Signal strength as percentage (0-100)
            let total_power = signal_power + noise_power;
            let strength = if total_power > 1e-10 {
                (signal_power / total_power * 100.0).min(100.0)
            } else {
                0.0
            };

            result[i] = strength;
        }

        result
    }
}

impl TechnicalIndicator for SignalStrengthMeter {
    fn name(&self) -> &str {
        "Signal Strength Meter"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Frequency Response Indicator - Measures frequency response of price
///
/// Analyzes the frequency content of price movements across different
/// time scales, returning a composite frequency response measure.
#[derive(Debug, Clone)]
pub struct FrequencyResponseIndicator {
    period: usize,
    num_bands: usize,
}

impl FrequencyResponseIndicator {
    pub fn new(period: usize, num_bands: usize) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        if num_bands < 2 || num_bands > 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "num_bands".to_string(),
                reason: "must be between 2 and 10".to_string(),
            });
        }
        Ok(Self { period, num_bands })
    }

    /// Calculate frequency response at a specific period
    fn band_power(&self, slice: &[f64], band_period: usize) -> f64 {
        let len = slice.len();
        if band_period >= len / 2 {
            return 0.0;
        }

        let mean: f64 = slice.iter().sum::<f64>() / len as f64;

        // Calculate power via autocorrelation at the band period
        let mut corr = 0.0;
        let mut norm = 0.0;

        for j in 0..(len - band_period) {
            let x = slice[j] - mean;
            let y = slice[j + band_period] - mean;
            corr += x * y;
            norm += x * x;
        }

        if norm > 1e-10 {
            (corr / norm).abs()
        } else {
            0.0
        }
    }

    /// Calculate frequency response indicator
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // Calculate band periods (logarithmically spaced)
        let min_band = 3;
        let max_band = self.period / 2;
        let band_periods: Vec<usize> = (0..self.num_bands)
            .map(|i| {
                let t = i as f64 / (self.num_bands - 1) as f64;
                let log_period = (min_band as f64).ln() + t * ((max_band as f64).ln() - (min_band as f64).ln());
                log_period.exp() as usize
            })
            .collect();

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let slice = &close[start..=i];

            // Calculate power at each frequency band
            let mut total_power = 0.0;
            let mut weighted_freq = 0.0;

            for (band_idx, &band_period) in band_periods.iter().enumerate() {
                let power = self.band_power(slice, band_period);
                total_power += power;
                // Higher frequency (lower period) gets higher weight
                let freq_weight = (self.num_bands - band_idx) as f64;
                weighted_freq += power * freq_weight;
            }

            // Frequency response: weighted average normalized by total power
            if total_power > 1e-10 {
                result[i] = (weighted_freq / total_power / self.num_bands as f64) * 100.0;
            }
        }

        result
    }
}

impl TechnicalIndicator for FrequencyResponseIndicator {
    fn name(&self) -> &str {
        "Frequency Response Indicator"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Phase Synchronization - Detects phase synchronization in price
///
/// Measures the degree of phase synchronization between price and its
/// detrended component, indicating cycle coherence.
#[derive(Debug, Clone)]
pub struct PhaseSynchronization {
    period: usize,
    reference_period: usize,
}

impl PhaseSynchronization {
    pub fn new(period: usize, reference_period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if reference_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "reference_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if reference_period >= period {
            return Err(IndicatorError::InvalidParameter {
                name: "reference_period".to_string(),
                reason: "must be less than period".to_string(),
            });
        }
        Ok(Self { period, reference_period })
    }

    /// Calculate phase at a given index using Hilbert transform approximation
    fn calculate_phase(&self, slice: &[f64], period: usize) -> f64 {
        let len = slice.len();
        if len < period {
            return 0.0;
        }

        let mean: f64 = slice.iter().sum::<f64>() / len as f64;

        // Calculate in-phase and quadrature components
        let mut in_phase = 0.0;
        let mut quadrature = 0.0;

        for (j, &val) in slice.iter().enumerate() {
            let angle = 2.0 * std::f64::consts::PI * j as f64 / period as f64;
            let centered = val - mean;
            in_phase += centered * angle.cos();
            quadrature += centered * angle.sin();
        }

        quadrature.atan2(in_phase)
    }

    /// Calculate phase synchronization index
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let slice = &close[start..=i];
            let len = slice.len() as f64;

            // Detrend using linear regression
            let x_mean = (len - 1.0) / 2.0;
            let y_mean: f64 = slice.iter().sum::<f64>() / len;

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

            // Create detrended series
            let detrended: Vec<f64> = slice.iter().enumerate()
                .map(|(j, &val)| val - (intercept + slope * j as f64))
                .collect();

            // Calculate phases at the reference period
            let phase1 = self.calculate_phase(slice, self.reference_period);
            let phase2 = self.calculate_phase(&detrended, self.reference_period);

            // Phase difference
            let phase_diff = (phase1 - phase2).abs();

            // Synchronization index: 1 when phases aligned, 0 when orthogonal
            // cos^2 of phase difference gives synchronization measure
            let sync_index = phase_diff.cos().powi(2);

            result[i] = sync_index * 100.0; // Scale to 0-100
        }

        result
    }
}

impl TechnicalIndicator for PhaseSynchronization {
    fn name(&self) -> &str {
        "Phase Synchronization"
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
        // Create data with trend + cycle component
        (0..60)
            .map(|i| {
                let trend = 100.0 + i as f64 * 0.5;
                let cycle = (i as f64 * 0.3).sin() * 5.0;
                let noise = ((i * 17) % 7) as f64 * 0.1 - 0.3;
                trend + cycle + noise
            })
            .collect()
    }

    fn make_ohlcv_series() -> OHLCVSeries {
        let close = make_test_data();
        OHLCVSeries::from_close(close)
    }

    // =====================================================================
    // AdaptiveFrequencyFilter Tests
    // =====================================================================

    #[test]
    fn test_adaptive_frequency_filter_basic() {
        let close = make_test_data();
        let aff = AdaptiveFrequencyFilter::new(5, 20, 0.5).unwrap();
        let result = aff.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Values after warmup should be non-zero
        for i in 25..result.len() {
            assert!(result[i] > 0.0, "Expected positive value at index {}", i);
        }
    }

    #[test]
    fn test_adaptive_frequency_filter_smoothing() {
        let close = make_test_data();
        let aff = AdaptiveFrequencyFilter::new(5, 20, 0.8).unwrap();
        let result = aff.calculate(&close);

        // Filter should produce values close to price
        for i in 25..result.len() {
            let diff = (result[i] - close[i]).abs();
            assert!(diff < 20.0, "Filter output too far from price at index {}", i);
        }
    }

    #[test]
    fn test_adaptive_frequency_filter_validation() {
        // min_period too small
        assert!(AdaptiveFrequencyFilter::new(2, 20, 0.5).is_err());
        // max_period <= min_period
        assert!(AdaptiveFrequencyFilter::new(10, 10, 0.5).is_err());
        assert!(AdaptiveFrequencyFilter::new(10, 5, 0.5).is_err());
        // invalid smoothing_factor
        assert!(AdaptiveFrequencyFilter::new(5, 20, 0.0).is_err());
        assert!(AdaptiveFrequencyFilter::new(5, 20, 1.5).is_err());
    }

    #[test]
    fn test_adaptive_frequency_filter_trait() {
        let data = make_ohlcv_series();
        let aff = AdaptiveFrequencyFilter::new(5, 20, 0.5).unwrap();

        assert_eq!(aff.name(), "Adaptive Frequency Filter");
        assert_eq!(aff.min_periods(), 21);

        let output = aff.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // =====================================================================
    // TrendPhaseAnalysis Tests
    // =====================================================================

    #[test]
    fn test_trend_phase_analysis_basic() {
        let close = make_test_data();
        let tpa = TrendPhaseAnalysis::new(20).unwrap();
        let result = tpa.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Phase should be between 0 and 360
        for i in 25..result.len() {
            assert!(result[i] >= 0.0 && result[i] < 360.0,
                    "Phase {} at index {} out of range", result[i], i);
        }
    }

    #[test]
    fn test_trend_phase_analysis_validation() {
        assert!(TrendPhaseAnalysis::new(5).is_err());
        assert!(TrendPhaseAnalysis::new(10).is_ok());
    }

    #[test]
    fn test_trend_phase_analysis_trait() {
        let data = make_ohlcv_series();
        let tpa = TrendPhaseAnalysis::new(15).unwrap();

        assert_eq!(tpa.name(), "Trend Phase Analysis");
        assert_eq!(tpa.min_periods(), 16);

        let output = tpa.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // =====================================================================
    // CycleModeIndicator Tests
    // =====================================================================

    #[test]
    fn test_cycle_mode_indicator_basic() {
        let close = make_test_data();
        let cmi = CycleModeIndicator::new(20).unwrap();
        let result = cmi.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Mode should be between 0 and 100
        for i in 25..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0,
                    "Mode {} at index {} out of range", result[i], i);
        }
    }

    #[test]
    fn test_cycle_mode_indicator_trending_data() {
        // Strong trend data
        let close: Vec<f64> = (0..60).map(|i| 100.0 + i as f64 * 2.0).collect();
        let cmi = CycleModeIndicator::new(20).unwrap();
        let result = cmi.calculate(&close);

        // Trending data should have high mode values
        for i in 25..result.len() {
            assert!(result[i] >= 50.0, "Expected high trend mode for trending data at index {}", i);
        }
    }

    #[test]
    fn test_cycle_mode_indicator_validation() {
        assert!(CycleModeIndicator::new(5).is_err());
        assert!(CycleModeIndicator::new(10).is_ok());
    }

    #[test]
    fn test_cycle_mode_indicator_trait() {
        let data = make_ohlcv_series();
        let cmi = CycleModeIndicator::new(15).unwrap();

        assert_eq!(cmi.name(), "Cycle Mode Indicator");
        assert_eq!(cmi.min_periods(), 16);

        let output = cmi.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // =====================================================================
    // SignalStrengthMeter Tests
    // =====================================================================

    #[test]
    fn test_signal_strength_meter_basic() {
        let close = make_test_data();
        let ssm = SignalStrengthMeter::new(20, 5).unwrap();
        let result = ssm.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Strength should be between 0 and 100
        for i in 25..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0,
                    "Strength {} at index {} out of range", result[i], i);
        }
    }

    #[test]
    fn test_signal_strength_meter_smooth_data() {
        // Smooth trending data (high signal)
        let close: Vec<f64> = (0..60).map(|i| 100.0 + i as f64 * 0.5).collect();
        let ssm = SignalStrengthMeter::new(20, 5).unwrap();
        let result = ssm.calculate(&close);

        // Smooth data should have high signal strength
        for i in 25..result.len() {
            assert!(result[i] >= 80.0, "Expected high signal strength for smooth data at index {}", i);
        }
    }

    #[test]
    fn test_signal_strength_meter_validation() {
        // period too small
        assert!(SignalStrengthMeter::new(5, 3).is_err());
        // noise_period too small
        assert!(SignalStrengthMeter::new(20, 2).is_err());
        // noise_period >= period
        assert!(SignalStrengthMeter::new(20, 20).is_err());
        assert!(SignalStrengthMeter::new(20, 25).is_err());
        // valid
        assert!(SignalStrengthMeter::new(20, 5).is_ok());
    }

    #[test]
    fn test_signal_strength_meter_trait() {
        let data = make_ohlcv_series();
        let ssm = SignalStrengthMeter::new(20, 5).unwrap();

        assert_eq!(ssm.name(), "Signal Strength Meter");
        assert_eq!(ssm.min_periods(), 21);

        let output = ssm.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // =====================================================================
    // FrequencyResponseIndicator Tests
    // =====================================================================

    #[test]
    fn test_frequency_response_indicator_basic() {
        let close = make_test_data();
        let fri = FrequencyResponseIndicator::new(30, 4).unwrap();
        let result = fri.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Values should be non-negative
        for i in 35..result.len() {
            assert!(result[i] >= 0.0, "Expected non-negative value at index {}", i);
        }
    }

    #[test]
    fn test_frequency_response_indicator_validation() {
        // period too small
        assert!(FrequencyResponseIndicator::new(10, 4).is_err());
        // num_bands out of range
        assert!(FrequencyResponseIndicator::new(30, 1).is_err());
        assert!(FrequencyResponseIndicator::new(30, 11).is_err());
        // valid
        assert!(FrequencyResponseIndicator::new(30, 2).is_ok());
        assert!(FrequencyResponseIndicator::new(30, 10).is_ok());
    }

    #[test]
    fn test_frequency_response_indicator_trait() {
        let data = make_ohlcv_series();
        let fri = FrequencyResponseIndicator::new(25, 4).unwrap();

        assert_eq!(fri.name(), "Frequency Response Indicator");
        assert_eq!(fri.min_periods(), 26);

        let output = fri.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // =====================================================================
    // PhaseSynchronization Tests
    // =====================================================================

    #[test]
    fn test_phase_synchronization_basic() {
        let close = make_test_data();
        let ps = PhaseSynchronization::new(20, 10).unwrap();
        let result = ps.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Synchronization should be between 0 and 100
        for i in 25..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0,
                    "Sync {} at index {} out of range", result[i], i);
        }
    }

    #[test]
    fn test_phase_synchronization_validation() {
        // period too small
        assert!(PhaseSynchronization::new(5, 3).is_err());
        // reference_period too small
        assert!(PhaseSynchronization::new(20, 3).is_err());
        // reference_period >= period
        assert!(PhaseSynchronization::new(20, 20).is_err());
        assert!(PhaseSynchronization::new(20, 25).is_err());
        // valid
        assert!(PhaseSynchronization::new(20, 10).is_ok());
    }

    #[test]
    fn test_phase_synchronization_trait() {
        let data = make_ohlcv_series();
        let ps = PhaseSynchronization::new(20, 10).unwrap();

        assert_eq!(ps.name(), "Phase Synchronization");
        assert_eq!(ps.min_periods(), 21);

        let output = ps.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // =====================================================================
    // Integration Tests
    // =====================================================================

    #[test]
    fn test_all_indicators_consistent_length() {
        let data = make_ohlcv_series();

        let aff = AdaptiveFrequencyFilter::new(5, 20, 0.5).unwrap();
        let tpa = TrendPhaseAnalysis::new(15).unwrap();
        let cmi = CycleModeIndicator::new(15).unwrap();
        let ssm = SignalStrengthMeter::new(20, 5).unwrap();
        let fri = FrequencyResponseIndicator::new(25, 4).unwrap();
        let ps = PhaseSynchronization::new(20, 10).unwrap();

        let results = vec![
            aff.compute(&data).unwrap().primary,
            tpa.compute(&data).unwrap().primary,
            cmi.compute(&data).unwrap().primary,
            ssm.compute(&data).unwrap().primary,
            fri.compute(&data).unwrap().primary,
            ps.compute(&data).unwrap().primary,
        ];

        for result in results {
            assert_eq!(result.len(), data.close.len());
        }
    }
}
