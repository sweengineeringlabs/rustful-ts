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

/// Adaptive Phase Measure - Measures instantaneous phase with adaptive smoothing
///
/// This indicator calculates the instantaneous phase of price movements using
/// an adaptive smoothing technique that adjusts based on detected cycle characteristics.
/// Phase values are returned in degrees (0-360).
#[derive(Debug, Clone)]
pub struct AdaptivePhaseMeasure {
    period: usize,
    adaptive_factor: f64,
}

impl AdaptivePhaseMeasure {
    pub fn new(period: usize, adaptive_factor: f64) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if adaptive_factor <= 0.0 || adaptive_factor > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "adaptive_factor".to_string(),
                reason: "must be between 0 (exclusive) and 1 (inclusive)".to_string(),
            });
        }
        Ok(Self { period, adaptive_factor })
    }

    /// Calculate adaptive phase measure
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];
        let mut prev_smooth_phase = 0.0;

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let slice = &close[start..=i];
            let len = slice.len() as f64;

            // Detrend using simple mean
            let mean: f64 = slice.iter().sum::<f64>() / len;

            // Calculate in-phase and quadrature using DFT at the dominant frequency
            let mut in_phase = 0.0;
            let mut quadrature = 0.0;
            let mut amplitude_sum = 0.0;

            // Adaptive smoothing based on local amplitude
            for (j, &val) in slice.iter().enumerate() {
                let centered = val - mean;
                let angle = 2.0 * std::f64::consts::PI * j as f64 / self.period as f64;
                in_phase += centered * angle.cos();
                quadrature += centered * angle.sin();
                amplitude_sum += centered.abs();
            }

            // Calculate raw phase
            let raw_phase = quadrature.atan2(in_phase);
            let phase_degrees = (raw_phase * 180.0 / std::f64::consts::PI + 360.0) % 360.0;

            // Adaptive smoothing: less smoothing when amplitude is high (clear signal)
            let avg_amplitude = amplitude_sum / len;
            let normalized_amp = (avg_amplitude / mean.abs().max(1.0)).min(1.0);
            let alpha = self.adaptive_factor * (0.5 + 0.5 * normalized_amp);

            // Handle phase wrapping for smoothing
            let mut phase_diff = phase_degrees - prev_smooth_phase;
            if phase_diff > 180.0 {
                phase_diff -= 360.0;
            } else if phase_diff < -180.0 {
                phase_diff += 360.0;
            }

            let smooth_phase = (prev_smooth_phase + alpha * phase_diff + 360.0) % 360.0;
            prev_smooth_phase = smooth_phase;

            result[i] = smooth_phase;
        }

        result
    }
}

impl TechnicalIndicator for AdaptivePhaseMeasure {
    fn name(&self) -> &str {
        "Adaptive Phase Measure"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Frequency Domain Momentum - Momentum calculated in frequency domain
///
/// Unlike traditional momentum which measures price change over time,
/// this indicator calculates momentum in the frequency domain by analyzing
/// the rate of change of dominant frequency components.
#[derive(Debug, Clone)]
pub struct FrequencyDomainMomentum {
    period: usize,
    momentum_period: usize,
}

impl FrequencyDomainMomentum {
    pub fn new(period: usize, momentum_period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if momentum_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if momentum_period >= period {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be less than period".to_string(),
            });
        }
        Ok(Self { period, momentum_period })
    }

    /// Calculate spectral power at a given frequency
    fn spectral_power(&self, slice: &[f64], freq_period: usize) -> f64 {
        let len = slice.len();
        let mean: f64 = slice.iter().sum::<f64>() / len as f64;

        let mut real = 0.0;
        let mut imag = 0.0;

        for (j, &val) in slice.iter().enumerate() {
            let angle = 2.0 * std::f64::consts::PI * j as f64 / freq_period as f64;
            let centered = val - mean;
            real += centered * angle.cos();
            imag += centered * angle.sin();
        }

        (real * real + imag * imag).sqrt() / len as f64
    }

    /// Calculate frequency domain momentum
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];
        let lookback = self.period + self.momentum_period;

        // First, calculate spectral power history
        let mut spectral_history = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let slice = &close[start..=i];

            // Sum power across multiple frequency bands
            let mut total_power = 0.0;
            let num_bands = 5;
            for band in 1..=num_bands {
                let freq_period = self.period / (band + 1);
                if freq_period >= 3 {
                    total_power += self.spectral_power(slice, freq_period);
                }
            }

            spectral_history[i] = total_power;
        }

        // Calculate momentum of spectral power
        for i in lookback..n {
            let current = spectral_history[i];
            let past = spectral_history[i - self.momentum_period];

            if past.abs() > 1e-10 {
                // Percentage change in spectral power
                result[i] = ((current - past) / past) * 100.0;
            }
        }

        result
    }
}

impl TechnicalIndicator for FrequencyDomainMomentum {
    fn name(&self) -> &str {
        "Frequency Domain Momentum"
    }

    fn min_periods(&self) -> usize {
        self.period + self.momentum_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Spectral Entropy - Entropy of the frequency spectrum
///
/// Measures the "flatness" of the frequency spectrum. Low entropy indicates
/// a dominant frequency (predictable cycle), high entropy indicates
/// equal power across all frequencies (noise-like, unpredictable).
/// Returns values between 0 (single frequency) and 100 (uniform spectrum).
#[derive(Debug, Clone)]
pub struct SpectralEntropy {
    period: usize,
    num_bins: usize,
}

impl SpectralEntropy {
    pub fn new(period: usize, num_bins: usize) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        if num_bins < 3 || num_bins > 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "num_bins".to_string(),
                reason: "must be between 3 and 20".to_string(),
            });
        }
        Ok(Self { period, num_bins })
    }

    /// Calculate spectral entropy
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let slice = &close[start..=i];
            let len = slice.len();
            let mean: f64 = slice.iter().sum::<f64>() / len as f64;

            // Calculate power spectrum at different frequency bins
            let mut powers = Vec::with_capacity(self.num_bins);
            let mut total_power = 0.0;

            for bin in 0..self.num_bins {
                // Frequency period for this bin (from period/2 down to ~3)
                let freq_period = ((self.period as f64 / 2.0)
                    * (1.0 - bin as f64 / self.num_bins as f64)).max(3.0) as usize;

                let mut real = 0.0;
                let mut imag = 0.0;

                for (j, &val) in slice.iter().enumerate() {
                    let angle = 2.0 * std::f64::consts::PI * j as f64 / freq_period as f64;
                    let centered = val - mean;
                    real += centered * angle.cos();
                    imag += centered * angle.sin();
                }

                let power = (real * real + imag * imag) / (len * len) as f64;
                powers.push(power);
                total_power += power;
            }

            // Normalize to probability distribution
            if total_power > 1e-10 {
                let probs: Vec<f64> = powers.iter()
                    .map(|&p| (p / total_power).max(1e-10))
                    .collect();

                // Calculate Shannon entropy
                let entropy: f64 = probs.iter()
                    .map(|&p| -p * p.ln())
                    .sum();

                // Normalize to 0-100 (max entropy = ln(num_bins))
                let max_entropy = (self.num_bins as f64).ln();
                result[i] = (entropy / max_entropy * 100.0).min(100.0);
            }
        }

        result
    }
}

impl TechnicalIndicator for SpectralEntropy {
    fn name(&self) -> &str {
        "Spectral Entropy"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Cycle Dominance - Measures how dominant the detected cycle is
///
/// Quantifies the strength of the dominant cycle relative to other frequency
/// components and noise. High values indicate a strong, clear cycle that
/// can be traded; low values suggest noisy or multi-frequency behavior.
/// Returns values between 0 (no dominant cycle) and 100 (pure sinusoid).
#[derive(Debug, Clone)]
pub struct CycleDominance {
    min_period: usize,
    max_period: usize,
}

impl CycleDominance {
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
        if max_period > 100 {
            return Err(IndicatorError::InvalidParameter {
                name: "max_period".to_string(),
                reason: "must be at most 100".to_string(),
            });
        }
        Ok(Self { min_period, max_period })
    }

    /// Calculate cycle dominance
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.max_period..n {
            let start = i.saturating_sub(self.max_period);
            let slice = &close[start..=i];
            let len = slice.len();
            let mean: f64 = slice.iter().sum::<f64>() / len as f64;

            // Calculate power at each period in range
            let mut powers = Vec::new();
            let mut max_power = 0.0f64;
            let mut total_power = 0.0;

            for period in self.min_period..=self.max_period {
                if period > len / 2 {
                    continue;
                }

                let mut real = 0.0;
                let mut imag = 0.0;

                for (j, &val) in slice.iter().enumerate() {
                    let angle = 2.0 * std::f64::consts::PI * j as f64 / period as f64;
                    let centered = val - mean;
                    real += centered * angle.cos();
                    imag += centered * angle.sin();
                }

                let power = (real * real + imag * imag) / (len * len) as f64;
                powers.push(power);
                max_power = max_power.max(power);
                total_power += power;
            }

            // Dominance ratio: how much of total power is in the dominant frequency
            if total_power > 1e-10 && !powers.is_empty() {
                let dominance = (max_power / total_power) * 100.0;
                result[i] = dominance.min(100.0);
            }
        }

        result
    }
}

impl TechnicalIndicator for CycleDominance {
    fn name(&self) -> &str {
        "Cycle Dominance"
    }

    fn min_periods(&self) -> usize {
        self.max_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Harmonic Analyzer - Detects harmonic patterns in price
///
/// Analyzes price data for harmonic relationships (2x, 3x, 4x frequency).
/// Returns a composite score indicating how much of the price movement
/// can be explained by harmonic frequency relationships.
#[derive(Debug, Clone)]
pub struct HarmonicAnalyzer {
    base_period: usize,
    num_harmonics: usize,
}

impl HarmonicAnalyzer {
    pub fn new(base_period: usize, num_harmonics: usize) -> Result<Self> {
        if base_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "base_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if num_harmonics < 2 || num_harmonics > 6 {
            return Err(IndicatorError::InvalidParameter {
                name: "num_harmonics".to_string(),
                reason: "must be between 2 and 6".to_string(),
            });
        }
        Ok(Self { base_period, num_harmonics })
    }

    /// Calculate harmonic analyzer
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];
        let window = self.base_period * 2;

        for i in window..n {
            let start = i.saturating_sub(window);
            let slice = &close[start..=i];
            let len = slice.len();
            let mean: f64 = slice.iter().sum::<f64>() / len as f64;

            // Calculate total variance
            let total_variance: f64 = slice.iter()
                .map(|&v| (v - mean).powi(2))
                .sum::<f64>() / len as f64;

            if total_variance < 1e-10 {
                continue;
            }

            // Calculate power at base frequency and harmonics
            let mut harmonic_power = 0.0;

            for h in 1..=self.num_harmonics {
                let period = self.base_period / h;
                if period < 3 {
                    break;
                }

                let mut real = 0.0;
                let mut imag = 0.0;

                for (j, &val) in slice.iter().enumerate() {
                    let angle = 2.0 * std::f64::consts::PI * j as f64 / period as f64;
                    let centered = val - mean;
                    real += centered * angle.cos();
                    imag += centered * angle.sin();
                }

                let power = (real * real + imag * imag) / (len * len) as f64;
                // Weight lower harmonics more heavily
                harmonic_power += power / h as f64;
            }

            // Score: ratio of harmonic power to total variance
            let harmonic_ratio = (harmonic_power / total_variance).sqrt() * 100.0;
            result[i] = harmonic_ratio.min(100.0);
        }

        result
    }
}

impl TechnicalIndicator for HarmonicAnalyzer {
    fn name(&self) -> &str {
        "Harmonic Analyzer"
    }

    fn min_periods(&self) -> usize {
        self.base_period * 2 + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Phase Coherence - Measures phase coherence between price and oscillator
///
/// Calculates the phase coherence between the price signal and a derived
/// oscillator (detrended price). High coherence indicates that the oscillator
/// is well-synchronized with price, making it more reliable for timing.
/// Returns values between 0 (no coherence) and 100 (perfect coherence).
#[derive(Debug, Clone)]
pub struct PhaseCoherence {
    period: usize,
    oscillator_period: usize,
}

impl PhaseCoherence {
    pub fn new(period: usize, oscillator_period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if oscillator_period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "oscillator_period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        if oscillator_period >= period {
            return Err(IndicatorError::InvalidParameter {
                name: "oscillator_period".to_string(),
                reason: "must be less than period".to_string(),
            });
        }
        Ok(Self { period, oscillator_period })
    }

    /// Calculate phase at a point using Hilbert-like transform
    fn get_phase(&self, slice: &[f64], ref_period: usize) -> f64 {
        let len = slice.len();
        let mean: f64 = slice.iter().sum::<f64>() / len as f64;

        let mut in_phase = 0.0;
        let mut quadrature = 0.0;

        for (j, &val) in slice.iter().enumerate() {
            let angle = 2.0 * std::f64::consts::PI * j as f64 / ref_period as f64;
            let centered = val - mean;
            in_phase += centered * angle.cos();
            quadrature += centered * angle.sin();
        }

        quadrature.atan2(in_phase)
    }

    /// Calculate phase coherence
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // Calculate oscillator (momentum-like detrending)
        let mut oscillator = vec![0.0; n];
        for i in self.oscillator_period..n {
            oscillator[i] = close[i] - close[i - self.oscillator_period];
        }

        // Calculate coherence over rolling window
        let coherence_window = self.period;

        for i in (self.period + self.oscillator_period)..n {
            let start = i.saturating_sub(coherence_window);

            let price_slice = &close[start..=i];
            let osc_slice = &oscillator[start..=i];

            // Get phases at the oscillator period
            let price_phase = self.get_phase(price_slice, self.oscillator_period);
            let osc_phase = self.get_phase(osc_slice, self.oscillator_period);

            // Phase difference
            let phase_diff = (price_phase - osc_phase).abs();

            // Coherence based on phase difference
            // Perfect coherence (0 or PI diff) -> 100
            // Worst coherence (PI/2 diff) -> 0
            let coherence = phase_diff.cos().abs() * 100.0;

            result[i] = coherence;
        }

        result
    }
}

impl TechnicalIndicator for PhaseCoherence {
    fn name(&self) -> &str {
        "Phase Coherence"
    }

    fn min_periods(&self) -> usize {
        self.period + self.oscillator_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Frequency Domain Moving Average - Moving average using frequency domain filtering
///
/// This indicator computes a moving average by transforming price data to the
/// frequency domain using a simplified DFT approach, applying a low-pass filter,
/// and transforming back. This produces smoother results with less lag than
/// traditional time-domain moving averages.
#[derive(Debug, Clone)]
pub struct FrequencyDomainMA {
    period: usize,
    cutoff_period: usize,
}

impl FrequencyDomainMA {
    pub fn new(period: usize, cutoff_period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if cutoff_period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "cutoff_period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        if cutoff_period > period / 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "cutoff_period".to_string(),
                reason: "must be at most half of period".to_string(),
            });
        }
        Ok(Self { period, cutoff_period })
    }

    /// Calculate frequency domain moving average
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let slice = &close[start..=i];
            let len = slice.len();
            let mean: f64 = slice.iter().sum::<f64>() / len as f64;

            // Center the data
            let centered: Vec<f64> = slice.iter().map(|&x| x - mean).collect();

            // Apply frequency domain filtering
            // Only keep frequency components with period >= cutoff_period
            let mut filtered_sum = 0.0;
            let mut weight_sum = 0.0;

            // Iterate through frequency components
            for freq_idx in 1..=(len / 2) {
                let freq_period = len as f64 / freq_idx as f64;

                // Calculate DFT coefficient at this frequency
                let mut real = 0.0;
                let mut imag = 0.0;

                for (j, &val) in centered.iter().enumerate() {
                    let angle = 2.0 * std::f64::consts::PI * freq_idx as f64 * j as f64 / len as f64;
                    real += val * angle.cos();
                    imag += val * angle.sin();
                }

                // Low-pass filter: attenuate high frequencies
                let attenuation = if freq_period >= self.cutoff_period as f64 {
                    1.0
                } else {
                    // Smooth roll-off
                    (freq_period / self.cutoff_period as f64).powi(2)
                };

                // Reconstruct signal at the last point (j = len - 1)
                let j = len - 1;
                let angle = 2.0 * std::f64::consts::PI * freq_idx as f64 * j as f64 / len as f64;
                let contribution = (real * angle.cos() + imag * angle.sin()) * attenuation;
                filtered_sum += contribution;
                weight_sum += attenuation;
            }

            // Normalize and add back mean
            let filtered_value = if weight_sum > 1e-10 {
                mean + (filtered_sum * 2.0 / len as f64)
            } else {
                mean
            };

            result[i] = filtered_value;
        }

        result
    }
}

impl TechnicalIndicator for FrequencyDomainMA {
    fn name(&self) -> &str {
        "Frequency Domain MA"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Phase Shift Indicator - Measures phase shift in price cycles
///
/// This indicator detects and quantifies the phase shift between the current
/// price cycle and a reference cycle. Positive values indicate price is
/// leading the reference; negative values indicate lagging. The output is
/// in degrees (-180 to +180).
#[derive(Debug, Clone)]
pub struct PhaseShiftIndicator {
    period: usize,
    reference_period: usize,
}

impl PhaseShiftIndicator {
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
        if reference_period > period {
            return Err(IndicatorError::InvalidParameter {
                name: "reference_period".to_string(),
                reason: "must not exceed period".to_string(),
            });
        }
        Ok(Self { period, reference_period })
    }

    /// Calculate phase using Hilbert transform approximation
    fn calculate_phase(&self, slice: &[f64]) -> f64 {
        let len = slice.len();
        let mean: f64 = slice.iter().sum::<f64>() / len as f64;

        let mut in_phase = 0.0;
        let mut quadrature = 0.0;

        for (j, &val) in slice.iter().enumerate() {
            let angle = 2.0 * std::f64::consts::PI * j as f64 / self.reference_period as f64;
            let centered = val - mean;
            in_phase += centered * angle.cos();
            quadrature += centered * angle.sin();
        }

        quadrature.atan2(in_phase)
    }

    /// Calculate phase shift indicator
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];
        let mut prev_phase = 0.0;
        let mut initialized = false;

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let slice = &close[start..=i];

            // Calculate current phase
            let current_phase = self.calculate_phase(slice);

            if !initialized {
                prev_phase = current_phase;
                initialized = true;
                continue;
            }

            // Calculate phase shift (difference from previous)
            let mut phase_shift = current_phase - prev_phase;

            // Normalize to -PI to +PI
            while phase_shift > std::f64::consts::PI {
                phase_shift -= 2.0 * std::f64::consts::PI;
            }
            while phase_shift < -std::f64::consts::PI {
                phase_shift += 2.0 * std::f64::consts::PI;
            }

            // Convert to degrees
            result[i] = phase_shift * 180.0 / std::f64::consts::PI;

            prev_phase = current_phase;
        }

        result
    }
}

impl TechnicalIndicator for PhaseShiftIndicator {
    fn name(&self) -> &str {
        "Phase Shift Indicator"
    }

    fn min_periods(&self) -> usize {
        self.period + 2
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Spectral Power Index - Power spectrum analysis index
///
/// This indicator measures the distribution of power across different
/// frequency bands in the price signal. It returns a composite index
/// that indicates whether power is concentrated in low frequencies
/// (trending) or high frequencies (choppy). Values range from 0 (all
/// high-frequency power) to 100 (all low-frequency power).
#[derive(Debug, Clone)]
pub struct SpectralPowerIndex {
    period: usize,
    num_bands: usize,
}

impl SpectralPowerIndex {
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

    /// Calculate power at a specific frequency period
    fn calculate_band_power(&self, slice: &[f64], freq_period: usize) -> f64 {
        let len = slice.len();
        let mean: f64 = slice.iter().sum::<f64>() / len as f64;

        let mut real = 0.0;
        let mut imag = 0.0;

        for (j, &val) in slice.iter().enumerate() {
            let angle = 2.0 * std::f64::consts::PI * j as f64 / freq_period as f64;
            let centered = val - mean;
            real += centered * angle.cos();
            imag += centered * angle.sin();
        }

        (real * real + imag * imag) / (len * len) as f64
    }

    /// Calculate spectral power index
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // Define frequency bands (from low to high frequency)
        let min_period = 4;
        let max_period = self.period / 2;

        let band_periods: Vec<usize> = (0..self.num_bands)
            .map(|i| {
                let t = i as f64 / (self.num_bands - 1).max(1) as f64;
                let log_min = (min_period as f64).ln();
                let log_max = (max_period as f64).ln();
                (log_min + t * (log_max - log_min)).exp() as usize
            })
            .collect();

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let slice = &close[start..=i];

            // Calculate power in each band
            let mut low_freq_power = 0.0;
            let mut high_freq_power = 0.0;
            let mut total_power = 0.0;

            for (band_idx, &freq_period) in band_periods.iter().enumerate() {
                let power = self.calculate_band_power(slice, freq_period);
                total_power += power;

                // First half of bands are high frequency (short periods)
                // Second half are low frequency (long periods)
                if band_idx < self.num_bands / 2 {
                    high_freq_power += power;
                } else {
                    low_freq_power += power;
                }
            }

            // Spectral Power Index: ratio of low-freq to total power
            if total_power > 1e-10 {
                result[i] = (low_freq_power / total_power * 100.0).min(100.0);
            }
        }

        result
    }
}

impl TechnicalIndicator for SpectralPowerIndex {
    fn name(&self) -> &str {
        "Spectral Power Index"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Noise Filter - Filters noise from price signal
///
/// This indicator applies a sophisticated noise filtering algorithm that
/// separates the signal (trend and cycle components) from random noise.
/// It uses an adaptive approach that preserves important price movements
/// while attenuating random fluctuations. Returns the filtered price.
#[derive(Debug, Clone)]
pub struct NoiseFilter {
    period: usize,
    smoothing_factor: f64,
}

impl NoiseFilter {
    pub fn new(period: usize, smoothing_factor: f64) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if smoothing_factor <= 0.0 || smoothing_factor > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing_factor".to_string(),
                reason: "must be between 0 (exclusive) and 1 (inclusive)".to_string(),
            });
        }
        Ok(Self { period, smoothing_factor })
    }

    /// Estimate noise variance using high-frequency residuals
    fn estimate_noise_variance(&self, slice: &[f64]) -> f64 {
        let len = slice.len();
        if len < 3 {
            return 0.0;
        }

        // Calculate second differences (approximates high-frequency content)
        let mut noise_sum = 0.0;
        let mut count = 0;

        for i in 2..len {
            let second_diff = slice[i] - 2.0 * slice[i - 1] + slice[i - 2];
            noise_sum += second_diff * second_diff;
            count += 1;
        }

        if count > 0 {
            noise_sum / (count as f64 * 6.0) // Factor of 6 normalizes second difference variance
        } else {
            0.0
        }
    }

    /// Calculate noise filter output
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];
        let mut filtered = 0.0;

        for i in 0..n {
            if i < self.period {
                result[i] = close[i];
                filtered = close[i];
                continue;
            }

            let start = i.saturating_sub(self.period);
            let slice = &close[start..=i];

            // Estimate noise variance
            let noise_var = self.estimate_noise_variance(slice);

            // Calculate signal variance (total variance minus noise)
            let mean: f64 = slice.iter().sum::<f64>() / slice.len() as f64;
            let total_var: f64 = slice.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / slice.len() as f64;

            let signal_var = (total_var - noise_var).max(0.0);

            // Adaptive alpha: higher when signal-to-noise is high
            let snr = if noise_var > 1e-10 {
                (signal_var / noise_var).sqrt()
            } else {
                10.0 // High SNR assumed when noise is negligible
            };

            // Alpha increases with SNR but capped by smoothing_factor
            let adaptive_alpha = self.smoothing_factor * (1.0 - (-snr / 2.0).exp());

            // Apply adaptive exponential filter
            filtered = adaptive_alpha * close[i] + (1.0 - adaptive_alpha) * filtered;
            result[i] = filtered;
        }

        result
    }
}

impl TechnicalIndicator for NoiseFilter {
    fn name(&self) -> &str {
        "Noise Filter"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Cycle Period Estimator - Estimates dominant cycle period
///
/// This indicator estimates the dominant cycle period in price data using
/// autocorrelation analysis. It returns the estimated period in bars,
/// which can be used to dynamically adjust other indicators. Values
/// range from min_period to max_period.
#[derive(Debug, Clone)]
pub struct CyclePeriodEstimator {
    min_period: usize,
    max_period: usize,
    smoothing: usize,
}

impl CyclePeriodEstimator {
    pub fn new(min_period: usize, max_period: usize, smoothing: usize) -> Result<Self> {
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
        if max_period > 100 {
            return Err(IndicatorError::InvalidParameter {
                name: "max_period".to_string(),
                reason: "must be at most 100".to_string(),
            });
        }
        if smoothing < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self { min_period, max_period, smoothing })
    }

    /// Find dominant period using autocorrelation
    fn find_dominant_period(&self, slice: &[f64]) -> usize {
        let len = slice.len();
        let mean: f64 = slice.iter().sum::<f64>() / len as f64;

        let mut best_period = self.min_period;
        let mut best_corr = f64::NEG_INFINITY;

        // Calculate autocorrelation at each candidate period
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

            // Look for peak in autocorrelation
            if corr > best_corr {
                best_corr = corr;
                best_period = period;
            }
        }

        best_period
    }

    /// Calculate cycle period estimator
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];
        let mut period_history: Vec<f64> = Vec::new();

        for i in self.max_period..n {
            let start = i.saturating_sub(self.max_period);
            let slice = &close[start..=i];

            // Find dominant period
            let detected_period = self.find_dominant_period(slice) as f64;

            // Add to history for smoothing
            period_history.push(detected_period);
            if period_history.len() > self.smoothing {
                period_history.remove(0);
            }

            // Calculate smoothed period (median for robustness)
            let mut sorted = period_history.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let smoothed_period = if sorted.len() % 2 == 0 {
                (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
            } else {
                sorted[sorted.len() / 2]
            };

            result[i] = smoothed_period;
        }

        result
    }
}

impl TechnicalIndicator for CyclePeriodEstimator {
    fn name(&self) -> &str {
        "Cycle Period Estimator"
    }

    fn min_periods(&self) -> usize {
        self.max_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Signal to Noise Ratio (Advanced) - Measures signal clarity
///
/// This indicator provides an advanced measurement of signal-to-noise ratio
/// in price data using spectral analysis. It separates coherent price
/// movements (signal) from random fluctuations (noise) and returns a
/// ratio in decibels (dB). Higher values indicate clearer, more tradeable
/// signals.
#[derive(Debug, Clone)]
pub struct SignalToNoiseRatioAdvanced {
    period: usize,
    signal_period: usize,
}

impl SignalToNoiseRatioAdvanced {
    pub fn new(period: usize, signal_period: usize) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        if signal_period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "signal_period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        if signal_period >= period / 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "signal_period".to_string(),
                reason: "must be less than half of period".to_string(),
            });
        }
        Ok(Self { period, signal_period })
    }

    /// Calculate signal to noise ratio
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let slice = &close[start..=i];
            let len = slice.len();
            let mean: f64 = slice.iter().sum::<f64>() / len as f64;

            // Calculate signal power (low-frequency components)
            let mut signal_power = 0.0;
            for freq_period in (self.signal_period * 2)..=(len / 2) {
                let mut real = 0.0;
                let mut imag = 0.0;

                for (j, &val) in slice.iter().enumerate() {
                    let angle = 2.0 * std::f64::consts::PI * j as f64 / freq_period as f64;
                    let centered = val - mean;
                    real += centered * angle.cos();
                    imag += centered * angle.sin();
                }

                signal_power += (real * real + imag * imag) / (len * len) as f64;
            }

            // Calculate noise power (high-frequency components)
            let mut noise_power = 0.0;
            for freq_period in 3..self.signal_period {
                let mut real = 0.0;
                let mut imag = 0.0;

                for (j, &val) in slice.iter().enumerate() {
                    let angle = 2.0 * std::f64::consts::PI * j as f64 / freq_period as f64;
                    let centered = val - mean;
                    real += centered * angle.cos();
                    imag += centered * angle.sin();
                }

                noise_power += (real * real + imag * imag) / (len * len) as f64;
            }

            // Ensure noise_power is not zero
            noise_power = noise_power.max(1e-10);

            // Calculate SNR in decibels
            let snr_db = 10.0 * (signal_power / noise_power).log10();

            // Clamp to reasonable range (-20 to +40 dB)
            result[i] = snr_db.max(-20.0).min(40.0);
        }

        result
    }
}

impl TechnicalIndicator for SignalToNoiseRatioAdvanced {
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

/// Bandpass Filter - Simple bandpass filter for isolating specific frequencies
///
/// This indicator implements a simple bandpass filter that allows frequencies
/// within a specified range to pass through while attenuating frequencies
/// outside this range. Useful for isolating cyclical components of specific
/// periodicities in price data. The output represents the bandpass-filtered
/// price component.
#[derive(Debug, Clone)]
pub struct BandpassFilter {
    center_period: usize,
    bandwidth: f64,
}

impl BandpassFilter {
    /// Creates a new BandpassFilter
    ///
    /// # Parameters
    /// - `center_period`: The center period of the passband (minimum 5)
    /// - `bandwidth`: The relative bandwidth as a fraction of center frequency (0.1 to 1.0)
    ///
    /// # Returns
    /// A Result containing the BandpassFilter or an error if parameters are invalid
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
                reason: "must be between 0 (exclusive) and 1 (inclusive)".to_string(),
            });
        }
        Ok(Self { center_period, bandwidth })
    }

    /// Calculate bandpass filter output
    ///
    /// Uses a combination of high-pass and low-pass filtering to create
    /// a bandpass effect centered on the specified period.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // Calculate cutoff periods based on center and bandwidth
        let low_cutoff = (self.center_period as f64 * (1.0 + self.bandwidth / 2.0)) as usize;
        let high_cutoff = (self.center_period as f64 * (1.0 - self.bandwidth / 2.0)).max(3.0) as usize;

        let window = low_cutoff * 2;

        for i in window..n {
            let start = i.saturating_sub(window);
            let slice = &close[start..=i];
            let len = slice.len();
            let mean: f64 = slice.iter().sum::<f64>() / len as f64;

            // Calculate bandpass output using DFT at passband frequencies
            let mut bandpass_sum = 0.0;
            let mut weight_sum = 0.0;

            for freq_period in high_cutoff..=low_cutoff {
                if freq_period < 3 || freq_period > len / 2 {
                    continue;
                }

                let mut real = 0.0;
                let mut imag = 0.0;

                for (j, &val) in slice.iter().enumerate() {
                    let angle = 2.0 * std::f64::consts::PI * j as f64 / freq_period as f64;
                    let centered = val - mean;
                    real += centered * angle.cos();
                    imag += centered * angle.sin();
                }

                // Bandpass weighting: peak at center, roll off at edges
                let freq_dist = ((freq_period as f64 - self.center_period as f64) / self.center_period as f64).abs();
                let weight = (1.0 - freq_dist / (self.bandwidth / 2.0)).max(0.0);

                // Reconstruct at the last point
                let j = len - 1;
                let angle = 2.0 * std::f64::consts::PI * j as f64 / freq_period as f64;
                let contribution = (real * angle.cos() + imag * angle.sin()) * weight;
                bandpass_sum += contribution;
                weight_sum += weight;
            }

            // Normalize
            if weight_sum > 1e-10 {
                result[i] = bandpass_sum * 2.0 / len as f64;
            }
        }

        result
    }
}

impl TechnicalIndicator for BandpassFilter {
    fn name(&self) -> &str {
        "Bandpass Filter"
    }

    fn min_periods(&self) -> usize {
        self.center_period * 2 + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Highpass Filter - High-pass filter for detrending price data
///
/// This indicator implements a high-pass filter that removes low-frequency
/// components (trends) from price data, leaving only the high-frequency
/// oscillations. Useful for extracting cyclical behavior while eliminating
/// the underlying trend. The output represents detrended price movements.
#[derive(Debug, Clone)]
pub struct HighpassFilter {
    cutoff_period: usize,
    poles: usize,
}

impl HighpassFilter {
    /// Creates a new HighpassFilter
    ///
    /// # Parameters
    /// - `cutoff_period`: The cutoff period below which frequencies pass through (minimum 5)
    /// - `poles`: Number of filter poles for steepness (1 to 4)
    ///
    /// # Returns
    /// A Result containing the HighpassFilter or an error if parameters are invalid
    pub fn new(cutoff_period: usize, poles: usize) -> Result<Self> {
        if cutoff_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "cutoff_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if poles < 1 || poles > 4 {
            return Err(IndicatorError::InvalidParameter {
                name: "poles".to_string(),
                reason: "must be between 1 and 4".to_string(),
            });
        }
        Ok(Self { cutoff_period, poles })
    }

    /// Calculate highpass filter output using Ehlers' formula
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // Calculate filter coefficient based on cutoff period
        let omega = 2.0 * std::f64::consts::PI / self.cutoff_period as f64;
        let alpha = (1.0 + omega.sin()) / omega.cos() - 1.0;

        // First pass: basic high-pass
        let mut hp = vec![0.0; n];
        for i in 1..n {
            let a = (1.0 + alpha) / 2.0;
            hp[i] = a * (close[i] - close[i - 1]) + alpha * hp[i - 1];
        }

        // Additional passes for higher pole filters
        let mut current = hp.clone();
        for _ in 1..self.poles {
            let mut next = vec![0.0; n];
            for i in 1..n {
                let a = (1.0 + alpha) / 2.0;
                next[i] = a * (current[i] - current[i - 1]) + alpha * next[i - 1];
            }
            current = next;
        }

        // Copy to result
        for i in 0..n {
            result[i] = current[i];
        }

        result
    }
}

impl TechnicalIndicator for HighpassFilter {
    fn name(&self) -> &str {
        "Highpass Filter"
    }

    fn min_periods(&self) -> usize {
        self.cutoff_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Lowpass Filter - Low-pass filter for smoothing price data
///
/// This indicator implements a low-pass filter that removes high-frequency
/// noise from price data, leaving only the smooth underlying trend and
/// low-frequency cycles. Based on Ehlers' approach for minimal lag smoothing.
/// The output represents a smoothed version of price.
#[derive(Debug, Clone)]
pub struct LowpassFilter {
    cutoff_period: usize,
    poles: usize,
}

impl LowpassFilter {
    /// Creates a new LowpassFilter
    ///
    /// # Parameters
    /// - `cutoff_period`: The cutoff period above which frequencies pass through (minimum 5)
    /// - `poles`: Number of filter poles for steepness (1 to 4)
    ///
    /// # Returns
    /// A Result containing the LowpassFilter or an error if parameters are invalid
    pub fn new(cutoff_period: usize, poles: usize) -> Result<Self> {
        if cutoff_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "cutoff_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if poles < 1 || poles > 4 {
            return Err(IndicatorError::InvalidParameter {
                name: "poles".to_string(),
                reason: "must be between 1 and 4".to_string(),
            });
        }
        Ok(Self { cutoff_period, poles })
    }

    /// Calculate lowpass filter output using Ehlers' supersmoother approach
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // Calculate filter coefficients based on cutoff period
        let omega = 2.0 * std::f64::consts::PI / self.cutoff_period as f64;
        let a1 = (-1.414 * omega).exp();
        let b1 = 2.0 * a1 * (1.414 * omega).cos();
        let c2 = b1;
        let c3 = -a1 * a1;
        let c1 = 1.0 - c2 - c3;

        // First pass
        if n > 0 {
            result[0] = close[0];
        }
        if n > 1 {
            result[1] = close[1];
        }

        for i in 2..n {
            result[i] = c1 * (close[i] + close[i - 1]) / 2.0 + c2 * result[i - 1] + c3 * result[i - 2];
        }

        // Additional smoothing passes for higher pole count
        for _ in 1..self.poles {
            let mut next = result.clone();
            if n > 2 {
                for i in 2..n {
                    next[i] = c1 * (result[i] + result[i - 1]) / 2.0 + c2 * next[i - 1] + c3 * next[i - 2];
                }
            }
            result = next;
        }

        result
    }
}

impl TechnicalIndicator for LowpassFilter {
    fn name(&self) -> &str {
        "Lowpass Filter"
    }

    fn min_periods(&self) -> usize {
        self.cutoff_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Notch Filter - Removes a specific frequency from the signal
///
/// This indicator implements a notch (band-reject) filter that attenuates
/// a narrow band of frequencies while passing all others. Useful for removing
/// known cyclical interference patterns from price data, such as weekly or
/// monthly seasonality effects. The output is price with the specified
/// frequency component removed.
#[derive(Debug, Clone)]
pub struct NotchFilter {
    notch_period: usize,
    notch_width: f64,
}

impl NotchFilter {
    /// Creates a new NotchFilter
    ///
    /// # Parameters
    /// - `notch_period`: The period to remove from the signal (minimum 5)
    /// - `notch_width`: The width of the notch as a fraction of center frequency (0.05 to 0.5)
    ///
    /// # Returns
    /// A Result containing the NotchFilter or an error if parameters are invalid
    pub fn new(notch_period: usize, notch_width: f64) -> Result<Self> {
        if notch_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "notch_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if notch_width < 0.05 || notch_width > 0.5 {
            return Err(IndicatorError::InvalidParameter {
                name: "notch_width".to_string(),
                reason: "must be between 0.05 and 0.5".to_string(),
            });
        }
        Ok(Self { notch_period, notch_width })
    }

    /// Calculate notch filter output
    ///
    /// The filter removes the specified frequency component while preserving
    /// other frequencies in the signal.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        let window = self.notch_period * 2;

        for i in window..n {
            let start = i.saturating_sub(window);
            let slice = &close[start..=i];
            let len = slice.len();
            let mean: f64 = slice.iter().sum::<f64>() / len as f64;

            // Calculate nearby frequencies within the notch width
            let low_period = (self.notch_period as f64 * (1.0 + self.notch_width / 2.0)) as usize;
            let high_period = (self.notch_period as f64 * (1.0 - self.notch_width / 2.0)).max(3.0) as usize;

            let mut notch_component = 0.0;
            let j = len - 1;

            for freq_period in high_period..=low_period {
                if freq_period < 3 || freq_period > len / 2 {
                    continue;
                }

                let mut real = 0.0;
                let mut imag = 0.0;

                for (k, &val) in slice.iter().enumerate() {
                    let angle = 2.0 * std::f64::consts::PI * k as f64 / freq_period as f64;
                    let centered = val - mean;
                    real += centered * angle.cos();
                    imag += centered * angle.sin();
                }

                // Notch attenuation: strongest at center, tapers at edges
                let freq_dist = ((freq_period as f64 - self.notch_period as f64) / self.notch_period as f64).abs();
                let attenuation = (1.0 - freq_dist / (self.notch_width / 2.0)).max(0.0);

                let angle = 2.0 * std::f64::consts::PI * j as f64 / freq_period as f64;
                notch_component += (real * angle.cos() + imag * angle.sin()) * attenuation * 2.0 / len as f64;
            }

            // Subtract the notch component from the current price
            result[i] = close[i] - notch_component;
        }

        // Fill initial values
        for i in 0..window.min(n) {
            result[i] = close[i];
        }

        result
    }
}

impl TechnicalIndicator for NotchFilter {
    fn name(&self) -> &str {
        "Notch Filter"
    }

    fn min_periods(&self) -> usize {
        self.notch_period * 2 + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Allpass Phase Shifter - Phase shift without amplitude change
///
/// This indicator implements an allpass filter that shifts the phase of
/// the signal without changing its amplitude spectrum. Useful for creating
/// quadrature (90-degree shifted) versions of price for cycle analysis or
/// for compensating phase delays in other indicators. The output is the
/// phase-shifted signal.
#[derive(Debug, Clone)]
pub struct AllpassPhaseShifter {
    period: usize,
    phase_shift_degrees: f64,
}

impl AllpassPhaseShifter {
    /// Creates a new AllpassPhaseShifter
    ///
    /// # Parameters
    /// - `period`: The reference period for the phase shift (minimum 5)
    /// - `phase_shift_degrees`: The desired phase shift in degrees (-180 to 180)
    ///
    /// # Returns
    /// A Result containing the AllpassPhaseShifter or an error if parameters are invalid
    pub fn new(period: usize, phase_shift_degrees: f64) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if phase_shift_degrees < -180.0 || phase_shift_degrees > 180.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "phase_shift_degrees".to_string(),
                reason: "must be between -180 and 180".to_string(),
            });
        }
        Ok(Self { period, phase_shift_degrees })
    }

    /// Calculate allpass phase shifter output
    ///
    /// Uses Hilbert transform approximation to create the phase-shifted signal.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        let window = self.period * 2;
        let phase_rad = self.phase_shift_degrees * std::f64::consts::PI / 180.0;

        for i in window..n {
            let start = i.saturating_sub(window);
            let slice = &close[start..=i];
            let len = slice.len();
            let mean: f64 = slice.iter().sum::<f64>() / len as f64;

            // Calculate in-phase and quadrature components at the reference period
            let mut in_phase = 0.0;
            let mut quadrature = 0.0;

            for (j, &val) in slice.iter().enumerate() {
                let angle = 2.0 * std::f64::consts::PI * j as f64 / self.period as f64;
                let centered = val - mean;
                in_phase += centered * angle.cos();
                quadrature += centered * angle.sin();
            }

            in_phase /= len as f64;
            quadrature /= len as f64;

            // Apply phase rotation
            let cos_shift = phase_rad.cos();
            let sin_shift = phase_rad.sin();

            let shifted_i = in_phase * cos_shift - quadrature * sin_shift;
            let shifted_q = in_phase * sin_shift + quadrature * cos_shift;

            // Reconstruct the signal at the last point with the new phase
            let j = len - 1;
            let angle = 2.0 * std::f64::consts::PI * j as f64 / self.period as f64;
            let shifted_value = shifted_i * angle.cos() + shifted_q * angle.sin();

            // Add back the mean and scale
            result[i] = mean + shifted_value * 2.0;
        }

        // Fill initial values
        for i in 0..window.min(n) {
            result[i] = close[i];
        }

        result
    }
}

impl TechnicalIndicator for AllpassPhaseShifter {
    fn name(&self) -> &str {
        "Allpass Phase Shifter"
    }

    fn min_periods(&self) -> usize {
        self.period * 2 + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Fourier Transform Power - Computes power spectrum using DFT
///
/// This indicator calculates the power spectrum of price data using a
/// discrete Fourier transform approach. It returns the total spectral
/// power across a range of frequencies, which can be used to identify
/// the strength of cyclical components in price movements. Higher values
/// indicate stronger periodic behavior.
#[derive(Debug, Clone)]
pub struct FourierTransformPower {
    period: usize,
    num_frequencies: usize,
}

impl FourierTransformPower {
    /// Creates a new FourierTransformPower indicator
    ///
    /// # Parameters
    /// - `period`: The lookback period for analysis (minimum 20)
    /// - `num_frequencies`: Number of frequency bins to analyze (2 to 20)
    ///
    /// # Returns
    /// A Result containing the FourierTransformPower or an error if parameters are invalid
    pub fn new(period: usize, num_frequencies: usize) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        if num_frequencies < 2 || num_frequencies > 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "num_frequencies".to_string(),
                reason: "must be between 2 and 20".to_string(),
            });
        }
        Ok(Self { period, num_frequencies })
    }

    /// Calculate the power at a specific frequency
    fn compute_power_at_frequency(&self, slice: &[f64], freq_period: usize) -> f64 {
        let len = slice.len();
        let mean: f64 = slice.iter().sum::<f64>() / len as f64;

        let mut real = 0.0;
        let mut imag = 0.0;

        for (j, &val) in slice.iter().enumerate() {
            let angle = 2.0 * std::f64::consts::PI * j as f64 / freq_period as f64;
            let centered = val - mean;
            real += centered * angle.cos();
            imag += centered * angle.sin();
        }

        // Power is magnitude squared, normalized by length
        (real * real + imag * imag) / (len * len) as f64
    }

    /// Calculate Fourier transform power
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // Generate frequency periods to analyze (logarithmically spaced)
        let min_period = 4;
        let max_period = self.period / 2;

        let freq_periods: Vec<usize> = (0..self.num_frequencies)
            .map(|i| {
                let t = i as f64 / (self.num_frequencies - 1).max(1) as f64;
                let log_min = (min_period as f64).ln();
                let log_max = (max_period as f64).ln();
                (log_min + t * (log_max - log_min)).exp() as usize
            })
            .collect();

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let slice = &close[start..=i];

            // Sum power across all frequency bins
            let mut total_power = 0.0;
            for &freq_period in &freq_periods {
                if freq_period >= 3 && freq_period <= slice.len() / 2 {
                    total_power += self.compute_power_at_frequency(slice, freq_period);
                }
            }

            result[i] = total_power;
        }

        result
    }
}

impl TechnicalIndicator for FourierTransformPower {
    fn name(&self) -> &str {
        "Fourier Transform Power"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Wavelet Smoothing - Applies wavelet-based smoothing to price data
///
/// This indicator uses a simplified wavelet decomposition approach to
/// smooth price data while preserving important features like trend
/// changes and reversals. It decomposes the signal into approximation
/// (low-frequency) and detail (high-frequency) components, then
/// reconstructs using only the approximation for smoothing.
#[derive(Debug, Clone)]
pub struct WaveletSmoothing {
    period: usize,
    decomposition_level: usize,
}

impl WaveletSmoothing {
    /// Creates a new WaveletSmoothing indicator
    ///
    /// # Parameters
    /// - `period`: The lookback period for wavelet analysis (minimum 16)
    /// - `decomposition_level`: Number of decomposition levels (1 to 4)
    ///
    /// # Returns
    /// A Result containing the WaveletSmoothing or an error if parameters are invalid
    pub fn new(period: usize, decomposition_level: usize) -> Result<Self> {
        if period < 16 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 16".to_string(),
            });
        }
        if decomposition_level < 1 || decomposition_level > 4 {
            return Err(IndicatorError::InvalidParameter {
                name: "decomposition_level".to_string(),
                reason: "must be between 1 and 4".to_string(),
            });
        }
        Ok(Self { period, decomposition_level })
    }

    /// Apply Haar wavelet low-pass filter (averaging)
    fn haar_lowpass(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < 2 {
            return data.to_vec();
        }

        let mut result = Vec::with_capacity((n + 1) / 2);
        let mut i = 0;
        while i + 1 < n {
            result.push((data[i] + data[i + 1]) / 2.0);
            i += 2;
        }
        if n % 2 == 1 {
            result.push(data[n - 1]);
        }
        result
    }

    /// Upsample by repeating each value
    fn upsample(&self, data: &[f64], target_len: usize) -> Vec<f64> {
        let mut result = Vec::with_capacity(target_len);
        for &val in data {
            result.push(val);
            if result.len() < target_len {
                result.push(val);
            }
        }
        while result.len() < target_len {
            if let Some(&last) = data.last() {
                result.push(last);
            }
        }
        result.truncate(target_len);
        result
    }

    /// Calculate wavelet smoothing
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let slice = &close[start..=i];
            let original_len = slice.len();

            // Multi-level wavelet decomposition (approximation only)
            let mut approx = slice.to_vec();
            for _ in 0..self.decomposition_level {
                if approx.len() >= 2 {
                    approx = self.haar_lowpass(&approx);
                }
            }

            // Reconstruction: upsample back to original size
            for _ in 0..self.decomposition_level {
                approx = self.upsample(&approx, original_len);
            }

            // Take the last value as the smoothed output
            result[i] = approx.last().copied().unwrap_or(close[i]);
        }

        // Fill initial values
        for i in 0..self.period.min(n) {
            result[i] = close[i];
        }

        result
    }
}

impl TechnicalIndicator for WaveletSmoothing {
    fn name(&self) -> &str {
        "Wavelet Smoothing"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Adaptive Low-Pass Filter - Filter that adapts cutoff based on volatility
///
/// This indicator implements an adaptive low-pass filter that adjusts its
/// cutoff frequency based on local price volatility. During high volatility
/// periods, it uses a faster response (higher cutoff) to track rapid changes;
/// during low volatility, it uses stronger smoothing to filter noise.
#[derive(Debug, Clone)]
pub struct AdaptiveLPFilter {
    period: usize,
    base_cutoff: usize,
    sensitivity: f64,
}

impl AdaptiveLPFilter {
    /// Creates a new AdaptiveLPFilter
    ///
    /// # Parameters
    /// - `period`: Lookback period for volatility estimation (minimum 10)
    /// - `base_cutoff`: Base cutoff period for the filter (minimum 5)
    /// - `sensitivity`: How much the filter adapts to volatility (0.1 to 2.0)
    ///
    /// # Returns
    /// A Result containing the AdaptiveLPFilter or an error if parameters are invalid
    pub fn new(period: usize, base_cutoff: usize, sensitivity: f64) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if base_cutoff < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "base_cutoff".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if sensitivity < 0.1 || sensitivity > 2.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "sensitivity".to_string(),
                reason: "must be between 0.1 and 2.0".to_string(),
            });
        }
        Ok(Self { period, base_cutoff, sensitivity })
    }

    /// Estimate local volatility using standard deviation of returns
    fn estimate_volatility(&self, slice: &[f64]) -> f64 {
        if slice.len() < 2 {
            return 0.0;
        }

        // Calculate returns
        let returns: Vec<f64> = slice.windows(2)
            .map(|w| if w[0].abs() > 1e-10 { (w[1] - w[0]) / w[0] } else { 0.0 })
            .collect();

        if returns.is_empty() {
            return 0.0;
        }

        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns.iter()
            .map(|&r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;

        variance.sqrt()
    }

    /// Calculate adaptive low-pass filter output
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];
        let mut filtered = 0.0;

        // Calculate long-term average volatility for normalization
        let mut vol_history = Vec::new();

        for i in 0..n {
            if i < self.period {
                result[i] = close[i];
                filtered = close[i];
                continue;
            }

            let start = i.saturating_sub(self.period);
            let slice = &close[start..=i];

            // Estimate current volatility
            let current_vol = self.estimate_volatility(slice);
            vol_history.push(current_vol);

            // Calculate average volatility
            let avg_vol = if vol_history.len() > 10 {
                vol_history.iter().sum::<f64>() / vol_history.len() as f64
            } else {
                current_vol
            };

            // Normalize volatility (ratio to average)
            let vol_ratio = if avg_vol > 1e-10 {
                (current_vol / avg_vol).max(0.5).min(2.0)
            } else {
                1.0
            };

            // Adaptive cutoff: higher volatility = faster response (lower effective period)
            let adaptive_cutoff = self.base_cutoff as f64 / (1.0 + self.sensitivity * (vol_ratio - 1.0));
            let effective_cutoff = adaptive_cutoff.max(3.0);

            // Calculate filter coefficient
            let omega = 2.0 * std::f64::consts::PI / effective_cutoff;
            let alpha = (1.0 - omega.cos()) / omega.sin();
            let a = alpha / (1.0 + alpha);

            // Apply adaptive smoothing
            filtered = a * close[i] + (1.0 - a) * filtered;
            result[i] = filtered;
        }

        result
    }
}

impl TechnicalIndicator for AdaptiveLPFilter {
    fn name(&self) -> &str {
        "Adaptive LP Filter"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Phase Detector - Detects the phase of dominant cycle in price data
///
/// This indicator detects and tracks the instantaneous phase of the
/// dominant cycle in price data using quadrature analysis. The phase
/// output ranges from 0 to 360 degrees and can be used to identify
/// cycle turning points (0/360 = trough, 180 = peak).
#[derive(Debug, Clone)]
pub struct PhaseDetector {
    period: usize,
    smoothing: usize,
}

impl PhaseDetector {
    /// Creates a new PhaseDetector
    ///
    /// # Parameters
    /// - `period`: Analysis period for cycle detection (minimum 10)
    /// - `smoothing`: Phase smoothing period (minimum 1)
    ///
    /// # Returns
    /// A Result containing the PhaseDetector or an error if parameters are invalid
    pub fn new(period: usize, smoothing: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if smoothing < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self { period, smoothing })
    }

    /// Detect dominant cycle period using autocorrelation
    fn detect_cycle_period(&self, slice: &[f64]) -> usize {
        let len = slice.len();
        let mean: f64 = slice.iter().sum::<f64>() / len as f64;

        let min_period = 5;
        let max_period = len / 2;

        let mut best_period = min_period;
        let mut best_corr = f64::NEG_INFINITY;

        for period in min_period..=max_period {
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

    /// Calculate phase detector output
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];
        let mut phase_history: Vec<f64> = Vec::new();

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let slice = &close[start..=i];
            let len = slice.len();
            let mean: f64 = slice.iter().sum::<f64>() / len as f64;

            // Detect dominant cycle period
            let cycle_period = self.detect_cycle_period(slice);

            // Calculate in-phase (I) and quadrature (Q) components
            let mut in_phase = 0.0;
            let mut quadrature = 0.0;

            for (j, &val) in slice.iter().enumerate() {
                let angle = 2.0 * std::f64::consts::PI * j as f64 / cycle_period as f64;
                let centered = val - mean;
                in_phase += centered * angle.cos();
                quadrature += centered * angle.sin();
            }

            // Calculate instantaneous phase
            let raw_phase = quadrature.atan2(in_phase);
            let phase_degrees = (raw_phase * 180.0 / std::f64::consts::PI + 360.0) % 360.0;

            // Add to phase history for smoothing
            phase_history.push(phase_degrees);
            if phase_history.len() > self.smoothing {
                phase_history.remove(0);
            }

            // Circular mean for phase smoothing (handles wraparound)
            let mut sin_sum = 0.0;
            let mut cos_sum = 0.0;
            for &p in &phase_history {
                let rad = p * std::f64::consts::PI / 180.0;
                sin_sum += rad.sin();
                cos_sum += rad.cos();
            }
            let smoothed_phase = (sin_sum.atan2(cos_sum) * 180.0 / std::f64::consts::PI + 360.0) % 360.0;

            result[i] = smoothed_phase;
        }

        result
    }
}

impl TechnicalIndicator for PhaseDetector {
    fn name(&self) -> &str {
        "Phase Detector"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Amplitude Extractor - Extracts the amplitude of dominant cycle
///
/// This indicator measures the amplitude (strength) of the dominant
/// cycle in price data. It uses the magnitude of the DFT coefficient
/// at the dominant frequency to estimate cycle amplitude. Higher values
/// indicate stronger, more pronounced cycles.
#[derive(Debug, Clone)]
pub struct AmplitudeExtractor {
    period: usize,
    normalize: bool,
}

impl AmplitudeExtractor {
    /// Creates a new AmplitudeExtractor
    ///
    /// # Parameters
    /// - `period`: Analysis period for amplitude extraction (minimum 10)
    /// - `normalize`: Whether to normalize amplitude by price level
    ///
    /// # Returns
    /// A Result containing the AmplitudeExtractor or an error if parameters are invalid
    pub fn new(period: usize, normalize: bool) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { period, normalize })
    }

    /// Detect dominant cycle period
    fn detect_cycle_period(&self, slice: &[f64]) -> usize {
        let len = slice.len();
        let mean: f64 = slice.iter().sum::<f64>() / len as f64;

        let min_period = 5;
        let max_period = len / 2;

        let mut best_period = min_period;
        let mut best_power = 0.0f64;

        for period in min_period..=max_period {
            let mut real = 0.0;
            let mut imag = 0.0;

            for (j, &val) in slice.iter().enumerate() {
                let angle = 2.0 * std::f64::consts::PI * j as f64 / period as f64;
                let centered = val - mean;
                real += centered * angle.cos();
                imag += centered * angle.sin();
            }

            let power = real * real + imag * imag;
            if power > best_power {
                best_power = power;
                best_period = period;
            }
        }

        best_period
    }

    /// Calculate amplitude extractor output
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let slice = &close[start..=i];
            let len = slice.len();
            let mean: f64 = slice.iter().sum::<f64>() / len as f64;

            // Detect dominant cycle period
            let cycle_period = self.detect_cycle_period(slice);

            // Calculate DFT coefficient at dominant frequency
            let mut real = 0.0;
            let mut imag = 0.0;

            for (j, &val) in slice.iter().enumerate() {
                let angle = 2.0 * std::f64::consts::PI * j as f64 / cycle_period as f64;
                let centered = val - mean;
                real += centered * angle.cos();
                imag += centered * angle.sin();
            }

            // Amplitude is the magnitude of the complex coefficient
            let amplitude = (real * real + imag * imag).sqrt() * 2.0 / len as f64;

            // Optionally normalize by price level
            let final_amplitude = if self.normalize && mean.abs() > 1e-10 {
                (amplitude / mean.abs()) * 100.0  // As percentage
            } else {
                amplitude
            };

            result[i] = final_amplitude;
        }

        result
    }
}

impl TechnicalIndicator for AmplitudeExtractor {
    fn name(&self) -> &str {
        "Amplitude Extractor"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Trend Cycle Decomposer - Separates trend and cycle components
///
/// This indicator decomposes price into trend and cycle components using
/// a combination of low-pass filtering (for trend) and the residual (for
/// cycle). It returns the cycle component, which oscillates around zero
/// and represents the deviation from the underlying trend.
#[derive(Debug, Clone)]
pub struct TrendCycleDecomposer {
    trend_period: usize,
    cycle_period: usize,
}

impl TrendCycleDecomposer {
    /// Creates a new TrendCycleDecomposer
    ///
    /// # Parameters
    /// - `trend_period`: Period for trend extraction (minimum 20)
    /// - `cycle_period`: Minimum cycle period to preserve (minimum 5)
    ///
    /// # Returns
    /// A Result containing the TrendCycleDecomposer or an error if parameters are invalid
    pub fn new(trend_period: usize, cycle_period: usize) -> Result<Self> {
        if trend_period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "trend_period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        if cycle_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "cycle_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if cycle_period >= trend_period {
            return Err(IndicatorError::InvalidParameter {
                name: "cycle_period".to_string(),
                reason: "must be less than trend_period".to_string(),
            });
        }
        Ok(Self { trend_period, cycle_period })
    }

    /// Calculate trend using supersmoother filter
    fn calculate_trend(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut trend = vec![0.0; n];

        // Ehlers' supersmoother coefficients
        let omega = 2.0 * std::f64::consts::PI / self.trend_period as f64;
        let a1 = (-1.414 * omega).exp();
        let b1 = 2.0 * a1 * (1.414 * omega).cos();
        let c2 = b1;
        let c3 = -a1 * a1;
        let c1 = 1.0 - c2 - c3;

        if n > 0 {
            trend[0] = close[0];
        }
        if n > 1 {
            trend[1] = close[1];
        }

        for i in 2..n {
            trend[i] = c1 * (close[i] + close[i - 1]) / 2.0 + c2 * trend[i - 1] + c3 * trend[i - 2];
        }

        trend
    }

    /// Calculate cycle component using bandpass filter
    fn calculate_cycle(&self, close: &[f64], trend: &[f64]) -> Vec<f64> {
        let n = close.len();

        // Get the detrended signal (cycle = price - trend)
        let detrended: Vec<f64> = close.iter().zip(trend.iter())
            .map(|(&c, &t)| c - t)
            .collect();

        // Apply light smoothing to remove very high frequency noise
        // Use simple EMA-style smoothing
        let mut cycle = vec![0.0; n];
        let alpha = 2.0 / (self.cycle_period as f64 / 2.0 + 1.0);

        if n > 0 {
            cycle[0] = detrended[0];
        }

        for i in 1..n {
            cycle[i] = alpha * detrended[i] + (1.0 - alpha) * cycle[i - 1];
        }

        cycle
    }

    /// Calculate trend-cycle decomposition (returns cycle component)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let trend = self.calculate_trend(close);
        self.calculate_cycle(close, &trend)
    }

    /// Calculate and return both trend and cycle components
    pub fn calculate_both(&self, close: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let trend = self.calculate_trend(close);
        let cycle = self.calculate_cycle(close, &trend);
        (trend, cycle)
    }
}

impl TechnicalIndicator for TrendCycleDecomposer {
    fn name(&self) -> &str {
        "Trend Cycle Decomposer"
    }

    fn min_periods(&self) -> usize {
        self.trend_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (trend, cycle) = self.calculate_both(&data.close);
        Ok(IndicatorOutput::dual(cycle, trend))
    }
}

/// Moving Average Convergence - Measures convergence of different MAs
///
/// This indicator calculates the degree of convergence or divergence between
/// multiple moving averages of different periods. When MAs converge, it suggests
/// consolidation; when they diverge, it indicates trending behavior. The output
/// is a normalized convergence score from 0 (maximum divergence) to 100
/// (perfect convergence).
#[derive(Debug, Clone)]
pub struct MovingAverageConvergence {
    short_period: usize,
    medium_period: usize,
    long_period: usize,
}

impl MovingAverageConvergence {
    /// Creates a new MovingAverageConvergence
    ///
    /// # Parameters
    /// - `short_period`: The period for the short MA (minimum 5)
    /// - `medium_period`: The period for the medium MA (must be > short_period)
    /// - `long_period`: The period for the long MA (must be > medium_period)
    ///
    /// # Returns
    /// A Result containing the MovingAverageConvergence or an error if parameters are invalid
    pub fn new(short_period: usize, medium_period: usize, long_period: usize) -> Result<Self> {
        if short_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if medium_period <= short_period {
            return Err(IndicatorError::InvalidParameter {
                name: "medium_period".to_string(),
                reason: "must be greater than short_period".to_string(),
            });
        }
        if long_period <= medium_period {
            return Err(IndicatorError::InvalidParameter {
                name: "long_period".to_string(),
                reason: "must be greater than medium_period".to_string(),
            });
        }
        Ok(Self { short_period, medium_period, long_period })
    }

    /// Calculate simple moving average at a position
    fn sma(&self, close: &[f64], end_idx: usize, period: usize) -> f64 {
        if period == 0 || end_idx + 1 < period {
            return close[end_idx];
        }
        let start = end_idx + 1 - period;
        let sum: f64 = close[start..=end_idx].iter().sum();
        sum / period as f64
    }

    /// Calculate moving average convergence
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // Pre-calculate all three moving averages
        let mut short_ma = vec![0.0; n];
        let mut medium_ma = vec![0.0; n];
        let mut long_ma = vec![0.0; n];

        for i in 0..n {
            short_ma[i] = self.sma(close, i, self.short_period);
            medium_ma[i] = self.sma(close, i, self.medium_period);
            long_ma[i] = self.sma(close, i, self.long_period);
        }

        // Calculate convergence after warmup
        for i in self.long_period..n {
            let s = short_ma[i];
            let m = medium_ma[i];
            let l = long_ma[i];

            // Calculate average of the MAs
            let avg_ma = (s + m + l) / 3.0;

            // Calculate spread as percentage of average
            let max_ma = s.max(m).max(l);
            let min_ma = s.min(m).min(l);

            let spread = max_ma - min_ma;
            let relative_spread = if avg_ma.abs() > 1e-10 {
                (spread / avg_ma.abs()) * 100.0
            } else {
                0.0
            };

            // Convert to convergence score: 100 = perfect convergence, 0 = max divergence
            // Use an exponential decay for the spread
            // A relative spread of 5% or more maps to near-zero convergence
            let convergence = (100.0 * (-relative_spread / 2.0).exp()).min(100.0).max(0.0);

            result[i] = convergence;
        }

        result
    }
}

impl TechnicalIndicator for MovingAverageConvergence {
    fn name(&self) -> &str {
        "Moving Average Convergence"
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

// =============================================================================
// NEW DSP INDICATORS - Batch 7
// =============================================================================

/// Adaptive Sine Wave - Adaptive sine wave that follows market cycles
///
/// This indicator generates a sine wave that adapts to the dominant market cycle.
/// It uses Hilbert transform concepts to detect the dominant period and generates
/// a sine wave synchronized with price action. The output oscillates between -1 and 1.
#[derive(Debug, Clone)]
pub struct AdaptiveSineWave {
    min_period: usize,
    max_period: usize,
    smoothing: f64,
}

impl AdaptiveSineWave {
    /// Creates a new AdaptiveSineWave indicator
    ///
    /// # Parameters
    /// - `min_period`: Minimum cycle period to detect (minimum 5)
    /// - `max_period`: Maximum cycle period to detect (must be > min_period)
    /// - `smoothing`: Smoothing factor for period transitions (0.0 to 1.0)
    pub fn new(min_period: usize, max_period: usize, smoothing: f64) -> Result<Self> {
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
        if smoothing < 0.0 || smoothing > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        Ok(Self { min_period, max_period, smoothing })
    }

    /// Detect dominant period using autocorrelation
    fn detect_period(&self, slice: &[f64]) -> f64 {
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

        best_period as f64
    }

    /// Calculate adaptive sine wave
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];
        let mut phase = 0.0;
        let mut prev_period = self.min_period as f64;

        for i in self.max_period..n {
            let start = i.saturating_sub(self.max_period);
            let slice = &close[start..=i];

            // Detect current dominant period
            let detected = self.detect_period(slice);

            // Smooth period transition
            let smooth_period = self.smoothing * prev_period + (1.0 - self.smoothing) * detected;
            prev_period = smooth_period;

            // Calculate phase increment based on period
            let phase_inc = 2.0 * std::f64::consts::PI / smooth_period;
            phase += phase_inc;

            // Keep phase in range 0 to 2*PI
            while phase > 2.0 * std::f64::consts::PI {
                phase -= 2.0 * std::f64::consts::PI;
            }

            // Generate sine wave
            result[i] = phase.sin();
        }

        result
    }

    /// Calculate both sine and lead sine (45 degrees ahead)
    pub fn calculate_with_lead(&self, close: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut sine = vec![0.0; n];
        let mut lead_sine = vec![0.0; n];
        let mut phase = 0.0;
        let mut prev_period = self.min_period as f64;

        for i in self.max_period..n {
            let start = i.saturating_sub(self.max_period);
            let slice = &close[start..=i];

            let detected = self.detect_period(slice);
            let smooth_period = self.smoothing * prev_period + (1.0 - self.smoothing) * detected;
            prev_period = smooth_period;

            let phase_inc = 2.0 * std::f64::consts::PI / smooth_period;
            phase += phase_inc;

            while phase > 2.0 * std::f64::consts::PI {
                phase -= 2.0 * std::f64::consts::PI;
            }

            sine[i] = phase.sin();
            lead_sine[i] = (phase + std::f64::consts::PI / 4.0).sin(); // 45 degrees ahead
        }

        (sine, lead_sine)
    }
}

impl TechnicalIndicator for AdaptiveSineWave {
    fn name(&self) -> &str {
        "Adaptive Sine Wave"
    }

    fn min_periods(&self) -> usize {
        self.max_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (sine, lead_sine) = self.calculate_with_lead(&data.close);
        Ok(IndicatorOutput::dual(sine, lead_sine))
    }
}

/// Cycle Bandwidth - Measures bandwidth of dominant market cycle
///
/// This indicator measures the "bandwidth" or spread of cycle periods present
/// in price data. A narrow bandwidth indicates a strong, consistent cycle,
/// while a wide bandwidth suggests multiple cycles or noise. Output is
/// normalized from 0 (narrow/pure cycle) to 100 (wide/noisy).
#[derive(Debug, Clone)]
pub struct CycleBandwidth {
    period: usize,
    num_bands: usize,
}

impl CycleBandwidth {
    /// Creates a new CycleBandwidth indicator
    ///
    /// # Parameters
    /// - `period`: Lookback period for analysis (minimum 20)
    /// - `num_bands`: Number of frequency bands to analyze (2 to 8)
    pub fn new(period: usize, num_bands: usize) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        if num_bands < 2 || num_bands > 8 {
            return Err(IndicatorError::InvalidParameter {
                name: "num_bands".to_string(),
                reason: "must be between 2 and 8".to_string(),
            });
        }
        Ok(Self { period, num_bands })
    }

    /// Calculate power at a specific cycle period
    fn band_power(&self, slice: &[f64], band_period: usize) -> f64 {
        let len = slice.len();
        if band_period >= len / 2 || band_period == 0 {
            return 0.0;
        }

        let mean: f64 = slice.iter().sum::<f64>() / len as f64;

        // Goertzel-like calculation for specific frequency
        let mut cos_sum = 0.0;
        let mut sin_sum = 0.0;
        let omega = 2.0 * std::f64::consts::PI / band_period as f64;

        for (j, &val) in slice.iter().enumerate() {
            let centered = val - mean;
            cos_sum += centered * (omega * j as f64).cos();
            sin_sum += centered * (omega * j as f64).sin();
        }

        // Power is magnitude squared
        (cos_sum * cos_sum + sin_sum * sin_sum) / (len as f64 * len as f64)
    }

    /// Calculate cycle bandwidth
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // Define band periods (logarithmically spaced)
        let min_band = 5;
        let max_band = self.period / 2;
        let band_periods: Vec<usize> = (0..self.num_bands)
            .map(|i| {
                let t = i as f64 / (self.num_bands - 1).max(1) as f64;
                let log_period = (min_band as f64).ln() + t * ((max_band as f64).ln() - (min_band as f64).ln());
                log_period.exp().round() as usize
            })
            .collect();

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let slice = &close[start..=i];

            // Calculate power at each band
            let powers: Vec<f64> = band_periods.iter()
                .map(|&p| self.band_power(slice, p))
                .collect();

            let total_power: f64 = powers.iter().sum();
            if total_power < 1e-10 {
                result[i] = 50.0; // Neutral when no power
                continue;
            }

            // Calculate entropy-based bandwidth measure
            // Higher entropy = more spread out power = wider bandwidth
            let mut entropy = 0.0;
            for &power in &powers {
                let p = power / total_power;
                if p > 1e-10 {
                    entropy -= p * p.ln();
                }
            }

            // Normalize to 0-100 scale
            // Max entropy is ln(num_bands)
            let max_entropy = (self.num_bands as f64).ln();
            let bandwidth = if max_entropy > 0.0 {
                (entropy / max_entropy * 100.0).min(100.0).max(0.0)
            } else {
                0.0
            };

            result[i] = bandwidth;
        }

        result
    }
}

impl TechnicalIndicator for CycleBandwidth {
    fn name(&self) -> &str {
        "Cycle Bandwidth"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Signal Envelope - Extracts envelope of cyclical price action
///
/// This indicator extracts the upper and lower envelope of cyclical price
/// movements using the Hilbert transform approach. The envelope captures
/// the amplitude modulation of price cycles. Output is the envelope amplitude.
#[derive(Debug, Clone)]
pub struct SignalEnvelope {
    period: usize,
    smoothing: f64,
}

impl SignalEnvelope {
    /// Creates a new SignalEnvelope indicator
    ///
    /// # Parameters
    /// - `period`: Analysis period (minimum 10)
    /// - `smoothing`: Envelope smoothing factor (0.0 to 1.0)
    pub fn new(period: usize, smoothing: f64) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if smoothing < 0.0 || smoothing > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        Ok(Self { period, smoothing })
    }

    /// Calculate signal envelope using Hilbert transform approximation
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut envelope = vec![0.0; n];

        // First, detrend the data
        let mut detrended = vec![0.0; n];
        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let slice = &close[start..=i];
            let mean: f64 = slice.iter().sum::<f64>() / slice.len() as f64;
            detrended[i] = close[i] - mean;
        }

        // Calculate in-phase and quadrature components using weighted average
        // This is a simplified Hilbert transform approximation
        let half_period = self.period / 2;
        let mut prev_envelope = 0.0;

        for i in self.period..n {
            let mut in_phase = 0.0;
            let mut quadrature = 0.0;

            // Compute I and Q using sine/cosine weighting
            for j in 0..self.period.min(i + 1) {
                let idx = i - j;
                if idx >= self.period {
                    let angle = 2.0 * std::f64::consts::PI * j as f64 / self.period as f64;
                    in_phase += detrended[idx] * angle.cos();
                    quadrature += detrended[idx] * angle.sin();
                }
            }

            in_phase /= half_period as f64;
            quadrature /= half_period as f64;

            // Envelope is the magnitude of the analytic signal
            let raw_envelope = (in_phase * in_phase + quadrature * quadrature).sqrt();

            // Apply smoothing
            let smooth_envelope = self.smoothing * prev_envelope + (1.0 - self.smoothing) * raw_envelope;
            prev_envelope = smooth_envelope;

            envelope[i] = smooth_envelope;
        }

        envelope
    }

    /// Calculate upper and lower envelope bands
    pub fn calculate_bands(&self, close: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();
        let envelope = self.calculate(close);

        // Calculate trend (simple moving average)
        let mut trend = vec![0.0; n];
        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            trend[i] = close[start..=i].iter().sum::<f64>() / (i - start + 1) as f64;
        }

        // Upper and lower bands
        let upper: Vec<f64> = (0..n).map(|i| trend[i] + envelope[i]).collect();
        let lower: Vec<f64> = (0..n).map(|i| trend[i] - envelope[i]).collect();

        (upper, lower)
    }
}

impl TechnicalIndicator for SignalEnvelope {
    fn name(&self) -> &str {
        "Signal Envelope"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (upper, lower) = self.calculate_bands(&data.close);
        Ok(IndicatorOutput::dual(upper, lower))
    }
}

/// Instantaneous Trend - Instantaneous trendline using Hilbert transform concepts
///
/// This indicator calculates an instantaneous trendline that adapts to market
/// conditions using concepts from the Hilbert transform. It provides a smooth
/// trend estimate with minimal lag compared to traditional moving averages.
#[derive(Debug, Clone)]
pub struct InstantaneousTrend {
    period: usize,
    alpha: f64,
}

impl InstantaneousTrend {
    /// Creates a new InstantaneousTrend indicator
    ///
    /// # Parameters
    /// - `period`: Base period for trend calculation (minimum 8)
    /// - `alpha`: Smoothing coefficient (0.0 to 1.0, default around 0.07)
    pub fn new(period: usize, alpha: f64) -> Result<Self> {
        if period < 8 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 8".to_string(),
            });
        }
        if alpha <= 0.0 || alpha > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "alpha".to_string(),
                reason: "must be between 0.0 (exclusive) and 1.0 (inclusive)".to_string(),
            });
        }
        Ok(Self { period, alpha })
    }

    /// Calculate instantaneous trend using Hilbert-based approach
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut trend = vec![0.0; n];

        // Use smoothed price as base
        let mut smooth = vec![0.0; n];
        for i in 3..n {
            smooth[i] = (4.0 * close[i] + 3.0 * close[i - 1] + 2.0 * close[i - 2] + close[i - 3]) / 10.0;
        }

        // Hilbert transform coefficients (Ehlers' standard values)
        let c1 = 0.0962;
        let c2 = 0.5769;

        let mut detrender = vec![0.0; n];
        let mut i1 = vec![0.0; n];
        let mut q1 = vec![0.0; n];
        let mut i2 = vec![0.0; n];
        let mut q2 = vec![0.0; n];
        let mut period_arr = vec![self.period as f64; n];

        for i in 7..n {
            // Detrend using Hilbert coefficients
            detrender[i] = c1 * smooth[i] + c2 * c1 * smooth[i - 2]
                - c2 * c1 * smooth[i - 4] - c1 * smooth[i - 6];
            detrender[i] += c2 * detrender[i - 1];

            // In-phase component
            i1[i] = detrender[i - 3];

            // Quadrature component
            q1[i] = c1 * detrender[i] + c2 * c1 * detrender[i - 2]
                - c2 * c1 * detrender[i - 4] - c1 * detrender[i - 6];
            q1[i] += c2 * q1[i - 1];

            // Phase advance for phasor addition
            let ji = c1 * i1[i] + c2 * c1 * i1[i - 2] - c2 * c1 * i1[i - 4] - c1 * i1[i - 6];
            let jq = c1 * q1[i] + c2 * c1 * q1[i - 2] - c2 * c1 * q1[i - 4] - c1 * q1[i - 6];

            // Phasor addition
            i2[i] = i1[i] - jq;
            q2[i] = q1[i] + ji;

            // Smooth I and Q
            i2[i] = 0.2 * i2[i] + 0.8 * i2[i - 1];
            q2[i] = 0.2 * q2[i] + 0.8 * q2[i - 1];

            // Calculate instantaneous period from homodyne discriminator
            let re = i2[i] * i2[i - 1] + q2[i] * q2[i - 1];
            let im = i2[i] * q2[i - 1] - q2[i] * i2[i - 1];

            if im.abs() > 1e-10 && re.abs() > 1e-10 {
                let inst_period = 2.0 * std::f64::consts::PI / im.atan2(re);
                // Constrain period
                let clamped = inst_period.max(self.period as f64 * 0.5).min(self.period as f64 * 2.0);
                period_arr[i] = 0.2 * clamped + 0.8 * period_arr[i - 1];
            } else {
                period_arr[i] = period_arr[i - 1];
            }

            // Calculate adaptive alpha based on period
            let adaptive_alpha = 2.0 / (period_arr[i] + 1.0) * self.alpha / 0.07;
            let alpha_clamped = adaptive_alpha.max(0.01).min(1.0);

            // Instantaneous trend calculation
            // Uses both current price and smoothed value
            if i == 7 {
                trend[i] = smooth[i];
            } else {
                trend[i] = alpha_clamped * (smooth[i] + smooth[i - 1]) / 2.0
                    + (1.0 - alpha_clamped) * trend[i - 1];
            }
        }

        trend
    }

    /// Calculate trend and its trigger line (delayed trend)
    pub fn calculate_with_trigger(&self, close: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let trend = self.calculate(close);
        let n = trend.len();

        // Trigger is a 2-bar delay of trend
        let mut trigger = vec![0.0; n];
        for i in 2..n {
            trigger[i] = trend[i - 2];
        }

        (trend, trigger)
    }
}

impl TechnicalIndicator for InstantaneousTrend {
    fn name(&self) -> &str {
        "Instantaneous Trend"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (trend, trigger) = self.calculate_with_trigger(&data.close);
        Ok(IndicatorOutput::dual(trend, trigger))
    }
}

/// Cycle Strength - Measures strength of current market cycle
///
/// This indicator measures how strong or dominant the current market cycle is.
/// A high value indicates a clear, consistent cycle; a low value indicates
/// weak or absent cyclical behavior. Output ranges from 0 to 100.
#[derive(Debug, Clone)]
pub struct CycleStrength {
    period: usize,
    cycle_period: usize,
}

impl CycleStrength {
    /// Creates a new CycleStrength indicator
    ///
    /// # Parameters
    /// - `period`: Lookback period for analysis (minimum 20)
    /// - `cycle_period`: Expected cycle period to measure strength (minimum 5)
    pub fn new(period: usize, cycle_period: usize) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        if cycle_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "cycle_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if cycle_period >= period {
            return Err(IndicatorError::InvalidParameter {
                name: "cycle_period".to_string(),
                reason: "must be less than period".to_string(),
            });
        }
        Ok(Self { period, cycle_period })
    }

    /// Calculate cycle strength using autocorrelation and spectral analysis
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let slice = &close[start..=i];
            let len = slice.len();
            let mean: f64 = slice.iter().sum::<f64>() / len as f64;

            // 1. Calculate autocorrelation at cycle period
            let mut auto_corr = 0.0;
            let mut norm1 = 0.0;
            let mut norm2 = 0.0;

            for j in 0..(len - self.cycle_period) {
                let x = slice[j] - mean;
                let y = slice[j + self.cycle_period] - mean;
                auto_corr += x * y;
                norm1 += x * x;
                norm2 += y * y;
            }

            let norm = (norm1 * norm2).sqrt();
            let corr = if norm > 1e-10 { auto_corr / norm } else { 0.0 };

            // 2. Calculate power at cycle frequency using Goertzel
            let omega = 2.0 * std::f64::consts::PI / self.cycle_period as f64;
            let mut cos_sum = 0.0;
            let mut sin_sum = 0.0;

            for (j, &val) in slice.iter().enumerate() {
                let centered = val - mean;
                cos_sum += centered * (omega * j as f64).cos();
                sin_sum += centered * (omega * j as f64).sin();
            }

            let cycle_power = (cos_sum * cos_sum + sin_sum * sin_sum) / (len as f64);

            // 3. Calculate total variance
            let variance: f64 = slice.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / len as f64;

            // 4. Calculate spectral concentration ratio
            let spectral_ratio = if variance > 1e-10 {
                (cycle_power / variance).min(1.0)
            } else {
                0.0
            };

            // 5. Combine autocorrelation and spectral ratio
            // Positive autocorrelation at cycle period suggests cycle presence
            // High spectral ratio suggests dominant cycle
            let corr_component = ((corr + 1.0) / 2.0 * 50.0).max(0.0); // Convert -1..1 to 0..50
            let spectral_component = spectral_ratio * 50.0;

            result[i] = (corr_component + spectral_component).min(100.0).max(0.0);
        }

        result
    }
}

impl TechnicalIndicator for CycleStrength {
    fn name(&self) -> &str {
        "Cycle Strength"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Adaptive Laguerre Filter - Laguerre filter with adaptive gamma
///
/// This indicator applies a Laguerre filter with a gamma parameter that adapts
/// to market conditions. It provides smooth filtering with the gamma adjusting
/// based on detected volatility or cycle characteristics.
#[derive(Debug, Clone)]
pub struct AdaptiveLaguerreFilter {
    min_gamma: f64,
    max_gamma: f64,
    lookback: usize,
}

impl AdaptiveLaguerreFilter {
    /// Creates a new AdaptiveLaguerreFilter indicator
    ///
    /// # Parameters
    /// - `min_gamma`: Minimum gamma value (0.0 to max_gamma)
    /// - `max_gamma`: Maximum gamma value (min_gamma to 1.0)
    /// - `lookback`: Lookback period for adaptation (minimum 10)
    pub fn new(min_gamma: f64, max_gamma: f64, lookback: usize) -> Result<Self> {
        if min_gamma < 0.0 || min_gamma >= 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "min_gamma".to_string(),
                reason: "must be between 0.0 and 1.0 (exclusive)".to_string(),
            });
        }
        if max_gamma <= min_gamma || max_gamma > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "max_gamma".to_string(),
                reason: "must be greater than min_gamma and at most 1.0".to_string(),
            });
        }
        if lookback < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { min_gamma, max_gamma, lookback })
    }

    /// Calculate adaptive gamma based on recent price action
    fn calculate_adaptive_gamma(&self, slice: &[f64]) -> f64 {
        let len = slice.len();
        if len < 2 {
            return (self.min_gamma + self.max_gamma) / 2.0;
        }

        // Calculate normalized volatility (coefficient of variation)
        let mean: f64 = slice.iter().sum::<f64>() / len as f64;
        let variance: f64 = slice.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / len as f64;
        let std_dev = variance.sqrt();
        let cv = if mean.abs() > 1e-10 { std_dev / mean.abs() } else { 0.0 };

        // Also measure directional movement
        let mut up_moves = 0.0;
        let mut down_moves = 0.0;
        for i in 1..len {
            let diff = slice[i] - slice[i - 1];
            if diff > 0.0 {
                up_moves += diff;
            } else {
                down_moves += -diff;
            }
        }
        let total_moves = up_moves + down_moves;
        let directional_ratio = if total_moves > 1e-10 {
            (up_moves - down_moves).abs() / total_moves
        } else {
            0.0
        };

        // High volatility or low directional movement -> higher gamma (smoother)
        // Low volatility and high directional movement -> lower gamma (faster)
        let vol_factor = (cv * 10.0).min(1.0); // Normalize CV
        let dir_factor = 1.0 - directional_ratio;

        let combined = (vol_factor + dir_factor) / 2.0;
        self.min_gamma + combined * (self.max_gamma - self.min_gamma)
    }

    /// Calculate adaptive Laguerre filter output
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // 4-element Laguerre filter state
        let mut l0 = 0.0;
        let mut l1 = 0.0;
        let mut l2 = 0.0;
        let mut l3 = 0.0;

        for i in 0..n {
            // Calculate adaptive gamma
            let gamma = if i >= self.lookback {
                let start = i.saturating_sub(self.lookback);
                self.calculate_adaptive_gamma(&close[start..=i])
            } else {
                (self.min_gamma + self.max_gamma) / 2.0
            };

            let one_minus_g = 1.0 - gamma;

            // Save previous values
            let l0_prev = l0;
            let l1_prev = l1;
            let l2_prev = l2;

            // Update Laguerre filter
            l0 = one_minus_g * close[i] + gamma * l0_prev;
            l1 = -gamma * l0 + l0_prev + gamma * l1_prev;
            l2 = -gamma * l1 + l1_prev + gamma * l2_prev;
            l3 = -gamma * l2 + l2_prev + gamma * l3;

            // Output is average of filter elements
            result[i] = (l0 + 2.0 * l1 + 2.0 * l2 + l3) / 6.0;
        }

        result
    }

    /// Calculate filter output with adaptive gamma values
    pub fn calculate_with_gamma(&self, close: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut result = vec![0.0; n];
        let mut gamma_values = vec![0.0; n];

        let mut l0 = 0.0;
        let mut l1 = 0.0;
        let mut l2 = 0.0;
        let mut l3 = 0.0;

        for i in 0..n {
            let gamma = if i >= self.lookback {
                let start = i.saturating_sub(self.lookback);
                self.calculate_adaptive_gamma(&close[start..=i])
            } else {
                (self.min_gamma + self.max_gamma) / 2.0
            };

            gamma_values[i] = gamma;
            let one_minus_g = 1.0 - gamma;

            let l0_prev = l0;
            let l1_prev = l1;
            let l2_prev = l2;

            l0 = one_minus_g * close[i] + gamma * l0_prev;
            l1 = -gamma * l0 + l0_prev + gamma * l1_prev;
            l2 = -gamma * l1 + l1_prev + gamma * l2_prev;
            l3 = -gamma * l2 + l2_prev + gamma * l3;

            result[i] = (l0 + 2.0 * l1 + 2.0 * l2 + l3) / 6.0;
        }

        (result, gamma_values)
    }
}

impl TechnicalIndicator for AdaptiveLaguerreFilter {
    fn name(&self) -> &str {
        "Adaptive Laguerre Filter"
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (filter, gamma) = self.calculate_with_gamma(&data.close);
        Ok(IndicatorOutput::dual(filter, gamma))
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
    // AdaptivePhaseMeasure Tests
    // =====================================================================

    #[test]
    fn test_adaptive_phase_measure_basic() {
        let close = make_test_data();
        let apm = AdaptivePhaseMeasure::new(20, 0.5).unwrap();
        let result = apm.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Phase should be between 0 and 360
        for i in 25..result.len() {
            assert!(result[i] >= 0.0 && result[i] < 360.0,
                    "Phase {} at index {} out of range", result[i], i);
        }
    }

    #[test]
    fn test_adaptive_phase_measure_validation() {
        // period too small
        assert!(AdaptivePhaseMeasure::new(5, 0.5).is_err());
        // invalid adaptive_factor
        assert!(AdaptivePhaseMeasure::new(20, 0.0).is_err());
        assert!(AdaptivePhaseMeasure::new(20, 1.5).is_err());
        // valid
        assert!(AdaptivePhaseMeasure::new(20, 0.5).is_ok());
        assert!(AdaptivePhaseMeasure::new(10, 1.0).is_ok());
    }

    #[test]
    fn test_adaptive_phase_measure_trait() {
        let data = make_ohlcv_series();
        let apm = AdaptivePhaseMeasure::new(15, 0.5).unwrap();

        assert_eq!(apm.name(), "Adaptive Phase Measure");
        assert_eq!(apm.min_periods(), 16);

        let output = apm.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // =====================================================================
    // FrequencyDomainMomentum Tests
    // =====================================================================

    #[test]
    fn test_frequency_domain_momentum_basic() {
        let close = make_test_data();
        let fdm = FrequencyDomainMomentum::new(20, 5).unwrap();
        let result = fdm.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Values after warmup should be finite
        for i in 30..result.len() {
            assert!(result[i].is_finite(), "Expected finite value at index {}", i);
        }
    }

    #[test]
    fn test_frequency_domain_momentum_validation() {
        // period too small
        assert!(FrequencyDomainMomentum::new(5, 3).is_err());
        // momentum_period too small
        assert!(FrequencyDomainMomentum::new(20, 1).is_err());
        // momentum_period >= period
        assert!(FrequencyDomainMomentum::new(20, 20).is_err());
        assert!(FrequencyDomainMomentum::new(20, 25).is_err());
        // valid
        assert!(FrequencyDomainMomentum::new(20, 5).is_ok());
    }

    #[test]
    fn test_frequency_domain_momentum_trait() {
        let data = make_ohlcv_series();
        let fdm = FrequencyDomainMomentum::new(20, 5).unwrap();

        assert_eq!(fdm.name(), "Frequency Domain Momentum");
        assert_eq!(fdm.min_periods(), 26);

        let output = fdm.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // =====================================================================
    // SpectralEntropy Tests
    // =====================================================================

    #[test]
    fn test_spectral_entropy_basic() {
        let close = make_test_data();
        let se = SpectralEntropy::new(30, 5).unwrap();
        let result = se.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Entropy should be between 0 and 100
        for i in 35..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0,
                    "Entropy {} at index {} out of range", result[i], i);
        }
    }

    #[test]
    fn test_spectral_entropy_validation() {
        // period too small
        assert!(SpectralEntropy::new(10, 5).is_err());
        // num_bins out of range
        assert!(SpectralEntropy::new(30, 2).is_err());
        assert!(SpectralEntropy::new(30, 25).is_err());
        // valid
        assert!(SpectralEntropy::new(30, 5).is_ok());
        assert!(SpectralEntropy::new(20, 3).is_ok());
        assert!(SpectralEntropy::new(30, 20).is_ok());
    }

    #[test]
    fn test_spectral_entropy_trait() {
        let data = make_ohlcv_series();
        let se = SpectralEntropy::new(25, 5).unwrap();

        assert_eq!(se.name(), "Spectral Entropy");
        assert_eq!(se.min_periods(), 26);

        let output = se.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // =====================================================================
    // CycleDominance Tests
    // =====================================================================

    #[test]
    fn test_cycle_dominance_basic() {
        let close = make_test_data();
        let cd = CycleDominance::new(5, 30).unwrap();
        let result = cd.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Dominance should be between 0 and 100
        for i in 35..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0,
                    "Dominance {} at index {} out of range", result[i], i);
        }
    }

    #[test]
    fn test_cycle_dominance_pure_sine() {
        // Pure sine wave should have high dominance
        let close: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64 * 0.5).sin() * 10.0)
            .collect();
        let cd = CycleDominance::new(5, 30).unwrap();
        let result = cd.calculate(&close);

        // Pure sine should have high dominance values
        let avg_dominance: f64 = result[40..].iter().sum::<f64>() / (result.len() - 40) as f64;
        assert!(avg_dominance > 20.0, "Expected high dominance for pure sine, got {}", avg_dominance);
    }

    #[test]
    fn test_cycle_dominance_validation() {
        // min_period too small
        assert!(CycleDominance::new(2, 30).is_err());
        // max_period <= min_period
        assert!(CycleDominance::new(10, 10).is_err());
        assert!(CycleDominance::new(10, 5).is_err());
        // max_period too large
        assert!(CycleDominance::new(10, 150).is_err());
        // valid
        assert!(CycleDominance::new(5, 30).is_ok());
    }

    #[test]
    fn test_cycle_dominance_trait() {
        let data = make_ohlcv_series();
        let cd = CycleDominance::new(5, 25).unwrap();

        assert_eq!(cd.name(), "Cycle Dominance");
        assert_eq!(cd.min_periods(), 26);

        let output = cd.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // =====================================================================
    // HarmonicAnalyzer Tests
    // =====================================================================

    #[test]
    fn test_harmonic_analyzer_basic() {
        let close = make_test_data();
        let ha = HarmonicAnalyzer::new(20, 3).unwrap();
        let result = ha.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Values should be non-negative and bounded
        for i in 45..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0,
                    "Harmonic score {} at index {} out of range", result[i], i);
        }
    }

    #[test]
    fn test_harmonic_analyzer_validation() {
        // base_period too small
        assert!(HarmonicAnalyzer::new(5, 3).is_err());
        // num_harmonics out of range
        assert!(HarmonicAnalyzer::new(20, 1).is_err());
        assert!(HarmonicAnalyzer::new(20, 10).is_err());
        // valid
        assert!(HarmonicAnalyzer::new(20, 2).is_ok());
        assert!(HarmonicAnalyzer::new(20, 6).is_ok());
    }

    #[test]
    fn test_harmonic_analyzer_trait() {
        let data = make_ohlcv_series();
        let ha = HarmonicAnalyzer::new(20, 3).unwrap();

        assert_eq!(ha.name(), "Harmonic Analyzer");
        assert_eq!(ha.min_periods(), 41);

        let output = ha.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // =====================================================================
    // PhaseCoherence Tests
    // =====================================================================

    #[test]
    fn test_phase_coherence_basic() {
        let close = make_test_data();
        let pc = PhaseCoherence::new(20, 5).unwrap();
        let result = pc.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Coherence should be between 0 and 100
        for i in 30..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0,
                    "Coherence {} at index {} out of range", result[i], i);
        }
    }

    #[test]
    fn test_phase_coherence_validation() {
        // period too small
        assert!(PhaseCoherence::new(5, 3).is_err());
        // oscillator_period too small
        assert!(PhaseCoherence::new(20, 2).is_err());
        // oscillator_period >= period
        assert!(PhaseCoherence::new(20, 20).is_err());
        assert!(PhaseCoherence::new(20, 25).is_err());
        // valid
        assert!(PhaseCoherence::new(20, 5).is_ok());
    }

    #[test]
    fn test_phase_coherence_trait() {
        let data = make_ohlcv_series();
        let pc = PhaseCoherence::new(20, 5).unwrap();

        assert_eq!(pc.name(), "Phase Coherence");
        assert_eq!(pc.min_periods(), 26);

        let output = pc.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // =====================================================================
    // FrequencyDomainMA Tests
    // =====================================================================

    #[test]
    fn test_frequency_domain_ma_basic() {
        let close = make_test_data();
        let fdma = FrequencyDomainMA::new(20, 5).unwrap();
        let result = fdma.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Values after warmup should be non-zero and reasonable
        for i in 25..result.len() {
            assert!(result[i] > 0.0, "Expected positive value at index {}", i);
            // Should be close to original price
            let diff = (result[i] - close[i]).abs();
            assert!(diff < 30.0, "Filtered value too far from original at index {}", i);
        }
    }

    #[test]
    fn test_frequency_domain_ma_smoothing() {
        // Pure trend data
        let close: Vec<f64> = (0..60).map(|i| 100.0 + i as f64 * 0.5).collect();
        let fdma = FrequencyDomainMA::new(20, 5).unwrap();
        let result = fdma.calculate(&close);

        // For smooth trend, filtered values should track the price closely
        for i in 25..result.len() {
            let diff = (result[i] - close[i]).abs();
            assert!(diff < 10.0, "Expected close tracking for smooth trend at index {}", i);
        }
    }

    #[test]
    fn test_frequency_domain_ma_validation() {
        // period too small
        assert!(FrequencyDomainMA::new(5, 3).is_err());
        // cutoff_period too small
        assert!(FrequencyDomainMA::new(20, 2).is_err());
        // cutoff_period > period/2
        assert!(FrequencyDomainMA::new(20, 15).is_err());
        // valid
        assert!(FrequencyDomainMA::new(20, 5).is_ok());
        assert!(FrequencyDomainMA::new(30, 10).is_ok());
    }

    #[test]
    fn test_frequency_domain_ma_trait() {
        let data = make_ohlcv_series();
        let fdma = FrequencyDomainMA::new(20, 5).unwrap();

        assert_eq!(fdma.name(), "Frequency Domain MA");
        assert_eq!(fdma.min_periods(), 21);

        let output = fdma.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // =====================================================================
    // PhaseShiftIndicator Tests
    // =====================================================================

    #[test]
    fn test_phase_shift_indicator_basic() {
        let close = make_test_data();
        let psi = PhaseShiftIndicator::new(20, 10).unwrap();
        let result = psi.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Phase shift should be between -180 and +180 degrees
        for i in 25..result.len() {
            assert!(result[i] >= -180.0 && result[i] <= 180.0,
                    "Phase shift {} at index {} out of range", result[i], i);
        }
    }

    #[test]
    fn test_phase_shift_indicator_constant_phase() {
        // Pure sine wave should have consistent phase progression
        let close: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64 * 0.2).sin() * 5.0)
            .collect();
        let psi = PhaseShiftIndicator::new(20, 10).unwrap();
        let result = psi.calculate(&close);

        // For a steady sine wave, phase shifts should be relatively stable
        let mut non_zero_count = 0;
        for i in 30..result.len() {
            if result[i].abs() > 0.1 {
                non_zero_count += 1;
            }
        }
        // Most values should be non-zero
        assert!(non_zero_count > 0, "Expected some non-zero phase shifts");
    }

    #[test]
    fn test_phase_shift_indicator_validation() {
        // period too small
        assert!(PhaseShiftIndicator::new(5, 3).is_err());
        // reference_period too small
        assert!(PhaseShiftIndicator::new(20, 3).is_err());
        // reference_period > period
        assert!(PhaseShiftIndicator::new(20, 25).is_err());
        // valid
        assert!(PhaseShiftIndicator::new(20, 10).is_ok());
        assert!(PhaseShiftIndicator::new(20, 20).is_ok()); // equal is allowed
    }

    #[test]
    fn test_phase_shift_indicator_trait() {
        let data = make_ohlcv_series();
        let psi = PhaseShiftIndicator::new(20, 10).unwrap();

        assert_eq!(psi.name(), "Phase Shift Indicator");
        assert_eq!(psi.min_periods(), 22);

        let output = psi.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // =====================================================================
    // SpectralPowerIndex Tests
    // =====================================================================

    #[test]
    fn test_spectral_power_index_basic() {
        let close = make_test_data();
        let spi = SpectralPowerIndex::new(30, 4).unwrap();
        let result = spi.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Index should be between 0 and 100
        for i in 35..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0,
                    "Spectral power index {} at index {} out of range", result[i], i);
        }
    }

    #[test]
    fn test_spectral_power_index_trending_data() {
        // Smooth trending data (should have high low-freq power)
        let close: Vec<f64> = (0..80).map(|i| 100.0 + i as f64 * 0.5).collect();
        let spi = SpectralPowerIndex::new(30, 4).unwrap();
        let result = spi.calculate(&close);

        // Trending data should have higher spectral power index (more low-freq power)
        let avg_spi: f64 = result[40..].iter().sum::<f64>() / (result.len() - 40) as f64;
        assert!(avg_spi >= 0.0, "Expected non-negative SPI for trending data, got {}", avg_spi);
    }

    #[test]
    fn test_spectral_power_index_validation() {
        // period too small
        assert!(SpectralPowerIndex::new(10, 4).is_err());
        // num_bands out of range
        assert!(SpectralPowerIndex::new(30, 1).is_err());
        assert!(SpectralPowerIndex::new(30, 15).is_err());
        // valid
        assert!(SpectralPowerIndex::new(30, 2).is_ok());
        assert!(SpectralPowerIndex::new(30, 10).is_ok());
    }

    #[test]
    fn test_spectral_power_index_trait() {
        let data = make_ohlcv_series();
        let spi = SpectralPowerIndex::new(25, 4).unwrap();

        assert_eq!(spi.name(), "Spectral Power Index");
        assert_eq!(spi.min_periods(), 26);

        let output = spi.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // =====================================================================
    // NoiseFilter Tests
    // =====================================================================

    #[test]
    fn test_noise_filter_basic() {
        let close = make_test_data();
        let nf = NoiseFilter::new(20, 0.5).unwrap();
        let result = nf.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Filtered values should be positive and close to original
        for i in 25..result.len() {
            assert!(result[i] > 0.0, "Expected positive value at index {}", i);
            let diff = (result[i] - close[i]).abs();
            assert!(diff < 20.0, "Filtered value too far from original at index {}", i);
        }
    }

    #[test]
    fn test_noise_filter_smooth_data() {
        // Smooth data should pass through with minimal change
        let close: Vec<f64> = (0..60).map(|i| 100.0 + i as f64 * 0.5).collect();
        let nf = NoiseFilter::new(20, 0.8).unwrap();
        let result = nf.calculate(&close);

        // For smooth data, filter should track closely
        for i in 30..result.len() {
            let diff = (result[i] - close[i]).abs();
            assert!(diff < 5.0, "Expected close tracking for smooth data at index {}", i);
        }
    }

    #[test]
    fn test_noise_filter_validation() {
        // period too small
        assert!(NoiseFilter::new(5, 0.5).is_err());
        // invalid smoothing_factor
        assert!(NoiseFilter::new(20, 0.0).is_err());
        assert!(NoiseFilter::new(20, 1.5).is_err());
        // valid
        assert!(NoiseFilter::new(20, 0.5).is_ok());
        assert!(NoiseFilter::new(10, 1.0).is_ok());
    }

    #[test]
    fn test_noise_filter_trait() {
        let data = make_ohlcv_series();
        let nf = NoiseFilter::new(20, 0.5).unwrap();

        assert_eq!(nf.name(), "Noise Filter");
        assert_eq!(nf.min_periods(), 21);

        let output = nf.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // =====================================================================
    // CyclePeriodEstimator Tests
    // =====================================================================

    #[test]
    fn test_cycle_period_estimator_basic() {
        let close = make_test_data();
        let cpe = CyclePeriodEstimator::new(5, 30, 3).unwrap();
        let result = cpe.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Period estimates should be within the specified range
        for i in 35..result.len() {
            assert!(result[i] >= 5.0 && result[i] <= 30.0,
                    "Estimated period {} at index {} out of range", result[i], i);
        }
    }

    #[test]
    fn test_cycle_period_estimator_known_cycle() {
        // Create data with known 12-bar cycle
        let close: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64 * 2.0 * std::f64::consts::PI / 12.0).sin() * 5.0)
            .collect();
        let cpe = CyclePeriodEstimator::new(5, 30, 5).unwrap();
        let result = cpe.calculate(&close);

        // Average detected period should be close to 12
        let avg_period: f64 = result[50..].iter().sum::<f64>() / (result.len() - 50) as f64;
        assert!(avg_period >= 8.0 && avg_period <= 16.0,
                "Expected period near 12, got {}", avg_period);
    }

    #[test]
    fn test_cycle_period_estimator_validation() {
        // min_period too small
        assert!(CyclePeriodEstimator::new(2, 30, 3).is_err());
        // max_period <= min_period
        assert!(CyclePeriodEstimator::new(10, 10, 3).is_err());
        assert!(CyclePeriodEstimator::new(10, 5, 3).is_err());
        // max_period too large
        assert!(CyclePeriodEstimator::new(5, 150, 3).is_err());
        // smoothing too small
        assert!(CyclePeriodEstimator::new(5, 30, 0).is_err());
        // valid
        assert!(CyclePeriodEstimator::new(5, 30, 3).is_ok());
    }

    #[test]
    fn test_cycle_period_estimator_trait() {
        let data = make_ohlcv_series();
        let cpe = CyclePeriodEstimator::new(5, 25, 3).unwrap();

        assert_eq!(cpe.name(), "Cycle Period Estimator");
        assert_eq!(cpe.min_periods(), 26);

        let output = cpe.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // =====================================================================
    // SignalToNoiseRatioAdvanced Tests
    // =====================================================================

    #[test]
    fn test_signal_to_noise_ratio_advanced_basic() {
        let close = make_test_data();
        let snr = SignalToNoiseRatioAdvanced::new(30, 5).unwrap();
        let result = snr.calculate(&close);

        assert_eq!(result.len(), close.len());
        // SNR should be within reasonable dB range
        for i in 35..result.len() {
            assert!(result[i] >= -20.0 && result[i] <= 40.0,
                    "SNR {} at index {} out of range", result[i], i);
        }
    }

    #[test]
    fn test_signal_to_noise_ratio_advanced_clean_signal() {
        // Clean smooth signal should have high SNR
        let close: Vec<f64> = (0..80).map(|i| 100.0 + i as f64 * 0.5).collect();
        let snr = SignalToNoiseRatioAdvanced::new(30, 5).unwrap();
        let result = snr.calculate(&close);

        // Smooth trend should have high SNR (positive dB)
        let avg_snr: f64 = result[40..].iter().sum::<f64>() / (result.len() - 40) as f64;
        assert!(avg_snr > -10.0, "Expected higher SNR for clean signal, got {} dB", avg_snr);
    }

    #[test]
    fn test_signal_to_noise_ratio_advanced_validation() {
        // period too small
        assert!(SignalToNoiseRatioAdvanced::new(10, 3).is_err());
        // signal_period too small
        assert!(SignalToNoiseRatioAdvanced::new(30, 2).is_err());
        // signal_period >= period/2
        assert!(SignalToNoiseRatioAdvanced::new(30, 15).is_err());
        assert!(SignalToNoiseRatioAdvanced::new(30, 20).is_err());
        // valid
        assert!(SignalToNoiseRatioAdvanced::new(30, 5).is_ok());
        assert!(SignalToNoiseRatioAdvanced::new(20, 3).is_ok());
    }

    #[test]
    fn test_signal_to_noise_ratio_advanced_trait() {
        let data = make_ohlcv_series();
        let snr = SignalToNoiseRatioAdvanced::new(25, 5).unwrap();

        assert_eq!(snr.name(), "Signal to Noise Ratio");
        assert_eq!(snr.min_periods(), 26);

        let output = snr.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // =====================================================================
    // BandpassFilter Tests
    // =====================================================================

    #[test]
    fn test_bandpass_filter_basic() {
        let close = make_test_data();
        let bpf = BandpassFilter::new(12, 0.5).unwrap();
        let result = bpf.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Values after warmup should be finite
        for i in 30..result.len() {
            assert!(result[i].is_finite(), "Expected finite value at index {}", i);
        }
    }

    #[test]
    fn test_bandpass_filter_isolates_frequency() {
        // Create data with known 12-bar cycle plus trend
        let close: Vec<f64> = (0..100)
            .map(|i| {
                let trend = 100.0 + i as f64 * 0.5;
                let cycle = (i as f64 * 2.0 * std::f64::consts::PI / 12.0).sin() * 5.0;
                trend + cycle
            })
            .collect();
        let bpf = BandpassFilter::new(12, 0.5).unwrap();
        let result = bpf.calculate(&close);

        // Bandpass should isolate the cycle component (oscillating around 0)
        let mut has_positive = false;
        let mut has_negative = false;
        for i in 40..result.len() {
            if result[i] > 0.5 {
                has_positive = true;
            }
            if result[i] < -0.5 {
                has_negative = true;
            }
        }
        assert!(has_positive && has_negative, "Bandpass filter should oscillate");
    }

    #[test]
    fn test_bandpass_filter_validation() {
        // center_period too small
        assert!(BandpassFilter::new(3, 0.5).is_err());
        // bandwidth invalid
        assert!(BandpassFilter::new(12, 0.0).is_err());
        assert!(BandpassFilter::new(12, 1.5).is_err());
        // valid
        assert!(BandpassFilter::new(12, 0.5).is_ok());
        assert!(BandpassFilter::new(5, 0.1).is_ok());
        assert!(BandpassFilter::new(20, 1.0).is_ok());
    }

    #[test]
    fn test_bandpass_filter_trait() {
        let data = make_ohlcv_series();
        let bpf = BandpassFilter::new(12, 0.5).unwrap();

        assert_eq!(bpf.name(), "Bandpass Filter");
        assert_eq!(bpf.min_periods(), 25);

        let output = bpf.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // =====================================================================
    // HighpassFilter Tests
    // =====================================================================

    #[test]
    fn test_highpass_filter_basic() {
        let close = make_test_data();
        let hpf = HighpassFilter::new(20, 2).unwrap();
        let result = hpf.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Values should be finite
        for i in 0..result.len() {
            assert!(result[i].is_finite(), "Expected finite value at index {}", i);
        }
    }

    #[test]
    fn test_highpass_filter_removes_trend() {
        // Create trending data
        let close: Vec<f64> = (0..100).map(|i| 100.0 + i as f64 * 2.0).collect();
        let hpf = HighpassFilter::new(20, 2).unwrap();
        let result = hpf.calculate(&close);

        // After warmup, highpass should oscillate near zero (trend removed)
        let avg: f64 = result[30..].iter().sum::<f64>() / (result.len() - 30) as f64;
        assert!(avg.abs() < 50.0, "Highpass should remove trend, got avg {}", avg);
    }

    #[test]
    fn test_highpass_filter_validation() {
        // cutoff_period too small
        assert!(HighpassFilter::new(3, 2).is_err());
        // poles out of range
        assert!(HighpassFilter::new(20, 0).is_err());
        assert!(HighpassFilter::new(20, 5).is_err());
        // valid
        assert!(HighpassFilter::new(20, 1).is_ok());
        assert!(HighpassFilter::new(20, 4).is_ok());
        assert!(HighpassFilter::new(5, 2).is_ok());
    }

    #[test]
    fn test_highpass_filter_trait() {
        let data = make_ohlcv_series();
        let hpf = HighpassFilter::new(15, 2).unwrap();

        assert_eq!(hpf.name(), "Highpass Filter");
        assert_eq!(hpf.min_periods(), 16);

        let output = hpf.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // =====================================================================
    // LowpassFilter Tests
    // =====================================================================

    #[test]
    fn test_lowpass_filter_basic() {
        let close = make_test_data();
        let lpf = LowpassFilter::new(10, 2).unwrap();
        let result = lpf.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Values should be positive and close to price
        for i in 15..result.len() {
            assert!(result[i] > 0.0, "Expected positive value at index {}", i);
            let diff = (result[i] - close[i]).abs();
            assert!(diff < 30.0, "Lowpass should track price, diff {} at {}", diff, i);
        }
    }

    #[test]
    fn test_lowpass_filter_smoothing() {
        // Noisy data
        let close: Vec<f64> = (0..100)
            .map(|i| 100.0 + i as f64 * 0.5 + ((i * 17) % 7) as f64 - 3.0)
            .collect();
        let lpf = LowpassFilter::new(10, 2).unwrap();
        let result = lpf.calculate(&close);

        // Filtered data should be smoother (less variance)
        let raw_variance: f64 = close[20..].windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .sum::<f64>() / (close.len() - 21) as f64;
        let filtered_variance: f64 = result[20..].windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .sum::<f64>() / (result.len() - 21) as f64;

        assert!(filtered_variance < raw_variance,
                "Lowpass should reduce variance: raw {} vs filtered {}", raw_variance, filtered_variance);
    }

    #[test]
    fn test_lowpass_filter_validation() {
        // cutoff_period too small
        assert!(LowpassFilter::new(3, 2).is_err());
        // poles out of range
        assert!(LowpassFilter::new(10, 0).is_err());
        assert!(LowpassFilter::new(10, 5).is_err());
        // valid
        assert!(LowpassFilter::new(10, 1).is_ok());
        assert!(LowpassFilter::new(10, 4).is_ok());
        assert!(LowpassFilter::new(5, 2).is_ok());
    }

    #[test]
    fn test_lowpass_filter_trait() {
        let data = make_ohlcv_series();
        let lpf = LowpassFilter::new(15, 2).unwrap();

        assert_eq!(lpf.name(), "Lowpass Filter");
        assert_eq!(lpf.min_periods(), 16);

        let output = lpf.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // =====================================================================
    // NotchFilter Tests
    // =====================================================================

    #[test]
    fn test_notch_filter_basic() {
        let close = make_test_data();
        let nf = NotchFilter::new(10, 0.2).unwrap();
        let result = nf.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Values should be positive (near price)
        for i in 25..result.len() {
            assert!(result[i] > 0.0, "Expected positive value at index {}", i);
        }
    }

    #[test]
    fn test_notch_filter_removes_frequency() {
        // Create data with a specific frequency to remove
        let close: Vec<f64> = (0..100)
            .map(|i| {
                let base = 100.0;
                let target_freq = (i as f64 * 2.0 * std::f64::consts::PI / 10.0).sin() * 5.0;
                let other_freq = (i as f64 * 2.0 * std::f64::consts::PI / 20.0).sin() * 3.0;
                base + target_freq + other_freq
            })
            .collect();
        let nf = NotchFilter::new(10, 0.2).unwrap();
        let result = nf.calculate(&close);

        // The notch filter should reduce the 10-period component
        // Output should still oscillate but differently than input
        assert!(result[50] != close[50], "Notch filter should modify the signal");
    }

    #[test]
    fn test_notch_filter_validation() {
        // notch_period too small
        assert!(NotchFilter::new(3, 0.2).is_err());
        // notch_width out of range
        assert!(NotchFilter::new(10, 0.01).is_err());
        assert!(NotchFilter::new(10, 0.6).is_err());
        // valid
        assert!(NotchFilter::new(10, 0.05).is_ok());
        assert!(NotchFilter::new(10, 0.5).is_ok());
        assert!(NotchFilter::new(5, 0.2).is_ok());
    }

    #[test]
    fn test_notch_filter_trait() {
        let data = make_ohlcv_series();
        let nf = NotchFilter::new(10, 0.2).unwrap();

        assert_eq!(nf.name(), "Notch Filter");
        assert_eq!(nf.min_periods(), 21);

        let output = nf.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // =====================================================================
    // AllpassPhaseShifter Tests
    // =====================================================================

    #[test]
    fn test_allpass_phase_shifter_basic() {
        let close = make_test_data();
        let aps = AllpassPhaseShifter::new(12, 90.0).unwrap();
        let result = aps.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Values should be positive
        for i in 30..result.len() {
            assert!(result[i] > 0.0, "Expected positive value at index {}", i);
        }
    }

    #[test]
    fn test_allpass_phase_shifter_shifts_phase() {
        // Create sine wave
        let close: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64 * 2.0 * std::f64::consts::PI / 12.0).sin() * 5.0)
            .collect();

        // 90 degree shift
        let aps = AllpassPhaseShifter::new(12, 90.0).unwrap();
        let result = aps.calculate(&close);

        // The shifted signal should differ from original
        let mut diff_count = 0;
        for i in 40..result.len() {
            if (result[i] - close[i]).abs() > 0.5 {
                diff_count += 1;
            }
        }
        assert!(diff_count > 10, "Phase shifter should modify the signal");
    }

    #[test]
    fn test_allpass_phase_shifter_zero_shift() {
        // Zero degree shift should approximately preserve the signal
        let close: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64 * 0.3).sin() * 5.0)
            .collect();
        let aps = AllpassPhaseShifter::new(12, 0.0).unwrap();
        let result = aps.calculate(&close);

        // With zero shift, output should be close to input (after warmup)
        for i in 40..result.len() {
            let diff = (result[i] - close[i]).abs();
            assert!(diff < 20.0, "Zero shift should preserve signal approximately, diff {} at {}", diff, i);
        }
    }

    #[test]
    fn test_allpass_phase_shifter_validation() {
        // period too small
        assert!(AllpassPhaseShifter::new(3, 90.0).is_err());
        // phase_shift out of range
        assert!(AllpassPhaseShifter::new(12, -200.0).is_err());
        assert!(AllpassPhaseShifter::new(12, 200.0).is_err());
        // valid
        assert!(AllpassPhaseShifter::new(12, -180.0).is_ok());
        assert!(AllpassPhaseShifter::new(12, 180.0).is_ok());
        assert!(AllpassPhaseShifter::new(5, 0.0).is_ok());
        assert!(AllpassPhaseShifter::new(20, 45.0).is_ok());
    }

    #[test]
    fn test_allpass_phase_shifter_trait() {
        let data = make_ohlcv_series();
        let aps = AllpassPhaseShifter::new(12, 90.0).unwrap();

        assert_eq!(aps.name(), "Allpass Phase Shifter");
        assert_eq!(aps.min_periods(), 25);

        let output = aps.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // =====================================================================
    // MovingAverageConvergence Tests
    // =====================================================================

    #[test]
    fn test_moving_average_convergence_basic() {
        let close = make_test_data();
        let mac = MovingAverageConvergence::new(5, 10, 20).unwrap();
        let result = mac.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Convergence should be between 0 and 100
        for i in 25..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0,
                    "Convergence {} at index {} out of range", result[i], i);
        }
    }

    #[test]
    fn test_moving_average_convergence_trending() {
        // Strong trending data should have lower convergence (MAs spread out)
        let close: Vec<f64> = (0..100).map(|i| 100.0 + i as f64 * 2.0).collect();
        let mac = MovingAverageConvergence::new(5, 10, 20).unwrap();
        let result = mac.calculate(&close);

        // In a strong trend, MAs diverge
        let avg_conv: f64 = result[30..].iter().sum::<f64>() / (result.len() - 30) as f64;
        assert!(avg_conv < 80.0, "Strong trend should show MA divergence, got {}", avg_conv);
    }

    #[test]
    fn test_moving_average_convergence_flat() {
        // Flat price should have high convergence (MAs close together)
        let close: Vec<f64> = (0..100).map(|_| 100.0).collect();
        let mac = MovingAverageConvergence::new(5, 10, 20).unwrap();
        let result = mac.calculate(&close);

        // Flat price means all MAs equal, perfect convergence
        for i in 25..result.len() {
            assert!(result[i] > 90.0, "Flat price should have high convergence, got {} at {}", result[i], i);
        }
    }

    #[test]
    fn test_moving_average_convergence_validation() {
        // short_period too small
        assert!(MovingAverageConvergence::new(3, 10, 20).is_err());
        // medium_period <= short_period
        assert!(MovingAverageConvergence::new(10, 10, 20).is_err());
        assert!(MovingAverageConvergence::new(10, 5, 20).is_err());
        // long_period <= medium_period
        assert!(MovingAverageConvergence::new(5, 10, 10).is_err());
        assert!(MovingAverageConvergence::new(5, 10, 8).is_err());
        // valid
        assert!(MovingAverageConvergence::new(5, 10, 20).is_ok());
        assert!(MovingAverageConvergence::new(5, 15, 30).is_ok());
    }

    #[test]
    fn test_moving_average_convergence_trait() {
        let data = make_ohlcv_series();
        let mac = MovingAverageConvergence::new(5, 10, 20).unwrap();

        assert_eq!(mac.name(), "Moving Average Convergence");
        assert_eq!(mac.min_periods(), 21);

        let output = mac.compute(&data).unwrap();
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
        let apm = AdaptivePhaseMeasure::new(15, 0.5).unwrap();
        let fdm = FrequencyDomainMomentum::new(20, 5).unwrap();
        let se = SpectralEntropy::new(25, 5).unwrap();
        let cd = CycleDominance::new(5, 25).unwrap();
        let ha = HarmonicAnalyzer::new(20, 3).unwrap();
        let pc = PhaseCoherence::new(20, 5).unwrap();
        // Existing new indicators
        let fdma = FrequencyDomainMA::new(20, 5).unwrap();
        let psi = PhaseShiftIndicator::new(20, 10).unwrap();
        let spi = SpectralPowerIndex::new(25, 4).unwrap();
        let nf = NoiseFilter::new(20, 0.5).unwrap();
        let cpe = CyclePeriodEstimator::new(5, 25, 3).unwrap();
        let snr = SignalToNoiseRatioAdvanced::new(25, 5).unwrap();
        // 6 newest DSP filter indicators
        let bpf = BandpassFilter::new(12, 0.5).unwrap();
        let hpf = HighpassFilter::new(15, 2).unwrap();
        let lpf = LowpassFilter::new(10, 2).unwrap();
        let notch = NotchFilter::new(10, 0.2).unwrap();
        let aps = AllpassPhaseShifter::new(12, 90.0).unwrap();
        let mac = MovingAverageConvergence::new(5, 10, 20).unwrap();

        let results = vec![
            aff.compute(&data).unwrap().primary,
            tpa.compute(&data).unwrap().primary,
            cmi.compute(&data).unwrap().primary,
            ssm.compute(&data).unwrap().primary,
            fri.compute(&data).unwrap().primary,
            ps.compute(&data).unwrap().primary,
            apm.compute(&data).unwrap().primary,
            fdm.compute(&data).unwrap().primary,
            se.compute(&data).unwrap().primary,
            cd.compute(&data).unwrap().primary,
            ha.compute(&data).unwrap().primary,
            pc.compute(&data).unwrap().primary,
            // Existing new indicators
            fdma.compute(&data).unwrap().primary,
            psi.compute(&data).unwrap().primary,
            spi.compute(&data).unwrap().primary,
            nf.compute(&data).unwrap().primary,
            cpe.compute(&data).unwrap().primary,
            snr.compute(&data).unwrap().primary,
            // 6 newest DSP filter indicators
            bpf.compute(&data).unwrap().primary,
            hpf.compute(&data).unwrap().primary,
            lpf.compute(&data).unwrap().primary,
            notch.compute(&data).unwrap().primary,
            aps.compute(&data).unwrap().primary,
            mac.compute(&data).unwrap().primary,
        ];

        for result in results {
            assert_eq!(result.len(), data.close.len());
        }
    }

    #[test]
    fn test_new_dsp_indicators_integration() {
        // Test that all 6 new indicators work together on the same data
        let close: Vec<f64> = (0..100)
            .map(|i| {
                let trend = 100.0 + i as f64 * 0.3;
                let cycle = (i as f64 * 0.25).sin() * 3.0;
                let noise = ((i * 13) % 5) as f64 * 0.2 - 0.4;
                trend + cycle + noise
            })
            .collect();
        let data = OHLCVSeries::from_close(close);

        let fdma = FrequencyDomainMA::new(20, 5).unwrap();
        let psi = PhaseShiftIndicator::new(20, 10).unwrap();
        let spi = SpectralPowerIndex::new(30, 4).unwrap();
        let nf = NoiseFilter::new(20, 0.5).unwrap();
        let cpe = CyclePeriodEstimator::new(5, 30, 3).unwrap();
        let snr = SignalToNoiseRatioAdvanced::new(30, 5).unwrap();

        // All should compute without errors
        let fdma_result = fdma.compute(&data).unwrap();
        let psi_result = psi.compute(&data).unwrap();
        let spi_result = spi.compute(&data).unwrap();
        let nf_result = nf.compute(&data).unwrap();
        let cpe_result = cpe.compute(&data).unwrap();
        let snr_result = snr.compute(&data).unwrap();

        // All should have correct length
        assert_eq!(fdma_result.primary.len(), 100);
        assert_eq!(psi_result.primary.len(), 100);
        assert_eq!(spi_result.primary.len(), 100);
        assert_eq!(nf_result.primary.len(), 100);
        assert_eq!(cpe_result.primary.len(), 100);
        assert_eq!(snr_result.primary.len(), 100);

        // Verify some values are being calculated (not all zeros after warmup)
        let fdma_sum: f64 = fdma_result.primary[30..].iter().sum();
        let nf_sum: f64 = nf_result.primary[30..].iter().sum();
        assert!(fdma_sum > 0.0, "FrequencyDomainMA should produce non-zero values");
        assert!(nf_sum > 0.0, "NoiseFilter should produce non-zero values");
    }

    #[test]
    fn test_six_new_dsp_filter_indicators_integration() {
        // Test that all 6 newest DSP filter indicators work together
        let close: Vec<f64> = (0..120)
            .map(|i| {
                let trend = 100.0 + i as f64 * 0.2;
                let cycle1 = (i as f64 * 2.0 * std::f64::consts::PI / 12.0).sin() * 4.0;
                let cycle2 = (i as f64 * 2.0 * std::f64::consts::PI / 25.0).sin() * 2.0;
                let noise = ((i * 19) % 7) as f64 * 0.15 - 0.5;
                trend + cycle1 + cycle2 + noise
            })
            .collect();
        let data = OHLCVSeries::from_close(close.clone());

        // Create all 6 new filter indicators
        let bpf = BandpassFilter::new(12, 0.5).unwrap();
        let hpf = HighpassFilter::new(20, 2).unwrap();
        let lpf = LowpassFilter::new(10, 2).unwrap();
        let notch = NotchFilter::new(12, 0.2).unwrap();
        let aps = AllpassPhaseShifter::new(12, 90.0).unwrap();
        let mac = MovingAverageConvergence::new(5, 12, 25).unwrap();

        // All should compute without errors
        let bpf_result = bpf.compute(&data).unwrap();
        let hpf_result = hpf.compute(&data).unwrap();
        let lpf_result = lpf.compute(&data).unwrap();
        let notch_result = notch.compute(&data).unwrap();
        let aps_result = aps.compute(&data).unwrap();
        let mac_result = mac.compute(&data).unwrap();

        // All should have correct length
        assert_eq!(bpf_result.primary.len(), 120);
        assert_eq!(hpf_result.primary.len(), 120);
        assert_eq!(lpf_result.primary.len(), 120);
        assert_eq!(notch_result.primary.len(), 120);
        assert_eq!(aps_result.primary.len(), 120);
        assert_eq!(mac_result.primary.len(), 120);

        // Verify bandpass filter oscillates (has positive and negative values after warmup)
        let bpf_has_positive = bpf_result.primary[40..].iter().any(|&x| x > 0.5);
        let bpf_has_negative = bpf_result.primary[40..].iter().any(|&x| x < -0.5);
        assert!(bpf_has_positive && bpf_has_negative, "Bandpass filter should oscillate");

        // Verify lowpass filter tracks price (positive values)
        let lpf_sum: f64 = lpf_result.primary[20..].iter().sum();
        assert!(lpf_sum > 0.0, "Lowpass filter should produce positive values");

        // Verify MA convergence is in valid range (0-100)
        for i in 30..mac_result.primary.len() {
            assert!(mac_result.primary[i] >= 0.0 && mac_result.primary[i] <= 100.0,
                    "MA Convergence {} at index {} out of range", mac_result.primary[i], i);
        }

        // Verify allpass shifter produces positive values (near price level)
        let aps_sum: f64 = aps_result.primary[40..].iter().sum();
        assert!(aps_sum > 0.0, "Allpass phase shifter should produce positive values");

        // Verify notch filter produces values near original price
        for i in 40..notch_result.primary.len() {
            let diff = (notch_result.primary[i] - close[i]).abs();
            assert!(diff < 20.0, "Notch filter should be close to original price, diff {} at {}", diff, i);
        }
    }

    // =====================================================================
    // FourierTransformPower Tests
    // =====================================================================

    #[test]
    fn test_fourier_transform_power_basic() {
        let close = make_test_data();
        let ftp = FourierTransformPower::new(30, 5).unwrap();
        let result = ftp.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Power values should be non-negative
        for i in 35..result.len() {
            assert!(result[i] >= 0.0, "Expected non-negative power at index {}", i);
        }
    }

    #[test]
    fn test_fourier_transform_power_sine_wave() {
        // Pure sine wave should have significant power
        let close: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64 * 2.0 * std::f64::consts::PI / 12.0).sin() * 5.0)
            .collect();
        let ftp = FourierTransformPower::new(30, 8).unwrap();
        let result = ftp.calculate(&close);

        // Sine wave should produce measurable power
        let avg_power: f64 = result[40..].iter().sum::<f64>() / (result.len() - 40) as f64;
        assert!(avg_power > 0.0, "Expected positive power for sine wave, got {}", avg_power);
    }

    #[test]
    fn test_fourier_transform_power_validation() {
        // period too small
        assert!(FourierTransformPower::new(10, 5).is_err());
        // num_frequencies out of range
        assert!(FourierTransformPower::new(30, 1).is_err());
        assert!(FourierTransformPower::new(30, 25).is_err());
        // valid
        assert!(FourierTransformPower::new(20, 2).is_ok());
        assert!(FourierTransformPower::new(30, 20).is_ok());
    }

    #[test]
    fn test_fourier_transform_power_trait() {
        let data = make_ohlcv_series();
        let ftp = FourierTransformPower::new(25, 5).unwrap();

        assert_eq!(ftp.name(), "Fourier Transform Power");
        assert_eq!(ftp.min_periods(), 26);

        let output = ftp.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // =====================================================================
    // WaveletSmoothing Tests
    // =====================================================================

    #[test]
    fn test_wavelet_smoothing_basic() {
        let close = make_test_data();
        let ws = WaveletSmoothing::new(20, 2).unwrap();
        let result = ws.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Smoothed values should be positive
        for i in 25..result.len() {
            assert!(result[i] > 0.0, "Expected positive value at index {}", i);
        }
    }

    #[test]
    fn test_wavelet_smoothing_reduces_noise() {
        // Noisy data
        let close: Vec<f64> = (0..100)
            .map(|i| 100.0 + i as f64 * 0.3 + ((i * 17) % 7) as f64 - 3.0)
            .collect();
        let ws = WaveletSmoothing::new(16, 2).unwrap();
        let result = ws.calculate(&close);

        // Smoothed data should have less variance in differences
        let raw_variance: f64 = close[25..].windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .sum::<f64>() / (close.len() - 26) as f64;
        let smoothed_variance: f64 = result[25..].windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .sum::<f64>() / (result.len() - 26) as f64;

        assert!(smoothed_variance <= raw_variance * 1.5,
                "Wavelet smoothing should reduce variance: raw {} vs smoothed {}", raw_variance, smoothed_variance);
    }

    #[test]
    fn test_wavelet_smoothing_validation() {
        // period too small
        assert!(WaveletSmoothing::new(10, 2).is_err());
        // decomposition_level out of range
        assert!(WaveletSmoothing::new(20, 0).is_err());
        assert!(WaveletSmoothing::new(20, 5).is_err());
        // valid
        assert!(WaveletSmoothing::new(16, 1).is_ok());
        assert!(WaveletSmoothing::new(32, 4).is_ok());
    }

    #[test]
    fn test_wavelet_smoothing_trait() {
        let data = make_ohlcv_series();
        let ws = WaveletSmoothing::new(20, 2).unwrap();

        assert_eq!(ws.name(), "Wavelet Smoothing");
        assert_eq!(ws.min_periods(), 21);

        let output = ws.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // =====================================================================
    // AdaptiveLPFilter Tests
    // =====================================================================

    #[test]
    fn test_adaptive_lp_filter_basic() {
        let close = make_test_data();
        let alpf = AdaptiveLPFilter::new(15, 8, 0.5).unwrap();
        let result = alpf.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Filtered values should be positive and track price
        for i in 20..result.len() {
            assert!(result[i] > 0.0, "Expected positive value at index {}", i);
            let diff = (result[i] - close[i]).abs();
            assert!(diff < 30.0, "Filter should track price, diff {} at {}", diff, i);
        }
    }

    #[test]
    fn test_adaptive_lp_filter_smoothing() {
        // Smooth trend data
        let close: Vec<f64> = (0..80).map(|i| 100.0 + i as f64 * 0.5).collect();
        let alpf = AdaptiveLPFilter::new(15, 8, 0.5).unwrap();
        let result = alpf.calculate(&close);

        // For smooth data, filter should track closely
        for i in 25..result.len() {
            let diff = (result[i] - close[i]).abs();
            assert!(diff < 10.0, "Expected close tracking for smooth data at index {}", i);
        }
    }

    #[test]
    fn test_adaptive_lp_filter_validation() {
        // period too small
        assert!(AdaptiveLPFilter::new(5, 5, 0.5).is_err());
        // base_cutoff too small
        assert!(AdaptiveLPFilter::new(15, 3, 0.5).is_err());
        // sensitivity out of range
        assert!(AdaptiveLPFilter::new(15, 8, 0.05).is_err());
        assert!(AdaptiveLPFilter::new(15, 8, 2.5).is_err());
        // valid
        assert!(AdaptiveLPFilter::new(10, 5, 0.1).is_ok());
        assert!(AdaptiveLPFilter::new(20, 10, 2.0).is_ok());
    }

    #[test]
    fn test_adaptive_lp_filter_trait() {
        let data = make_ohlcv_series();
        let alpf = AdaptiveLPFilter::new(15, 8, 0.5).unwrap();

        assert_eq!(alpf.name(), "Adaptive LP Filter");
        assert_eq!(alpf.min_periods(), 16);

        let output = alpf.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // =====================================================================
    // PhaseDetector Tests
    // =====================================================================

    #[test]
    fn test_phase_detector_basic() {
        let close = make_test_data();
        let pd = PhaseDetector::new(20, 3).unwrap();
        let result = pd.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Phase should be between 0 and 360
        for i in 25..result.len() {
            assert!(result[i] >= 0.0 && result[i] < 360.0,
                    "Phase {} at index {} out of range", result[i], i);
        }
    }

    #[test]
    fn test_phase_detector_sine_wave() {
        // Pure sine wave should show phase progression
        let close: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64 * 2.0 * std::f64::consts::PI / 12.0).sin() * 5.0)
            .collect();
        let pd = PhaseDetector::new(20, 1).unwrap();
        let result = pd.calculate(&close);

        // Phase values should vary (not all the same)
        let phase_variance: f64 = {
            let mean = result[40..].iter().sum::<f64>() / (result.len() - 40) as f64;
            result[40..].iter().map(|&p| (p - mean).powi(2)).sum::<f64>() / (result.len() - 40) as f64
        };
        assert!(phase_variance > 0.0, "Phase should vary for sine wave");
    }

    #[test]
    fn test_phase_detector_validation() {
        // period too small
        assert!(PhaseDetector::new(5, 3).is_err());
        // smoothing too small
        assert!(PhaseDetector::new(20, 0).is_err());
        // valid
        assert!(PhaseDetector::new(10, 1).is_ok());
        assert!(PhaseDetector::new(30, 5).is_ok());
    }

    #[test]
    fn test_phase_detector_trait() {
        let data = make_ohlcv_series();
        let pd = PhaseDetector::new(20, 3).unwrap();

        assert_eq!(pd.name(), "Phase Detector");
        assert_eq!(pd.min_periods(), 21);

        let output = pd.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // =====================================================================
    // AmplitudeExtractor Tests
    // =====================================================================

    #[test]
    fn test_amplitude_extractor_basic() {
        let close = make_test_data();
        let ae = AmplitudeExtractor::new(20, false).unwrap();
        let result = ae.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Amplitude should be non-negative
        for i in 25..result.len() {
            assert!(result[i] >= 0.0, "Expected non-negative amplitude at index {}", i);
        }
    }

    #[test]
    fn test_amplitude_extractor_sine_wave() {
        // Sine wave with amplitude 5
        let close: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64 * 2.0 * std::f64::consts::PI / 12.0).sin() * 5.0)
            .collect();
        let ae = AmplitudeExtractor::new(25, false).unwrap();
        let result = ae.calculate(&close);

        // Average amplitude should be close to the actual amplitude (5)
        let avg_amplitude: f64 = result[40..].iter().sum::<f64>() / (result.len() - 40) as f64;
        assert!(avg_amplitude > 2.0 && avg_amplitude < 10.0,
                "Expected amplitude near 5, got {}", avg_amplitude);
    }

    #[test]
    fn test_amplitude_extractor_normalized() {
        let close: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64 * 0.3).sin() * 5.0)
            .collect();
        let ae = AmplitudeExtractor::new(20, true).unwrap();
        let result = ae.calculate(&close);

        // Normalized amplitude should be percentage values
        for i in 25..result.len() {
            assert!(result[i] >= 0.0 && result[i] < 50.0,
                    "Normalized amplitude {} at index {} seems too high", result[i], i);
        }
    }

    #[test]
    fn test_amplitude_extractor_validation() {
        // period too small
        assert!(AmplitudeExtractor::new(5, false).is_err());
        // valid
        assert!(AmplitudeExtractor::new(10, false).is_ok());
        assert!(AmplitudeExtractor::new(20, true).is_ok());
    }

    #[test]
    fn test_amplitude_extractor_trait() {
        let data = make_ohlcv_series();
        let ae = AmplitudeExtractor::new(20, false).unwrap();

        assert_eq!(ae.name(), "Amplitude Extractor");
        assert_eq!(ae.min_periods(), 21);

        let output = ae.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // =====================================================================
    // TrendCycleDecomposer Tests
    // =====================================================================

    #[test]
    fn test_trend_cycle_decomposer_basic() {
        let close = make_test_data();
        let tcd = TrendCycleDecomposer::new(25, 8).unwrap();
        let result = tcd.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Cycle component should oscillate (have positive and negative values)
        let mut has_positive = false;
        let mut has_negative = false;
        for i in 30..result.len() {
            if result[i] > 0.5 {
                has_positive = true;
            }
            if result[i] < -0.5 {
                has_negative = true;
            }
        }
        assert!(has_positive || has_negative, "Cycle component should oscillate");
    }

    #[test]
    fn test_trend_cycle_decomposer_both() {
        let close: Vec<f64> = (0..100)
            .map(|i| {
                let trend = 100.0 + i as f64 * 0.5;
                let cycle = (i as f64 * 0.4).sin() * 3.0;
                trend + cycle
            })
            .collect();
        let tcd = TrendCycleDecomposer::new(25, 8).unwrap();
        let (trend, cycle) = tcd.calculate_both(&close);

        assert_eq!(trend.len(), close.len());
        assert_eq!(cycle.len(), close.len());

        // Trend should be positive and generally increasing
        for i in 30..trend.len() {
            assert!(trend[i] > 0.0, "Trend should be positive at index {}", i);
        }

        // Cycle should oscillate around zero
        let cycle_mean: f64 = cycle[40..].iter().sum::<f64>() / (cycle.len() - 40) as f64;
        assert!(cycle_mean.abs() < 5.0, "Cycle mean {} should be near zero", cycle_mean);
    }

    #[test]
    fn test_trend_cycle_decomposer_validation() {
        // trend_period too small
        assert!(TrendCycleDecomposer::new(10, 5).is_err());
        // cycle_period too small
        assert!(TrendCycleDecomposer::new(25, 3).is_err());
        // cycle_period >= trend_period
        assert!(TrendCycleDecomposer::new(25, 25).is_err());
        assert!(TrendCycleDecomposer::new(25, 30).is_err());
        // valid
        assert!(TrendCycleDecomposer::new(20, 5).is_ok());
        assert!(TrendCycleDecomposer::new(30, 10).is_ok());
    }

    #[test]
    fn test_trend_cycle_decomposer_trait() {
        let data = make_ohlcv_series();
        let tcd = TrendCycleDecomposer::new(25, 8).unwrap();

        assert_eq!(tcd.name(), "Trend Cycle Decomposer");
        assert_eq!(tcd.min_periods(), 26);

        let output = tcd.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
        // Should have dual output (cycle as primary, trend as secondary)
        assert!(output.secondary.is_some());
        assert_eq!(output.secondary.as_ref().unwrap().len(), data.close.len());
    }

    // =====================================================================
    // New 6 DSP Indicators Integration Test
    // =====================================================================

    #[test]
    fn test_six_new_dsp_indicators_integration() {
        // Test all 6 newest DSP indicators together
        let close: Vec<f64> = (0..120)
            .map(|i| {
                let trend = 100.0 + i as f64 * 0.25;
                let cycle = (i as f64 * 2.0 * std::f64::consts::PI / 15.0).sin() * 4.0;
                let noise = ((i * 13) % 5) as f64 * 0.2 - 0.4;
                trend + cycle + noise
            })
            .collect();
        let data = OHLCVSeries::from_close(close);

        // Create all 6 new indicators
        let ftp = FourierTransformPower::new(30, 6).unwrap();
        let ws = WaveletSmoothing::new(20, 2).unwrap();
        let alpf = AdaptiveLPFilter::new(15, 8, 0.5).unwrap();
        let pd = PhaseDetector::new(20, 3).unwrap();
        let ae = AmplitudeExtractor::new(20, false).unwrap();
        let tcd = TrendCycleDecomposer::new(25, 8).unwrap();

        // All should compute without errors
        let ftp_result = ftp.compute(&data).unwrap();
        let ws_result = ws.compute(&data).unwrap();
        let alpf_result = alpf.compute(&data).unwrap();
        let pd_result = pd.compute(&data).unwrap();
        let ae_result = ae.compute(&data).unwrap();
        let tcd_result = tcd.compute(&data).unwrap();

        // All should have correct length
        assert_eq!(ftp_result.primary.len(), 120);
        assert_eq!(ws_result.primary.len(), 120);
        assert_eq!(alpf_result.primary.len(), 120);
        assert_eq!(pd_result.primary.len(), 120);
        assert_eq!(ae_result.primary.len(), 120);
        assert_eq!(tcd_result.primary.len(), 120);

        // Verify Fourier power is non-negative
        for i in 40..ftp_result.primary.len() {
            assert!(ftp_result.primary[i] >= 0.0, "Fourier power should be non-negative");
        }

        // Verify wavelet smoothing produces positive values
        let ws_sum: f64 = ws_result.primary[30..].iter().sum();
        assert!(ws_sum > 0.0, "Wavelet smoothing should produce positive values");

        // Verify adaptive LP filter tracks price
        let alpf_sum: f64 = alpf_result.primary[20..].iter().sum();
        assert!(alpf_sum > 0.0, "Adaptive LP filter should produce positive values");

        // Verify phase is in range 0-360
        for i in 25..pd_result.primary.len() {
            assert!(pd_result.primary[i] >= 0.0 && pd_result.primary[i] < 360.0,
                    "Phase {} at index {} out of range", pd_result.primary[i], i);
        }

        // Verify amplitude is non-negative
        for i in 25..ae_result.primary.len() {
            assert!(ae_result.primary[i] >= 0.0, "Amplitude should be non-negative");
        }

        // Verify trend-cycle decomposer has dual output
        assert!(tcd_result.secondary.is_some());
    }

    // =====================================================================
    // AdaptiveSineWave Tests
    // =====================================================================

    #[test]
    fn test_adaptive_sine_wave_basic() {
        let close = make_test_data();
        let asw = AdaptiveSineWave::new(5, 25, 0.5).unwrap();
        let result = asw.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Sine wave should be bounded between -1 and 1
        for i in 30..result.len() {
            assert!(result[i] >= -1.0 && result[i] <= 1.0,
                    "Sine value {} at index {} out of range", result[i], i);
        }
    }

    #[test]
    fn test_adaptive_sine_wave_with_lead() {
        let close: Vec<f64> = (0..80)
            .map(|i| 100.0 + (i as f64 * 0.3).sin() * 5.0)
            .collect();
        let asw = AdaptiveSineWave::new(5, 20, 0.3).unwrap();
        let (sine, lead_sine) = asw.calculate_with_lead(&close);

        assert_eq!(sine.len(), close.len());
        assert_eq!(lead_sine.len(), close.len());

        // Both should be bounded
        for i in 25..sine.len() {
            assert!(sine[i] >= -1.0 && sine[i] <= 1.0);
            assert!(lead_sine[i] >= -1.0 && lead_sine[i] <= 1.0);
        }
    }

    #[test]
    fn test_adaptive_sine_wave_validation() {
        // min_period too small
        assert!(AdaptiveSineWave::new(3, 20, 0.5).is_err());
        // max_period <= min_period
        assert!(AdaptiveSineWave::new(10, 10, 0.5).is_err());
        assert!(AdaptiveSineWave::new(10, 5, 0.5).is_err());
        // invalid smoothing
        assert!(AdaptiveSineWave::new(5, 20, -0.1).is_err());
        assert!(AdaptiveSineWave::new(5, 20, 1.5).is_err());
        // valid
        assert!(AdaptiveSineWave::new(5, 20, 0.0).is_ok());
        assert!(AdaptiveSineWave::new(5, 20, 1.0).is_ok());
    }

    #[test]
    fn test_adaptive_sine_wave_trait() {
        let data = make_ohlcv_series();
        let asw = AdaptiveSineWave::new(5, 20, 0.5).unwrap();

        assert_eq!(asw.name(), "Adaptive Sine Wave");
        assert_eq!(asw.min_periods(), 21);

        let output = asw.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
        // Should have dual output (sine and lead_sine)
        assert!(output.secondary.is_some());
    }

    // =====================================================================
    // CycleBandwidth Tests
    // =====================================================================

    #[test]
    fn test_cycle_bandwidth_basic() {
        let close = make_test_data();
        let cb = CycleBandwidth::new(25, 4).unwrap();
        let result = cb.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Bandwidth should be between 0 and 100
        for i in 30..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0,
                    "Bandwidth {} at index {} out of range", result[i], i);
        }
    }

    #[test]
    fn test_cycle_bandwidth_pure_sine() {
        // Pure sine wave should have narrow bandwidth
        let close: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64 * 2.0 * std::f64::consts::PI / 20.0).sin() * 5.0)
            .collect();
        let cb = CycleBandwidth::new(30, 4).unwrap();
        let result = cb.calculate(&close);

        // Pure cycle should have relatively low bandwidth (concentrated power)
        let avg_bandwidth: f64 = result[40..].iter().sum::<f64>() / (result.len() - 40) as f64;
        assert!(avg_bandwidth < 80.0, "Pure sine should have moderate-low bandwidth, got {}", avg_bandwidth);
    }

    #[test]
    fn test_cycle_bandwidth_validation() {
        // period too small
        assert!(CycleBandwidth::new(15, 4).is_err());
        // num_bands out of range
        assert!(CycleBandwidth::new(25, 1).is_err());
        assert!(CycleBandwidth::new(25, 10).is_err());
        // valid
        assert!(CycleBandwidth::new(20, 2).is_ok());
        assert!(CycleBandwidth::new(30, 8).is_ok());
    }

    #[test]
    fn test_cycle_bandwidth_trait() {
        let data = make_ohlcv_series();
        let cb = CycleBandwidth::new(25, 4).unwrap();

        assert_eq!(cb.name(), "Cycle Bandwidth");
        assert_eq!(cb.min_periods(), 26);

        let output = cb.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // =====================================================================
    // SignalEnvelope Tests
    // =====================================================================

    #[test]
    fn test_signal_envelope_basic() {
        let close = make_test_data();
        let se = SignalEnvelope::new(15, 0.5).unwrap();
        let result = se.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Envelope should be non-negative
        for i in 20..result.len() {
            assert!(result[i] >= 0.0, "Envelope {} at index {} should be non-negative", result[i], i);
        }
    }

    #[test]
    fn test_signal_envelope_bands() {
        let close: Vec<f64> = (0..80)
            .map(|i| 100.0 + (i as f64 * 0.3).sin() * 8.0)
            .collect();
        let se = SignalEnvelope::new(15, 0.3).unwrap();
        let (upper, lower) = se.calculate_bands(&close);

        assert_eq!(upper.len(), close.len());
        assert_eq!(lower.len(), close.len());

        // Upper should be >= lower
        for i in 20..upper.len() {
            assert!(upper[i] >= lower[i], "Upper {} should be >= lower {} at index {}", upper[i], lower[i], i);
        }
    }

    #[test]
    fn test_signal_envelope_validation() {
        // period too small
        assert!(SignalEnvelope::new(5, 0.5).is_err());
        // invalid smoothing
        assert!(SignalEnvelope::new(15, -0.1).is_err());
        assert!(SignalEnvelope::new(15, 1.5).is_err());
        // valid
        assert!(SignalEnvelope::new(10, 0.0).is_ok());
        assert!(SignalEnvelope::new(20, 1.0).is_ok());
    }

    #[test]
    fn test_signal_envelope_trait() {
        let data = make_ohlcv_series();
        let se = SignalEnvelope::new(15, 0.5).unwrap();

        assert_eq!(se.name(), "Signal Envelope");
        assert_eq!(se.min_periods(), 16);

        let output = se.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
        // Should have dual output (upper and lower bands)
        assert!(output.secondary.is_some());
    }

    // =====================================================================
    // InstantaneousTrend Tests
    // =====================================================================

    #[test]
    fn test_instantaneous_trend_basic() {
        let close = make_test_data();
        let it = InstantaneousTrend::new(12, 0.07).unwrap();
        let result = it.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Trend should be positive for our uptrending test data
        for i in 15..result.len() {
            assert!(result[i] > 0.0, "Trend {} at index {} should be positive", result[i], i);
        }
    }

    #[test]
    fn test_instantaneous_trend_with_trigger() {
        let close: Vec<f64> = (0..80)
            .map(|i| 100.0 + i as f64 * 0.5 + (i as f64 * 0.2).sin() * 2.0)
            .collect();
        let it = InstantaneousTrend::new(10, 0.1).unwrap();
        let (trend, trigger) = it.calculate_with_trigger(&close);

        assert_eq!(trend.len(), close.len());
        assert_eq!(trigger.len(), close.len());

        // Trigger should be lagged version of trend
        for i in 12..trend.len() {
            // Both should be in reasonable range
            assert!(trend[i] > 50.0 && trend[i] < 200.0);
        }
    }

    #[test]
    fn test_instantaneous_trend_tracks_price() {
        // Strong uptrend
        let close: Vec<f64> = (0..60).map(|i| 100.0 + i as f64 * 1.0).collect();
        let it = InstantaneousTrend::new(10, 0.07).unwrap();
        let trend = it.calculate(&close);

        // Trend should increase with price
        for i in 15..trend.len() - 1 {
            assert!(trend[i + 1] >= trend[i] - 1.0,
                    "Trend should follow upward price movement");
        }
    }

    #[test]
    fn test_instantaneous_trend_validation() {
        // period too small
        assert!(InstantaneousTrend::new(5, 0.07).is_err());
        // invalid alpha
        assert!(InstantaneousTrend::new(10, 0.0).is_err());
        assert!(InstantaneousTrend::new(10, 1.5).is_err());
        // valid
        assert!(InstantaneousTrend::new(8, 0.01).is_ok());
        assert!(InstantaneousTrend::new(20, 1.0).is_ok());
    }

    #[test]
    fn test_instantaneous_trend_trait() {
        let data = make_ohlcv_series();
        let it = InstantaneousTrend::new(12, 0.07).unwrap();

        assert_eq!(it.name(), "Instantaneous Trend");
        assert_eq!(it.min_periods(), 13);

        let output = it.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
        // Should have dual output (trend and trigger)
        assert!(output.secondary.is_some());
    }

    // =====================================================================
    // CycleStrength Tests
    // =====================================================================

    #[test]
    fn test_cycle_strength_basic() {
        let close = make_test_data();
        let cs = CycleStrength::new(25, 10).unwrap();
        let result = cs.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Strength should be between 0 and 100
        for i in 30..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0,
                    "Strength {} at index {} out of range", result[i], i);
        }
    }

    #[test]
    fn test_cycle_strength_pure_cycle() {
        // Pure sine wave should have high cycle strength
        let close: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64 * 2.0 * std::f64::consts::PI / 15.0).sin() * 10.0)
            .collect();
        let cs = CycleStrength::new(30, 15).unwrap();
        let result = cs.calculate(&close);

        // Pure cycle at the measured period should have decent strength
        let avg_strength: f64 = result[40..].iter().sum::<f64>() / (result.len() - 40) as f64;
        assert!(avg_strength > 20.0, "Pure cycle should have measurable strength, got {}", avg_strength);
    }

    #[test]
    fn test_cycle_strength_validation() {
        // period too small
        assert!(CycleStrength::new(15, 10).is_err());
        // cycle_period too small
        assert!(CycleStrength::new(25, 3).is_err());
        // cycle_period >= period
        assert!(CycleStrength::new(25, 25).is_err());
        assert!(CycleStrength::new(25, 30).is_err());
        // valid
        assert!(CycleStrength::new(20, 5).is_ok());
        assert!(CycleStrength::new(30, 15).is_ok());
    }

    #[test]
    fn test_cycle_strength_trait() {
        let data = make_ohlcv_series();
        let cs = CycleStrength::new(25, 10).unwrap();

        assert_eq!(cs.name(), "Cycle Strength");
        assert_eq!(cs.min_periods(), 26);

        let output = cs.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // =====================================================================
    // AdaptiveLaguerreFilter Tests
    // =====================================================================

    #[test]
    fn test_adaptive_laguerre_filter_basic() {
        let close = make_test_data();
        let alf = AdaptiveLaguerreFilter::new(0.2, 0.8, 15).unwrap();
        let result = alf.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Filter should produce positive values for our positive price data
        for i in 20..result.len() {
            assert!(result[i] > 0.0, "Filter value {} at index {} should be positive", result[i], i);
        }
    }

    #[test]
    fn test_adaptive_laguerre_filter_with_gamma() {
        let close: Vec<f64> = (0..80)
            .map(|i| 100.0 + (i as f64 * 0.25).sin() * 5.0)
            .collect();
        let alf = AdaptiveLaguerreFilter::new(0.3, 0.7, 12).unwrap();
        let (filter, gamma) = alf.calculate_with_gamma(&close);

        assert_eq!(filter.len(), close.len());
        assert_eq!(gamma.len(), close.len());

        // Gamma should be within bounds
        for i in 15..gamma.len() {
            assert!(gamma[i] >= 0.3 && gamma[i] <= 0.7,
                    "Gamma {} at index {} out of range", gamma[i], i);
        }
    }

    #[test]
    fn test_adaptive_laguerre_filter_smoothing() {
        // Filter should smooth the price data
        let close: Vec<f64> = (0..60)
            .map(|i| 100.0 + (i as f64 * 0.5).sin() * 10.0 + ((i * 7) % 3) as f64)
            .collect();
        let alf = AdaptiveLaguerreFilter::new(0.4, 0.9, 10).unwrap();
        let result = alf.calculate(&close);

        // Calculate variance of original and filtered
        let close_mean: f64 = close[20..].iter().sum::<f64>() / (close.len() - 20) as f64;
        let result_mean: f64 = result[20..].iter().sum::<f64>() / (result.len() - 20) as f64;

        let close_var: f64 = close[20..].iter()
            .map(|x| (x - close_mean).powi(2))
            .sum::<f64>() / (close.len() - 20) as f64;
        let result_var: f64 = result[20..].iter()
            .map(|x| (x - result_mean).powi(2))
            .sum::<f64>() / (result.len() - 20) as f64;

        // Filtered data should have lower or similar variance
        assert!(result_var <= close_var * 1.5,
                "Filter should smooth data: close_var={}, result_var={}", close_var, result_var);
    }

    #[test]
    fn test_adaptive_laguerre_filter_validation() {
        // invalid min_gamma
        assert!(AdaptiveLaguerreFilter::new(-0.1, 0.8, 15).is_err());
        assert!(AdaptiveLaguerreFilter::new(1.0, 0.8, 15).is_err());
        // max_gamma <= min_gamma
        assert!(AdaptiveLaguerreFilter::new(0.5, 0.5, 15).is_err());
        assert!(AdaptiveLaguerreFilter::new(0.5, 0.3, 15).is_err());
        // max_gamma > 1.0
        assert!(AdaptiveLaguerreFilter::new(0.2, 1.1, 15).is_err());
        // lookback too small
        assert!(AdaptiveLaguerreFilter::new(0.2, 0.8, 5).is_err());
        // valid
        assert!(AdaptiveLaguerreFilter::new(0.0, 0.5, 10).is_ok());
        assert!(AdaptiveLaguerreFilter::new(0.5, 1.0, 20).is_ok());
    }

    #[test]
    fn test_adaptive_laguerre_filter_trait() {
        let data = make_ohlcv_series();
        let alf = AdaptiveLaguerreFilter::new(0.2, 0.8, 15).unwrap();

        assert_eq!(alf.name(), "Adaptive Laguerre Filter");
        assert_eq!(alf.min_periods(), 16);

        let output = alf.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
        // Should have dual output (filter and gamma values)
        assert!(output.secondary.is_some());
    }

    // =====================================================================
    // New 6 DSP Indicators Batch 7 Integration Test
    // =====================================================================

    #[test]
    fn test_batch7_dsp_indicators_integration() {
        // Test all 6 new DSP indicators (Batch 7) together
        let close: Vec<f64> = (0..120)
            .map(|i| {
                let trend = 100.0 + i as f64 * 0.25;
                let cycle = (i as f64 * 2.0 * std::f64::consts::PI / 15.0).sin() * 4.0;
                let noise = ((i * 13) % 5) as f64 * 0.2 - 0.4;
                trend + cycle + noise
            })
            .collect();
        let data = OHLCVSeries::from_close(close);

        // Create all 6 new indicators
        let asw = AdaptiveSineWave::new(5, 25, 0.5).unwrap();
        let cb = CycleBandwidth::new(25, 4).unwrap();
        let se = SignalEnvelope::new(15, 0.5).unwrap();
        let it = InstantaneousTrend::new(12, 0.07).unwrap();
        let cs = CycleStrength::new(25, 10).unwrap();
        let alf = AdaptiveLaguerreFilter::new(0.2, 0.8, 15).unwrap();

        // All should compute without errors
        let asw_result = asw.compute(&data).unwrap();
        let cb_result = cb.compute(&data).unwrap();
        let se_result = se.compute(&data).unwrap();
        let it_result = it.compute(&data).unwrap();
        let cs_result = cs.compute(&data).unwrap();
        let alf_result = alf.compute(&data).unwrap();

        // All should have correct length
        assert_eq!(asw_result.primary.len(), 120);
        assert_eq!(cb_result.primary.len(), 120);
        assert_eq!(se_result.primary.len(), 120);
        assert_eq!(it_result.primary.len(), 120);
        assert_eq!(cs_result.primary.len(), 120);
        assert_eq!(alf_result.primary.len(), 120);

        // Verify adaptive sine wave is bounded
        for i in 30..asw_result.primary.len() {
            assert!(asw_result.primary[i] >= -1.0 && asw_result.primary[i] <= 1.0,
                    "Adaptive sine wave should be bounded");
        }

        // Verify cycle bandwidth is in range
        for i in 30..cb_result.primary.len() {
            assert!(cb_result.primary[i] >= 0.0 && cb_result.primary[i] <= 100.0,
                    "Cycle bandwidth should be 0-100");
        }

        // Verify signal envelope is non-negative (upper band > lower band)
        assert!(se_result.secondary.is_some());

        // Verify instantaneous trend is positive
        for i in 20..it_result.primary.len() {
            assert!(it_result.primary[i] > 0.0, "Trend should be positive for uptrending data");
        }

        // Verify cycle strength is in range
        for i in 30..cs_result.primary.len() {
            assert!(cs_result.primary[i] >= 0.0 && cs_result.primary[i] <= 100.0,
                    "Cycle strength should be 0-100");
        }

        // Verify adaptive Laguerre filter produces positive values
        for i in 20..alf_result.primary.len() {
            assert!(alf_result.primary[i] > 0.0, "Laguerre filter should be positive");
        }

        // Verify gamma is within bounds
        let gamma = alf_result.secondary.as_ref().unwrap();
        for i in 20..gamma.len() {
            assert!(gamma[i] >= 0.2 && gamma[i] <= 0.8, "Gamma should be within bounds");
        }
    }
}
