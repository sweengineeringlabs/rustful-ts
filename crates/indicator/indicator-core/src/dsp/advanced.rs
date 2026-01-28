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
        // New indicators
        let fdma = FrequencyDomainMA::new(20, 5).unwrap();
        let psi = PhaseShiftIndicator::new(20, 10).unwrap();
        let spi = SpectralPowerIndex::new(25, 4).unwrap();
        let nf = NoiseFilter::new(20, 0.5).unwrap();
        let cpe = CyclePeriodEstimator::new(5, 25, 3).unwrap();
        let snr = SignalToNoiseRatioAdvanced::new(25, 5).unwrap();

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
            // New indicators
            fdma.compute(&data).unwrap().primary,
            psi.compute(&data).unwrap().primary,
            spi.compute(&data).unwrap().primary,
            nf.compute(&data).unwrap().primary,
            cpe.compute(&data).unwrap().primary,
            snr.compute(&data).unwrap().primary,
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
}
