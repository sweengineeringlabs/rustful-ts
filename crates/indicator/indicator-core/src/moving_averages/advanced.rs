//! Advanced Adaptive Moving Average Indicators
//!
//! This module contains sophisticated adaptive moving averages that adjust their
//! smoothing factors based on various market conditions including fractal dimension,
//! volume, trend strength, noise levels, momentum, and efficiency.

use indicator_spi::{IndicatorError, IndicatorOutput, OHLCVSeries, Result, TechnicalIndicator};

// ============================================================================
// Fractal Adaptive Moving Average
// ============================================================================

/// Fractal Adaptive Moving Average (FractalAdaptiveMA)
///
/// Uses fractal dimension to dynamically adapt the smoothing factor.
/// When the market is trending (low fractal dimension), it responds quickly.
/// When the market is choppy (high fractal dimension), it applies more smoothing.
#[derive(Debug, Clone)]
pub struct FractalAdaptiveMA {
    period: usize,
    fast_alpha: f64,
    slow_alpha: f64,
}

impl FractalAdaptiveMA {
    /// Create a new FractalAdaptiveMA with the given period.
    ///
    /// # Arguments
    /// * `period` - The lookback period (must be at least 4 and even)
    /// * `fast_alpha` - Fast smoothing factor (0.0 to 1.0)
    /// * `slow_alpha` - Slow smoothing factor (0.0 to 1.0, must be less than fast_alpha)
    pub fn new(period: usize, fast_alpha: f64, slow_alpha: f64) -> Result<Self> {
        if period < 4 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 4".to_string(),
            });
        }
        if period % 2 != 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be an even number".to_string(),
            });
        }
        if fast_alpha <= 0.0 || fast_alpha > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "fast_alpha".to_string(),
                reason: "must be between 0.0 (exclusive) and 1.0 (inclusive)".to_string(),
            });
        }
        if slow_alpha <= 0.0 || slow_alpha >= fast_alpha {
            return Err(IndicatorError::InvalidParameter {
                name: "slow_alpha".to_string(),
                reason: "must be between 0.0 (exclusive) and fast_alpha (exclusive)".to_string(),
            });
        }
        Ok(Self {
            period,
            fast_alpha,
            slow_alpha,
        })
    }

    /// Calculate the fractal adaptive moving average values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period {
            return vec![f64::NAN; n];
        }

        let half = self.period / 2;
        let mut result = vec![f64::NAN; n];

        // Initialize with the first valid price
        result[self.period - 1] = data[self.period - 1];
        let mut fama = data[self.period - 1];

        for i in self.period..n {
            let window = &data[(i + 1 - self.period)..=i];
            let alpha = self.calculate_fractal_alpha(window, half);

            fama = alpha * data[i] + (1.0 - alpha) * fama;
            result[i] = fama;
        }

        result
    }

    /// Calculate the adaptive alpha based on fractal dimension.
    fn calculate_fractal_alpha(&self, window: &[f64], half: usize) -> f64 {
        let n = window.len();
        if n < 2 || half == 0 {
            return self.slow_alpha;
        }

        // N1: range of first half
        let first_half = &window[0..half];
        let n1 = self.range(first_half);

        // N2: range of second half
        let second_half = &window[half..];
        let n2 = self.range(second_half);

        // N3: range of full period
        let n3 = self.range(window);

        if n3 < 1e-10 || (n1 + n2) < 1e-10 {
            return self.slow_alpha;
        }

        // Calculate fractal dimension D
        let d = ((n1 + n2) / n3).ln() / 2.0_f64.ln();

        // Map D to alpha: D near 1 = trending (fast), D near 2 = choppy (slow)
        let alpha = self.fast_alpha - (d - 1.0) * (self.fast_alpha - self.slow_alpha);
        alpha.clamp(self.slow_alpha, self.fast_alpha)
    }

    fn range(&self, data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        max - min
    }
}

impl TechnicalIndicator for FractalAdaptiveMA {
    fn name(&self) -> &str {
        "Fractal Adaptive MA"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

// ============================================================================
// Volume Adaptive Moving Average
// ============================================================================

/// Volume Adaptive Moving Average (VolumeAdaptiveMA)
///
/// Adapts the smoothing factor based on relative volume changes.
/// Higher volume leads to faster response, lower volume leads to more smoothing.
#[derive(Debug, Clone)]
pub struct VolumeAdaptiveMA {
    period: usize,
    volume_period: usize,
    fast_alpha: f64,
    slow_alpha: f64,
}

impl VolumeAdaptiveMA {
    /// Create a new VolumeAdaptiveMA.
    ///
    /// # Arguments
    /// * `period` - The base MA period (must be at least 2)
    /// * `volume_period` - Period for volume averaging (must be at least 2)
    /// * `fast_alpha` - Fast smoothing factor (0.0 to 1.0)
    /// * `slow_alpha` - Slow smoothing factor (0.0 to 1.0, must be less than fast_alpha)
    pub fn new(
        period: usize,
        volume_period: usize,
        fast_alpha: f64,
        slow_alpha: f64,
    ) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if volume_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "volume_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if fast_alpha <= 0.0 || fast_alpha > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "fast_alpha".to_string(),
                reason: "must be between 0.0 (exclusive) and 1.0 (inclusive)".to_string(),
            });
        }
        if slow_alpha <= 0.0 || slow_alpha >= fast_alpha {
            return Err(IndicatorError::InvalidParameter {
                name: "slow_alpha".to_string(),
                reason: "must be between 0.0 (exclusive) and fast_alpha (exclusive)".to_string(),
            });
        }
        Ok(Self {
            period,
            volume_period,
            fast_alpha,
            slow_alpha,
        })
    }

    /// Calculate the volume adaptive moving average values.
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        let min_required = self.period.max(self.volume_period);

        if n < min_required || volume.len() < n {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; n];

        // Initialize
        result[min_required - 1] = close[min_required - 1];
        let mut vama = close[min_required - 1];

        for i in min_required..n {
            // Calculate average volume over volume_period
            let start = i + 1 - self.volume_period;
            let avg_volume: f64 = volume[start..=i].iter().sum::<f64>() / self.volume_period as f64;

            // Calculate volume ratio
            let volume_ratio = if avg_volume > 1e-10 {
                (volume[i] / avg_volume).min(3.0)
            } else {
                1.0
            };

            // Map volume ratio to alpha: high volume = fast, low volume = slow
            let alpha = if volume_ratio >= 1.0 {
                // Above average volume: interpolate toward fast
                let t = ((volume_ratio - 1.0) / 2.0).min(1.0);
                self.slow_alpha + t * (self.fast_alpha - self.slow_alpha)
            } else {
                // Below average volume: use slow alpha
                self.slow_alpha * volume_ratio
            };

            let alpha = alpha.clamp(self.slow_alpha * 0.5, self.fast_alpha);

            vama = alpha * close[i] + (1.0 - alpha) * vama;
            result[i] = vama;
        }

        result
    }
}

impl TechnicalIndicator for VolumeAdaptiveMA {
    fn name(&self) -> &str {
        "Volume Adaptive MA"
    }

    fn min_periods(&self) -> usize {
        self.period.max(self.volume_period)
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.min_periods();
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }
        Ok(IndicatorOutput::single(
            self.calculate(&data.close, &data.volume),
        ))
    }
}

// ============================================================================
// Trend Adaptive Moving Average
// ============================================================================

/// Trend Adaptive Moving Average (TrendAdaptiveMA)
///
/// Adjusts smoothing based on trend strength measured by ADX-like calculation.
/// Strong trends lead to faster response, weak trends lead to more smoothing.
#[derive(Debug, Clone)]
pub struct TrendAdaptiveMA {
    period: usize,
    adx_period: usize,
    fast_alpha: f64,
    slow_alpha: f64,
}

impl TrendAdaptiveMA {
    /// Create a new TrendAdaptiveMA.
    ///
    /// # Arguments
    /// * `period` - The base MA period (must be at least 2)
    /// * `adx_period` - Period for trend strength calculation (must be at least 2)
    /// * `fast_alpha` - Fast smoothing factor (0.0 to 1.0)
    /// * `slow_alpha` - Slow smoothing factor (0.0 to 1.0, must be less than fast_alpha)
    pub fn new(period: usize, adx_period: usize, fast_alpha: f64, slow_alpha: f64) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if adx_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "adx_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if fast_alpha <= 0.0 || fast_alpha > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "fast_alpha".to_string(),
                reason: "must be between 0.0 (exclusive) and 1.0 (inclusive)".to_string(),
            });
        }
        if slow_alpha <= 0.0 || slow_alpha >= fast_alpha {
            return Err(IndicatorError::InvalidParameter {
                name: "slow_alpha".to_string(),
                reason: "must be between 0.0 (exclusive) and fast_alpha (exclusive)".to_string(),
            });
        }
        Ok(Self {
            period,
            adx_period,
            fast_alpha,
            slow_alpha,
        })
    }

    /// Calculate the trend adaptive moving average values.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let min_required = self.period.max(self.adx_period + 1);

        if n < min_required || high.len() < n || low.len() < n {
            return vec![f64::NAN; n];
        }

        // Calculate trend strength (simplified ADX-like measure)
        let trend_strength = self.calculate_trend_strength(high, low, close);

        let mut result = vec![f64::NAN; n];
        result[min_required - 1] = close[min_required - 1];
        let mut tama = close[min_required - 1];

        for i in min_required..n {
            // Normalize trend strength to 0-1 range (ADX typically 0-100)
            let normalized_trend = (trend_strength[i] / 100.0).clamp(0.0, 1.0);

            // Strong trend = fast alpha, weak trend = slow alpha
            let alpha = self.slow_alpha + normalized_trend * (self.fast_alpha - self.slow_alpha);

            tama = alpha * close[i] + (1.0 - alpha) * tama;
            result[i] = tama;
        }

        result
    }

    /// Calculate trend strength using a simplified ADX-like approach.
    fn calculate_trend_strength(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < 2 {
            return result;
        }

        // Calculate True Range and Directional Movement
        let mut tr = vec![0.0; n];
        let mut plus_dm = vec![0.0; n];
        let mut minus_dm = vec![0.0; n];

        for i in 1..n {
            let high_low = high[i] - low[i];
            let high_close = (high[i] - close[i - 1]).abs();
            let low_close = (low[i] - close[i - 1]).abs();
            tr[i] = high_low.max(high_close).max(low_close);

            let up_move = high[i] - high[i - 1];
            let down_move = low[i - 1] - low[i];

            if up_move > down_move && up_move > 0.0 {
                plus_dm[i] = up_move;
            }
            if down_move > up_move && down_move > 0.0 {
                minus_dm[i] = down_move;
            }
        }

        // Smooth TR and DM
        let alpha = 1.0 / self.adx_period as f64;
        let mut smoothed_tr = 0.0;
        let mut smoothed_plus_dm = 0.0;
        let mut smoothed_minus_dm = 0.0;
        let mut dx_values = Vec::new();

        for i in 1..n {
            smoothed_tr = smoothed_tr - (smoothed_tr * alpha) + tr[i];
            smoothed_plus_dm = smoothed_plus_dm - (smoothed_plus_dm * alpha) + plus_dm[i];
            smoothed_minus_dm = smoothed_minus_dm - (smoothed_minus_dm * alpha) + minus_dm[i];

            if i >= self.adx_period {
                let plus_di = if smoothed_tr > 1e-10 {
                    100.0 * smoothed_plus_dm / smoothed_tr
                } else {
                    0.0
                };
                let minus_di = if smoothed_tr > 1e-10 {
                    100.0 * smoothed_minus_dm / smoothed_tr
                } else {
                    0.0
                };

                let di_sum = plus_di + minus_di;
                let dx = if di_sum > 1e-10 {
                    100.0 * (plus_di - minus_di).abs() / di_sum
                } else {
                    0.0
                };

                dx_values.push(dx);

                // Calculate ADX as smoothed DX
                if dx_values.len() >= self.adx_period {
                    let adx: f64 = dx_values[dx_values.len() - self.adx_period..]
                        .iter()
                        .sum::<f64>()
                        / self.adx_period as f64;
                    result[i] = adx;
                }
            }
        }

        result
    }
}

impl TechnicalIndicator for TrendAdaptiveMA {
    fn name(&self) -> &str {
        "Trend Adaptive MA"
    }

    fn min_periods(&self) -> usize {
        self.period.max(self.adx_period * 2)
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.min_periods();
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }
        Ok(IndicatorOutput::single(self.calculate(
            &data.high,
            &data.low,
            &data.close,
        )))
    }
}

// ============================================================================
// Noise Adaptive Moving Average
// ============================================================================

/// Noise Adaptive Moving Average (NoiseAdaptiveMA)
///
/// Adapts smoothing based on noise levels in price data.
/// Higher noise leads to more smoothing to filter out randomness.
#[derive(Debug, Clone)]
pub struct NoiseAdaptiveMA {
    period: usize,
    noise_period: usize,
    fast_alpha: f64,
    slow_alpha: f64,
}

impl NoiseAdaptiveMA {
    /// Create a new NoiseAdaptiveMA.
    ///
    /// # Arguments
    /// * `period` - The base MA period (must be at least 2)
    /// * `noise_period` - Period for noise calculation (must be at least 2)
    /// * `fast_alpha` - Fast smoothing factor (0.0 to 1.0)
    /// * `slow_alpha` - Slow smoothing factor (0.0 to 1.0, must be less than fast_alpha)
    pub fn new(
        period: usize,
        noise_period: usize,
        fast_alpha: f64,
        slow_alpha: f64,
    ) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if noise_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "noise_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if fast_alpha <= 0.0 || fast_alpha > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "fast_alpha".to_string(),
                reason: "must be between 0.0 (exclusive) and 1.0 (inclusive)".to_string(),
            });
        }
        if slow_alpha <= 0.0 || slow_alpha >= fast_alpha {
            return Err(IndicatorError::InvalidParameter {
                name: "slow_alpha".to_string(),
                reason: "must be between 0.0 (exclusive) and fast_alpha (exclusive)".to_string(),
            });
        }
        Ok(Self {
            period,
            noise_period,
            fast_alpha,
            slow_alpha,
        })
    }

    /// Calculate the noise adaptive moving average values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let min_required = self.period.max(self.noise_period);

        if n < min_required {
            return vec![f64::NAN; n];
        }

        // Calculate noise levels
        let noise_levels = self.calculate_noise(data);

        let mut result = vec![f64::NAN; n];
        result[min_required - 1] = data[min_required - 1];
        let mut nama = data[min_required - 1];

        for i in min_required..n {
            // Inverse relationship: high noise = slow alpha, low noise = fast alpha
            let noise_factor = (1.0 - noise_levels[i]).clamp(0.0, 1.0);
            let alpha = self.slow_alpha + noise_factor * (self.fast_alpha - self.slow_alpha);

            nama = alpha * data[i] + (1.0 - alpha) * nama;
            result[i] = nama;
        }

        result
    }

    /// Calculate noise level using standard deviation of price changes relative to price range.
    fn calculate_noise(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![0.5; n]; // Default to medium noise

        if n < self.noise_period {
            return result;
        }

        for i in self.noise_period..n {
            let start = i + 1 - self.noise_period;
            let window = &data[start..=i];

            // Calculate price changes
            let changes: Vec<f64> = (1..window.len())
                .map(|j| (window[j] - window[j - 1]).abs())
                .collect();

            if changes.is_empty() {
                continue;
            }

            // Mean and standard deviation of changes
            let mean_change: f64 = changes.iter().sum::<f64>() / changes.len() as f64;
            let variance: f64 = changes
                .iter()
                .map(|c| (c - mean_change).powi(2))
                .sum::<f64>()
                / changes.len() as f64;
            let std_dev = variance.sqrt();

            // Calculate overall direction
            let net_change = (window.last().unwrap() - window.first().unwrap()).abs();
            let total_movement: f64 = changes.iter().sum();

            // Noise ratio: how much movement was "wasted" (not contributing to direction)
            if total_movement > 1e-10 {
                let efficiency = net_change / total_movement;
                // Combine efficiency with volatility clustering
                let noise =
                    (1.0 - efficiency) * (1.0 + std_dev / mean_change.max(1e-10)).min(2.0) / 2.0;
                result[i] = noise.clamp(0.0, 1.0);
            }
        }

        result
    }
}

impl TechnicalIndicator for NoiseAdaptiveMA {
    fn name(&self) -> &str {
        "Noise Adaptive MA"
    }

    fn min_periods(&self) -> usize {
        self.period.max(self.noise_period)
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.min_periods();
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

// ============================================================================
// Momentum Adaptive Moving Average
// ============================================================================

/// Momentum Adaptive Moving Average (MomentumAdaptiveMA)
///
/// Uses price momentum to adjust the smoothing factor.
/// Strong momentum leads to faster response, weak momentum leads to more smoothing.
#[derive(Debug, Clone)]
pub struct MomentumAdaptiveMA {
    period: usize,
    momentum_period: usize,
    fast_alpha: f64,
    slow_alpha: f64,
}

impl MomentumAdaptiveMA {
    /// Create a new MomentumAdaptiveMA.
    ///
    /// # Arguments
    /// * `period` - The base MA period (must be at least 2)
    /// * `momentum_period` - Period for momentum calculation (must be at least 2)
    /// * `fast_alpha` - Fast smoothing factor (0.0 to 1.0)
    /// * `slow_alpha` - Slow smoothing factor (0.0 to 1.0, must be less than fast_alpha)
    pub fn new(
        period: usize,
        momentum_period: usize,
        fast_alpha: f64,
        slow_alpha: f64,
    ) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if momentum_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if fast_alpha <= 0.0 || fast_alpha > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "fast_alpha".to_string(),
                reason: "must be between 0.0 (exclusive) and 1.0 (inclusive)".to_string(),
            });
        }
        if slow_alpha <= 0.0 || slow_alpha >= fast_alpha {
            return Err(IndicatorError::InvalidParameter {
                name: "slow_alpha".to_string(),
                reason: "must be between 0.0 (exclusive) and fast_alpha (exclusive)".to_string(),
            });
        }
        Ok(Self {
            period,
            momentum_period,
            fast_alpha,
            slow_alpha,
        })
    }

    /// Calculate the momentum adaptive moving average values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let min_required = self.period.max(self.momentum_period);

        if n < min_required {
            return vec![f64::NAN; n];
        }

        // Calculate normalized momentum
        let momentum = self.calculate_normalized_momentum(data);

        let mut result = vec![f64::NAN; n];
        result[min_required - 1] = data[min_required - 1];
        let mut mama = data[min_required - 1];

        for i in min_required..n {
            // Strong momentum (high absolute value) = fast alpha
            let momentum_factor = momentum[i].abs().clamp(0.0, 1.0);
            let alpha = self.slow_alpha + momentum_factor * (self.fast_alpha - self.slow_alpha);

            mama = alpha * data[i] + (1.0 - alpha) * mama;
            result[i] = mama;
        }

        result
    }

    /// Calculate normalized momentum (ROC normalized to 0-1 range).
    fn calculate_normalized_momentum(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![0.0; n];

        if n < self.momentum_period {
            return result;
        }

        // Calculate ROC (Rate of Change)
        let mut roc_values = Vec::new();
        for i in self.momentum_period..n {
            if data[i - self.momentum_period].abs() > 1e-10 {
                let roc =
                    (data[i] - data[i - self.momentum_period]) / data[i - self.momentum_period];
                roc_values.push(roc.abs());
                result[i] = roc;
            }
        }

        // Normalize ROC values using recent max
        if !roc_values.is_empty() {
            let lookback = (self.momentum_period * 2).min(roc_values.len());
            for i in self.momentum_period..n {
                let idx = i - self.momentum_period;
                if idx < roc_values.len() {
                    let start = idx.saturating_sub(lookback);
                    let max_roc = roc_values[start..=idx]
                        .iter()
                        .cloned()
                        .fold(0.0_f64, f64::max)
                        .max(0.01);
                    result[i] = (result[i].abs() / max_roc).clamp(0.0, 1.0);
                }
            }
        }

        result
    }
}

impl TechnicalIndicator for MomentumAdaptiveMA {
    fn name(&self) -> &str {
        "Momentum Adaptive MA"
    }

    fn min_periods(&self) -> usize {
        self.period.max(self.momentum_period)
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.min_periods();
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

// ============================================================================
// Efficiency Adaptive Moving Average
// ============================================================================

/// Efficiency Adaptive Moving Average (EfficiencyAdaptiveMA)
///
/// Uses the efficiency ratio (similar to KAMA) for adaptive smoothing.
/// High efficiency (trending) leads to faster response, low efficiency (choppy) leads to more smoothing.
#[derive(Debug, Clone)]
pub struct EfficiencyAdaptiveMA {
    period: usize,
    fast_period: usize,
    slow_period: usize,
}

impl EfficiencyAdaptiveMA {
    /// Create a new EfficiencyAdaptiveMA.
    ///
    /// # Arguments
    /// * `period` - The efficiency ratio period (must be at least 2)
    /// * `fast_period` - Fast EMA period for high efficiency (must be at least 2)
    /// * `slow_period` - Slow EMA period for low efficiency (must be greater than fast_period)
    pub fn new(period: usize, fast_period: usize, slow_period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if fast_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "fast_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if slow_period <= fast_period {
            return Err(IndicatorError::InvalidParameter {
                name: "slow_period".to_string(),
                reason: "must be greater than fast_period".to_string(),
            });
        }
        Ok(Self {
            period,
            fast_period,
            slow_period,
        })
    }

    /// Calculate the efficiency adaptive moving average values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n <= self.period {
            return vec![f64::NAN; n];
        }

        let fast_sc = 2.0 / (self.fast_period as f64 + 1.0);
        let slow_sc = 2.0 / (self.slow_period as f64 + 1.0);

        let mut result = vec![f64::NAN; n];
        result[self.period] = data[self.period];
        let mut eama = data[self.period];

        for i in (self.period + 1)..n {
            // Calculate Efficiency Ratio
            let change = (data[i] - data[i - self.period]).abs();
            let mut volatility = 0.0;
            for j in (i - self.period + 1)..=i {
                volatility += (data[j] - data[j - 1]).abs();
            }

            let er = if volatility > 1e-10 {
                change / volatility
            } else {
                0.0
            };

            // Calculate smoothing constant (squared for extra smoothing)
            let sc = (er * (fast_sc - slow_sc) + slow_sc).powi(2);

            // Update EAMA
            eama = eama + sc * (data[i] - eama);
            result[i] = eama;
        }

        result
    }
}

impl TechnicalIndicator for EfficiencyAdaptiveMA {
    fn name(&self) -> &str {
        "Efficiency Adaptive MA"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() <= self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

// ============================================================================
// Volatility Adaptive Moving Average
// ============================================================================

/// Volatility Adaptive Moving Average (VolatilityAdaptiveMA)
///
/// Adjusts the smoothing factor based on price volatility (ATR-like measure).
/// High volatility leads to slower response (more smoothing) to avoid whipsaws.
/// Low volatility leads to faster response to capture breakouts.
#[derive(Debug, Clone)]
pub struct VolatilityAdaptiveMA {
    period: usize,
    atr_period: usize,
    fast_alpha: f64,
    slow_alpha: f64,
}

impl VolatilityAdaptiveMA {
    /// Create a new VolatilityAdaptiveMA.
    ///
    /// # Arguments
    /// * `period` - The base MA period (must be at least 2)
    /// * `atr_period` - Period for ATR calculation (must be at least 2)
    /// * `fast_alpha` - Fast smoothing factor (0.0 to 1.0)
    /// * `slow_alpha` - Slow smoothing factor (0.0 to 1.0, must be less than fast_alpha)
    pub fn new(period: usize, atr_period: usize, fast_alpha: f64, slow_alpha: f64) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if atr_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "atr_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if fast_alpha <= 0.0 || fast_alpha > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "fast_alpha".to_string(),
                reason: "must be between 0.0 (exclusive) and 1.0 (inclusive)".to_string(),
            });
        }
        if slow_alpha <= 0.0 || slow_alpha >= fast_alpha {
            return Err(IndicatorError::InvalidParameter {
                name: "slow_alpha".to_string(),
                reason: "must be between 0.0 (exclusive) and fast_alpha (exclusive)".to_string(),
            });
        }
        Ok(Self {
            period,
            atr_period,
            fast_alpha,
            slow_alpha,
        })
    }

    /// Calculate the volatility adaptive moving average values.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let min_required = self.period.max(self.atr_period + 1);

        if n < min_required || high.len() < n || low.len() < n {
            return vec![f64::NAN; n];
        }

        // Calculate ATR
        let atr = self.calculate_atr(high, low, close);

        let mut result = vec![f64::NAN; n];
        result[min_required - 1] = close[min_required - 1];
        let mut vama = close[min_required - 1];

        // Calculate baseline ATR for normalization
        let mut atr_sum = 0.0;
        let mut atr_count = 0;
        for i in min_required..n {
            if atr[i] > 0.0 {
                atr_sum += atr[i];
                atr_count += 1;
            }
        }
        let baseline_atr = if atr_count > 0 { atr_sum / atr_count as f64 } else { 1.0 };

        for i in min_required..n {
            // Normalize ATR: high volatility = high ratio
            let vol_ratio = if baseline_atr > 1e-10 {
                (atr[i] / baseline_atr).clamp(0.0, 3.0)
            } else {
                1.0
            };

            // Inverse relationship: high volatility = slow alpha
            let normalized_vol = (vol_ratio / 3.0).clamp(0.0, 1.0);
            let alpha = self.fast_alpha - normalized_vol * (self.fast_alpha - self.slow_alpha);

            vama = alpha * close[i] + (1.0 - alpha) * vama;
            result[i] = vama;
        }

        result
    }

    /// Calculate Average True Range.
    fn calculate_atr(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut tr = vec![0.0; n];
        let mut atr = vec![0.0; n];

        // Calculate True Range
        for i in 1..n {
            let high_low = high[i] - low[i];
            let high_close = (high[i] - close[i - 1]).abs();
            let low_close = (low[i] - close[i - 1]).abs();
            tr[i] = high_low.max(high_close).max(low_close);
        }

        // Calculate ATR using EMA
        let alpha = 2.0 / (self.atr_period as f64 + 1.0);
        let mut atr_val = 0.0;

        for i in 1..n {
            if i <= self.atr_period {
                atr_val = tr[1..=i].iter().sum::<f64>() / i as f64;
            } else {
                atr_val = alpha * tr[i] + (1.0 - alpha) * atr_val;
            }
            atr[i] = atr_val;
        }

        atr
    }
}

impl TechnicalIndicator for VolatilityAdaptiveMA {
    fn name(&self) -> &str {
        "Volatility Adaptive MA"
    }

    fn min_periods(&self) -> usize {
        self.period.max(self.atr_period + 1)
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.min_periods();
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }
        Ok(IndicatorOutput::single(self.calculate(
            &data.high,
            &data.low,
            &data.close,
        )))
    }
}

// ============================================================================
// Cycle Adaptive Moving Average
// ============================================================================

/// Cycle Adaptive Moving Average (CycleAdaptiveMA)
///
/// Adapts the smoothing factor based on detected price cycles.
/// Uses a simplified cycle detection to adjust responsiveness.
/// Short cycles lead to faster response, long cycles lead to more smoothing.
#[derive(Debug, Clone)]
pub struct CycleAdaptiveMA {
    period: usize,
    cycle_period: usize,
    fast_alpha: f64,
    slow_alpha: f64,
}

impl CycleAdaptiveMA {
    /// Create a new CycleAdaptiveMA.
    ///
    /// # Arguments
    /// * `period` - The base MA period (must be at least 2)
    /// * `cycle_period` - Period for cycle detection (must be at least 10)
    /// * `fast_alpha` - Fast smoothing factor (0.0 to 1.0)
    /// * `slow_alpha` - Slow smoothing factor (0.0 to 1.0, must be less than fast_alpha)
    pub fn new(period: usize, cycle_period: usize, fast_alpha: f64, slow_alpha: f64) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if cycle_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "cycle_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if fast_alpha <= 0.0 || fast_alpha > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "fast_alpha".to_string(),
                reason: "must be between 0.0 (exclusive) and 1.0 (inclusive)".to_string(),
            });
        }
        if slow_alpha <= 0.0 || slow_alpha >= fast_alpha {
            return Err(IndicatorError::InvalidParameter {
                name: "slow_alpha".to_string(),
                reason: "must be between 0.0 (exclusive) and fast_alpha (exclusive)".to_string(),
            });
        }
        Ok(Self {
            period,
            cycle_period,
            fast_alpha,
            slow_alpha,
        })
    }

    /// Calculate the cycle adaptive moving average values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let min_required = self.period.max(self.cycle_period);

        if n < min_required {
            return vec![f64::NAN; n];
        }

        // Calculate cycle strength
        let cycle_strength = self.calculate_cycle_strength(data);

        let mut result = vec![f64::NAN; n];
        result[min_required - 1] = data[min_required - 1];
        let mut cama = data[min_required - 1];

        for i in min_required..n {
            // High cycle strength = more predictable = fast alpha
            // Low cycle strength = less predictable = slow alpha
            let alpha = self.slow_alpha + cycle_strength[i] * (self.fast_alpha - self.slow_alpha);

            cama = alpha * data[i] + (1.0 - alpha) * cama;
            result[i] = cama;
        }

        result
    }

    /// Calculate cycle strength using autocorrelation.
    fn calculate_cycle_strength(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![0.5; n]; // Default to medium

        if n < self.cycle_period {
            return result;
        }

        for i in self.cycle_period..n {
            let start = i + 1 - self.cycle_period;
            let window = &data[start..=i];

            // Calculate mean
            let mean: f64 = window.iter().sum::<f64>() / window.len() as f64;

            // Calculate autocorrelation at different lags
            let mut max_corr = 0.0_f64;

            for lag in 2..self.cycle_period / 2 {
                let mut num = 0.0;
                let mut den1 = 0.0;
                let mut den2 = 0.0;

                for j in lag..window.len() {
                    let x = window[j] - mean;
                    let y = window[j - lag] - mean;
                    num += x * y;
                    den1 += x * x;
                    den2 += y * y;
                }

                let denom = (den1 * den2).sqrt();
                let corr = if denom > 1e-10 { num / denom } else { 0.0 };
                max_corr = max_corr.max(corr.abs());
            }

            result[i] = max_corr.clamp(0.0, 1.0);
        }

        result
    }
}

impl TechnicalIndicator for CycleAdaptiveMA {
    fn name(&self) -> &str {
        "Cycle Adaptive MA"
    }

    fn min_periods(&self) -> usize {
        self.period.max(self.cycle_period)
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.min_periods();
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trending_data() -> Vec<f64> {
        (0..50).map(|i| 100.0 + i as f64 * 1.5).collect()
    }

    fn make_choppy_data() -> Vec<f64> {
        (0..50)
            .map(|i| 100.0 + if i % 2 == 0 { 2.0 } else { -2.0 })
            .collect()
    }

    fn make_ohlcv_data() -> OHLCVSeries {
        let close: Vec<f64> = (0..50).map(|i| 100.0 + i as f64).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
        let open: Vec<f64> = close.iter().map(|c| c - 0.5).collect();
        let volume: Vec<f64> = vec![1000.0; 50];

        OHLCVSeries {
            open,
            high,
            low,
            close,
            volume,
        }
    }

    // ========================================================================
    // FractalAdaptiveMA Tests
    // ========================================================================

    #[test]
    fn test_fractal_adaptive_ma_new_valid() {
        let fama = FractalAdaptiveMA::new(10, 0.5, 0.1);
        assert!(fama.is_ok());
    }

    #[test]
    fn test_fractal_adaptive_ma_invalid_period() {
        let result = FractalAdaptiveMA::new(3, 0.5, 0.1);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "period");
        }
    }

    #[test]
    fn test_fractal_adaptive_ma_odd_period() {
        let result = FractalAdaptiveMA::new(11, 0.5, 0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_fractal_adaptive_ma_invalid_fast_alpha() {
        let result = FractalAdaptiveMA::new(10, 0.0, 0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_fractal_adaptive_ma_invalid_slow_alpha() {
        let result = FractalAdaptiveMA::new(10, 0.5, 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_fractal_adaptive_ma_calculate() {
        let fama = FractalAdaptiveMA::new(10, 0.5, 0.1).unwrap();
        let data = make_trending_data();
        let result = fama.calculate(&data);

        assert_eq!(result.len(), data.len());
        // First 9 values should be NaN
        for i in 0..9 {
            assert!(result[i].is_nan());
        }
        // Subsequent values should be valid
        for i in 9..50 {
            assert!(!result[i].is_nan());
        }
    }

    #[test]
    fn test_fractal_adaptive_ma_insufficient_data() {
        let fama = FractalAdaptiveMA::new(10, 0.5, 0.1).unwrap();
        let data = vec![1.0, 2.0, 3.0];
        let result = fama.calculate(&data);

        assert!(result.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn test_fractal_adaptive_ma_trait() {
        let fama = FractalAdaptiveMA::new(10, 0.5, 0.1).unwrap();
        assert_eq!(fama.name(), "Fractal Adaptive MA");
        assert_eq!(fama.min_periods(), 10);
    }

    #[test]
    fn test_fractal_adaptive_ma_compute() {
        let fama = FractalAdaptiveMA::new(10, 0.5, 0.1).unwrap();
        let data = make_ohlcv_data();
        let result = fama.compute(&data);
        assert!(result.is_ok());
    }

    // ========================================================================
    // VolumeAdaptiveMA Tests
    // ========================================================================

    #[test]
    fn test_volume_adaptive_ma_new_valid() {
        let vama = VolumeAdaptiveMA::new(10, 10, 0.5, 0.1);
        assert!(vama.is_ok());
    }

    #[test]
    fn test_volume_adaptive_ma_invalid_period() {
        let result = VolumeAdaptiveMA::new(1, 10, 0.5, 0.1);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "period");
        }
    }

    #[test]
    fn test_volume_adaptive_ma_invalid_volume_period() {
        let result = VolumeAdaptiveMA::new(10, 1, 0.5, 0.1);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "volume_period");
        }
    }

    #[test]
    fn test_volume_adaptive_ma_calculate() {
        let vama = VolumeAdaptiveMA::new(10, 10, 0.5, 0.1).unwrap();
        let close = make_trending_data();
        let volume: Vec<f64> = (0..50).map(|i| 1000.0 + (i as f64 * 10.0)).collect();
        let result = vama.calculate(&close, &volume);

        assert_eq!(result.len(), close.len());
        // Values after min_periods should be valid
        for i in 10..50 {
            assert!(!result[i].is_nan());
        }
    }

    #[test]
    fn test_volume_adaptive_ma_high_volume_response() {
        let vama = VolumeAdaptiveMA::new(5, 5, 0.8, 0.1).unwrap();
        let close: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        // High volume at the end
        let mut volume = vec![1000.0; 20];
        volume[19] = 5000.0;

        let result = vama.calculate(&close, &volume);
        // With high volume, should track price more closely
        assert!(!result[19].is_nan());
    }

    #[test]
    fn test_volume_adaptive_ma_trait() {
        let vama = VolumeAdaptiveMA::new(10, 15, 0.5, 0.1).unwrap();
        assert_eq!(vama.name(), "Volume Adaptive MA");
        assert_eq!(vama.min_periods(), 15);
    }

    #[test]
    fn test_volume_adaptive_ma_compute() {
        let vama = VolumeAdaptiveMA::new(10, 10, 0.5, 0.1).unwrap();
        let data = make_ohlcv_data();
        let result = vama.compute(&data);
        assert!(result.is_ok());
    }

    // ========================================================================
    // TrendAdaptiveMA Tests
    // ========================================================================

    #[test]
    fn test_trend_adaptive_ma_new_valid() {
        let tama = TrendAdaptiveMA::new(10, 14, 0.5, 0.1);
        assert!(tama.is_ok());
    }

    #[test]
    fn test_trend_adaptive_ma_invalid_period() {
        let result = TrendAdaptiveMA::new(1, 14, 0.5, 0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_trend_adaptive_ma_invalid_adx_period() {
        let result = TrendAdaptiveMA::new(10, 1, 0.5, 0.1);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "adx_period");
        }
    }

    #[test]
    fn test_trend_adaptive_ma_calculate() {
        let tama = TrendAdaptiveMA::new(10, 14, 0.5, 0.1).unwrap();
        let data = make_ohlcv_data();
        let result = tama.calculate(&data.high, &data.low, &data.close);

        assert_eq!(result.len(), data.close.len());
    }

    #[test]
    fn test_trend_adaptive_ma_trait() {
        let tama = TrendAdaptiveMA::new(10, 14, 0.5, 0.1).unwrap();
        assert_eq!(tama.name(), "Trend Adaptive MA");
        assert_eq!(tama.min_periods(), 28); // max(10, 14*2)
    }

    #[test]
    fn test_trend_adaptive_ma_compute() {
        let tama = TrendAdaptiveMA::new(10, 14, 0.5, 0.1).unwrap();
        let data = make_ohlcv_data();
        let result = tama.compute(&data);
        assert!(result.is_ok());
    }

    // ========================================================================
    // NoiseAdaptiveMA Tests
    // ========================================================================

    #[test]
    fn test_noise_adaptive_ma_new_valid() {
        let nama = NoiseAdaptiveMA::new(10, 10, 0.5, 0.1);
        assert!(nama.is_ok());
    }

    #[test]
    fn test_noise_adaptive_ma_invalid_period() {
        let result = NoiseAdaptiveMA::new(1, 10, 0.5, 0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_noise_adaptive_ma_invalid_noise_period() {
        let result = NoiseAdaptiveMA::new(10, 1, 0.5, 0.1);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "noise_period");
        }
    }

    #[test]
    fn test_noise_adaptive_ma_trending_vs_choppy() {
        let nama = NoiseAdaptiveMA::new(10, 10, 0.8, 0.1).unwrap();

        let trending = make_trending_data();
        let choppy = make_choppy_data();

        let result_trending = nama.calculate(&trending);
        let result_choppy = nama.calculate(&choppy);

        // Both should have valid values after warmup
        assert!(!result_trending[49].is_nan());
        assert!(!result_choppy[49].is_nan());
    }

    #[test]
    fn test_noise_adaptive_ma_trait() {
        let nama = NoiseAdaptiveMA::new(10, 15, 0.5, 0.1).unwrap();
        assert_eq!(nama.name(), "Noise Adaptive MA");
        assert_eq!(nama.min_periods(), 15);
    }

    #[test]
    fn test_noise_adaptive_ma_compute() {
        let nama = NoiseAdaptiveMA::new(10, 10, 0.5, 0.1).unwrap();
        let data = make_ohlcv_data();
        let result = nama.compute(&data);
        assert!(result.is_ok());
    }

    // ========================================================================
    // MomentumAdaptiveMA Tests
    // ========================================================================

    #[test]
    fn test_momentum_adaptive_ma_new_valid() {
        let mama = MomentumAdaptiveMA::new(10, 10, 0.5, 0.1);
        assert!(mama.is_ok());
    }

    #[test]
    fn test_momentum_adaptive_ma_invalid_period() {
        let result = MomentumAdaptiveMA::new(1, 10, 0.5, 0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_momentum_adaptive_ma_invalid_momentum_period() {
        let result = MomentumAdaptiveMA::new(10, 1, 0.5, 0.1);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "momentum_period");
        }
    }

    #[test]
    fn test_momentum_adaptive_ma_calculate() {
        let mama = MomentumAdaptiveMA::new(10, 10, 0.5, 0.1).unwrap();
        let data = make_trending_data();
        let result = mama.calculate(&data);

        assert_eq!(result.len(), data.len());
        // Values after min_periods should be valid
        for i in 10..50 {
            assert!(!result[i].is_nan());
        }
    }

    #[test]
    fn test_momentum_adaptive_ma_trait() {
        let mama = MomentumAdaptiveMA::new(10, 15, 0.5, 0.1).unwrap();
        assert_eq!(mama.name(), "Momentum Adaptive MA");
        assert_eq!(mama.min_periods(), 15);
    }

    #[test]
    fn test_momentum_adaptive_ma_compute() {
        let mama = MomentumAdaptiveMA::new(10, 10, 0.5, 0.1).unwrap();
        let data = make_ohlcv_data();
        let result = mama.compute(&data);
        assert!(result.is_ok());
    }

    // ========================================================================
    // EfficiencyAdaptiveMA Tests
    // ========================================================================

    #[test]
    fn test_efficiency_adaptive_ma_new_valid() {
        let eama = EfficiencyAdaptiveMA::new(10, 2, 30);
        assert!(eama.is_ok());
    }

    #[test]
    fn test_efficiency_adaptive_ma_invalid_period() {
        let result = EfficiencyAdaptiveMA::new(1, 2, 30);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "period");
        }
    }

    #[test]
    fn test_efficiency_adaptive_ma_invalid_fast_period() {
        let result = EfficiencyAdaptiveMA::new(10, 1, 30);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "fast_period");
        }
    }

    #[test]
    fn test_efficiency_adaptive_ma_invalid_slow_period() {
        let result = EfficiencyAdaptiveMA::new(10, 2, 2);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "slow_period");
        }
    }

    #[test]
    fn test_efficiency_adaptive_ma_calculate() {
        let eama = EfficiencyAdaptiveMA::new(10, 2, 30).unwrap();
        let data = make_trending_data();
        let result = eama.calculate(&data);

        assert_eq!(result.len(), data.len());
        // First 10 values should be NaN
        for i in 0..10 {
            assert!(result[i].is_nan());
        }
        // Subsequent values should be valid
        for i in 11..50 {
            assert!(!result[i].is_nan());
        }
    }

    #[test]
    fn test_efficiency_adaptive_ma_trending() {
        let eama = EfficiencyAdaptiveMA::new(10, 2, 30).unwrap();
        let data = make_trending_data();
        let result = eama.calculate(&data);

        // In a strong trend, EAMA should track price closely
        let last_value = result[49];
        assert!(!last_value.is_nan());
        assert!(last_value > 130.0); // Should be tracking upward
    }

    #[test]
    fn test_efficiency_adaptive_ma_trait() {
        let eama = EfficiencyAdaptiveMA::new(10, 2, 30).unwrap();
        assert_eq!(eama.name(), "Efficiency Adaptive MA");
        assert_eq!(eama.min_periods(), 11);
    }

    #[test]
    fn test_efficiency_adaptive_ma_compute() {
        let eama = EfficiencyAdaptiveMA::new(10, 2, 30).unwrap();
        let data = make_ohlcv_data();
        let result = eama.compute(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_efficiency_adaptive_ma_insufficient_data() {
        let eama = EfficiencyAdaptiveMA::new(10, 2, 30).unwrap();
        let data = OHLCVSeries {
            open: vec![1.0, 2.0],
            high: vec![1.5, 2.5],
            low: vec![0.5, 1.5],
            close: vec![1.0, 2.0],
            volume: vec![100.0, 100.0],
        };
        let result = eama.compute(&data);
        assert!(result.is_err());
        if let Err(IndicatorError::InsufficientData { required, got }) = result {
            assert_eq!(required, 11);
            assert_eq!(got, 2);
        }
    }

    // ========================================================================
    // VolatilityAdaptiveMA Tests
    // ========================================================================

    #[test]
    fn test_volatility_adaptive_ma_new_valid() {
        let vama = VolatilityAdaptiveMA::new(10, 14, 0.5, 0.1);
        assert!(vama.is_ok());
    }

    #[test]
    fn test_volatility_adaptive_ma_invalid_period() {
        let result = VolatilityAdaptiveMA::new(1, 14, 0.5, 0.1);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "period");
        }
    }

    #[test]
    fn test_volatility_adaptive_ma_invalid_atr_period() {
        let result = VolatilityAdaptiveMA::new(10, 1, 0.5, 0.1);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "atr_period");
        }
    }

    #[test]
    fn test_volatility_adaptive_ma_calculate() {
        let vama = VolatilityAdaptiveMA::new(10, 14, 0.5, 0.1).unwrap();
        let data = make_ohlcv_data();
        let result = vama.calculate(&data.high, &data.low, &data.close);

        assert_eq!(result.len(), data.close.len());
        // Values after min_periods should be valid
        for i in 15..50 {
            assert!(!result[i].is_nan());
        }
    }

    #[test]
    fn test_volatility_adaptive_ma_trait() {
        let vama = VolatilityAdaptiveMA::new(10, 14, 0.5, 0.1).unwrap();
        assert_eq!(vama.name(), "Volatility Adaptive MA");
        assert_eq!(vama.min_periods(), 15);
    }

    #[test]
    fn test_volatility_adaptive_ma_compute() {
        let vama = VolatilityAdaptiveMA::new(10, 14, 0.5, 0.1).unwrap();
        let data = make_ohlcv_data();
        let result = vama.compute(&data);
        assert!(result.is_ok());
    }

    // ========================================================================
    // CycleAdaptiveMA Tests
    // ========================================================================

    #[test]
    fn test_cycle_adaptive_ma_new_valid() {
        let cama = CycleAdaptiveMA::new(10, 20, 0.5, 0.1);
        assert!(cama.is_ok());
    }

    #[test]
    fn test_cycle_adaptive_ma_invalid_period() {
        let result = CycleAdaptiveMA::new(1, 20, 0.5, 0.1);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "period");
        }
    }

    #[test]
    fn test_cycle_adaptive_ma_invalid_cycle_period() {
        let result = CycleAdaptiveMA::new(10, 5, 0.5, 0.1);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "cycle_period");
        }
    }

    #[test]
    fn test_cycle_adaptive_ma_calculate() {
        let cama = CycleAdaptiveMA::new(10, 20, 0.5, 0.1).unwrap();
        let data = make_trending_data();
        let result = cama.calculate(&data);

        assert_eq!(result.len(), data.len());
        // Values after min_periods should be valid
        for i in 20..50 {
            assert!(!result[i].is_nan());
        }
    }

    #[test]
    fn test_cycle_adaptive_ma_trait() {
        let cama = CycleAdaptiveMA::new(10, 20, 0.5, 0.1).unwrap();
        assert_eq!(cama.name(), "Cycle Adaptive MA");
        assert_eq!(cama.min_periods(), 20);
    }

    #[test]
    fn test_cycle_adaptive_ma_compute() {
        let cama = CycleAdaptiveMA::new(10, 20, 0.5, 0.1).unwrap();
        let data = make_ohlcv_data();
        let result = cama.compute(&data);
        assert!(result.is_ok());
    }
}
