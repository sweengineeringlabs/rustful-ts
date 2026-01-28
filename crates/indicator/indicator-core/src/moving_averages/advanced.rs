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
// Regime Adaptive Moving Average
// ============================================================================

/// Regime Adaptive Moving Average (RegimeAdaptiveMA)
///
/// Adapts the smoothing factor based on detected market regime.
/// Identifies trending, ranging, and volatile regimes to adjust responsiveness.
/// Trending regimes use faster response, ranging regimes use more smoothing.
#[derive(Debug, Clone)]
pub struct RegimeAdaptiveMA {
    period: usize,
    regime_period: usize,
    fast_alpha: f64,
    slow_alpha: f64,
}

impl RegimeAdaptiveMA {
    /// Create a new RegimeAdaptiveMA.
    ///
    /// # Arguments
    /// * `period` - The base MA period (must be at least 2)
    /// * `regime_period` - Period for regime detection (must be at least 10)
    /// * `fast_alpha` - Fast smoothing factor (0.0 to 1.0)
    /// * `slow_alpha` - Slow smoothing factor (0.0 to 1.0, must be less than fast_alpha)
    pub fn new(period: usize, regime_period: usize, fast_alpha: f64, slow_alpha: f64) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if regime_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "regime_period".to_string(),
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
            regime_period,
            fast_alpha,
            slow_alpha,
        })
    }

    /// Calculate the regime adaptive moving average values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let min_required = self.period.max(self.regime_period);

        if n < min_required {
            return vec![f64::NAN; n];
        }

        // Calculate regime scores
        let regime_scores = self.calculate_regime_score(data);

        let mut result = vec![f64::NAN; n];
        result[min_required - 1] = data[min_required - 1];
        let mut rama = data[min_required - 1];

        for i in min_required..n {
            // Regime score: 1.0 = strong trend, 0.0 = ranging
            let alpha = self.slow_alpha + regime_scores[i] * (self.fast_alpha - self.slow_alpha);

            rama = alpha * data[i] + (1.0 - alpha) * rama;
            result[i] = rama;
        }

        result
    }

    /// Calculate regime score based on trend strength and directional consistency.
    fn calculate_regime_score(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![0.5; n];

        if n < self.regime_period {
            return result;
        }

        for i in self.regime_period..n {
            let start = i + 1 - self.regime_period;
            let window = &data[start..=i];

            // Calculate efficiency ratio (direction vs total movement)
            let net_change = (window.last().unwrap() - window.first().unwrap()).abs();
            let total_movement: f64 = (1..window.len())
                .map(|j| (window[j] - window[j - 1]).abs())
                .sum();

            let efficiency = if total_movement > 1e-10 {
                net_change / total_movement
            } else {
                0.0
            };

            // Calculate directional consistency (how often price moves in same direction)
            let mut same_direction_count = 0;
            let overall_direction = if net_change > 0.0 { 1 } else { -1 };
            for j in 1..window.len() {
                let change = window[j] - window[j - 1];
                let dir = if change > 0.0 { 1 } else if change < 0.0 { -1 } else { 0 };
                if dir == overall_direction {
                    same_direction_count += 1;
                }
            }
            let consistency = same_direction_count as f64 / (window.len() - 1) as f64;

            // Combine efficiency and consistency for regime score
            result[i] = (efficiency * 0.6 + consistency * 0.4).clamp(0.0, 1.0);
        }

        result
    }
}

impl TechnicalIndicator for RegimeAdaptiveMA {
    fn name(&self) -> &str {
        "Regime Adaptive MA"
    }

    fn min_periods(&self) -> usize {
        self.period.max(self.regime_period)
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
// Volume Price Moving Average
// ============================================================================

/// Volume Price Moving Average (VolumePriceMA)
///
/// Weights the moving average by the volume-price relationship.
/// High volume on significant price moves carries more weight.
/// Captures the importance of volume-confirmed price action.
#[derive(Debug, Clone)]
pub struct VolumePriceMA {
    period: usize,
}

impl VolumePriceMA {
    /// Create a new VolumePriceMA.
    ///
    /// # Arguments
    /// * `period` - The lookback period (must be at least 2)
    pub fn new(period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate the volume-price weighted moving average values.
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < self.period || volume.len() < n {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; n];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window_close = &close[start..=i];
            let window_volume = &volume[start..=i];

            // Calculate volume-price weights
            let mut weights = Vec::with_capacity(self.period);
            let mut weight_sum = 0.0;

            for j in 0..self.period {
                // Price change magnitude
                let price_change = if j > 0 {
                    (window_close[j] - window_close[j - 1]).abs()
                } else if start > 0 {
                    (window_close[j] - close[start - 1]).abs()
                } else {
                    0.0
                };

                // Normalize volume
                let avg_volume: f64 = window_volume.iter().sum::<f64>() / self.period as f64;
                let vol_ratio = if avg_volume > 1e-10 {
                    window_volume[j] / avg_volume
                } else {
                    1.0
                };

                // Weight = volume ratio * (1 + price change factor)
                let avg_price = window_close.iter().sum::<f64>() / self.period as f64;
                let price_factor = if avg_price > 1e-10 {
                    price_change / avg_price
                } else {
                    0.0
                };

                let weight = vol_ratio * (1.0 + price_factor * 10.0);
                weights.push(weight);
                weight_sum += weight;
            }

            // Calculate weighted average
            if weight_sum > 1e-10 {
                let weighted_sum: f64 = window_close
                    .iter()
                    .zip(weights.iter())
                    .map(|(p, w)| p * w)
                    .sum();
                result[i] = weighted_sum / weight_sum;
            } else {
                result[i] = window_close.iter().sum::<f64>() / self.period as f64;
            }
        }

        result
    }
}

impl TechnicalIndicator for VolumePriceMA {
    fn name(&self) -> &str {
        "Volume Price MA"
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
        Ok(IndicatorOutput::single(
            self.calculate(&data.close, &data.volume),
        ))
    }
}

// ============================================================================
// Momentum Filtered Moving Average
// ============================================================================

/// Momentum Filtered Moving Average (MomentumFilteredMA)
///
/// Filters the moving average based on momentum conditions.
/// Only updates the MA when momentum exceeds a threshold.
/// Helps reduce whipsaws during low-momentum periods.
#[derive(Debug, Clone)]
pub struct MomentumFilteredMA {
    period: usize,
    momentum_period: usize,
    threshold: f64,
}

impl MomentumFilteredMA {
    /// Create a new MomentumFilteredMA.
    ///
    /// # Arguments
    /// * `period` - The base MA period (must be at least 2)
    /// * `momentum_period` - Period for momentum calculation (must be at least 2)
    /// * `threshold` - Momentum threshold for updates (0.0 to 1.0)
    pub fn new(period: usize, momentum_period: usize, threshold: f64) -> Result<Self> {
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
        if threshold < 0.0 || threshold > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "threshold".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        Ok(Self {
            period,
            momentum_period,
            threshold,
        })
    }

    /// Calculate the momentum filtered moving average values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let min_required = self.period.max(self.momentum_period);

        if n < min_required {
            return vec![f64::NAN; n];
        }

        // Calculate normalized momentum
        let momentum = self.calculate_normalized_momentum(data);

        let alpha = 2.0 / (self.period as f64 + 1.0);
        let mut result = vec![f64::NAN; n];
        result[min_required - 1] = data[min_required - 1];
        let mut mfma = data[min_required - 1];

        for i in min_required..n {
            // Only update if momentum exceeds threshold
            if momentum[i] >= self.threshold {
                mfma = alpha * data[i] + (1.0 - alpha) * mfma;
            }
            // If momentum below threshold, MA stays flat (no update)
            result[i] = mfma;
        }

        result
    }

    /// Calculate normalized momentum.
    fn calculate_normalized_momentum(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![0.0; n];

        if n < self.momentum_period {
            return result;
        }

        // Calculate ROC values
        let mut roc_abs = Vec::new();
        for i in self.momentum_period..n {
            if data[i - self.momentum_period].abs() > 1e-10 {
                let roc = ((data[i] - data[i - self.momentum_period]) / data[i - self.momentum_period]).abs();
                roc_abs.push(roc);
            }
        }

        // Normalize using recent max
        if !roc_abs.is_empty() {
            let lookback = (self.momentum_period * 2).min(roc_abs.len());
            for i in self.momentum_period..n {
                let idx = i - self.momentum_period;
                if idx < roc_abs.len() {
                    let start = idx.saturating_sub(lookback);
                    let max_roc = roc_abs[start..=idx]
                        .iter()
                        .cloned()
                        .fold(0.01_f64, f64::max);
                    result[i] = (roc_abs[idx] / max_roc).clamp(0.0, 1.0);
                }
            }
        }

        result
    }
}

impl TechnicalIndicator for MomentumFilteredMA {
    fn name(&self) -> &str {
        "Momentum Filtered MA"
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
// Trend Strength Moving Average
// ============================================================================

/// Trend Strength Moving Average (TrendStrengthMA)
///
/// Weights the moving average by trend strength at each point.
/// Prices during strong trends carry more weight than ranging periods.
/// Emphasizes trend-confirming price action.
#[derive(Debug, Clone)]
pub struct TrendStrengthMA {
    period: usize,
    strength_period: usize,
}

impl TrendStrengthMA {
    /// Create a new TrendStrengthMA.
    ///
    /// # Arguments
    /// * `period` - The base MA period (must be at least 2)
    /// * `strength_period` - Period for trend strength calculation (must be at least 2)
    pub fn new(period: usize, strength_period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if strength_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "strength_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self {
            period,
            strength_period,
        })
    }

    /// Calculate the trend strength weighted moving average values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let min_required = self.period.max(self.strength_period);

        if n < min_required {
            return vec![f64::NAN; n];
        }

        // Calculate trend strength for each point
        let strength = self.calculate_trend_strength(data);

        let mut result = vec![f64::NAN; n];

        for i in (min_required - 1)..n {
            let start = i + 1 - self.period;
            let window_data = &data[start..=i];
            let window_strength = &strength[start..=i];

            // Calculate weighted average using trend strength as weights
            let mut weight_sum = 0.0;
            let mut weighted_sum = 0.0;

            for j in 0..self.period {
                // Add small base weight to avoid zero weights
                let weight = window_strength[j] + 0.1;
                weight_sum += weight;
                weighted_sum += window_data[j] * weight;
            }

            result[i] = if weight_sum > 1e-10 {
                weighted_sum / weight_sum
            } else {
                window_data.iter().sum::<f64>() / self.period as f64
            };
        }

        result
    }

    /// Calculate trend strength at each point using efficiency ratio.
    fn calculate_trend_strength(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![0.5; n];

        if n < self.strength_period {
            return result;
        }

        for i in self.strength_period..n {
            let start = i + 1 - self.strength_period;
            let window = &data[start..=i];

            let net_change = (window.last().unwrap() - window.first().unwrap()).abs();
            let total_movement: f64 = (1..window.len())
                .map(|j| (window[j] - window[j - 1]).abs())
                .sum();

            result[i] = if total_movement > 1e-10 {
                (net_change / total_movement).clamp(0.0, 1.0)
            } else {
                0.5
            };
        }

        result
    }
}

impl TechnicalIndicator for TrendStrengthMA {
    fn name(&self) -> &str {
        "Trend Strength MA"
    }

    fn min_periods(&self) -> usize {
        self.period.max(self.strength_period)
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
// Cycle Adjusted Moving Average
// ============================================================================

/// Cycle Adjusted Moving Average (CycleAdjustedMA)
///
/// Adjusts the moving average period based on the dominant cycle length.
/// Detects the dominant cycle using autocorrelation and adjusts accordingly.
/// Shorter detected cycles lead to shorter MA periods.
#[derive(Debug, Clone)]
pub struct CycleAdjustedMA {
    base_period: usize,
    cycle_period: usize,
    min_period: usize,
    max_period: usize,
}

impl CycleAdjustedMA {
    /// Create a new CycleAdjustedMA.
    ///
    /// # Arguments
    /// * `base_period` - The base MA period (must be at least 2)
    /// * `cycle_period` - Period for cycle detection (must be at least 10)
    /// * `min_period` - Minimum adaptive period (must be at least 2)
    /// * `max_period` - Maximum adaptive period (must be greater than min_period)
    pub fn new(base_period: usize, cycle_period: usize, min_period: usize, max_period: usize) -> Result<Self> {
        if base_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "base_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if cycle_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "cycle_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if min_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "min_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if max_period <= min_period {
            return Err(IndicatorError::InvalidParameter {
                name: "max_period".to_string(),
                reason: "must be greater than min_period".to_string(),
            });
        }
        Ok(Self {
            base_period,
            cycle_period,
            min_period,
            max_period,
        })
    }

    /// Calculate the cycle adjusted moving average values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let min_required = self.base_period.max(self.cycle_period);

        if n < min_required {
            return vec![f64::NAN; n];
        }

        // Detect dominant cycle lengths
        let cycle_lengths = self.detect_cycle_length(data);

        let mut result = vec![f64::NAN; n];

        for i in min_required..n {
            // Adaptive period based on detected cycle
            let cycle_len = cycle_lengths[i];
            let adaptive_period = if cycle_len > 0 {
                // Use half the cycle length as the MA period
                ((cycle_len / 2) as usize).clamp(self.min_period, self.max_period)
            } else {
                self.base_period
            };

            // Calculate SMA with adaptive period
            if i >= adaptive_period - 1 {
                let start = i + 1 - adaptive_period;
                let sum: f64 = data[start..=i].iter().sum();
                result[i] = sum / adaptive_period as f64;
            }
        }

        result
    }

    /// Detect dominant cycle length using autocorrelation.
    fn detect_cycle_length(&self, data: &[f64]) -> Vec<i32> {
        let n = data.len();
        let mut result = vec![0; n];

        if n < self.cycle_period {
            return result;
        }

        for i in self.cycle_period..n {
            let start = i + 1 - self.cycle_period;
            let window = &data[start..=i];

            // Calculate mean
            let mean: f64 = window.iter().sum::<f64>() / window.len() as f64;

            // Find lag with maximum autocorrelation
            let mut max_corr = 0.0_f64;
            let mut best_lag = 0;

            for lag in 2..(self.cycle_period / 2) {
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

                if corr > max_corr && corr > 0.3 {
                    max_corr = corr;
                    best_lag = lag as i32;
                }
            }

            result[i] = best_lag * 2; // Cycle length is approximately 2x the lag
        }

        result
    }
}

impl TechnicalIndicator for CycleAdjustedMA {
    fn name(&self) -> &str {
        "Cycle Adjusted MA"
    }

    fn min_periods(&self) -> usize {
        self.base_period.max(self.cycle_period)
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
// Adaptive Lag Moving Average
// ============================================================================

/// Adaptive Lag Moving Average (AdaptiveLagMA)
///
/// Adjusts the effective lag based on price volatility.
/// High volatility increases lag to reduce noise, low volatility decreases lag.
/// Balances responsiveness and smoothness based on market conditions.
#[derive(Debug, Clone)]
pub struct AdaptiveLagMA {
    period: usize,
    volatility_period: usize,
    min_lag: f64,
    max_lag: f64,
}

impl AdaptiveLagMA {
    /// Create a new AdaptiveLagMA.
    ///
    /// # Arguments
    /// * `period` - The base MA period (must be at least 2)
    /// * `volatility_period` - Period for volatility calculation (must be at least 2)
    /// * `min_lag` - Minimum lag factor (0.0 to 1.0)
    /// * `max_lag` - Maximum lag factor (must be greater than min_lag and at most 1.0)
    pub fn new(period: usize, volatility_period: usize, min_lag: f64, max_lag: f64) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if volatility_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if min_lag < 0.0 || min_lag > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "min_lag".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        if max_lag <= min_lag || max_lag > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "max_lag".to_string(),
                reason: "must be greater than min_lag and at most 1.0".to_string(),
            });
        }
        Ok(Self {
            period,
            volatility_period,
            min_lag,
            max_lag,
        })
    }

    /// Calculate the adaptive lag moving average values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let min_required = self.period.max(self.volatility_period);

        if n < min_required {
            return vec![f64::NAN; n];
        }

        // Calculate normalized volatility
        let volatility = self.calculate_normalized_volatility(data);

        let mut result = vec![f64::NAN; n];
        result[min_required - 1] = data[min_required - 1];
        let mut alma = data[min_required - 1];

        for i in min_required..n {
            // Adaptive lag: high volatility = high lag (slow), low volatility = low lag (fast)
            let lag_factor = self.min_lag + volatility[i] * (self.max_lag - self.min_lag);

            // Convert lag factor to alpha (inverse relationship)
            let alpha = (1.0 - lag_factor) * (2.0 / (self.period as f64 + 1.0));

            alma = alpha * data[i] + (1.0 - alpha) * alma;
            result[i] = alma;
        }

        result
    }

    /// Calculate normalized volatility using standard deviation of returns.
    fn calculate_normalized_volatility(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![0.5; n];

        if n < self.volatility_period {
            return result;
        }

        // Calculate rolling volatility
        let mut vol_values = Vec::new();

        for i in self.volatility_period..n {
            let start = i + 1 - self.volatility_period;
            let window = &data[start..=i];

            // Calculate returns
            let returns: Vec<f64> = (1..window.len())
                .map(|j| {
                    if window[j - 1].abs() > 1e-10 {
                        (window[j] - window[j - 1]) / window[j - 1]
                    } else {
                        0.0
                    }
                })
                .collect();

            if returns.is_empty() {
                vol_values.push(0.0);
                continue;
            }

            // Calculate standard deviation of returns
            let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
            let std_dev = variance.sqrt();

            vol_values.push(std_dev);
        }

        // Normalize volatility
        if !vol_values.is_empty() {
            let lookback = (self.volatility_period * 2).min(vol_values.len());
            for i in self.volatility_period..n {
                let idx = i - self.volatility_period;
                if idx < vol_values.len() {
                    let start = idx.saturating_sub(lookback);
                    let max_vol = vol_values[start..=idx]
                        .iter()
                        .cloned()
                        .fold(0.001_f64, f64::max);
                    result[i] = (vol_values[idx] / max_vol).clamp(0.0, 1.0);
                }
            }
        }

        result
    }
}

impl TechnicalIndicator for AdaptiveLagMA {
    fn name(&self) -> &str {
        "Adaptive Lag MA"
    }

    fn min_periods(&self) -> usize {
        self.period.max(self.volatility_period)
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

    // ========================================================================
    // RegimeAdaptiveMA Tests
    // ========================================================================

    #[test]
    fn test_regime_adaptive_ma_new_valid() {
        let rama = RegimeAdaptiveMA::new(10, 20, 0.5, 0.1);
        assert!(rama.is_ok());
    }

    #[test]
    fn test_regime_adaptive_ma_invalid_period() {
        let result = RegimeAdaptiveMA::new(1, 20, 0.5, 0.1);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "period");
        }
    }

    #[test]
    fn test_regime_adaptive_ma_invalid_regime_period() {
        let result = RegimeAdaptiveMA::new(10, 5, 0.5, 0.1);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "regime_period");
        }
    }

    #[test]
    fn test_regime_adaptive_ma_invalid_fast_alpha() {
        let result = RegimeAdaptiveMA::new(10, 20, 0.0, 0.1);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "fast_alpha");
        }
    }

    #[test]
    fn test_regime_adaptive_ma_invalid_slow_alpha() {
        let result = RegimeAdaptiveMA::new(10, 20, 0.5, 0.6);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "slow_alpha");
        }
    }

    #[test]
    fn test_regime_adaptive_ma_calculate() {
        let rama = RegimeAdaptiveMA::new(10, 20, 0.5, 0.1).unwrap();
        let data = make_trending_data();
        let result = rama.calculate(&data);

        assert_eq!(result.len(), data.len());
        // Values after min_periods should be valid
        for i in 20..50 {
            assert!(!result[i].is_nan());
        }
    }

    #[test]
    fn test_regime_adaptive_ma_trait() {
        let rama = RegimeAdaptiveMA::new(10, 20, 0.5, 0.1).unwrap();
        assert_eq!(rama.name(), "Regime Adaptive MA");
        assert_eq!(rama.min_periods(), 20);
    }

    #[test]
    fn test_regime_adaptive_ma_compute() {
        let rama = RegimeAdaptiveMA::new(10, 20, 0.5, 0.1).unwrap();
        let data = make_ohlcv_data();
        let result = rama.compute(&data);
        assert!(result.is_ok());
    }

    // ========================================================================
    // VolumePriceMA Tests
    // ========================================================================

    #[test]
    fn test_volume_price_ma_new_valid() {
        let vpma = VolumePriceMA::new(10);
        assert!(vpma.is_ok());
    }

    #[test]
    fn test_volume_price_ma_invalid_period() {
        let result = VolumePriceMA::new(1);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "period");
        }
    }

    #[test]
    fn test_volume_price_ma_calculate() {
        let vpma = VolumePriceMA::new(10).unwrap();
        let close = make_trending_data();
        let volume: Vec<f64> = (0..50).map(|i| 1000.0 + (i as f64 * 10.0)).collect();
        let result = vpma.calculate(&close, &volume);

        assert_eq!(result.len(), close.len());
        // Values after min_periods should be valid
        for i in 9..50 {
            assert!(!result[i].is_nan());
        }
    }

    #[test]
    fn test_volume_price_ma_trait() {
        let vpma = VolumePriceMA::new(10).unwrap();
        assert_eq!(vpma.name(), "Volume Price MA");
        assert_eq!(vpma.min_periods(), 10);
    }

    #[test]
    fn test_volume_price_ma_compute() {
        let vpma = VolumePriceMA::new(10).unwrap();
        let data = make_ohlcv_data();
        let result = vpma.compute(&data);
        assert!(result.is_ok());
    }

    // ========================================================================
    // MomentumFilteredMA Tests
    // ========================================================================

    #[test]
    fn test_momentum_filtered_ma_new_valid() {
        let mfma = MomentumFilteredMA::new(10, 10, 0.3);
        assert!(mfma.is_ok());
    }

    #[test]
    fn test_momentum_filtered_ma_invalid_period() {
        let result = MomentumFilteredMA::new(1, 10, 0.3);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "period");
        }
    }

    #[test]
    fn test_momentum_filtered_ma_invalid_momentum_period() {
        let result = MomentumFilteredMA::new(10, 1, 0.3);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "momentum_period");
        }
    }

    #[test]
    fn test_momentum_filtered_ma_invalid_threshold() {
        let result = MomentumFilteredMA::new(10, 10, 1.5);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "threshold");
        }
    }

    #[test]
    fn test_momentum_filtered_ma_calculate() {
        let mfma = MomentumFilteredMA::new(10, 10, 0.3).unwrap();
        let data = make_trending_data();
        let result = mfma.calculate(&data);

        assert_eq!(result.len(), data.len());
        // Values after min_periods should be valid
        for i in 10..50 {
            assert!(!result[i].is_nan());
        }
    }

    #[test]
    fn test_momentum_filtered_ma_trait() {
        let mfma = MomentumFilteredMA::new(10, 15, 0.3).unwrap();
        assert_eq!(mfma.name(), "Momentum Filtered MA");
        assert_eq!(mfma.min_periods(), 15);
    }

    #[test]
    fn test_momentum_filtered_ma_compute() {
        let mfma = MomentumFilteredMA::new(10, 10, 0.3).unwrap();
        let data = make_ohlcv_data();
        let result = mfma.compute(&data);
        assert!(result.is_ok());
    }

    // ========================================================================
    // TrendStrengthMA Tests
    // ========================================================================

    #[test]
    fn test_trend_strength_ma_new_valid() {
        let tsma = TrendStrengthMA::new(10, 10);
        assert!(tsma.is_ok());
    }

    #[test]
    fn test_trend_strength_ma_invalid_period() {
        let result = TrendStrengthMA::new(1, 10);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "period");
        }
    }

    #[test]
    fn test_trend_strength_ma_invalid_strength_period() {
        let result = TrendStrengthMA::new(10, 1);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "strength_period");
        }
    }

    #[test]
    fn test_trend_strength_ma_calculate() {
        let tsma = TrendStrengthMA::new(10, 10).unwrap();
        let data = make_trending_data();
        let result = tsma.calculate(&data);

        assert_eq!(result.len(), data.len());
        // Values after min_periods should be valid
        for i in 9..50 {
            assert!(!result[i].is_nan());
        }
    }

    #[test]
    fn test_trend_strength_ma_trait() {
        let tsma = TrendStrengthMA::new(10, 15).unwrap();
        assert_eq!(tsma.name(), "Trend Strength MA");
        assert_eq!(tsma.min_periods(), 15);
    }

    #[test]
    fn test_trend_strength_ma_compute() {
        let tsma = TrendStrengthMA::new(10, 10).unwrap();
        let data = make_ohlcv_data();
        let result = tsma.compute(&data);
        assert!(result.is_ok());
    }

    // ========================================================================
    // CycleAdjustedMA Tests
    // ========================================================================

    #[test]
    fn test_cycle_adjusted_ma_new_valid() {
        let cama = CycleAdjustedMA::new(10, 20, 5, 30);
        assert!(cama.is_ok());
    }

    #[test]
    fn test_cycle_adjusted_ma_invalid_base_period() {
        let result = CycleAdjustedMA::new(1, 20, 5, 30);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "base_period");
        }
    }

    #[test]
    fn test_cycle_adjusted_ma_invalid_cycle_period() {
        let result = CycleAdjustedMA::new(10, 5, 5, 30);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "cycle_period");
        }
    }

    #[test]
    fn test_cycle_adjusted_ma_invalid_min_period() {
        let result = CycleAdjustedMA::new(10, 20, 1, 30);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "min_period");
        }
    }

    #[test]
    fn test_cycle_adjusted_ma_invalid_max_period() {
        let result = CycleAdjustedMA::new(10, 20, 10, 10);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "max_period");
        }
    }

    #[test]
    fn test_cycle_adjusted_ma_calculate() {
        let cama = CycleAdjustedMA::new(10, 20, 5, 30).unwrap();
        let data = make_trending_data();
        let result = cama.calculate(&data);

        assert_eq!(result.len(), data.len());
        // Values after min_periods should be valid
        for i in 20..50 {
            assert!(!result[i].is_nan());
        }
    }

    #[test]
    fn test_cycle_adjusted_ma_trait() {
        let cama = CycleAdjustedMA::new(10, 20, 5, 30).unwrap();
        assert_eq!(cama.name(), "Cycle Adjusted MA");
        assert_eq!(cama.min_periods(), 20);
    }

    #[test]
    fn test_cycle_adjusted_ma_compute() {
        let cama = CycleAdjustedMA::new(10, 20, 5, 30).unwrap();
        let data = make_ohlcv_data();
        let result = cama.compute(&data);
        assert!(result.is_ok());
    }

    // ========================================================================
    // AdaptiveLagMA Tests
    // ========================================================================

    #[test]
    fn test_adaptive_lag_ma_new_valid() {
        let alma = AdaptiveLagMA::new(10, 10, 0.2, 0.8);
        assert!(alma.is_ok());
    }

    #[test]
    fn test_adaptive_lag_ma_invalid_period() {
        let result = AdaptiveLagMA::new(1, 10, 0.2, 0.8);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "period");
        }
    }

    #[test]
    fn test_adaptive_lag_ma_invalid_volatility_period() {
        let result = AdaptiveLagMA::new(10, 1, 0.2, 0.8);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "volatility_period");
        }
    }

    #[test]
    fn test_adaptive_lag_ma_invalid_min_lag() {
        let result = AdaptiveLagMA::new(10, 10, -0.1, 0.8);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "min_lag");
        }
    }

    #[test]
    fn test_adaptive_lag_ma_invalid_max_lag() {
        let result = AdaptiveLagMA::new(10, 10, 0.5, 0.3);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "max_lag");
        }
    }

    #[test]
    fn test_adaptive_lag_ma_calculate() {
        let alma = AdaptiveLagMA::new(10, 10, 0.2, 0.8).unwrap();
        let data = make_trending_data();
        let result = alma.calculate(&data);

        assert_eq!(result.len(), data.len());
        // Values after min_periods should be valid
        for i in 10..50 {
            assert!(!result[i].is_nan());
        }
    }

    #[test]
    fn test_adaptive_lag_ma_trait() {
        let alma = AdaptiveLagMA::new(10, 15, 0.2, 0.8).unwrap();
        assert_eq!(alma.name(), "Adaptive Lag MA");
        assert_eq!(alma.min_periods(), 15);
    }

    #[test]
    fn test_adaptive_lag_ma_compute() {
        let alma = AdaptiveLagMA::new(10, 10, 0.2, 0.8).unwrap();
        let data = make_ohlcv_data();
        let result = alma.compute(&data);
        assert!(result.is_ok());
    }
}
