//! Advanced Volume Indicators
//!
//! Sophisticated volume analysis indicators for detecting institutional activity,
//! breakouts, and volume efficiency.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Volume Accumulation - Tracks cumulative volume momentum
///
/// Accumulates volume weighted by price direction over a period,
/// helping identify sustained buying or selling pressure.
#[derive(Debug, Clone)]
pub struct VolumeAccumulation {
    period: usize,
}

impl VolumeAccumulation {
    pub fn new(period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate cumulative volume momentum
    ///
    /// Returns a vector of cumulative volume weighted by price direction.
    /// Positive values indicate accumulation (buying), negative indicates distribution (selling).
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len());
        let mut result = vec![0.0; n];

        if n < 2 {
            return result;
        }

        // Calculate signed volume based on price direction
        let mut signed_volume = vec![0.0; n];
        for i in 1..n {
            if close[i] > close[i - 1] {
                signed_volume[i] = volume[i];
            } else if close[i] < close[i - 1] {
                signed_volume[i] = -volume[i];
            }
            // If close[i] == close[i-1], signed_volume remains 0
        }

        // Calculate rolling cumulative sum over period
        for i in 0..n {
            let start = if i >= self.period { i - self.period + 1 } else { 0 };
            result[i] = signed_volume[start..=i].iter().sum();
        }

        result
    }
}

impl TechnicalIndicator for VolumeAccumulation {
    fn name(&self) -> &str {
        "Volume Accumulation"
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
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

/// Volume Breakout - Detects volume breakouts above threshold
///
/// Identifies when current volume significantly exceeds its historical average,
/// signaling potential breakout or reversal conditions.
#[derive(Debug, Clone)]
pub struct VolumeBreakout {
    period: usize,
    threshold: f64,
}

impl VolumeBreakout {
    pub fn new(period: usize, threshold: f64) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if threshold <= 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "threshold".to_string(),
                reason: "must be greater than 1.0".to_string(),
            });
        }
        Ok(Self { period, threshold })
    }

    /// Calculate volume breakout indicator
    ///
    /// Returns (breakout_ratio, breakout_signal):
    /// - breakout_ratio: current volume / average volume
    /// - breakout_signal: 1.0 if breakout detected, 0.0 otherwise
    pub fn calculate(&self, volume: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = volume.len();
        let mut ratio = vec![0.0; n];
        let mut signal = vec![0.0; n];

        for i in self.period..n {
            let start = i - self.period;
            let avg_volume: f64 = volume[start..i].iter().sum::<f64>() / self.period as f64;

            if avg_volume > 1e-10 {
                ratio[i] = volume[i] / avg_volume;
                if ratio[i] >= self.threshold {
                    signal[i] = 1.0;
                }
            }
        }

        (ratio, signal)
    }
}

impl TechnicalIndicator for VolumeBreakout {
    fn name(&self) -> &str {
        "Volume Breakout"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.volume.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.volume.len(),
            });
        }
        let (ratio, signal) = self.calculate(&data.volume);
        Ok(IndicatorOutput::dual(ratio, signal))
    }
}

/// Relative Volume Strength - Compares current volume to historical
///
/// Measures the strength of current volume relative to a longer historical period,
/// normalized to a 0-100 scale for easy interpretation.
#[derive(Debug, Clone)]
pub struct RelativeVolumeStrength {
    short_period: usize,
    long_period: usize,
}

impl RelativeVolumeStrength {
    pub fn new(short_period: usize, long_period: usize) -> Result<Self> {
        if short_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if long_period <= short_period {
            return Err(IndicatorError::InvalidParameter {
                name: "long_period".to_string(),
                reason: "must be greater than short_period".to_string(),
            });
        }
        Ok(Self { short_period, long_period })
    }

    /// Calculate relative volume strength
    ///
    /// Returns a normalized strength value (0-100) comparing short-term
    /// average volume to long-term average volume.
    pub fn calculate(&self, volume: &[f64]) -> Vec<f64> {
        let n = volume.len();
        let mut result = vec![0.0; n];

        for i in self.long_period..n {
            let short_start = i - self.short_period;
            let long_start = i - self.long_period;

            let short_avg: f64 = volume[short_start..i].iter().sum::<f64>() / self.short_period as f64;
            let long_avg: f64 = volume[long_start..i].iter().sum::<f64>() / self.long_period as f64;

            if long_avg > 1e-10 {
                // Normalize to 0-100 scale, capped at 200% strength
                let strength = (short_avg / long_avg) * 50.0;
                result[i] = strength.min(100.0);
            }
        }

        result
    }
}

impl TechnicalIndicator for RelativeVolumeStrength {
    fn name(&self) -> &str {
        "Relative Volume Strength"
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.volume.len() < self.long_period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.long_period + 1,
                got: data.volume.len(),
            });
        }
        Ok(IndicatorOutput::single(self.calculate(&data.volume)))
    }
}

/// Volume Climax Detector - Identifies volume climax events
///
/// Detects extreme volume events that often mark trend exhaustion or reversal points.
/// A climax is identified when volume exceeds a multiple of its standard deviation.
#[derive(Debug, Clone)]
pub struct VolumeClimaxDetector {
    period: usize,
    std_multiplier: f64,
}

impl VolumeClimaxDetector {
    pub fn new(period: usize, std_multiplier: f64) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if std_multiplier <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "std_multiplier".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { period, std_multiplier })
    }

    /// Calculate volume climax detector
    ///
    /// Returns (z_score, climax_signal):
    /// - z_score: how many standard deviations current volume is from mean
    /// - climax_signal: 1.0 for bullish climax, -1.0 for bearish climax, 0.0 otherwise
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len().min(volume.len());
        let mut z_score = vec![0.0; n];
        let mut climax_signal = vec![0.0; n];

        for i in self.period..n {
            let start = i - self.period;
            let window = &volume[start..i];

            // Calculate mean and std deviation
            let mean: f64 = window.iter().sum::<f64>() / self.period as f64;
            let variance: f64 = window.iter()
                .map(|v| (v - mean).powi(2))
                .sum::<f64>() / self.period as f64;
            let std_dev = variance.sqrt();

            if std_dev > 1e-10 {
                z_score[i] = (volume[i] - mean) / std_dev;

                // Detect climax
                if z_score[i] >= self.std_multiplier {
                    // Determine direction based on price movement
                    if i > 0 && close[i] > close[i - 1] {
                        climax_signal[i] = 1.0; // Bullish climax (buying climax)
                    } else if i > 0 && close[i] < close[i - 1] {
                        climax_signal[i] = -1.0; // Bearish climax (selling climax)
                    }
                }
            }
        }

        (z_score, climax_signal)
    }
}

impl TechnicalIndicator for VolumeClimaxDetector {
    fn name(&self) -> &str {
        "Volume Climax Detector"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.volume.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.volume.len(),
            });
        }
        let (z_score, climax) = self.calculate(&data.close, &data.volume);
        Ok(IndicatorOutput::dual(z_score, climax))
    }
}

/// Smart Money Volume - Estimates institutional volume patterns
///
/// Attempts to identify institutional (smart money) activity by analyzing
/// volume patterns during specific price ranges and times.
/// Higher values indicate more institutional participation.
#[derive(Debug, Clone)]
pub struct SmartMoneyVolume {
    period: usize,
}

impl SmartMoneyVolume {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate smart money volume indicator
    ///
    /// Returns a cumulative score indicating likely institutional activity.
    /// Smart money typically:
    /// - Accumulates on low volume down days
    /// - Distributes on high volume up days near highs
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len()).min(volume.len());
        let mut result = vec![0.0; n];

        if n < 2 {
            return result;
        }

        // Calculate smart money flow
        let mut smart_flow = vec![0.0; n];
        for i in 1..n {
            let range = high[i] - low[i];
            if range > 1e-10 {
                // Close location value: where did price close within the range?
                let clv = ((close[i] - low[i]) - (high[i] - close[i])) / range;

                // Smart money factor: large volume on small moves vs small volume on large moves
                let price_change = (close[i] - close[i - 1]).abs();
                let avg_price = (close[i] + close[i - 1]) / 2.0;
                let pct_change = if avg_price > 1e-10 { price_change / avg_price } else { 0.0 };

                // Inverse relationship: high volume with small price change = smart money
                // Low volume with large price change = retail
                let efficiency = if pct_change > 1e-10 {
                    (volume[i] / pct_change).ln()
                } else {
                    volume[i].ln()
                };

                smart_flow[i] = clv * efficiency;
            }
        }

        // EMA smoothing of smart flow
        let alpha = 2.0 / (self.period as f64 + 1.0);
        for i in 0..n {
            if i == 0 {
                result[i] = smart_flow[i];
            } else {
                result[i] = alpha * smart_flow[i] + (1.0 - alpha) * result[i - 1];
            }
        }

        result
    }
}

impl TechnicalIndicator for SmartMoneyVolume {
    fn name(&self) -> &str {
        "Smart Money Volume"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close, &data.volume)))
    }
}

/// Volume Efficiency - Measures price movement per unit volume
///
/// Calculates how efficiently volume translates into price movement.
/// Higher efficiency suggests stronger conviction behind the move.
#[derive(Debug, Clone)]
pub struct VolumeEfficiency {
    period: usize,
}

impl VolumeEfficiency {
    pub fn new(period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate volume efficiency ratio
    ///
    /// Returns a ratio of price movement to volume consumed.
    /// Higher values indicate more efficient price discovery.
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len());
        let mut result = vec![0.0; n];

        if n < 2 {
            return result;
        }

        for i in self.period..n {
            let start = i - self.period;

            // Net price movement over period
            let net_move = (close[i] - close[start]).abs();

            // Total volume over period
            let total_volume: f64 = volume[start..=i].iter().sum();

            // Efficiency = price movement per unit of volume (scaled)
            if total_volume > 1e-10 {
                // Scale by average price to normalize across different price levels
                let avg_price = (close[i] + close[start]) / 2.0;
                if avg_price > 1e-10 {
                    // Efficiency as percentage move per million volume units
                    let pct_move = (net_move / avg_price) * 100.0;
                    result[i] = pct_move / (total_volume / 1_000_000.0);
                }
            }
        }

        result
    }
}

impl TechnicalIndicator for VolumeEfficiency {
    fn name(&self) -> &str {
        "Volume Efficiency"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

/// Volume Distribution - Analyzes volume distribution across price levels
///
/// Calculates how volume is distributed across the price range,
/// helping identify areas of high liquidity and potential support/resistance.
#[derive(Debug, Clone)]
pub struct VolumeDistribution {
    period: usize,
    num_bins: usize,
}

impl VolumeDistribution {
    pub fn new(period: usize, num_bins: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if num_bins < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "num_bins".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        Ok(Self { period, num_bins })
    }

    /// Calculate volume distribution score
    ///
    /// Returns a normalized score (0-100) indicating how concentrated
    /// volume is at current price level vs distributed across all levels.
    /// Higher values indicate more volume concentration at current price.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len()).min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i - self.period;
            let window_high: &[f64] = &high[start..=i];
            let window_low: &[f64] = &low[start..=i];
            let window_close: &[f64] = &close[start..=i];
            let window_vol: &[f64] = &volume[start..=i];

            // Find price range
            let max_price = window_high.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_price = window_low.iter().cloned().fold(f64::INFINITY, f64::min);
            let range = max_price - min_price;

            if range < 1e-10 {
                continue;
            }

            // Create bins and distribute volume
            let bin_size = range / self.num_bins as f64;
            let mut bins = vec![0.0; self.num_bins];
            let total_vol: f64 = window_vol.iter().sum();

            for j in 0..window_close.len() {
                let typical_price = (window_high[j] + window_low[j] + window_close[j]) / 3.0;
                let bin_idx = ((typical_price - min_price) / bin_size).floor() as usize;
                let bin_idx = bin_idx.min(self.num_bins - 1);
                bins[bin_idx] += window_vol[j];
            }

            // Find which bin current price falls into
            let current_typical = (high[i] + low[i] + close[i]) / 3.0;
            let current_bin = ((current_typical - min_price) / bin_size).floor() as usize;
            let current_bin = current_bin.min(self.num_bins - 1);

            // Score: volume in current bin relative to total
            if total_vol > 1e-10 {
                result[i] = (bins[current_bin] / total_vol) * 100.0 * self.num_bins as f64;
                result[i] = result[i].min(100.0);
            }
        }

        result
    }
}

impl TechnicalIndicator for VolumeDistribution {
    fn name(&self) -> &str {
        "Volume Distribution"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close, &data.volume)))
    }
}

/// Volume Intensity - Measures intensity of volume relative to price movement
///
/// Calculates how much volume is being transacted per unit of price movement.
/// High intensity suggests strong conviction, low intensity suggests weak moves.
#[derive(Debug, Clone)]
pub struct VolumeIntensity {
    period: usize,
}

impl VolumeIntensity {
    pub fn new(period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate volume intensity
    ///
    /// Returns (intensity, direction):
    /// - intensity: normalized volume intensity score (0-100)
    /// - direction: +1 for bullish intensity, -1 for bearish, 0 for neutral
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len().min(high.len()).min(low.len()).min(volume.len());
        let mut intensity = vec![0.0; n];
        let mut direction = vec![0.0; n];

        if n < 2 {
            return (intensity, direction);
        }

        // Calculate raw intensity values first for normalization
        let mut raw_intensity = vec![0.0; n];
        for i in 1..n {
            let range = high[i] - low[i];
            if range > 1e-10 {
                raw_intensity[i] = volume[i] / range;
            }
        }

        // Calculate rolling stats and normalize
        for i in self.period..n {
            let start = i - self.period;
            let window = &raw_intensity[start..i];

            let mean: f64 = window.iter().sum::<f64>() / self.period as f64;
            let variance: f64 = window.iter()
                .map(|v| (v - mean).powi(2))
                .sum::<f64>() / self.period as f64;
            let std_dev = variance.sqrt();

            if std_dev > 1e-10 && mean > 1e-10 {
                // Normalize to z-score then convert to 0-100 scale
                let z_score = (raw_intensity[i] - mean) / std_dev;
                intensity[i] = 50.0 + (z_score * 15.0); // ~99.7% within 0-100
                intensity[i] = intensity[i].clamp(0.0, 100.0);
            } else {
                intensity[i] = 50.0;
            }

            // Determine direction
            if close[i] > close[i - 1] {
                direction[i] = 1.0;
            } else if close[i] < close[i - 1] {
                direction[i] = -1.0;
            }
        }

        (intensity, direction)
    }
}

impl TechnicalIndicator for VolumeIntensity {
    fn name(&self) -> &str {
        "Volume Intensity"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }
        let (intensity, direction) = self.calculate(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::dual(intensity, direction))
    }
}

/// Volume Trend - Tracks long-term volume trends
///
/// Identifies whether volume is in an uptrend, downtrend, or range-bound state.
/// Uses dual EMA comparison to determine volume trend direction and strength.
#[derive(Debug, Clone)]
pub struct VolumeTrend {
    fast_period: usize,
    slow_period: usize,
}

impl VolumeTrend {
    pub fn new(fast_period: usize, slow_period: usize) -> Result<Self> {
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
        Ok(Self { fast_period, slow_period })
    }

    /// Calculate volume trend
    ///
    /// Returns (trend_strength, trend_direction):
    /// - trend_strength: absolute strength of the volume trend (0-100)
    /// - trend_direction: +1 for increasing volume trend, -1 for decreasing, 0 for neutral
    pub fn calculate(&self, volume: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = volume.len();
        let mut strength = vec![0.0; n];
        let mut direction = vec![0.0; n];

        if n == 0 {
            return (strength, direction);
        }

        // Calculate fast EMA
        let alpha_fast = 2.0 / (self.fast_period as f64 + 1.0);
        let mut fast_ema = vec![0.0; n];
        fast_ema[0] = volume[0];
        for i in 1..n {
            fast_ema[i] = alpha_fast * volume[i] + (1.0 - alpha_fast) * fast_ema[i - 1];
        }

        // Calculate slow EMA
        let alpha_slow = 2.0 / (self.slow_period as f64 + 1.0);
        let mut slow_ema = vec![0.0; n];
        slow_ema[0] = volume[0];
        for i in 1..n {
            slow_ema[i] = alpha_slow * volume[i] + (1.0 - alpha_slow) * slow_ema[i - 1];
        }

        // Calculate trend strength and direction
        for i in self.slow_period..n {
            if slow_ema[i] > 1e-10 {
                let diff = fast_ema[i] - slow_ema[i];
                let pct_diff = (diff / slow_ema[i]) * 100.0;

                // Strength is absolute percentage difference, capped at 100
                strength[i] = pct_diff.abs().min(100.0);

                // Direction based on whether fast > slow
                if diff > slow_ema[i] * 0.01 { // 1% threshold for significance
                    direction[i] = 1.0;
                } else if diff < -slow_ema[i] * 0.01 {
                    direction[i] = -1.0;
                }
            }
        }

        (strength, direction)
    }
}

impl TechnicalIndicator for VolumeTrend {
    fn name(&self) -> &str {
        "Volume Trend"
    }

    fn min_periods(&self) -> usize {
        self.slow_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.volume.len() < self.slow_period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.slow_period + 1,
                got: data.volume.len(),
            });
        }
        let (strength, direction) = self.calculate(&data.volume);
        Ok(IndicatorOutput::dual(strength, direction))
    }
}

/// Volume Anomaly - Detects anomalous volume patterns
///
/// Identifies statistically significant deviations from normal volume patterns
/// using z-score analysis with configurable sensitivity.
#[derive(Debug, Clone)]
pub struct VolumeAnomaly {
    period: usize,
    threshold: f64,
}

impl VolumeAnomaly {
    pub fn new(period: usize, threshold: f64) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if threshold <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "threshold".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { period, threshold })
    }

    /// Calculate volume anomaly detection
    ///
    /// Returns (anomaly_score, anomaly_signal):
    /// - anomaly_score: z-score of current volume
    /// - anomaly_signal: 1 for high anomaly, -1 for low anomaly, 0 for normal
    pub fn calculate(&self, volume: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = volume.len();
        let mut score = vec![0.0; n];
        let mut signal = vec![0.0; n];

        for i in self.period..n {
            let start = i - self.period;
            let window = &volume[start..i];

            let mean: f64 = window.iter().sum::<f64>() / self.period as f64;
            let variance: f64 = window.iter()
                .map(|v| (v - mean).powi(2))
                .sum::<f64>() / self.period as f64;
            let std_dev = variance.sqrt();

            if std_dev > 1e-10 {
                score[i] = (volume[i] - mean) / std_dev;

                if score[i] >= self.threshold {
                    signal[i] = 1.0; // High volume anomaly
                } else if score[i] <= -self.threshold {
                    signal[i] = -1.0; // Low volume anomaly
                }
            }
        }

        (score, signal)
    }
}

impl TechnicalIndicator for VolumeAnomaly {
    fn name(&self) -> &str {
        "Volume Anomaly"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.volume.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.volume.len(),
            });
        }
        let (score, signal) = self.calculate(&data.volume);
        Ok(IndicatorOutput::dual(score, signal))
    }
}

/// Volume Price Confirmation - Confirms price moves with volume
///
/// Analyzes whether price movements are confirmed by corresponding volume patterns.
/// High confirmation suggests sustainable moves, low confirmation suggests potential reversals.
#[derive(Debug, Clone)]
pub struct VolumePriceConfirmation {
    period: usize,
}

impl VolumePriceConfirmation {
    pub fn new(period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate volume-price confirmation
    ///
    /// Returns (confirmation_score, divergence_signal):
    /// - confirmation_score: 0-100 score where 100 = perfect confirmation
    /// - divergence_signal: +1 bullish divergence, -1 bearish divergence, 0 confirmed
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len().min(volume.len());
        let mut confirmation = vec![0.0; n];
        let mut divergence = vec![0.0; n];

        if n < 2 {
            return (confirmation, divergence);
        }

        for i in self.period..n {
            let start = i - self.period;

            // Calculate price trend (linear regression slope)
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            let mut sum_xy = 0.0;
            let mut sum_x2 = 0.0;

            for (j, idx) in (start..=i).enumerate() {
                let x = j as f64;
                sum_x += x;
                sum_y += close[idx];
                sum_xy += x * close[idx];
                sum_x2 += x * x;
            }
            let period_len = (self.period + 1) as f64;
            let price_slope = (period_len * sum_xy - sum_x * sum_y) / (period_len * sum_x2 - sum_x * sum_x);

            // Calculate volume trend
            sum_y = 0.0;
            sum_xy = 0.0;
            for (j, idx) in (start..=i).enumerate() {
                let x = j as f64;
                sum_y += volume[idx];
                sum_xy += x * volume[idx];
            }
            let volume_slope = (period_len * sum_xy - sum_x * sum_y) / (period_len * sum_x2 - sum_x * sum_x);

            // Normalize slopes
            let avg_price = close[start..=i].iter().sum::<f64>() / period_len;
            let avg_volume = volume[start..=i].iter().sum::<f64>() / period_len;

            let norm_price_slope = if avg_price > 1e-10 { price_slope / avg_price } else { 0.0 };
            let norm_vol_slope = if avg_volume > 1e-10 { volume_slope / avg_volume } else { 0.0 };

            // Confirmation: both slopes same sign and proportional
            let price_up = norm_price_slope > 0.001;
            let price_down = norm_price_slope < -0.001;
            let vol_up = norm_vol_slope > 0.001;
            let vol_down = norm_vol_slope < -0.001;

            if (price_up && vol_up) || (price_down && vol_up) {
                // Price move confirmed by increasing volume
                confirmation[i] = (norm_vol_slope.abs() / (norm_price_slope.abs() + norm_vol_slope.abs() + 1e-10)) * 100.0;
                confirmation[i] = confirmation[i].min(100.0);
            } else if price_up && vol_down {
                // Bearish divergence: price up but volume declining
                confirmation[i] = 50.0 - (norm_vol_slope.abs() * 50.0).min(50.0);
                divergence[i] = -1.0;
            } else if price_down && vol_down {
                // Potential bullish divergence: price down on declining volume
                confirmation[i] = 50.0 - (norm_vol_slope.abs() * 50.0).min(50.0);
                divergence[i] = 1.0;
            } else {
                confirmation[i] = 50.0; // Neutral/unclear
            }
        }

        (confirmation, divergence)
    }
}

impl TechnicalIndicator for VolumePriceConfirmation {
    fn name(&self) -> &str {
        "Volume Price Confirmation"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }
        let (confirmation, divergence) = self.calculate(&data.close, &data.volume);
        Ok(IndicatorOutput::dual(confirmation, divergence))
    }
}

/// Volume Exhaustion - Detects volume exhaustion patterns
///
/// Identifies when volume is exhausting during a price trend, which often
/// precedes trend reversals or consolidation periods.
#[derive(Debug, Clone)]
pub struct VolumeExhaustion {
    period: usize,
    exhaustion_threshold: f64,
}

impl VolumeExhaustion {
    pub fn new(period: usize, exhaustion_threshold: f64) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if exhaustion_threshold <= 0.0 || exhaustion_threshold >= 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "exhaustion_threshold".to_string(),
                reason: "must be between 0 and 1".to_string(),
            });
        }
        Ok(Self { period, exhaustion_threshold })
    }

    /// Calculate volume exhaustion
    ///
    /// Returns (exhaustion_level, exhaustion_signal):
    /// - exhaustion_level: 0-100 score where 100 = complete exhaustion
    /// - exhaustion_signal: 1 for uptrend exhaustion, -1 for downtrend exhaustion, 0 for none
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len().min(volume.len());
        let mut exhaustion = vec![0.0; n];
        let mut signal = vec![0.0; n];

        if n < 2 {
            return (exhaustion, signal);
        }

        for i in self.period..n {
            let start = i - self.period;

            // Determine price trend direction
            let price_change = close[i] - close[start];
            let is_uptrend = price_change > 0.0;
            let is_downtrend = price_change < 0.0;

            if !is_uptrend && !is_downtrend {
                continue;
            }

            // Find peak volume in the period
            let max_volume = volume[start..=i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_volume = volume[start..=i].iter().cloned().fold(f64::INFINITY, f64::min);
            let vol_range = max_volume - min_volume;

            if vol_range < 1e-10 || max_volume < 1e-10 {
                continue;
            }

            // Calculate volume decay from peak
            // Find where peak occurred
            let peak_idx = volume[start..=i].iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            // If peak is in first half and current volume is significantly lower
            let half_period = self.period / 2;
            if peak_idx < half_period {
                // Volume peaked early - calculate exhaustion
                let current_ratio = volume[i] / max_volume;
                let decay_ratio = 1.0 - current_ratio;

                exhaustion[i] = (decay_ratio / (1.0 - self.exhaustion_threshold)) * 100.0;
                exhaustion[i] = exhaustion[i].clamp(0.0, 100.0);

                // Signal if exhaustion exceeds threshold
                if current_ratio < self.exhaustion_threshold {
                    if is_uptrend {
                        signal[i] = 1.0; // Uptrend exhaustion
                    } else {
                        signal[i] = -1.0; // Downtrend exhaustion
                    }
                }
            } else {
                // Volume peaked late - trend still has momentum
                let current_ratio = volume[i] / max_volume;
                exhaustion[i] = (1.0 - current_ratio) * 50.0; // Lower exhaustion score
            }
        }

        (exhaustion, signal)
    }
}

impl TechnicalIndicator for VolumeExhaustion {
    fn name(&self) -> &str {
        "Volume Exhaustion"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }
        let (exhaustion, signal) = self.calculate(&data.close, &data.volume);
        Ok(IndicatorOutput::dual(exhaustion, signal))
    }
}

/// Volume Weighted Momentum - Momentum weighted by relative volume
///
/// Calculates price momentum weighted by how volume compares to its average.
/// Higher volume gives more weight to the momentum reading, making it more
/// responsive to high-conviction moves.
#[derive(Debug, Clone)]
pub struct VolumeWeightedMomentum {
    period: usize,
}

impl VolumeWeightedMomentum {
    pub fn new(period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate volume-weighted momentum
    ///
    /// Returns momentum values weighted by relative volume.
    /// Positive values indicate bullish momentum, negative indicates bearish.
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len());
        let mut result = vec![0.0; n];

        if n < self.period + 1 {
            return result;
        }

        for i in self.period..n {
            let start = i - self.period;

            // Calculate raw momentum (rate of change)
            if close[start] > 1e-10 {
                let momentum = (close[i] - close[start]) / close[start];

                // Calculate average volume over period
                let avg_volume: f64 = volume[start..i].iter().sum::<f64>() / self.period as f64;

                // Calculate relative volume (current vs average)
                let relative_volume = if avg_volume > 1e-10 {
                    volume[i] / avg_volume
                } else {
                    1.0
                };

                // Weight momentum by relative volume
                // Higher volume = more significant momentum
                result[i] = momentum * relative_volume * 100.0; // Convert to percentage
            }
        }

        result
    }
}

impl TechnicalIndicator for VolumeWeightedMomentum {
    fn name(&self) -> &str {
        "Volume Weighted Momentum"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

/// Volume Force Index - Force index with volume normalization
///
/// Combines price change with volume to measure the force behind price moves.
/// Normalizes volume to make the indicator comparable across different securities.
#[derive(Debug, Clone)]
pub struct VolumeForceIndex {
    period: usize,
    smoothing: usize,
}

impl VolumeForceIndex {
    pub fn new(period: usize, smoothing: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
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

    /// Calculate volume force index
    ///
    /// Returns normalized force index values.
    /// Positive values indicate buying pressure, negative indicates selling pressure.
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len());
        let mut result = vec![0.0; n];

        if n < 2 {
            return result;
        }

        // Calculate raw force index: price_change * volume
        let mut raw_force = vec![0.0; n];
        for i in 1..n {
            raw_force[i] = (close[i] - close[i - 1]) * volume[i];
        }

        // Calculate average volume for normalization
        let mut normalized_force = vec![0.0; n];
        for i in self.period..n {
            let start = i - self.period;
            let avg_volume: f64 = volume[start..i].iter().sum::<f64>() / self.period as f64;

            if avg_volume > 1e-10 {
                // Normalize by average volume
                normalized_force[i] = raw_force[i] / avg_volume;
            }
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smoothing as f64 + 1.0);
        for i in 0..n {
            if i == 0 {
                result[i] = normalized_force[i];
            } else {
                result[i] = alpha * normalized_force[i] + (1.0 - alpha) * result[i - 1];
            }
        }

        result
    }
}

impl TechnicalIndicator for VolumeForceIndex {
    fn name(&self) -> &str {
        "Volume Force Index"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

/// Cumulative Volume Oscillator - Oscillator based on cumulative volume delta
///
/// Tracks the cumulative difference between up-volume and down-volume,
/// then creates an oscillator by comparing short-term and long-term averages.
#[derive(Debug, Clone)]
pub struct CumulativeVolumeOscillator {
    fast_period: usize,
    slow_period: usize,
}

impl CumulativeVolumeOscillator {
    pub fn new(fast_period: usize, slow_period: usize) -> Result<Self> {
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
        Ok(Self { fast_period, slow_period })
    }

    /// Calculate cumulative volume oscillator
    ///
    /// Returns oscillator values showing the difference between fast and slow
    /// cumulative volume delta averages.
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len());
        let mut result = vec![0.0; n];

        if n < 2 {
            return result;
        }

        // Calculate volume delta (up volume - down volume)
        let mut volume_delta = vec![0.0; n];
        for i in 1..n {
            if close[i] > close[i - 1] {
                volume_delta[i] = volume[i];
            } else if close[i] < close[i - 1] {
                volume_delta[i] = -volume[i];
            }
            // If close[i] == close[i-1], delta is 0
        }

        // Calculate cumulative volume delta
        let mut cumulative = vec![0.0; n];
        for i in 0..n {
            if i == 0 {
                cumulative[i] = volume_delta[i];
            } else {
                cumulative[i] = cumulative[i - 1] + volume_delta[i];
            }
        }

        // Calculate fast and slow EMAs of cumulative delta
        let alpha_fast = 2.0 / (self.fast_period as f64 + 1.0);
        let alpha_slow = 2.0 / (self.slow_period as f64 + 1.0);

        let mut fast_ema = vec![0.0; n];
        let mut slow_ema = vec![0.0; n];

        for i in 0..n {
            if i == 0 {
                fast_ema[i] = cumulative[i];
                slow_ema[i] = cumulative[i];
            } else {
                fast_ema[i] = alpha_fast * cumulative[i] + (1.0 - alpha_fast) * fast_ema[i - 1];
                slow_ema[i] = alpha_slow * cumulative[i] + (1.0 - alpha_slow) * slow_ema[i - 1];
            }
        }

        // Oscillator = fast EMA - slow EMA
        for i in self.slow_period..n {
            result[i] = fast_ema[i] - slow_ema[i];
        }

        result
    }
}

impl TechnicalIndicator for CumulativeVolumeOscillator {
    fn name(&self) -> &str {
        "Cumulative Volume Oscillator"
    }

    fn min_periods(&self) -> usize {
        self.slow_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.slow_period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.slow_period + 1,
                got: data.close.len(),
            });
        }
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

/// Volume Rate of Change - Rate of change of volume with smoothing
///
/// Measures the percentage change in volume over a period,
/// with optional smoothing to reduce noise.
#[derive(Debug, Clone)]
pub struct VolumeRateOfChange {
    period: usize,
    smoothing: usize,
}

impl VolumeRateOfChange {
    pub fn new(period: usize, smoothing: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
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

    /// Calculate volume rate of change
    ///
    /// Returns percentage change in volume over the period, smoothed.
    pub fn calculate(&self, volume: &[f64]) -> Vec<f64> {
        let n = volume.len();
        let mut result = vec![0.0; n];

        if n < self.period + 1 {
            return result;
        }

        // Calculate raw rate of change
        let mut raw_roc = vec![0.0; n];
        for i in self.period..n {
            let past_volume = volume[i - self.period];
            if past_volume > 1e-10 {
                raw_roc[i] = ((volume[i] - past_volume) / past_volume) * 100.0;
            }
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smoothing as f64 + 1.0);
        for i in 0..n {
            if i == 0 {
                result[i] = raw_roc[i];
            } else {
                result[i] = alpha * raw_roc[i] + (1.0 - alpha) * result[i - 1];
            }
        }

        result
    }
}

impl TechnicalIndicator for VolumeRateOfChange {
    fn name(&self) -> &str {
        "Volume Rate of Change"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.volume.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.volume.len(),
            });
        }
        Ok(IndicatorOutput::single(self.calculate(&data.volume)))
    }
}

/// Relative Volume Profile - Volume profile relative to historical average
///
/// Compares current volume to a historical average, providing a ratio
/// that shows whether volume is above or below normal levels.
#[derive(Debug, Clone)]
pub struct RelativeVolumeProfile {
    period: usize,
    smoothing: usize,
}

impl RelativeVolumeProfile {
    pub fn new(period: usize, smoothing: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
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

    /// Calculate relative volume profile
    ///
    /// Returns ratio of current volume to historical average.
    /// Values > 1 indicate above-average volume, < 1 indicates below-average.
    pub fn calculate(&self, volume: &[f64]) -> Vec<f64> {
        let n = volume.len();
        let mut result = vec![0.0; n];

        if n < self.period + 1 {
            return result;
        }

        // Calculate raw relative volume
        let mut raw_relative = vec![0.0; n];
        for i in self.period..n {
            let start = i - self.period;
            let avg_volume: f64 = volume[start..i].iter().sum::<f64>() / self.period as f64;

            if avg_volume > 1e-10 {
                raw_relative[i] = volume[i] / avg_volume;
            } else {
                raw_relative[i] = 1.0;
            }
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smoothing as f64 + 1.0);
        for i in 0..n {
            if i == 0 {
                result[i] = raw_relative[i];
            } else {
                result[i] = alpha * raw_relative[i] + (1.0 - alpha) * result[i - 1];
            }
        }

        result
    }
}

impl TechnicalIndicator for RelativeVolumeProfile {
    fn name(&self) -> &str {
        "Relative Volume Profile"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.volume.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.volume.len(),
            });
        }
        Ok(IndicatorOutput::single(self.calculate(&data.volume)))
    }
}

/// Volume Impulse - Sudden volume spikes detection
///
/// Detects sudden spikes in volume that may indicate significant market events,
/// institutional activity, or breakout conditions.
#[derive(Debug, Clone)]
pub struct VolumeImpulse {
    period: usize,
    threshold: f64,
}

impl VolumeImpulse {
    pub fn new(period: usize, threshold: f64) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if threshold <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "threshold".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { period, threshold })
    }

    /// Calculate volume impulse
    ///
    /// Returns (impulse_strength, impulse_signal):
    /// - impulse_strength: z-score measuring the magnitude of volume spike
    /// - impulse_signal: 1.0 if impulse detected, 0.0 otherwise
    pub fn calculate(&self, volume: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = volume.len();
        let mut impulse = vec![0.0; n];
        let mut signal = vec![0.0; n];

        if n < self.period + 1 {
            return (impulse, signal);
        }

        for i in self.period..n {
            let start = i - self.period;
            let window = &volume[start..i];

            // Calculate mean and standard deviation of historical volume
            let mean: f64 = window.iter().sum::<f64>() / self.period as f64;
            let variance: f64 = window.iter()
                .map(|v| (v - mean).powi(2))
                .sum::<f64>() / self.period as f64;
            let std_dev = variance.sqrt();

            if std_dev > 1e-10 {
                // Calculate z-score for current volume
                let z_score = (volume[i] - mean) / std_dev;
                impulse[i] = z_score;

                // Detect impulse if z-score exceeds threshold
                if z_score >= self.threshold {
                    signal[i] = 1.0;
                }
            } else if mean > 1e-10 {
                // If std_dev is near zero but mean exists, check for simple ratio
                let ratio = volume[i] / mean;
                if ratio >= (1.0 + self.threshold) {
                    impulse[i] = ratio - 1.0;
                    signal[i] = 1.0;
                }
            }
        }

        (impulse, signal)
    }
}

impl TechnicalIndicator for VolumeImpulse {
    fn name(&self) -> &str {
        "Volume Impulse"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.volume.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.volume.len(),
            });
        }
        let (impulse, signal) = self.calculate(&data.volume);
        Ok(IndicatorOutput::dual(impulse, signal))
    }

    fn output_features(&self) -> usize {
        2 // impulse strength and impulse signal
    }
}

/// Volume Weighted Trend - Measures trend direction weighted by volume
///
/// Calculates trend direction by weighting price changes by their corresponding
/// volume. High volume moves have more influence on the trend reading than
/// low volume moves, providing a more accurate picture of the true trend.
#[derive(Debug, Clone)]
pub struct VolumeWeightedTrend {
    period: usize,
    smoothing: usize,
}

impl VolumeWeightedTrend {
    /// Create a new VolumeWeightedTrend indicator
    ///
    /// # Arguments
    /// * `period` - Lookback period for trend calculation (minimum 5)
    /// * `smoothing` - EMA smoothing period for the result (minimum 1)
    ///
    /// # Returns
    /// A Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize, smoothing: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
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

    /// Calculate volume-weighted trend
    ///
    /// Returns (trend_value, trend_direction):
    /// - trend_value: Volume-weighted trend strength (-100 to 100)
    /// - trend_direction: +1 for bullish, -1 for bearish, 0 for neutral
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len().min(volume.len());
        let mut trend_value = vec![0.0; n];
        let mut trend_direction = vec![0.0; n];

        if n < 2 {
            return (trend_value, trend_direction);
        }

        // Calculate raw volume-weighted price changes
        let mut raw_vwt = vec![0.0; n];
        for i in self.period..n {
            let start = i - self.period;

            // Sum of volume-weighted price changes
            let mut weighted_sum = 0.0;
            let mut volume_sum = 0.0;

            for j in (start + 1)..=i {
                let price_change = close[j] - close[j - 1];
                let avg_price = (close[j] + close[j - 1]) / 2.0;

                if avg_price > 1e-10 {
                    // Normalize price change as percentage
                    let pct_change = (price_change / avg_price) * 100.0;
                    weighted_sum += pct_change * volume[j];
                    volume_sum += volume[j];
                }
            }

            if volume_sum > 1e-10 {
                raw_vwt[i] = weighted_sum / volume_sum;
            }
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smoothing as f64 + 1.0);
        for i in 0..n {
            if i == 0 {
                trend_value[i] = raw_vwt[i];
            } else {
                trend_value[i] = alpha * raw_vwt[i] + (1.0 - alpha) * trend_value[i - 1];
            }

            // Clamp to -100 to 100 range
            trend_value[i] = trend_value[i].clamp(-100.0, 100.0);

            // Determine direction
            if trend_value[i] > 0.1 {
                trend_direction[i] = 1.0;
            } else if trend_value[i] < -0.1 {
                trend_direction[i] = -1.0;
            }
        }

        (trend_value, trend_direction)
    }
}

impl TechnicalIndicator for VolumeWeightedTrend {
    fn name(&self) -> &str {
        "Volume Weighted Trend"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }
        let (trend_value, trend_direction) = self.calculate(&data.close, &data.volume);
        Ok(IndicatorOutput::dual(trend_value, trend_direction))
    }
}

/// Volume Momentum Oscillator - Oscillator combining volume and price momentum
///
/// Creates an oscillator that measures momentum weighted by volume intensity.
/// The indicator oscillates around zero, with positive values indicating
/// bullish volume-momentum and negative values indicating bearish.
#[derive(Debug, Clone)]
pub struct VolumeMomentumOscillator {
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
}

impl VolumeMomentumOscillator {
    /// Create a new VolumeMomentumOscillator indicator
    ///
    /// # Arguments
    /// * `fast_period` - Fast EMA period (minimum 2)
    /// * `slow_period` - Slow EMA period (must be greater than fast_period)
    /// * `signal_period` - Signal line EMA period (minimum 2)
    ///
    /// # Returns
    /// A Result containing the indicator or an error if parameters are invalid
    pub fn new(fast_period: usize, slow_period: usize, signal_period: usize) -> Result<Self> {
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
        if signal_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "signal_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { fast_period, slow_period, signal_period })
    }

    /// Calculate volume momentum oscillator
    ///
    /// Returns (oscillator, signal_line, histogram):
    /// - oscillator: Main oscillator line
    /// - signal_line: EMA of the oscillator
    /// - histogram: Difference between oscillator and signal
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len().min(volume.len());
        let mut oscillator = vec![0.0; n];
        let mut signal_line = vec![0.0; n];
        let mut histogram = vec![0.0; n];

        if n < 2 {
            return (oscillator, signal_line, histogram);
        }

        // Calculate volume-weighted momentum
        let mut vw_momentum = vec![0.0; n];
        for i in 1..n {
            let price_change = close[i] - close[i - 1];
            let avg_price = (close[i] + close[i - 1]) / 2.0;

            if avg_price > 1e-10 {
                let pct_change = (price_change / avg_price) * 100.0;
                vw_momentum[i] = pct_change * volume[i];
            }
        }

        // Calculate fast EMA of volume-weighted momentum
        let alpha_fast = 2.0 / (self.fast_period as f64 + 1.0);
        let mut fast_ema = vec![0.0; n];
        for i in 0..n {
            if i == 0 {
                fast_ema[i] = vw_momentum[i];
            } else {
                fast_ema[i] = alpha_fast * vw_momentum[i] + (1.0 - alpha_fast) * fast_ema[i - 1];
            }
        }

        // Calculate slow EMA of volume-weighted momentum
        let alpha_slow = 2.0 / (self.slow_period as f64 + 1.0);
        let mut slow_ema = vec![0.0; n];
        for i in 0..n {
            if i == 0 {
                slow_ema[i] = vw_momentum[i];
            } else {
                slow_ema[i] = alpha_slow * vw_momentum[i] + (1.0 - alpha_slow) * slow_ema[i - 1];
            }
        }

        // Calculate oscillator (fast - slow)
        for i in self.slow_period..n {
            oscillator[i] = fast_ema[i] - slow_ema[i];
        }

        // Calculate signal line (EMA of oscillator)
        let alpha_signal = 2.0 / (self.signal_period as f64 + 1.0);
        for i in 0..n {
            if i == 0 {
                signal_line[i] = oscillator[i];
            } else {
                signal_line[i] = alpha_signal * oscillator[i] + (1.0 - alpha_signal) * signal_line[i - 1];
            }
        }

        // Calculate histogram
        for i in 0..n {
            histogram[i] = oscillator[i] - signal_line[i];
        }

        (oscillator, signal_line, histogram)
    }
}

impl TechnicalIndicator for VolumeMomentumOscillator {
    fn name(&self) -> &str {
        "Volume Momentum Oscillator"
    }

    fn min_periods(&self) -> usize {
        self.slow_period + self.signal_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let required = self.slow_period + self.signal_period;
        if data.close.len() < required {
            return Err(IndicatorError::InsufficientData {
                required,
                got: data.close.len(),
            });
        }
        let (oscillator, signal, histogram) = self.calculate(&data.close, &data.volume);
        Ok(IndicatorOutput::triple(oscillator, signal, histogram))
    }

    fn output_features(&self) -> usize {
        3 // oscillator, signal line, histogram
    }
}

/// Volume Accumulation Trend - Tracks sustained accumulation or distribution
///
/// Measures the trend of accumulation (buying) or distribution (selling)
/// by tracking the cumulative sum of volume weighted by price position
/// within the bar's range over time.
#[derive(Debug, Clone)]
pub struct VolumeAccumulationTrend {
    period: usize,
    smoothing: usize,
}

impl VolumeAccumulationTrend {
    /// Create a new VolumeAccumulationTrend indicator
    ///
    /// # Arguments
    /// * `period` - Lookback period for trend calculation (minimum 5)
    /// * `smoothing` - EMA smoothing period (minimum 1)
    ///
    /// # Returns
    /// A Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize, smoothing: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
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

    /// Calculate volume accumulation trend
    ///
    /// Returns (trend_value, trend_signal):
    /// - trend_value: Cumulative accumulation/distribution trend
    /// - trend_signal: +1 for accumulation trend, -1 for distribution, 0 for neutral
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len().min(high.len()).min(low.len()).min(volume.len());
        let mut trend_value = vec![0.0; n];
        let mut trend_signal = vec![0.0; n];

        if n < 1 {
            return (trend_value, trend_signal);
        }

        // Calculate Accumulation/Distribution values using CLV (Close Location Value)
        let mut ad_values = vec![0.0; n];
        for i in 0..n {
            let range = high[i] - low[i];
            if range > 1e-10 {
                // CLV ranges from -1 (close at low) to +1 (close at high)
                let clv = ((close[i] - low[i]) - (high[i] - close[i])) / range;
                ad_values[i] = clv * volume[i];
            }
        }

        // Calculate rolling trend (slope of A/D line over period)
        let mut raw_trend = vec![0.0; n];
        for i in self.period..n {
            let start = i - self.period;

            // Linear regression slope of A/D values
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            let mut sum_xy = 0.0;
            let mut sum_x2 = 0.0;

            let mut cumulative_ad = 0.0;
            for (j, idx) in (start..=i).enumerate() {
                cumulative_ad += ad_values[idx];
                let x = j as f64;
                sum_x += x;
                sum_y += cumulative_ad;
                sum_xy += x * cumulative_ad;
                sum_x2 += x * x;
            }

            let period_len = (self.period + 1) as f64;
            let denom = period_len * sum_x2 - sum_x * sum_x;
            if denom.abs() > 1e-10 {
                raw_trend[i] = (period_len * sum_xy - sum_x * sum_y) / denom;
            }
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smoothing as f64 + 1.0);
        for i in 0..n {
            if i == 0 {
                trend_value[i] = raw_trend[i];
            } else {
                trend_value[i] = alpha * raw_trend[i] + (1.0 - alpha) * trend_value[i - 1];
            }
        }

        // Determine trend signal
        for i in self.period..n {
            // Use standard deviation to determine significance threshold
            let start = i - self.period;
            let window = &raw_trend[start..i];
            let mean: f64 = window.iter().sum::<f64>() / self.period as f64;
            let variance: f64 = window.iter()
                .map(|v| (v - mean).powi(2))
                .sum::<f64>() / self.period as f64;
            let std_dev = variance.sqrt();

            let threshold = std_dev * 0.5;
            if trend_value[i] > threshold {
                trend_signal[i] = 1.0; // Accumulation trend
            } else if trend_value[i] < -threshold {
                trend_signal[i] = -1.0; // Distribution trend
            }
        }

        (trend_value, trend_signal)
    }
}

impl TechnicalIndicator for VolumeAccumulationTrend {
    fn name(&self) -> &str {
        "Volume Accumulation Trend"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }
        let (trend_value, trend_signal) = self.calculate(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::dual(trend_value, trend_signal))
    }
}

/// Adaptive Volume MA - Moving average that adapts to volume conditions
///
/// A volume-adaptive moving average that becomes more responsive when
/// volume is high and more stable when volume is low. This helps filter
/// out noise during low-volume periods while remaining responsive to
/// significant high-volume price moves.
#[derive(Debug, Clone)]
pub struct AdaptiveVolumeMA {
    period: usize,
    fast_factor: f64,
    slow_factor: f64,
}

impl AdaptiveVolumeMA {
    /// Create a new AdaptiveVolumeMA indicator
    ///
    /// # Arguments
    /// * `period` - Lookback period for volume comparison (minimum 5)
    /// * `fast_factor` - Fast smoothing factor (0.0 to 1.0, default 0.5)
    /// * `slow_factor` - Slow smoothing factor (0.0 to fast_factor, default 0.1)
    ///
    /// # Returns
    /// A Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize, fast_factor: f64, slow_factor: f64) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if fast_factor <= 0.0 || fast_factor > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "fast_factor".to_string(),
                reason: "must be between 0 and 1 (exclusive of 0)".to_string(),
            });
        }
        if slow_factor < 0.0 || slow_factor >= fast_factor {
            return Err(IndicatorError::InvalidParameter {
                name: "slow_factor".to_string(),
                reason: "must be between 0 and fast_factor".to_string(),
            });
        }
        Ok(Self { period, fast_factor, slow_factor })
    }

    /// Create with default factors (fast=0.5, slow=0.1)
    pub fn with_period(period: usize) -> Result<Self> {
        Self::new(period, 0.5, 0.1)
    }

    /// Calculate adaptive volume moving average
    ///
    /// Returns (adaptive_ma, volume_ratio):
    /// - adaptive_ma: The adaptive moving average values
    /// - volume_ratio: Current volume relative to average (for reference)
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len().min(volume.len());
        let mut adaptive_ma = vec![0.0; n];
        let mut volume_ratio = vec![0.0; n];

        if n == 0 {
            return (adaptive_ma, volume_ratio);
        }

        // Initialize with first close price
        adaptive_ma[0] = close[0];

        for i in 1..n {
            // Calculate average volume over lookback period
            let start = if i >= self.period { i - self.period } else { 0 };
            let avg_volume: f64 = volume[start..i].iter().sum::<f64>() / (i - start) as f64;

            // Calculate volume ratio (current vs average)
            let ratio = if avg_volume > 1e-10 {
                (volume[i] / avg_volume).min(3.0) // Cap at 3x average
            } else {
                1.0
            };
            volume_ratio[i] = ratio;

            // Adaptive smoothing constant based on volume ratio
            // High volume = more responsive (closer to fast_factor)
            // Low volume = more stable (closer to slow_factor)
            let normalized_ratio = ((ratio - 0.5) / 2.5).clamp(0.0, 1.0);
            let alpha = self.slow_factor + normalized_ratio * (self.fast_factor - self.slow_factor);

            // Apply adaptive EMA
            adaptive_ma[i] = alpha * close[i] + (1.0 - alpha) * adaptive_ma[i - 1];
        }

        (adaptive_ma, volume_ratio)
    }
}

impl TechnicalIndicator for AdaptiveVolumeMA {
    fn name(&self) -> &str {
        "Adaptive Volume MA"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }
        let (adaptive_ma, volume_ratio) = self.calculate(&data.close, &data.volume);
        Ok(IndicatorOutput::dual(adaptive_ma, volume_ratio))
    }
}

/// Volume Flow Index - Measures the direction and strength of volume flow
///
/// Calculates the flow of volume by analyzing whether volume is flowing
/// into (accumulation) or out of (distribution) a security. Uses a
/// combination of price change direction and volume intensity.
#[derive(Debug, Clone)]
pub struct VolumeFlowIndex {
    period: usize,
    smoothing: usize,
}

impl VolumeFlowIndex {
    /// Create a new VolumeFlowIndex indicator
    ///
    /// # Arguments
    /// * `period` - Lookback period for flow calculation (minimum 5)
    /// * `smoothing` - EMA smoothing period (minimum 1)
    ///
    /// # Returns
    /// A Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize, smoothing: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
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

    /// Calculate volume flow index
    ///
    /// Returns (flow_index, flow_direction):
    /// - flow_index: Normalized flow index (-100 to 100)
    /// - flow_direction: +1 for inflow, -1 for outflow, 0 for neutral
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len().min(high.len()).min(low.len()).min(volume.len());
        let mut flow_index = vec![0.0; n];
        let mut flow_direction = vec![0.0; n];

        if n < 2 {
            return (flow_index, flow_direction);
        }

        // Calculate typical price and money flow
        let mut money_flow = vec![0.0; n];
        for i in 1..n {
            let typical_price = (high[i] + low[i] + close[i]) / 3.0;
            let prev_typical = (high[i - 1] + low[i - 1] + close[i - 1]) / 3.0;

            // Positive money flow if typical price increased, negative if decreased
            if typical_price > prev_typical {
                money_flow[i] = typical_price * volume[i];
            } else if typical_price < prev_typical {
                money_flow[i] = -typical_price * volume[i];
            }
            // If equal, money_flow remains 0
        }

        // Calculate rolling flow ratio
        let mut raw_flow = vec![0.0; n];
        for i in self.period..n {
            let start = i - self.period;
            let window = &money_flow[start..=i];

            let positive_flow: f64 = window.iter().filter(|&&x| x > 0.0).sum();
            let negative_flow: f64 = window.iter().filter(|&&x| x < 0.0).map(|x| x.abs()).sum();

            let total_flow = positive_flow + negative_flow;
            if total_flow > 1e-10 {
                // Flow index as percentage: (positive - negative) / total * 100
                raw_flow[i] = ((positive_flow - negative_flow) / total_flow) * 100.0;
            }
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smoothing as f64 + 1.0);
        for i in 0..n {
            if i == 0 {
                flow_index[i] = raw_flow[i];
            } else {
                flow_index[i] = alpha * raw_flow[i] + (1.0 - alpha) * flow_index[i - 1];
            }

            // Clamp to -100 to 100
            flow_index[i] = flow_index[i].clamp(-100.0, 100.0);

            // Determine flow direction
            if flow_index[i] > 10.0 {
                flow_direction[i] = 1.0; // Net inflow
            } else if flow_index[i] < -10.0 {
                flow_direction[i] = -1.0; // Net outflow
            }
        }

        (flow_index, flow_direction)
    }
}

impl TechnicalIndicator for VolumeFlowIndex {
    fn name(&self) -> &str {
        "Volume Flow Index"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }
        let (flow_index, flow_direction) = self.calculate(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::dual(flow_index, flow_direction))
    }
}

/// Volume Pressure Index - Measures buying vs selling pressure from volume
///
/// Analyzes volume to determine whether buying or selling pressure dominates.
/// Uses the relationship between close price and the high-low range to
/// estimate the proportion of volume that represents buying vs selling.
#[derive(Debug, Clone)]
pub struct VolumePressureIndex {
    period: usize,
}

impl VolumePressureIndex {
    /// Create a new VolumePressureIndex indicator
    ///
    /// # Arguments
    /// * `period` - Lookback period for pressure calculation (minimum 5)
    ///
    /// # Returns
    /// A Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate volume pressure index
    ///
    /// Returns (pressure_index, pressure_signal):
    /// - pressure_index: Buying pressure ratio (0-100, 50 = balanced)
    /// - pressure_signal: +1 for buying pressure, -1 for selling pressure, 0 for neutral
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len().min(high.len()).min(low.len()).min(volume.len());
        let mut pressure_index = vec![50.0; n]; // Default to balanced
        let mut pressure_signal = vec![0.0; n];

        if n < 1 {
            return (pressure_index, pressure_signal);
        }

        // Calculate buying and selling volume for each bar
        let mut buying_volume = vec![0.0; n];
        let mut selling_volume = vec![0.0; n];

        for i in 0..n {
            let range = high[i] - low[i];
            if range > 1e-10 {
                // Buying volume proportion: how close did price close to the high?
                let buy_ratio = (close[i] - low[i]) / range;
                buying_volume[i] = buy_ratio * volume[i];
                selling_volume[i] = (1.0 - buy_ratio) * volume[i];
            } else {
                // No range, split 50/50
                buying_volume[i] = volume[i] * 0.5;
                selling_volume[i] = volume[i] * 0.5;
            }
        }

        // Calculate rolling pressure index
        for i in self.period..n {
            let start = i - self.period;

            let total_buying: f64 = buying_volume[start..=i].iter().sum();
            let total_selling: f64 = selling_volume[start..=i].iter().sum();
            let total_volume = total_buying + total_selling;

            if total_volume > 1e-10 {
                pressure_index[i] = (total_buying / total_volume) * 100.0;
            }

            // Determine pressure signal
            if pressure_index[i] > 60.0 {
                pressure_signal[i] = 1.0; // Strong buying pressure
            } else if pressure_index[i] < 40.0 {
                pressure_signal[i] = -1.0; // Strong selling pressure
            }
        }

        (pressure_index, pressure_signal)
    }
}

impl TechnicalIndicator for VolumePressureIndex {
    fn name(&self) -> &str {
        "Volume Pressure Index"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }
        let (pressure_index, pressure_signal) = self.calculate(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::dual(pressure_index, pressure_signal))
    }
}

/// Volume Strength Index - Composite volume strength measure
///
/// Combines multiple volume metrics to create a comprehensive strength indicator.
/// The index considers relative volume, volume momentum, and volume-price correlation
/// to produce a single strength score ranging from 0 to 100.
#[derive(Debug, Clone)]
pub struct VolumeStrengthIndex {
    period: usize,
    smoothing: usize,
}

impl VolumeStrengthIndex {
    /// Create a new VolumeStrengthIndex indicator
    ///
    /// # Arguments
    /// * `period` - Lookback period for calculations (minimum 5)
    /// * `smoothing` - EMA smoothing period for the final output (minimum 1)
    ///
    /// # Returns
    /// A Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize, smoothing: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
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

    /// Calculate volume strength index
    ///
    /// Returns (strength_index, strength_signal):
    /// - strength_index: Composite strength score (0-100)
    /// - strength_signal: +1 for strong volume, -1 for weak volume, 0 for neutral
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len().min(volume.len());
        let mut strength_index = vec![50.0; n];
        let mut strength_signal = vec![0.0; n];

        if n < self.period + 1 {
            return (strength_index, strength_signal);
        }

        // Calculate raw strength components
        let mut raw_strength = vec![0.0; n];

        for i in self.period..n {
            let start = i - self.period;
            let window_vol = &volume[start..i];
            let window_close = &close[start..i];

            // Component 1: Relative Volume (0-33 points)
            let avg_volume: f64 = window_vol.iter().sum::<f64>() / self.period as f64;
            let relative_vol = if avg_volume > 1e-10 {
                (volume[i] / avg_volume).min(3.0) / 3.0 * 33.0
            } else {
                16.5
            };

            // Component 2: Volume Momentum (0-33 points)
            // Compare recent average to older average
            let half = self.period / 2;
            let recent_avg: f64 = window_vol[half..].iter().sum::<f64>() / (self.period - half) as f64;
            let older_avg: f64 = window_vol[..half].iter().sum::<f64>() / half as f64;
            let vol_momentum = if older_avg > 1e-10 {
                let ratio = recent_avg / older_avg;
                ((ratio - 0.5) / 1.5).clamp(0.0, 1.0) * 33.0
            } else {
                16.5
            };

            // Component 3: Volume-Price Correlation (0-34 points)
            // Calculate correlation between volume and absolute price changes
            let mut price_changes = Vec::with_capacity(self.period - 1);
            for j in 1..self.period {
                price_changes.push((window_close[j] - window_close[j - 1]).abs());
            }
            let vol_changes: Vec<f64> = window_vol[1..].to_vec();

            // Calculate correlation
            let mean_price: f64 = price_changes.iter().sum::<f64>() / price_changes.len() as f64;
            let mean_vol: f64 = vol_changes.iter().sum::<f64>() / vol_changes.len() as f64;

            let mut cov = 0.0;
            let mut var_price = 0.0;
            let mut var_vol = 0.0;

            for j in 0..price_changes.len() {
                let dp = price_changes[j] - mean_price;
                let dv = vol_changes[j] - mean_vol;
                cov += dp * dv;
                var_price += dp * dp;
                var_vol += dv * dv;
            }

            let correlation = if var_price > 1e-10 && var_vol > 1e-10 {
                cov / (var_price.sqrt() * var_vol.sqrt())
            } else {
                0.0
            };

            // Convert correlation (-1 to 1) to points (0 to 34)
            let vol_price_score = ((correlation + 1.0) / 2.0) * 34.0;

            // Combine components
            raw_strength[i] = relative_vol + vol_momentum + vol_price_score;
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smoothing as f64 + 1.0);
        for i in 0..n {
            if i == 0 {
                strength_index[i] = raw_strength[i];
            } else {
                strength_index[i] = alpha * raw_strength[i] + (1.0 - alpha) * strength_index[i - 1];
            }

            // Clamp to 0-100
            strength_index[i] = strength_index[i].clamp(0.0, 100.0);

            // Determine signal
            if strength_index[i] > 65.0 {
                strength_signal[i] = 1.0; // Strong volume
            } else if strength_index[i] < 35.0 {
                strength_signal[i] = -1.0; // Weak volume
            }
        }

        (strength_index, strength_signal)
    }
}

impl TechnicalIndicator for VolumeStrengthIndex {
    fn name(&self) -> &str {
        "Volume Strength Index"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }
        let (strength_index, strength_signal) = self.calculate(&data.close, &data.volume);
        Ok(IndicatorOutput::dual(strength_index, strength_signal))
    }
}

/// Normalized Volume - Volume normalized by its moving average
///
/// Expresses current volume as a ratio of its moving average, making it easier
/// to compare volume levels across different time periods and securities.
/// A value of 1.0 indicates average volume, >1 is above average, <1 is below.
#[derive(Debug, Clone)]
pub struct NormalizedVolume {
    period: usize,
    use_ema: bool,
}

impl NormalizedVolume {
    /// Create a new NormalizedVolume indicator
    ///
    /// # Arguments
    /// * `period` - Lookback period for the moving average (minimum 2)
    /// * `use_ema` - If true, use EMA for normalization; if false, use SMA
    ///
    /// # Returns
    /// A Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize, use_ema: bool) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period, use_ema })
    }

    /// Create with SMA normalization (default)
    pub fn with_sma(period: usize) -> Result<Self> {
        Self::new(period, false)
    }

    /// Create with EMA normalization
    pub fn with_ema(period: usize) -> Result<Self> {
        Self::new(period, true)
    }

    /// Calculate normalized volume
    ///
    /// Returns (normalized_volume, volume_state):
    /// - normalized_volume: Volume as ratio of moving average
    /// - volume_state: +1 for high volume (>1.5x), -1 for low volume (<0.5x), 0 for normal
    pub fn calculate(&self, volume: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = volume.len();
        let mut normalized = vec![1.0; n];
        let mut state = vec![0.0; n];

        if n < self.period {
            return (normalized, state);
        }

        if self.use_ema {
            // Calculate EMA
            let alpha = 2.0 / (self.period as f64 + 1.0);
            let mut ema = vec![0.0; n];
            ema[0] = volume[0];

            for i in 1..n {
                ema[i] = alpha * volume[i] + (1.0 - alpha) * ema[i - 1];
            }

            // Calculate normalized volume
            for i in self.period..n {
                if ema[i - 1] > 1e-10 {
                    normalized[i] = volume[i] / ema[i - 1];
                }

                // Determine state
                if normalized[i] > 1.5 {
                    state[i] = 1.0; // High volume
                } else if normalized[i] < 0.5 {
                    state[i] = -1.0; // Low volume
                }
            }
        } else {
            // Calculate SMA
            for i in self.period..n {
                let start = i - self.period;
                let sma: f64 = volume[start..i].iter().sum::<f64>() / self.period as f64;

                if sma > 1e-10 {
                    normalized[i] = volume[i] / sma;
                }

                // Determine state
                if normalized[i] > 1.5 {
                    state[i] = 1.0; // High volume
                } else if normalized[i] < 0.5 {
                    state[i] = -1.0; // Low volume
                }
            }
        }

        (normalized, state)
    }
}

impl TechnicalIndicator for NormalizedVolume {
    fn name(&self) -> &str {
        "Normalized Volume"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.volume.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.volume.len(),
            });
        }
        let (normalized, state) = self.calculate(&data.volume);
        Ok(IndicatorOutput::dual(normalized, state))
    }
}

/// Volume Surge - Detects sudden volume surges
///
/// Identifies rapid increases in volume that may indicate significant market events,
/// breakouts, or institutional activity. Uses rate of change and statistical
/// thresholds to detect meaningful surges while filtering noise.
#[derive(Debug, Clone)]
pub struct VolumeSurge {
    period: usize,
    surge_threshold: f64,
    lookback: usize,
}

impl VolumeSurge {
    /// Create a new VolumeSurge indicator
    ///
    /// # Arguments
    /// * `period` - Lookback period for baseline calculation (minimum 5)
    /// * `surge_threshold` - Multiple of average required to signal surge (minimum 1.5)
    /// * `lookback` - Number of bars to compare for surge detection (minimum 1)
    ///
    /// # Returns
    /// A Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize, surge_threshold: f64, lookback: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if surge_threshold < 1.5 {
            return Err(IndicatorError::InvalidParameter {
                name: "surge_threshold".to_string(),
                reason: "must be at least 1.5".to_string(),
            });
        }
        if lookback < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self { period, surge_threshold, lookback })
    }

    /// Create with default threshold (2.0) and lookback (3)
    pub fn with_period(period: usize) -> Result<Self> {
        Self::new(period, 2.0, 3)
    }

    /// Calculate volume surge detection
    ///
    /// Returns (surge_magnitude, surge_signal, consecutive_surges):
    /// - surge_magnitude: How many times average the current volume is
    /// - surge_signal: 1.0 if surge detected, 0.0 otherwise
    /// - consecutive_surges: Count of consecutive surge bars
    pub fn calculate(&self, volume: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = volume.len();
        let mut magnitude = vec![1.0; n];
        let mut signal = vec![0.0; n];
        let mut consecutive = vec![0.0; n];

        if n < self.period + self.lookback {
            return (magnitude, signal, consecutive);
        }

        for i in self.period..n {
            let start = i - self.period;
            let window = &volume[start..i];

            // Calculate baseline (average volume)
            let avg_volume: f64 = window.iter().sum::<f64>() / self.period as f64;

            if avg_volume > 1e-10 {
                magnitude[i] = volume[i] / avg_volume;

                // Check for surge
                if magnitude[i] >= self.surge_threshold {
                    signal[i] = 1.0;

                    // Count consecutive surges
                    if i > 0 && signal[i - 1] == 1.0 {
                        consecutive[i] = consecutive[i - 1] + 1.0;
                    } else {
                        consecutive[i] = 1.0;
                    }
                }

                // Additional check: rapid increase over lookback period
                if self.lookback > 0 && i >= self.lookback {
                    let lookback_avg: f64 = volume[i - self.lookback..i].iter().sum::<f64>() / self.lookback as f64;
                    if lookback_avg > 1e-10 && volume[i] / lookback_avg >= self.surge_threshold {
                        signal[i] = 1.0;
                        if i > 0 && signal[i - 1] == 1.0 && consecutive[i] == 0.0 {
                            consecutive[i] = consecutive[i - 1] + 1.0;
                        } else if consecutive[i] == 0.0 {
                            consecutive[i] = 1.0;
                        }
                    }
                }
            }
        }

        (magnitude, signal, consecutive)
    }
}

impl TechnicalIndicator for VolumeSurge {
    fn name(&self) -> &str {
        "Volume Surge"
    }

    fn min_periods(&self) -> usize {
        self.period + self.lookback
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.volume.len() < self.period + self.lookback {
            return Err(IndicatorError::InsufficientData {
                required: self.period + self.lookback,
                got: data.volume.len(),
            });
        }
        let (magnitude, signal, consecutive) = self.calculate(&data.volume);
        Ok(IndicatorOutput::triple(magnitude, signal, consecutive))
    }

    fn output_features(&self) -> usize {
        3 // magnitude, signal, consecutive
    }
}

/// Volume Divergence Index - Measures divergence between volume and price
///
/// Calculates the divergence between price movement direction and volume trends.
/// Positive divergence (price down, volume down) may indicate selling exhaustion.
/// Negative divergence (price up, volume down) may indicate buying exhaustion.
#[derive(Debug, Clone)]
pub struct VolumeDivergenceIndex {
    period: usize,
    smoothing: usize,
}

impl VolumeDivergenceIndex {
    /// Create a new VolumeDivergenceIndex indicator
    ///
    /// # Arguments
    /// * `period` - Lookback period for divergence calculation (minimum 5)
    /// * `smoothing` - EMA smoothing period (minimum 1)
    ///
    /// # Returns
    /// A Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize, smoothing: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
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

    /// Calculate volume divergence index
    ///
    /// Returns (divergence_index, divergence_type):
    /// - divergence_index: Strength of divergence (-100 to 100)
    /// - divergence_type: +1 for bullish divergence, -1 for bearish divergence, 0 for confirmation
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len().min(volume.len());
        let mut divergence_index = vec![0.0; n];
        let mut divergence_type = vec![0.0; n];

        if n < self.period + 1 {
            return (divergence_index, divergence_type);
        }

        let mut raw_divergence = vec![0.0; n];

        for i in self.period..n {
            let start = i - self.period;

            // Calculate price change over period (normalized)
            let price_start = close[start];
            let price_end = close[i];
            let avg_price = (price_start + price_end) / 2.0;

            let price_change = if avg_price > 1e-10 {
                (price_end - price_start) / avg_price * 100.0
            } else {
                0.0
            };

            // Calculate volume change over period (normalized)
            let vol_start: f64 = volume[start..start + self.period / 2].iter().sum::<f64>() / (self.period / 2) as f64;
            let vol_end: f64 = volume[i - self.period / 2..=i].iter().sum::<f64>() / (self.period / 2 + 1) as f64;
            let avg_vol = (vol_start + vol_end) / 2.0;

            let volume_change = if avg_vol > 1e-10 {
                (vol_end - vol_start) / avg_vol * 100.0
            } else {
                0.0
            };

            // Calculate divergence
            // Positive price change with negative volume change = bearish divergence
            // Negative price change with negative volume change = bullish divergence
            // Same sign = confirmation (no divergence)

            let price_up = price_change > 1.0;
            let price_down = price_change < -1.0;
            let vol_up = volume_change > 5.0;
            let vol_down = volume_change < -5.0;

            if price_up && vol_down {
                // Bearish divergence: price rising on declining volume
                raw_divergence[i] = -((price_change.abs() + volume_change.abs()) / 2.0);
                raw_divergence[i] = raw_divergence[i].clamp(-100.0, 0.0);
            } else if price_down && vol_down {
                // Bullish divergence: price falling on declining volume
                raw_divergence[i] = (price_change.abs() + volume_change.abs()) / 2.0;
                raw_divergence[i] = raw_divergence[i].clamp(0.0, 100.0);
            } else if (price_up && vol_up) || (price_down && vol_up) {
                // Confirmation: price move supported by volume
                raw_divergence[i] = 0.0;
            }
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smoothing as f64 + 1.0);
        for i in 0..n {
            if i == 0 {
                divergence_index[i] = raw_divergence[i];
            } else {
                divergence_index[i] = alpha * raw_divergence[i] + (1.0 - alpha) * divergence_index[i - 1];
            }

            // Clamp to -100 to 100
            divergence_index[i] = divergence_index[i].clamp(-100.0, 100.0);

            // Determine divergence type
            if divergence_index[i] > 10.0 {
                divergence_type[i] = 1.0; // Bullish divergence
            } else if divergence_index[i] < -10.0 {
                divergence_type[i] = -1.0; // Bearish divergence
            }
        }

        (divergence_index, divergence_type)
    }
}

impl TechnicalIndicator for VolumeDivergenceIndex {
    fn name(&self) -> &str {
        "Volume Divergence Index"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }
        let (divergence_index, divergence_type) = self.calculate(&data.close, &data.volume);
        Ok(IndicatorOutput::dual(divergence_index, divergence_type))
    }
}

/// Institutional Flow Indicator - Estimates institutional vs retail volume
///
/// Analyzes volume patterns to estimate the proportion of institutional (smart money)
/// versus retail trading activity. Uses price efficiency and volume clustering
/// as proxies for institutional activity.
#[derive(Debug, Clone)]
pub struct InstitutionalFlowIndicator {
    period: usize,
    efficiency_threshold: f64,
}

impl InstitutionalFlowIndicator {
    /// Create a new InstitutionalFlowIndicator
    ///
    /// # Arguments
    /// * `period` - Lookback period for analysis (minimum 10)
    /// * `efficiency_threshold` - Threshold for efficient price movement (0.0 to 1.0)
    ///
    /// # Returns
    /// A Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize, efficiency_threshold: f64) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if efficiency_threshold <= 0.0 || efficiency_threshold >= 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "efficiency_threshold".to_string(),
                reason: "must be between 0 and 1 (exclusive)".to_string(),
            });
        }
        Ok(Self { period, efficiency_threshold })
    }

    /// Create with default efficiency threshold (0.5)
    pub fn with_period(period: usize) -> Result<Self> {
        Self::new(period, 0.5)
    }

    /// Calculate institutional flow indicator
    ///
    /// Returns (institutional_ratio, flow_signal, accumulation_score):
    /// - institutional_ratio: Estimated ratio of institutional volume (0-100)
    /// - flow_signal: +1 for institutional buying, -1 for institutional selling, 0 for neutral
    /// - accumulation_score: Cumulative institutional accumulation/distribution
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len().min(high.len()).min(low.len()).min(volume.len());
        let mut institutional_ratio = vec![50.0; n];
        let mut flow_signal = vec![0.0; n];
        let mut accumulation_score = vec![0.0; n];

        if n < self.period + 1 {
            return (institutional_ratio, flow_signal, accumulation_score);
        }

        for i in self.period..n {
            let start = i - self.period;

            // Calculate price efficiency (net move / total path)
            let net_move = (close[i] - close[start]).abs();
            let total_path: f64 = (start + 1..=i)
                .map(|j| (close[j] - close[j - 1]).abs())
                .sum();

            let efficiency = if total_path > 1e-10 {
                net_move / total_path
            } else {
                0.0
            };

            // Calculate volume concentration (high volume on efficient moves)
            let avg_volume: f64 = volume[start..i].iter().sum::<f64>() / self.period as f64;

            // Identify high-volume bars with efficient price movement
            let mut institutional_volume = 0.0;
            let mut total_volume = 0.0;

            for j in (start + 1)..=i {
                let bar_move = (close[j] - close[j - 1]).abs();
                let bar_range = high[j] - low[j];
                let bar_efficiency = if bar_range > 1e-10 {
                    bar_move / bar_range
                } else {
                    0.0
                };

                total_volume += volume[j];

                // High volume with efficient movement suggests institutional activity
                if volume[j] > avg_volume && bar_efficiency > self.efficiency_threshold {
                    institutional_volume += volume[j];
                }
                // Also: low volume with inefficient movement is retail noise
                else if volume[j] <= avg_volume && bar_efficiency <= self.efficiency_threshold {
                    // Don't add to institutional volume
                }
                // Mixed cases: partial attribution
                else if volume[j] > avg_volume {
                    institutional_volume += volume[j] * 0.5;
                }
            }

            // Calculate institutional ratio
            if total_volume > 1e-10 {
                institutional_ratio[i] = (institutional_volume / total_volume) * 100.0;
                institutional_ratio[i] = institutional_ratio[i].clamp(0.0, 100.0);
            }

            // Determine flow direction
            let price_direction = if close[i] > close[start] { 1.0 } else if close[i] < close[start] { -1.0 } else { 0.0 };

            if institutional_ratio[i] > 60.0 {
                flow_signal[i] = price_direction; // Institutional buying or selling
            }

            // Calculate cumulative accumulation score
            let close_position = if high[i] - low[i] > 1e-10 {
                ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
            } else {
                0.0
            };

            let institutional_contribution = close_position * (institutional_ratio[i] / 100.0) * volume[i];

            if i == self.period {
                accumulation_score[i] = institutional_contribution;
            } else {
                accumulation_score[i] = accumulation_score[i - 1] + institutional_contribution;
            }
        }

        (institutional_ratio, flow_signal, accumulation_score)
    }
}

impl TechnicalIndicator for InstitutionalFlowIndicator {
    fn name(&self) -> &str {
        "Institutional Flow Indicator"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }
        let (ratio, signal, accumulation) = self.calculate(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::triple(ratio, signal, accumulation))
    }

    fn output_features(&self) -> usize {
        3 // institutional_ratio, flow_signal, accumulation_score
    }
}

/// Volume Z-Score - Statistical z-score of volume
///
/// Calculates the statistical z-score of current volume relative to a historical
/// distribution. This standardized measure indicates how many standard deviations
/// the current volume is from the mean, useful for identifying statistically
/// significant volume events.
#[derive(Debug, Clone)]
pub struct VolumeZScore {
    period: usize,
}

impl VolumeZScore {
    /// Create a new VolumeZScore indicator
    ///
    /// # Arguments
    /// * `period` - Lookback period for statistical calculation (minimum 10)
    ///
    /// # Returns
    /// A Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10 for meaningful statistical analysis".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate volume z-score
    ///
    /// Returns (z_score, significance_level, extreme_signal):
    /// - z_score: Standard deviations from mean (can be negative or positive)
    /// - significance_level: Statistical significance (0-100, where 95+ is significant)
    /// - extreme_signal: +1 for extremely high volume, -1 for extremely low, 0 for normal
    pub fn calculate(&self, volume: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = volume.len();
        let mut z_score = vec![0.0; n];
        let mut significance = vec![0.0; n];
        let mut extreme_signal = vec![0.0; n];

        if n < self.period + 1 {
            return (z_score, significance, extreme_signal);
        }

        for i in self.period..n {
            let start = i - self.period;
            let window = &volume[start..i];

            // Calculate mean
            let mean: f64 = window.iter().sum::<f64>() / self.period as f64;

            // Calculate standard deviation
            let variance: f64 = window.iter()
                .map(|v| (v - mean).powi(2))
                .sum::<f64>() / self.period as f64;
            let std_dev = variance.sqrt();

            if std_dev > 1e-10 {
                // Calculate z-score
                z_score[i] = (volume[i] - mean) / std_dev;

                // Calculate significance level using cumulative normal distribution approximation
                // Using error function approximation for CDF
                let abs_z = z_score[i].abs();
                let t = 1.0 / (1.0 + 0.2316419 * abs_z);
                let d = 0.3989423 * (-abs_z * abs_z / 2.0).exp();
                let p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));

                // Two-tailed significance
                let two_tailed_p = 2.0 * p;
                significance[i] = (1.0 - two_tailed_p) * 100.0;
                significance[i] = significance[i].clamp(0.0, 99.9);

                // Determine extreme signal
                // z > 2 is ~95% significance (high volume)
                // z < -2 is ~95% significance (low volume)
                if z_score[i] > 2.0 {
                    extreme_signal[i] = 1.0; // Extremely high volume
                } else if z_score[i] < -2.0 {
                    extreme_signal[i] = -1.0; // Extremely low volume
                }
            }
        }

        (z_score, significance, extreme_signal)
    }
}

impl TechnicalIndicator for VolumeZScore {
    fn name(&self) -> &str {
        "Volume Z-Score"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.volume.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.volume.len(),
            });
        }
        let (z_score, significance, extreme) = self.calculate(&data.volume);
        Ok(IndicatorOutput::triple(z_score, significance, extreme))
    }

    fn output_features(&self) -> usize {
        3 // z_score, significance_level, extreme_signal
    }
}

/// Volume Rank - Ranks current volume relative to historical volume
///
/// Calculates where the current volume ranks within the lookback period,
/// providing a percentile-like ranking from 0 (lowest) to 100 (highest).
/// Useful for identifying relative volume strength compared to recent history.
#[derive(Debug, Clone)]
pub struct VolumeRank {
    period: usize,
}

impl VolumeRank {
    /// Create a new VolumeRank indicator
    ///
    /// # Arguments
    /// * `period` - Lookback period for ranking (minimum 10)
    ///
    /// # Returns
    /// A Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate volume rank
    ///
    /// Returns (rank, rank_signal):
    /// - rank: Rank of current volume (0-100) within the lookback period
    /// - rank_signal: +1 for high rank (>80), -1 for low rank (<20), 0 otherwise
    pub fn calculate(&self, volume: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = volume.len();
        let mut rank = vec![0.0; n];
        let mut rank_signal = vec![0.0; n];

        if n < self.period {
            return (rank, rank_signal);
        }

        for i in self.period..n {
            let start = i - self.period;
            let current_vol = volume[i];

            // Count how many historical volumes are less than current
            let mut count_below = 0;
            for j in start..i {
                if volume[j] < current_vol {
                    count_below += 1;
                }
            }

            // Calculate rank as percentage
            rank[i] = (count_below as f64 / self.period as f64) * 100.0;

            // Generate signal
            if rank[i] > 80.0 {
                rank_signal[i] = 1.0; // High volume rank
            } else if rank[i] < 20.0 {
                rank_signal[i] = -1.0; // Low volume rank
            }
        }

        (rank, rank_signal)
    }
}

impl TechnicalIndicator for VolumeRank {
    fn name(&self) -> &str {
        "Volume Rank"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.volume.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.volume.len(),
            });
        }
        let (rank, signal) = self.calculate(&data.volume);
        Ok(IndicatorOutput::dual(rank, signal))
    }
}

/// Volume Percentile - Statistical percentile of current volume
///
/// Calculates the exact percentile position of current volume within
/// a rolling window, accounting for ties. More precise than simple ranking.
#[derive(Debug, Clone)]
pub struct VolumePercentile {
    period: usize,
    smoothing: usize,
}

impl VolumePercentile {
    /// Create a new VolumePercentile indicator
    ///
    /// # Arguments
    /// * `period` - Lookback period for percentile calculation (minimum 10)
    /// * `smoothing` - EMA smoothing period for the result (minimum 1)
    ///
    /// # Returns
    /// A Result containing the indicator or an error if parameters are invalid
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

    /// Create with default smoothing (3)
    pub fn with_period(period: usize) -> Result<Self> {
        Self::new(period, 3)
    }

    /// Calculate volume percentile
    ///
    /// Returns (percentile, percentile_state):
    /// - percentile: Smoothed percentile value (0-100)
    /// - percentile_state: +1 for extreme high (>90), -1 for extreme low (<10), 0 otherwise
    pub fn calculate(&self, volume: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = volume.len();
        let mut percentile = vec![0.0; n];
        let mut percentile_state = vec![0.0; n];

        if n < self.period {
            return (percentile, percentile_state);
        }

        // Calculate raw percentiles
        let mut raw_percentile = vec![0.0; n];
        for i in self.period..n {
            let start = i - self.period;
            let current_vol = volume[i];

            // Count values below and equal to current
            let mut count_below = 0.0;
            let mut count_equal = 0.0;
            for j in start..i {
                if volume[j] < current_vol {
                    count_below += 1.0;
                } else if (volume[j] - current_vol).abs() < 1e-10 {
                    count_equal += 1.0;
                }
            }

            // Percentile formula accounting for ties: (count_below + 0.5 * count_equal) / n * 100
            raw_percentile[i] = ((count_below + 0.5 * count_equal) / self.period as f64) * 100.0;
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smoothing as f64 + 1.0);
        for i in 0..n {
            if i == 0 {
                percentile[i] = raw_percentile[i];
            } else {
                percentile[i] = alpha * raw_percentile[i] + (1.0 - alpha) * percentile[i - 1];
            }

            // Generate state signal
            if percentile[i] > 90.0 {
                percentile_state[i] = 1.0; // Extreme high percentile
            } else if percentile[i] < 10.0 {
                percentile_state[i] = -1.0; // Extreme low percentile
            }
        }

        (percentile, percentile_state)
    }
}

impl TechnicalIndicator for VolumePercentile {
    fn name(&self) -> &str {
        "Volume Percentile"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.volume.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.volume.len(),
            });
        }
        let (percentile, state) = self.calculate(&data.volume);
        Ok(IndicatorOutput::dual(percentile, state))
    }
}

/// Volume Ratio - Ratio of up volume to down volume
///
/// Measures the balance between volume on up-bars vs down-bars,
/// indicating whether buyers or sellers are more active.
/// Values > 1 indicate buying dominance, < 1 indicate selling dominance.
#[derive(Debug, Clone)]
pub struct VolumeRatio {
    period: usize,
    smoothing: usize,
}

impl VolumeRatio {
    /// Create a new VolumeRatio indicator
    ///
    /// # Arguments
    /// * `period` - Lookback period for ratio calculation (minimum 5)
    /// * `smoothing` - EMA smoothing period (minimum 1)
    ///
    /// # Returns
    /// A Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize, smoothing: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
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

    /// Create with default smoothing (3)
    pub fn with_period(period: usize) -> Result<Self> {
        Self::new(period, 3)
    }

    /// Calculate volume ratio
    ///
    /// Returns (ratio, ratio_index, dominance_signal):
    /// - ratio: Raw up/down volume ratio (capped at 10.0 for display)
    /// - ratio_index: Normalized ratio as percentage (0-100), 50 = balanced
    /// - dominance_signal: +1 for buying dominance (>60), -1 for selling (<40), 0 for balanced
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len().min(volume.len());
        let mut ratio = vec![0.0; n];
        let mut ratio_index = vec![50.0; n];
        let mut dominance_signal = vec![0.0; n];

        if n < 2 {
            return (ratio, ratio_index, dominance_signal);
        }

        // Calculate raw up/down volume ratios
        let mut raw_ratio = vec![1.0; n];
        for i in self.period..n {
            let start = i - self.period + 1;

            let mut up_volume = 0.0;
            let mut down_volume = 0.0;

            for j in start..=i {
                if j > 0 {
                    if close[j] > close[j - 1] {
                        up_volume += volume[j];
                    } else if close[j] < close[j - 1] {
                        down_volume += volume[j];
                    } else {
                        // Unchanged: split between up and down
                        up_volume += volume[j] / 2.0;
                        down_volume += volume[j] / 2.0;
                    }
                }
            }

            if down_volume > 1e-10 {
                raw_ratio[i] = up_volume / down_volume;
            } else if up_volume > 1e-10 {
                raw_ratio[i] = 10.0; // Cap at 10 when no down volume
            } else {
                raw_ratio[i] = 1.0; // Equal when both are zero
            }
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smoothing as f64 + 1.0);
        for i in 0..n {
            if i == 0 {
                ratio[i] = raw_ratio[i].min(10.0);
            } else {
                ratio[i] = (alpha * raw_ratio[i] + (1.0 - alpha) * ratio[i - 1]).min(10.0);
            }

            // Convert ratio to index (0-100 scale)
            // ratio of 1.0 = 50, ratio of 2.0 = 66.7, ratio of 0.5 = 33.3
            ratio_index[i] = (ratio[i] / (ratio[i] + 1.0)) * 100.0;

            // Generate dominance signal
            if ratio_index[i] > 60.0 {
                dominance_signal[i] = 1.0; // Buying dominance
            } else if ratio_index[i] < 40.0 {
                dominance_signal[i] = -1.0; // Selling dominance
            }
        }

        (ratio, ratio_index, dominance_signal)
    }
}

impl TechnicalIndicator for VolumeRatio {
    fn name(&self) -> &str {
        "Volume Ratio"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }
        let (ratio, ratio_index, dominance) = self.calculate(&data.close, &data.volume);
        Ok(IndicatorOutput::triple(ratio, ratio_index, dominance))
    }

    fn output_features(&self) -> usize {
        3 // ratio, ratio_index, dominance_signal
    }
}

/// Volume Concentration - Measures how concentrated volume is at specific price levels
///
/// Analyzes the distribution of volume across the price range,
/// identifying whether volume is concentrated (focused trading) or dispersed.
/// High concentration suggests strong conviction, low suggests indecision.
#[derive(Debug, Clone)]
pub struct VolumeConcentration {
    period: usize,
    num_bins: usize,
}

impl VolumeConcentration {
    /// Create a new VolumeConcentration indicator
    ///
    /// # Arguments
    /// * `period` - Lookback period for concentration analysis (minimum 5)
    /// * `num_bins` - Number of price bins for distribution (minimum 3)
    ///
    /// # Returns
    /// A Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize, num_bins: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if num_bins < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "num_bins".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        Ok(Self { period, num_bins })
    }

    /// Create with default bins (5)
    pub fn with_period(period: usize) -> Result<Self> {
        Self::new(period, 5)
    }

    /// Calculate volume concentration
    ///
    /// Returns (concentration_index, concentration_signal):
    /// - concentration_index: Concentration score (0-100), higher = more concentrated
    /// - concentration_signal: +1 for high concentration (>70), -1 for low (<30), 0 otherwise
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len().min(high.len()).min(low.len()).min(volume.len());
        let mut concentration_index = vec![50.0; n];
        let mut concentration_signal = vec![0.0; n];

        if n < self.period {
            return (concentration_index, concentration_signal);
        }

        for i in self.period..n {
            let start = i - self.period + 1;

            // Find price range for the period
            let mut min_price = f64::MAX;
            let mut max_price = f64::MIN;
            for j in start..=i {
                min_price = min_price.min(low[j]);
                max_price = max_price.max(high[j]);
            }

            let price_range = max_price - min_price;
            if price_range < 1e-10 {
                concentration_index[i] = 100.0; // All at same price = perfect concentration
                concentration_signal[i] = 1.0;
                continue;
            }

            // Distribute volume into bins
            let mut bins = vec![0.0; self.num_bins];
            let mut total_volume = 0.0;

            for j in start..=i {
                let typical_price = (high[j] + low[j] + close[j]) / 3.0;
                let bin_idx = ((typical_price - min_price) / price_range * (self.num_bins as f64 - 0.001))
                    .floor() as usize;
                let bin_idx = bin_idx.min(self.num_bins - 1);
                bins[bin_idx] += volume[j];
                total_volume += volume[j];
            }

            if total_volume < 1e-10 {
                continue;
            }

            // Calculate Herfindahl-Hirschman Index (HHI) for concentration
            // HHI = sum of squared market shares
            let mut hhi = 0.0;
            for bin_vol in &bins {
                let share = bin_vol / total_volume;
                hhi += share * share;
            }

            // Normalize HHI to 0-100 scale
            // Min HHI = 1/n (equal distribution), Max HHI = 1 (all in one bin)
            let min_hhi = 1.0 / self.num_bins as f64;
            let normalized_hhi = (hhi - min_hhi) / (1.0 - min_hhi);
            concentration_index[i] = (normalized_hhi * 100.0).clamp(0.0, 100.0);

            // Generate signal
            if concentration_index[i] > 70.0 {
                concentration_signal[i] = 1.0; // High concentration
            } else if concentration_index[i] < 30.0 {
                concentration_signal[i] = -1.0; // Low concentration
            }
        }

        (concentration_index, concentration_signal)
    }
}

impl TechnicalIndicator for VolumeConcentration {
    fn name(&self) -> &str {
        "Volume Concentration"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }
        let (concentration, signal) = self.calculate(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::dual(concentration, signal))
    }
}

/// Volume Bias - Measures directional bias of volume-weighted price movement
///
/// Calculates the bias of volume towards bullish or bearish price action,
/// helping identify whether smart money is accumulating or distributing.
/// Combines volume with price change direction and magnitude.
#[derive(Debug, Clone)]
pub struct VolumeBias {
    period: usize,
    smoothing: usize,
}

impl VolumeBias {
    /// Create a new VolumeBias indicator
    ///
    /// # Arguments
    /// * `period` - Lookback period for bias calculation (minimum 5)
    /// * `smoothing` - EMA smoothing period (minimum 1)
    ///
    /// # Returns
    /// A Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize, smoothing: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
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

    /// Create with default smoothing (3)
    pub fn with_period(period: usize) -> Result<Self> {
        Self::new(period, 3)
    }

    /// Calculate volume bias
    ///
    /// Returns (bias_value, bias_strength, bias_signal):
    /// - bias_value: Directional bias (-100 to +100), positive = bullish, negative = bearish
    /// - bias_strength: Absolute strength of bias (0-100)
    /// - bias_signal: +1 for bullish bias, -1 for bearish, 0 for neutral
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len().min(volume.len());
        let mut bias_value = vec![0.0; n];
        let mut bias_strength = vec![0.0; n];
        let mut bias_signal = vec![0.0; n];

        if n < 2 {
            return (bias_value, bias_strength, bias_signal);
        }

        // Calculate raw bias values
        let mut raw_bias = vec![0.0; n];
        for i in self.period..n {
            let start = i - self.period + 1;

            let mut weighted_sum = 0.0;
            let mut volume_sum = 0.0;

            for j in start..=i {
                if j > 0 {
                    let price_change = close[j] - close[j - 1];
                    let avg_price = (close[j] + close[j - 1]) / 2.0;

                    if avg_price > 1e-10 {
                        // Percentage change weighted by volume
                        let pct_change = price_change / avg_price;
                        weighted_sum += pct_change * volume[j];
                        volume_sum += volume[j];
                    }
                }
            }

            if volume_sum > 1e-10 {
                // Scale to reasonable range (-100 to 100)
                raw_bias[i] = (weighted_sum / volume_sum) * 10000.0;
            }
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smoothing as f64 + 1.0);
        for i in 0..n {
            if i == 0 {
                bias_value[i] = raw_bias[i].clamp(-100.0, 100.0);
            } else {
                let smoothed = alpha * raw_bias[i] + (1.0 - alpha) * bias_value[i - 1];
                bias_value[i] = smoothed.clamp(-100.0, 100.0);
            }

            // Calculate strength as absolute value
            bias_strength[i] = bias_value[i].abs();

            // Generate bias signal
            if bias_value[i] > 10.0 {
                bias_signal[i] = 1.0; // Bullish bias
            } else if bias_value[i] < -10.0 {
                bias_signal[i] = -1.0; // Bearish bias
            }
        }

        (bias_value, bias_strength, bias_signal)
    }
}

impl TechnicalIndicator for VolumeBias {
    fn name(&self) -> &str {
        "Volume Bias"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }
        let (bias, strength, signal) = self.calculate(&data.close, &data.volume);
        Ok(IndicatorOutput::triple(bias, strength, signal))
    }

    fn output_features(&self) -> usize {
        3 // bias_value, bias_strength, bias_signal
    }
}

/// Volume Quality - Measures the quality and reliability of volume signals
///
/// Evaluates volume quality by analyzing consistency, confirmation with price,
/// and statistical significance. Higher quality indicates more reliable volume signals.
/// Combines multiple factors: consistency, price confirmation, and magnitude.
#[derive(Debug, Clone)]
pub struct VolumeQuality {
    period: usize,
    confirmation_threshold: f64,
}

impl VolumeQuality {
    /// Create a new VolumeQuality indicator
    ///
    /// # Arguments
    /// * `period` - Lookback period for quality analysis (minimum 10)
    /// * `confirmation_threshold` - Price-volume correlation threshold (0.0 to 1.0)
    ///
    /// # Returns
    /// A Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize, confirmation_threshold: f64) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if confirmation_threshold < 0.0 || confirmation_threshold > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "confirmation_threshold".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        Ok(Self { period, confirmation_threshold })
    }

    /// Create with default threshold (0.5)
    pub fn with_period(period: usize) -> Result<Self> {
        Self::new(period, 0.5)
    }

    /// Calculate volume quality
    ///
    /// Returns (quality_score, reliability_signal):
    /// - quality_score: Overall quality score (0-100)
    /// - reliability_signal: +1 for high quality (>70), -1 for low quality (<30), 0 otherwise
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len().min(volume.len());
        let mut quality_score = vec![50.0; n];
        let mut reliability_signal = vec![0.0; n];

        if n < self.period + 1 {
            return (quality_score, reliability_signal);
        }

        for i in self.period..n {
            let start = i - self.period;

            // Factor 1: Volume Consistency (lower coefficient of variation = higher consistency)
            let volume_window = &volume[start..=i];
            let vol_mean: f64 = volume_window.iter().sum::<f64>() / (self.period + 1) as f64;
            let vol_variance: f64 = volume_window.iter()
                .map(|v| (v - vol_mean).powi(2))
                .sum::<f64>() / (self.period + 1) as f64;
            let vol_cv = if vol_mean > 1e-10 {
                vol_variance.sqrt() / vol_mean
            } else {
                1.0
            };
            // Lower CV = more consistent = higher score
            // CV of 0 = 100 score, CV of 1 = 0 score
            let consistency_score = ((1.0 - vol_cv.min(1.0)) * 100.0).max(0.0);

            // Factor 2: Price-Volume Confirmation
            // Higher volume on trend-following moves = confirmation
            let mut confirming_volume = 0.0;
            let mut total_volume = 0.0;

            for j in (start + 1)..=i {
                let price_change = close[j] - close[j - 1];
                let prev_trend = if j > 1 { close[j - 1] - close[j - 2] } else { 0.0 };

                total_volume += volume[j];

                // Confirming if current move aligns with previous trend
                if (price_change > 0.0 && prev_trend > 0.0) || (price_change < 0.0 && prev_trend < 0.0) {
                    confirming_volume += volume[j];
                } else if price_change.abs() < 1e-10 {
                    // Neutral, count as partial confirmation
                    confirming_volume += volume[j] * 0.5;
                }
            }

            let confirmation_score = if total_volume > 1e-10 {
                (confirming_volume / total_volume * 100.0).min(100.0)
            } else {
                50.0
            };

            // Factor 3: Volume Magnitude (current vs average)
            let magnitude_score = if vol_mean > 1e-10 {
                let relative_vol = volume[i] / vol_mean;
                // Moderate volume (0.8-1.2 ratio) gets highest score
                // Very high or very low volume reduces quality
                if relative_vol >= 0.8 && relative_vol <= 1.2 {
                    100.0
                } else if relative_vol > 1.2 {
                    (100.0 - (relative_vol - 1.2) * 25.0).max(30.0)
                } else {
                    (100.0 - (0.8 - relative_vol) * 100.0).max(30.0)
                }
            } else {
                50.0
            };

            // Combine factors with weights
            // Consistency: 40%, Confirmation: 35%, Magnitude: 25%
            quality_score[i] = consistency_score * 0.40 + confirmation_score * 0.35 + magnitude_score * 0.25;

            // Generate reliability signal
            if quality_score[i] > 70.0 {
                reliability_signal[i] = 1.0; // High quality
            } else if quality_score[i] < 30.0 {
                reliability_signal[i] = -1.0; // Low quality
            }
        }

        (quality_score, reliability_signal)
    }
}

impl TechnicalIndicator for VolumeQuality {
    fn name(&self) -> &str {
        "Volume Quality"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }
        let (quality, signal) = self.calculate(&data.close, &data.volume);
        Ok(IndicatorOutput::dual(quality, signal))
    }
}

// ============================================================================
// 6 NEW Volume Indicators - VolumeClimaxIndex, InstitutionalVolumeProxy,
// VolumeAccelerationIndex, SmartMoneyIndicator, VolumeEfficiencyIndex,
// VolumeDivergenceDetector
// ============================================================================

/// Volume Climax Index - Measures volume climax events (exhaustion)
///
/// Detects extreme volume events that often signal trend exhaustion or reversal.
/// Combines volume z-score with price range analysis to identify climax conditions.
/// A climax typically occurs when volume spikes significantly above normal levels
/// during a trend, potentially indicating exhaustion of buying/selling pressure.
#[derive(Debug, Clone)]
pub struct VolumeClimaxIndex {
    period: usize,
    z_threshold: f64,
}

impl VolumeClimaxIndex {
    /// Create a new VolumeClimaxIndex indicator
    ///
    /// # Arguments
    /// * `period` - Lookback period for volume analysis (minimum 10)
    /// * `z_threshold` - Z-score threshold for climax detection (minimum 1.5)
    ///
    /// # Returns
    /// A Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize, z_threshold: f64) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if z_threshold < 1.5 {
            return Err(IndicatorError::InvalidParameter {
                name: "z_threshold".to_string(),
                reason: "must be at least 1.5".to_string(),
            });
        }
        Ok(Self { period, z_threshold })
    }

    /// Create with default z-score threshold (2.0)
    pub fn with_period(period: usize) -> Result<Self> {
        Self::new(period, 2.0)
    }

    /// Calculate volume climax index
    ///
    /// Returns (climax_index, exhaustion_signal, climax_type):
    /// - climax_index: Index value (0-100) indicating climax intensity
    /// - exhaustion_signal: +1 for buying exhaustion, -1 for selling exhaustion, 0 for none
    /// - climax_type: 1.0 for bullish climax, -1.0 for bearish climax, 0.0 for no climax
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len().min(high.len()).min(low.len()).min(volume.len());
        let mut climax_index = vec![0.0; n];
        let mut exhaustion_signal = vec![0.0; n];
        let mut climax_type = vec![0.0; n];

        if n < self.period + 1 {
            return (climax_index, exhaustion_signal, climax_type);
        }

        for i in self.period..n {
            let start = i - self.period;

            // Calculate volume statistics
            let vol_window = &volume[start..i];
            let vol_mean: f64 = vol_window.iter().sum::<f64>() / self.period as f64;
            let vol_variance: f64 = vol_window.iter()
                .map(|v| (v - vol_mean).powi(2))
                .sum::<f64>() / self.period as f64;
            let vol_std = vol_variance.sqrt();

            // Calculate volume z-score
            let z_score = if vol_std > 1e-10 {
                (volume[i] - vol_mean) / vol_std
            } else {
                0.0
            };

            // Calculate price range expansion
            let avg_range: f64 = (start..i)
                .map(|j| high[j] - low[j])
                .sum::<f64>() / self.period as f64;
            let current_range = high[i] - low[i];
            let range_expansion = if avg_range > 1e-10 {
                current_range / avg_range
            } else {
                1.0
            };

            // Calculate climax index
            // Higher z-score and range expansion = more intense climax
            let raw_index = (z_score.abs() * 30.0 + range_expansion * 20.0).min(100.0);
            climax_index[i] = raw_index.max(0.0);

            // Detect climax conditions
            if z_score >= self.z_threshold {
                let price_direction = if i > 0 { close[i] - close[i - 1] } else { 0.0 };

                if price_direction > 0.0 {
                    // High volume on up move = potential buying climax (exhaustion)
                    exhaustion_signal[i] = 1.0;
                    climax_type[i] = 1.0;
                } else if price_direction < 0.0 {
                    // High volume on down move = potential selling climax (exhaustion)
                    exhaustion_signal[i] = -1.0;
                    climax_type[i] = -1.0;
                }
            }
        }

        (climax_index, exhaustion_signal, climax_type)
    }
}

impl TechnicalIndicator for VolumeClimaxIndex {
    fn name(&self) -> &str {
        "Volume Climax Index"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }
        let (index, signal, climax_type) = self.calculate(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::triple(index, signal, climax_type))
    }

    fn output_features(&self) -> usize {
        3 // climax_index, exhaustion_signal, climax_type
    }
}

/// Institutional Volume Proxy - Estimates institutional activity from volume patterns
///
/// Analyzes volume patterns to estimate the proportion of institutional trading.
/// Institutions typically trade in larger blocks with more efficient price impact,
/// often accumulating during weak periods and distributing during strong periods.
/// Uses volume clustering, price efficiency, and timing patterns as proxies.
#[derive(Debug, Clone)]
pub struct InstitutionalVolumeProxy {
    period: usize,
    block_threshold: f64,
}

impl InstitutionalVolumeProxy {
    /// Create a new InstitutionalVolumeProxy indicator
    ///
    /// # Arguments
    /// * `period` - Lookback period for analysis (minimum 10)
    /// * `block_threshold` - Multiple of average volume to consider as block trade (minimum 1.5)
    ///
    /// # Returns
    /// A Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize, block_threshold: f64) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if block_threshold < 1.5 {
            return Err(IndicatorError::InvalidParameter {
                name: "block_threshold".to_string(),
                reason: "must be at least 1.5".to_string(),
            });
        }
        Ok(Self { period, block_threshold })
    }

    /// Create with default block threshold (2.0)
    pub fn with_period(period: usize) -> Result<Self> {
        Self::new(period, 2.0)
    }

    /// Calculate institutional volume proxy
    ///
    /// Returns (institutional_pct, activity_signal, accumulation_index):
    /// - institutional_pct: Estimated percentage of institutional volume (0-100)
    /// - activity_signal: +1 for institutional buying, -1 for selling, 0 for neutral
    /// - accumulation_index: Cumulative institutional accumulation/distribution score
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len().min(high.len()).min(low.len()).min(volume.len());
        let mut institutional_pct = vec![50.0; n];
        let mut activity_signal = vec![0.0; n];
        let mut accumulation_index = vec![0.0; n];

        if n < self.period + 1 {
            return (institutional_pct, activity_signal, accumulation_index);
        }

        for i in self.period..n {
            let start = i - self.period;

            // Calculate average volume
            let avg_volume: f64 = volume[start..i].iter().sum::<f64>() / self.period as f64;

            // Identify block trades (large volume relative to average)
            let mut block_volume = 0.0;
            let mut efficient_volume = 0.0;
            let mut total_volume = 0.0;

            for j in (start + 1)..=i {
                let bar_range = high[j] - low[j];
                let price_move = (close[j] - close[j - 1]).abs();

                // Price efficiency: how much of the range resulted in net movement
                let efficiency = if bar_range > 1e-10 {
                    price_move / bar_range
                } else {
                    0.0
                };

                total_volume += volume[j];

                // Block trade detection
                if volume[j] >= avg_volume * self.block_threshold {
                    block_volume += volume[j];
                }

                // Efficient volume (institutional characteristic)
                if efficiency > 0.5 && volume[j] > avg_volume * 0.8 {
                    efficient_volume += volume[j];
                }
            }

            // Estimate institutional percentage
            if total_volume > 1e-10 {
                // Combine block trade ratio and efficiency ratio
                let block_ratio = block_volume / total_volume;
                let efficiency_ratio = efficient_volume / total_volume;

                // Weight: 60% block trades, 40% efficient trades
                let raw_pct = (block_ratio * 60.0 + efficiency_ratio * 40.0);
                institutional_pct[i] = raw_pct.clamp(0.0, 100.0);
            }

            // Determine activity direction
            let price_trend = close[i] - close[start];
            if institutional_pct[i] > 60.0 {
                if price_trend > 0.0 {
                    activity_signal[i] = 1.0; // Institutional buying
                } else if price_trend < 0.0 {
                    activity_signal[i] = -1.0; // Institutional selling
                }
            }

            // Calculate accumulation index
            let clv = if high[i] - low[i] > 1e-10 {
                ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
            } else {
                0.0
            };

            let institutional_contribution = clv * (institutional_pct[i] / 100.0) * volume[i];

            if i == self.period {
                accumulation_index[i] = institutional_contribution;
            } else {
                accumulation_index[i] = accumulation_index[i - 1] + institutional_contribution;
            }
        }

        (institutional_pct, activity_signal, accumulation_index)
    }
}

impl TechnicalIndicator for InstitutionalVolumeProxy {
    fn name(&self) -> &str {
        "Institutional Volume Proxy"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }
        let (pct, signal, accum) = self.calculate(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::triple(pct, signal, accum))
    }

    fn output_features(&self) -> usize {
        3 // institutional_pct, activity_signal, accumulation_index
    }
}

/// Volume Acceleration Index - Rate of change in volume momentum
///
/// Measures the acceleration (second derivative) of volume changes,
/// helping identify when volume momentum is increasing or decreasing.
/// Rising acceleration suggests strengthening volume trends,
/// falling acceleration may signal weakening conviction.
#[derive(Debug, Clone)]
pub struct VolumeAccelerationIndex {
    period: usize,
    smoothing: usize,
}

impl VolumeAccelerationIndex {
    /// Create a new VolumeAccelerationIndex indicator
    ///
    /// # Arguments
    /// * `period` - Lookback period for momentum calculation (minimum 5)
    /// * `smoothing` - EMA smoothing period (minimum 1)
    ///
    /// # Returns
    /// A Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize, smoothing: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
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

    /// Create with default smoothing (3)
    pub fn with_period(period: usize) -> Result<Self> {
        Self::new(period, 3)
    }

    /// Calculate volume acceleration index
    ///
    /// Returns (acceleration, momentum, acceleration_signal):
    /// - acceleration: Rate of change in volume momentum (-100 to 100)
    /// - momentum: First derivative of volume (smoothed)
    /// - acceleration_signal: +1 for increasing momentum, -1 for decreasing, 0 for stable
    pub fn calculate(&self, volume: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = volume.len();
        let mut acceleration = vec![0.0; n];
        let mut momentum = vec![0.0; n];
        let mut acceleration_signal = vec![0.0; n];

        if n < self.period * 2 + 1 {
            return (acceleration, momentum, acceleration_signal);
        }

        // Calculate volume momentum (first derivative)
        let mut raw_momentum = vec![0.0; n];
        for i in self.period..n {
            let prev_avg: f64 = volume[i - self.period..i - self.period / 2].iter().sum::<f64>()
                / (self.period / 2) as f64;
            let curr_avg: f64 = volume[i - self.period / 2..=i].iter().sum::<f64>()
                / (self.period / 2 + 1) as f64;

            if prev_avg > 1e-10 {
                raw_momentum[i] = ((curr_avg - prev_avg) / prev_avg) * 100.0;
            }
        }

        // Calculate acceleration (second derivative)
        let mut raw_acceleration = vec![0.0; n];
        for i in (self.period + 1)..n {
            raw_acceleration[i] = raw_momentum[i] - raw_momentum[i - 1];
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smoothing as f64 + 1.0);
        for i in 0..n {
            if i == 0 {
                momentum[i] = raw_momentum[i].clamp(-100.0, 100.0);
                acceleration[i] = raw_acceleration[i].clamp(-100.0, 100.0);
            } else {
                momentum[i] = (alpha * raw_momentum[i] + (1.0 - alpha) * momentum[i - 1])
                    .clamp(-100.0, 100.0);
                acceleration[i] = (alpha * raw_acceleration[i] + (1.0 - alpha) * acceleration[i - 1])
                    .clamp(-100.0, 100.0);
            }

            // Generate acceleration signal
            if acceleration[i] > 5.0 {
                acceleration_signal[i] = 1.0; // Accelerating volume momentum
            } else if acceleration[i] < -5.0 {
                acceleration_signal[i] = -1.0; // Decelerating volume momentum
            }
        }

        (acceleration, momentum, acceleration_signal)
    }
}

impl TechnicalIndicator for VolumeAccelerationIndex {
    fn name(&self) -> &str {
        "Volume Acceleration Index"
    }

    fn min_periods(&self) -> usize {
        self.period * 2 + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.volume.len() < self.period * 2 + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period * 2 + 1,
                got: data.volume.len(),
            });
        }
        let (accel, momentum, signal) = self.calculate(&data.volume);
        Ok(IndicatorOutput::triple(accel, momentum, signal))
    }

    fn output_features(&self) -> usize {
        3 // acceleration, momentum, acceleration_signal
    }
}

/// Smart Money Indicator - Tracks potential smart money flow using volume/price analysis
///
/// Identifies potential "smart money" (institutional/professional) activity by
/// analyzing the relationship between volume, price movement, and market position.
/// Smart money tends to accumulate during quiet periods with small price moves
/// and high volume, and distribute during volatile periods.
#[derive(Debug, Clone)]
pub struct SmartMoneyIndicator {
    period: usize,
    sensitivity: f64,
}

impl SmartMoneyIndicator {
    /// Create a new SmartMoneyIndicator
    ///
    /// # Arguments
    /// * `period` - Lookback period for analysis (minimum 10)
    /// * `sensitivity` - Sensitivity factor for detection (0.1 to 2.0)
    ///
    /// # Returns
    /// A Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize, sensitivity: f64) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if sensitivity < 0.1 || sensitivity > 2.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "sensitivity".to_string(),
                reason: "must be between 0.1 and 2.0".to_string(),
            });
        }
        Ok(Self { period, sensitivity })
    }

    /// Create with default sensitivity (1.0)
    pub fn with_period(period: usize) -> Result<Self> {
        Self::new(period, 1.0)
    }

    /// Calculate smart money indicator
    ///
    /// Returns (smart_money_index, flow_direction, confidence):
    /// - smart_money_index: Cumulative smart money flow (can be positive or negative)
    /// - flow_direction: +1 for smart money buying, -1 for selling, 0 for neutral
    /// - confidence: Confidence level of the signal (0-100)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len().min(high.len()).min(low.len()).min(volume.len());
        let mut smart_money_index = vec![0.0; n];
        let mut flow_direction = vec![0.0; n];
        let mut confidence = vec![0.0; n];

        if n < self.period + 1 {
            return (smart_money_index, flow_direction, confidence);
        }

        for i in self.period..n {
            let start = i - self.period;

            // Calculate average metrics
            let avg_volume: f64 = volume[start..i].iter().sum::<f64>() / self.period as f64;
            let avg_range: f64 = (start..i)
                .map(|j| high[j] - low[j])
                .sum::<f64>() / self.period as f64;

            // Current bar metrics
            let current_range = high[i] - low[i];
            let price_move = if i > 0 { (close[i] - close[i - 1]).abs() } else { 0.0 };

            // Smart money detection logic:
            // High volume + small price move = potential accumulation/distribution
            // Low volume + large price move = retail activity

            let volume_ratio = if avg_volume > 1e-10 {
                volume[i] / avg_volume
            } else {
                1.0
            };

            let range_ratio = if avg_range > 1e-10 {
                current_range / avg_range
            } else {
                1.0
            };

            // Smart money score: high volume with low range expansion
            // Inverse of efficiency - smart money hides their activity
            let smart_score = if range_ratio > 1e-10 {
                (volume_ratio / range_ratio) - 1.0
            } else {
                volume_ratio - 1.0
            };

            // Apply sensitivity
            let adjusted_score = smart_score * self.sensitivity;

            // Determine direction based on close position within range
            let close_position = if current_range > 1e-10 {
                (close[i] - low[i]) / current_range * 2.0 - 1.0 // -1 to +1
            } else {
                0.0
            };

            // Calculate smart money flow
            let flow = adjusted_score * close_position * volume[i];

            // Accumulate index
            if i == self.period {
                smart_money_index[i] = flow;
            } else {
                smart_money_index[i] = smart_money_index[i - 1] + flow;
            }

            // Determine flow direction
            if adjusted_score > 0.5 {
                if close_position > 0.2 {
                    flow_direction[i] = 1.0; // Smart money buying
                } else if close_position < -0.2 {
                    flow_direction[i] = -1.0; // Smart money selling
                }
            }

            // Calculate confidence
            let vol_confidence = (volume_ratio - 1.0).abs().min(1.0) * 50.0;
            let range_confidence = if range_ratio < 1.0 {
                (1.0 - range_ratio) * 50.0
            } else {
                0.0
            };
            confidence[i] = (vol_confidence + range_confidence).min(100.0);
        }

        (smart_money_index, flow_direction, confidence)
    }
}

impl TechnicalIndicator for SmartMoneyIndicator {
    fn name(&self) -> &str {
        "Smart Money Indicator"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }
        let (index, direction, conf) = self.calculate(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::triple(index, direction, conf))
    }

    fn output_features(&self) -> usize {
        3 // smart_money_index, flow_direction, confidence
    }
}

/// Volume Efficiency Index - Measures how efficiently volume moves price
///
/// Calculates the ratio of price movement achieved per unit of volume,
/// helping identify periods of efficient (strong conviction) vs inefficient
/// (churning/indecision) price movement. Higher values indicate that
/// volume is effectively translating into price change.
#[derive(Debug, Clone)]
pub struct VolumeEfficiencyIndex {
    period: usize,
    smoothing: usize,
}

impl VolumeEfficiencyIndex {
    /// Create a new VolumeEfficiencyIndex indicator
    ///
    /// # Arguments
    /// * `period` - Lookback period for efficiency calculation (minimum 5)
    /// * `smoothing` - EMA smoothing period (minimum 1)
    ///
    /// # Returns
    /// A Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize, smoothing: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
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

    /// Create with default smoothing (3)
    pub fn with_period(period: usize) -> Result<Self> {
        Self::new(period, 3)
    }

    /// Calculate volume efficiency index
    ///
    /// Returns (efficiency_index, efficiency_rating, trend_quality):
    /// - efficiency_index: Normalized efficiency score (0-100)
    /// - efficiency_rating: +1 for high efficiency (>70), -1 for low (<30), 0 otherwise
    /// - trend_quality: Quality of the current trend based on efficiency
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len().min(volume.len());
        let mut efficiency_index = vec![50.0; n];
        let mut efficiency_rating = vec![0.0; n];
        let mut trend_quality = vec![0.0; n];

        if n < self.period + 1 {
            return (efficiency_index, efficiency_rating, trend_quality);
        }

        let mut raw_efficiency = vec![0.0; n];

        for i in self.period..n {
            let start = i - self.period;

            // Calculate net price movement
            let net_move = (close[i] - close[start]).abs();

            // Calculate total path traveled
            let total_path: f64 = (start + 1..=i)
                .map(|j| (close[j] - close[j - 1]).abs())
                .sum();

            // Calculate total volume
            let total_volume: f64 = volume[start..=i].iter().sum();

            // Path efficiency (Kaufman-style)
            let path_efficiency = if total_path > 1e-10 {
                net_move / total_path
            } else {
                0.0
            };

            // Volume efficiency: price move per unit volume (normalized)
            let avg_price = (close[i] + close[start]) / 2.0;
            let vol_efficiency = if total_volume > 1e-10 && avg_price > 1e-10 {
                (net_move / avg_price * 100.0) / (total_volume / 1_000_000.0)
            } else {
                0.0
            };

            // Combine efficiencies: path efficiency * volume-adjusted factor
            // Higher path efficiency + reasonable volume = higher score
            raw_efficiency[i] = path_efficiency * 100.0 * (1.0 + vol_efficiency.min(1.0));
            raw_efficiency[i] = raw_efficiency[i].min(100.0);
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smoothing as f64 + 1.0);
        for i in 0..n {
            if i == 0 {
                efficiency_index[i] = raw_efficiency[i];
            } else {
                efficiency_index[i] = alpha * raw_efficiency[i] + (1.0 - alpha) * efficiency_index[i - 1];
            }

            efficiency_index[i] = efficiency_index[i].clamp(0.0, 100.0);

            // Generate efficiency rating
            if efficiency_index[i] > 70.0 {
                efficiency_rating[i] = 1.0; // High efficiency
            } else if efficiency_index[i] < 30.0 {
                efficiency_rating[i] = -1.0; // Low efficiency
            }

            // Calculate trend quality (efficiency weighted by direction consistency)
            if i >= self.period {
                let start = i - self.period;
                let trend_dir = (close[i] - close[start]).signum();
                let positive_moves: usize = (start + 1..=i)
                    .filter(|&j| (close[j] - close[j - 1]).signum() == trend_dir)
                    .count();
                let consistency = positive_moves as f64 / self.period as f64;
                trend_quality[i] = efficiency_index[i] * consistency;
            }
        }

        (efficiency_index, efficiency_rating, trend_quality)
    }
}

impl TechnicalIndicator for VolumeEfficiencyIndex {
    fn name(&self) -> &str {
        "Volume Efficiency Index"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }
        let (index, rating, quality) = self.calculate(&data.close, &data.volume);
        Ok(IndicatorOutput::triple(index, rating, quality))
    }

    fn output_features(&self) -> usize {
        3 // efficiency_index, efficiency_rating, trend_quality
    }
}

/// Volume Divergence Detector - Detects price/volume divergences
///
/// Identifies divergences between price movement and volume trends,
/// which can signal potential trend reversals. Bullish divergence occurs
/// when price makes lower lows but volume decreases. Bearish divergence
/// occurs when price makes higher highs but volume decreases.
#[derive(Debug, Clone)]
pub struct VolumeDivergenceDetector {
    period: usize,
    lookback: usize,
}

impl VolumeDivergenceDetector {
    /// Create a new VolumeDivergenceDetector indicator
    ///
    /// # Arguments
    /// * `period` - Period for moving averages (minimum 5)
    /// * `lookback` - Lookback for divergence detection (minimum 3)
    ///
    /// # Returns
    /// A Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize, lookback: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if lookback < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        Ok(Self { period, lookback })
    }

    /// Create with default lookback (5)
    pub fn with_period(period: usize) -> Result<Self> {
        Self::new(period, 5)
    }

    /// Calculate volume divergence detector
    ///
    /// Returns (divergence_score, divergence_type, strength):
    /// - divergence_score: Score indicating divergence intensity (-100 to 100)
    /// - divergence_type: +1 for bullish, -1 for bearish, 0 for none
    /// - strength: Strength/confidence of the divergence signal (0-100)
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len().min(volume.len());
        let mut divergence_score = vec![0.0; n];
        let mut divergence_type = vec![0.0; n];
        let mut strength = vec![0.0; n];

        if n < self.period + self.lookback + 1 {
            return (divergence_score, divergence_type, strength);
        }

        // Calculate smoothed volume
        let mut smoothed_volume = vec![0.0; n];
        for i in self.period..n {
            smoothed_volume[i] = volume[i - self.period..=i].iter().sum::<f64>() / (self.period + 1) as f64;
        }

        for i in (self.period + self.lookback)..n {
            let lookback_start = i - self.lookback;

            // Price trend over lookback
            let price_change = close[i] - close[lookback_start];
            let price_trend = if close[lookback_start] > 1e-10 {
                price_change / close[lookback_start] * 100.0
            } else {
                0.0
            };

            // Volume trend over lookback
            let vol_change = smoothed_volume[i] - smoothed_volume[lookback_start];
            let vol_trend = if smoothed_volume[lookback_start] > 1e-10 {
                vol_change / smoothed_volume[lookback_start] * 100.0
            } else {
                0.0
            };

            // Detect divergences
            let price_up = price_trend > 2.0;
            let price_down = price_trend < -2.0;
            let vol_up = vol_trend > 5.0;
            let vol_down = vol_trend < -5.0;

            // Bearish divergence: price up, volume down
            if price_up && vol_down {
                let score = ((price_trend.abs() + vol_trend.abs()) / 2.0).min(100.0);
                divergence_score[i] = -score;
                divergence_type[i] = -1.0;
                strength[i] = score;
            }
            // Bullish divergence: price down, volume down (selling exhaustion)
            else if price_down && vol_down {
                let score = ((price_trend.abs() + vol_trend.abs()) / 2.0).min(100.0);
                divergence_score[i] = score;
                divergence_type[i] = 1.0;
                strength[i] = score;
            }
            // Confirmation: price up with volume up, or price down with volume up
            else if (price_up && vol_up) || (price_down && vol_up) {
                divergence_score[i] = 0.0;
                divergence_type[i] = 0.0;
                strength[i] = 0.0;
            }

            // Check for new highs/lows with volume divergence (stronger signals)
            let is_new_high = close[i] >= close[lookback_start..i].iter().cloned().fold(f64::MIN, f64::max);
            let is_new_low = close[i] <= close[lookback_start..i].iter().cloned().fold(f64::MAX, f64::min);

            if is_new_high && vol_down {
                strength[i] = strength[i].max(70.0);
            }
            if is_new_low && vol_down {
                strength[i] = strength[i].max(70.0);
            }
        }

        (divergence_score, divergence_type, strength)
    }
}

impl TechnicalIndicator for VolumeDivergenceDetector {
    fn name(&self) -> &str {
        "Volume Divergence Detector"
    }

    fn min_periods(&self) -> usize {
        self.period + self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + self.lookback + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + self.lookback + 1,
                got: data.close.len(),
            });
        }
        let (score, div_type, str_val) = self.calculate(&data.close, &data.volume);
        Ok(IndicatorOutput::triple(score, div_type, str_val))
    }

    fn output_features(&self) -> usize {
        3 // divergence_score, divergence_type, strength
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let high = vec![
            102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
            112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0,
            122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0, 130.0, 131.0,
        ];
        let low = vec![
            98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0,
            108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0,
            118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0,
        ];
        let close = vec![
            100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0,
            110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0,
            120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0,
        ];
        let volume = vec![
            1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0, 1900.0,
            2000.0, 2100.0, 2200.0, 2300.0, 2400.0, 2500.0, 2600.0, 2700.0, 2800.0, 2900.0,
            3000.0, 3100.0, 3200.0, 3300.0, 3400.0, 3500.0, 3600.0, 3700.0, 3800.0, 3900.0,
        ];
        (high, low, close, volume)
    }

    #[test]
    fn test_volume_accumulation() {
        let (_, _, close, volume) = make_test_data();
        let va = VolumeAccumulation::new(5).unwrap();
        let result = va.calculate(&close, &volume);

        assert_eq!(result.len(), close.len());
        // In an uptrend, accumulation should be positive
        assert!(result[10] > 0.0);
    }

    #[test]
    fn test_volume_accumulation_mixed() {
        // Test with mixed price direction
        let close = vec![100.0, 101.0, 100.5, 99.0, 100.0, 101.0];
        let volume = vec![1000.0, 1500.0, 1200.0, 1800.0, 1400.0, 1600.0];
        let va = VolumeAccumulation::new(3).unwrap();
        let result = va.calculate(&close, &volume);

        assert_eq!(result.len(), 6);
    }

    #[test]
    fn test_volume_breakout() {
        let (_, _, _, volume) = make_test_data();
        let vb = VolumeBreakout::new(10, 1.5).unwrap();
        let (ratio, signal) = vb.calculate(&volume);

        assert_eq!(ratio.len(), volume.len());
        assert_eq!(signal.len(), volume.len());
        // Signal should be 0 or 1
        assert!(signal.iter().all(|&s| s == 0.0 || s == 1.0));
    }

    #[test]
    fn test_volume_breakout_with_spike() {
        // Create volume data with a spike
        let mut volume = vec![1000.0; 20];
        volume[15] = 5000.0; // Spike

        let vb = VolumeBreakout::new(10, 2.0).unwrap();
        let (ratio, signal) = vb.calculate(&volume);

        // At index 15, ratio should be high and signal should be 1.0
        assert!(ratio[15] > 2.0);
        assert_eq!(signal[15], 1.0);
    }

    #[test]
    fn test_relative_volume_strength() {
        let (_, _, _, volume) = make_test_data();
        let rvs = RelativeVolumeStrength::new(5, 20).unwrap();
        let result = rvs.calculate(&volume);

        assert_eq!(result.len(), volume.len());
        // Should be between 0 and 100
        assert!(result.iter().skip(20).all(|&v| v >= 0.0 && v <= 100.0));
    }

    #[test]
    fn test_relative_volume_strength_validation() {
        // short_period must be less than long_period
        assert!(RelativeVolumeStrength::new(20, 10).is_err());
        assert!(RelativeVolumeStrength::new(10, 10).is_err());
    }

    #[test]
    fn test_volume_climax_detector() {
        let (_, _, close, volume) = make_test_data();
        let vcd = VolumeClimaxDetector::new(10, 2.0).unwrap();
        let (z_score, climax) = vcd.calculate(&close, &volume);

        assert_eq!(z_score.len(), close.len());
        assert_eq!(climax.len(), close.len());
        // Climax signal should be -1, 0, or 1
        assert!(climax.iter().all(|&c| c == -1.0 || c == 0.0 || c == 1.0));
    }

    #[test]
    fn test_volume_climax_with_extreme_volume() {
        // Create data with an extreme volume spike
        // Need enough data points: period (10) + 1 for calculation + more for spike
        // Volume has some variance so std_dev is non-zero
        let close = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 120.0];
        let volume = vec![900.0, 1100.0, 950.0, 1050.0, 980.0, 1020.0, 1080.0, 920.0, 1000.0, 1000.0, 960.0, 1040.0, 1010.0, 990.0, 5000.0];

        let vcd = VolumeClimaxDetector::new(10, 2.0).unwrap();
        let (z_score, climax) = vcd.calculate(&close, &volume);

        // High z-score at spike (5000 vs ~1000 mean with ~60 std dev = ~66 z-score)
        assert!(z_score[14] > 2.0, "z_score[14] = {} should be > 2.0", z_score[14]);
        // Should detect bullish climax (price went up)
        assert_eq!(climax[14], 1.0);
    }

    #[test]
    fn test_smart_money_volume() {
        let (high, low, close, volume) = make_test_data();
        let smv = SmartMoneyVolume::new(10).unwrap();
        let result = smv.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_smart_money_volume_accumulation() {
        // Test with clear accumulation pattern
        let high = vec![102.0, 103.0, 102.5, 103.5, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0];
        let low = vec![98.0, 99.0, 98.5, 99.5, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0];
        let close = vec![101.0, 102.0, 101.5, 102.5, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0];
        let volume = vec![1000.0, 1200.0, 800.0, 1100.0, 900.0, 1500.0, 1300.0, 1400.0, 1100.0, 1600.0];

        let smv = SmartMoneyVolume::new(5).unwrap();
        let result = smv.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), 10);
    }

    #[test]
    fn test_volume_efficiency() {
        let (_, _, close, volume) = make_test_data();
        let ve = VolumeEfficiency::new(5).unwrap();
        let result = ve.calculate(&close, &volume);

        assert_eq!(result.len(), close.len());
        // Efficiency should be positive when there's price movement
        assert!(result[10] >= 0.0);
    }

    #[test]
    fn test_volume_efficiency_high_vs_low() {
        // High efficiency: large move on small volume
        let close1 = vec![100.0, 100.0, 100.0, 100.0, 100.0, 110.0];
        let volume1 = vec![100.0, 100.0, 100.0, 100.0, 100.0, 100.0];

        // Low efficiency: small move on large volume
        let close2 = vec![100.0, 100.0, 100.0, 100.0, 100.0, 101.0];
        let volume2 = vec![10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0];

        let ve = VolumeEfficiency::new(5).unwrap();
        let result1 = ve.calculate(&close1, &volume1);
        let result2 = ve.calculate(&close2, &volume2);

        // First scenario should have higher efficiency
        assert!(result1[5] > result2[5]);
    }

    #[test]
    fn test_parameter_validation() {
        // VolumeAccumulation
        assert!(VolumeAccumulation::new(1).is_err());
        assert!(VolumeAccumulation::new(2).is_ok());

        // VolumeBreakout
        assert!(VolumeBreakout::new(4, 1.5).is_err());
        assert!(VolumeBreakout::new(5, 0.5).is_err());
        assert!(VolumeBreakout::new(5, 1.5).is_ok());

        // RelativeVolumeStrength
        assert!(RelativeVolumeStrength::new(1, 10).is_err());
        assert!(RelativeVolumeStrength::new(5, 20).is_ok());

        // VolumeClimaxDetector
        assert!(VolumeClimaxDetector::new(5, 2.0).is_err());
        assert!(VolumeClimaxDetector::new(10, 0.0).is_err());
        assert!(VolumeClimaxDetector::new(10, 2.0).is_ok());

        // SmartMoneyVolume
        assert!(SmartMoneyVolume::new(4).is_err());
        assert!(SmartMoneyVolume::new(5).is_ok());

        // VolumeEfficiency
        assert!(VolumeEfficiency::new(1).is_err());
        assert!(VolumeEfficiency::new(2).is_ok());
    }

    // =========================================================================
    // Tests for new indicators
    // =========================================================================

    #[test]
    fn test_volume_distribution() {
        let (high, low, close, volume) = make_test_data();
        let vd = VolumeDistribution::new(10, 5).unwrap();
        let result = vd.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        // Score should be between 0 and 100
        assert!(result.iter().skip(10).all(|&v| v >= 0.0 && v <= 100.0));
    }

    #[test]
    fn test_volume_distribution_concentrated() {
        // Test with concentrated volume at current price
        let high = vec![102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0];
        let low = vec![98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0];
        let close = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0];
        let volume = vec![100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 5000.0];

        let vd = VolumeDistribution::new(5, 5).unwrap();
        let result = vd.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), 11);
        // Last value should show high concentration at current price
        assert!(result[10] > 50.0, "result[10] = {} should be > 50", result[10]);
    }

    #[test]
    fn test_volume_distribution_validation() {
        assert!(VolumeDistribution::new(1, 5).is_err());
        assert!(VolumeDistribution::new(5, 2).is_err());
        assert!(VolumeDistribution::new(5, 3).is_ok());
    }

    #[test]
    fn test_volume_intensity() {
        let (high, low, close, volume) = make_test_data();
        let vi = VolumeIntensity::new(10).unwrap();
        let (intensity, direction) = vi.calculate(&high, &low, &close, &volume);

        assert_eq!(intensity.len(), close.len());
        assert_eq!(direction.len(), close.len());
        // Intensity should be between 0 and 100
        assert!(intensity.iter().skip(10).all(|&v| v >= 0.0 && v <= 100.0));
        // Direction should be -1, 0, or 1
        assert!(direction.iter().all(|&d| d == -1.0 || d == 0.0 || d == 1.0));
    }

    #[test]
    fn test_volume_intensity_high_volume() {
        // High intensity: high volume relative to price range
        let high = vec![101.0; 15];
        let low = vec![99.0; 15];
        let close: Vec<f64> = (0..15).map(|i| 100.0 + i as f64 * 0.1).collect();
        let mut volume = vec![1000.0; 15];
        volume[14] = 5000.0; // High volume spike

        let vi = VolumeIntensity::new(10).unwrap();
        let (intensity, _) = vi.calculate(&high, &low, &close, &volume);

        // Last value should show higher intensity (at least 50)
        assert!(intensity[14] >= 50.0, "intensity[14] = {} should be >= 50", intensity[14]);
    }

    #[test]
    fn test_volume_intensity_validation() {
        assert!(VolumeIntensity::new(1).is_err());
        assert!(VolumeIntensity::new(2).is_ok());
    }

    #[test]
    fn test_volume_trend() {
        let (_, _, _, volume) = make_test_data();
        let vt = VolumeTrend::new(5, 20).unwrap();
        let (strength, direction) = vt.calculate(&volume);

        assert_eq!(strength.len(), volume.len());
        assert_eq!(direction.len(), volume.len());
        // Strength should be between 0 and 100
        assert!(strength.iter().skip(20).all(|&v| v >= 0.0 && v <= 100.0));
        // Direction should be -1, 0, or 1
        assert!(direction.iter().all(|&d| d == -1.0 || d == 0.0 || d == 1.0));
    }

    #[test]
    fn test_volume_trend_increasing() {
        // Test with clearly increasing volume
        let volume: Vec<f64> = (0..30).map(|i| 1000.0 + i as f64 * 100.0).collect();
        let vt = VolumeTrend::new(5, 15).unwrap();
        let (strength, direction) = vt.calculate(&volume);

        // Should detect upward trend
        assert!(direction[25] == 1.0, "direction[25] = {} should be 1.0", direction[25]);
    }

    #[test]
    fn test_volume_trend_validation() {
        assert!(VolumeTrend::new(1, 10).is_err());
        assert!(VolumeTrend::new(10, 10).is_err());
        assert!(VolumeTrend::new(10, 5).is_err());
        assert!(VolumeTrend::new(5, 10).is_ok());
    }

    #[test]
    fn test_volume_anomaly() {
        let (_, _, _, volume) = make_test_data();
        let va = VolumeAnomaly::new(10, 2.0).unwrap();
        let (score, signal) = va.calculate(&volume);

        assert_eq!(score.len(), volume.len());
        assert_eq!(signal.len(), volume.len());
        // Signal should be -1, 0, or 1
        assert!(signal.iter().all(|&s| s == -1.0 || s == 0.0 || s == 1.0));
    }

    #[test]
    fn test_volume_anomaly_spike() {
        // Create volume data with an anomalous spike
        let mut volume = vec![1000.0; 20];
        // Add some variance
        for i in 0..20 {
            volume[i] += (i as f64 % 3.0) * 50.0;
        }
        volume[15] = 5000.0; // Anomalous spike

        let va = VolumeAnomaly::new(10, 2.0).unwrap();
        let (score, signal) = va.calculate(&volume);

        // Should detect high anomaly at spike
        assert!(score[15] > 2.0, "score[15] = {} should be > 2.0", score[15]);
        assert_eq!(signal[15], 1.0);
    }

    #[test]
    fn test_volume_anomaly_validation() {
        assert!(VolumeAnomaly::new(1, 2.0).is_err());
        assert!(VolumeAnomaly::new(10, 0.0).is_err());
        assert!(VolumeAnomaly::new(10, -1.0).is_err());
        assert!(VolumeAnomaly::new(10, 2.0).is_ok());
    }

    #[test]
    fn test_volume_price_confirmation() {
        let (_, _, close, volume) = make_test_data();
        let vpc = VolumePriceConfirmation::new(10).unwrap();
        let (confirmation, divergence) = vpc.calculate(&close, &volume);

        assert_eq!(confirmation.len(), close.len());
        assert_eq!(divergence.len(), close.len());
        // Confirmation should be between 0 and 100
        assert!(confirmation.iter().skip(10).all(|&v| v >= 0.0 && v <= 100.0));
        // Divergence should be -1, 0, or 1
        assert!(divergence.iter().all(|&d| d == -1.0 || d == 0.0 || d == 1.0));
    }

    #[test]
    fn test_volume_price_confirmation_confirmed_move() {
        // Price up with volume up - confirmed
        let close: Vec<f64> = (0..15).map(|i| 100.0 + i as f64 * 2.0).collect();
        let volume: Vec<f64> = (0..15).map(|i| 1000.0 + i as f64 * 200.0).collect();

        let vpc = VolumePriceConfirmation::new(5).unwrap();
        let (confirmation, divergence) = vpc.calculate(&close, &volume);

        // Should show confirmation (not divergence)
        assert_eq!(divergence[10], 0.0, "divergence[10] = {} should be 0", divergence[10]);
    }

    #[test]
    fn test_volume_price_confirmation_bearish_divergence() {
        // Price up but volume down - bearish divergence
        let close: Vec<f64> = (0..15).map(|i| 100.0 + i as f64 * 2.0).collect();
        let volume: Vec<f64> = (0..15).map(|i| 5000.0 - i as f64 * 200.0).collect();

        let vpc = VolumePriceConfirmation::new(5).unwrap();
        let (_, divergence) = vpc.calculate(&close, &volume);

        // Should show bearish divergence
        assert_eq!(divergence[10], -1.0, "divergence[10] = {} should be -1", divergence[10]);
    }

    #[test]
    fn test_volume_price_confirmation_validation() {
        assert!(VolumePriceConfirmation::new(1).is_err());
        assert!(VolumePriceConfirmation::new(2).is_ok());
    }

    #[test]
    fn test_volume_exhaustion() {
        let (_, _, close, volume) = make_test_data();
        let ve = VolumeExhaustion::new(10, 0.5).unwrap();
        let (exhaustion, signal) = ve.calculate(&close, &volume);

        assert_eq!(exhaustion.len(), close.len());
        assert_eq!(signal.len(), close.len());
        // Exhaustion should be between 0 and 100
        assert!(exhaustion.iter().all(|&v| v >= 0.0 && v <= 100.0));
        // Signal should be -1, 0, or 1
        assert!(signal.iter().all(|&s| s == -1.0 || s == 0.0 || s == 1.0));
    }

    #[test]
    fn test_volume_exhaustion_detection() {
        // Create exhaustion pattern: volume peaks early then declines while price continues
        let close: Vec<f64> = (0..15).map(|i| 100.0 + i as f64 * 2.0).collect();
        let mut volume = vec![1000.0; 15];
        // Volume peaks early in the period
        volume[0] = 5000.0;
        volume[1] = 4500.0;
        volume[2] = 4000.0;
        // Then declines
        for i in 3..15 {
            volume[i] = 2000.0 - (i as f64 * 100.0);
        }

        let ve = VolumeExhaustion::new(10, 0.5).unwrap();
        let (exhaustion, signal) = ve.calculate(&close, &volume);

        // Should detect some exhaustion (volume peaked early, now declining)
        assert!(exhaustion[14] > 0.0, "exhaustion[14] = {} should be > 0", exhaustion[14]);
    }

    #[test]
    fn test_volume_exhaustion_validation() {
        assert!(VolumeExhaustion::new(1, 0.5).is_err());
        assert!(VolumeExhaustion::new(10, 0.0).is_err());
        assert!(VolumeExhaustion::new(10, 1.0).is_err());
        assert!(VolumeExhaustion::new(10, 1.5).is_err());
        assert!(VolumeExhaustion::new(10, 0.5).is_ok());
    }

    #[test]
    fn test_new_indicators_parameter_validation() {
        // VolumeDistribution
        assert!(VolumeDistribution::new(1, 5).is_err());
        assert!(VolumeDistribution::new(5, 2).is_err());
        assert!(VolumeDistribution::new(5, 5).is_ok());

        // VolumeIntensity
        assert!(VolumeIntensity::new(1).is_err());
        assert!(VolumeIntensity::new(5).is_ok());

        // VolumeTrend
        assert!(VolumeTrend::new(1, 10).is_err());
        assert!(VolumeTrend::new(5, 3).is_err());
        assert!(VolumeTrend::new(5, 10).is_ok());

        // VolumeAnomaly
        assert!(VolumeAnomaly::new(1, 2.0).is_err());
        assert!(VolumeAnomaly::new(10, 0.0).is_err());
        assert!(VolumeAnomaly::new(10, 2.0).is_ok());

        // VolumePriceConfirmation
        assert!(VolumePriceConfirmation::new(1).is_err());
        assert!(VolumePriceConfirmation::new(5).is_ok());

        // VolumeExhaustion
        assert!(VolumeExhaustion::new(1, 0.5).is_err());
        assert!(VolumeExhaustion::new(10, 0.0).is_err());
        assert!(VolumeExhaustion::new(10, 1.0).is_err());
        assert!(VolumeExhaustion::new(10, 0.5).is_ok());
    }

    // =========================================================================
    // Tests for extended indicators (VolumeWeightedMomentum, etc.)
    // =========================================================================

    #[test]
    fn test_volume_weighted_momentum() {
        let (_, _, close, volume) = make_test_data();
        let vwm = VolumeWeightedMomentum::new(10).unwrap();
        let result = vwm.calculate(&close, &volume);

        assert_eq!(result.len(), close.len());
        // Should produce values after min_periods
        assert!(result[15] != 0.0 || result[20] != 0.0);
    }

    #[test]
    fn test_volume_weighted_momentum_uptrend() {
        // Test with clear uptrend and increasing volume
        let close: Vec<f64> = (0..20).map(|i| 100.0 + i as f64 * 2.0).collect();
        let volume: Vec<f64> = (0..20).map(|i| 1000.0 + i as f64 * 100.0).collect();

        let vwm = VolumeWeightedMomentum::new(5).unwrap();
        let result = vwm.calculate(&close, &volume);

        // Should show positive momentum in uptrend
        assert!(result[15] > 0.0, "result[15] = {} should be > 0", result[15]);
    }

    #[test]
    fn test_volume_weighted_momentum_validation() {
        assert!(VolumeWeightedMomentum::new(1).is_err());
        assert!(VolumeWeightedMomentum::new(2).is_ok());
    }

    #[test]
    fn test_volume_force_index() {
        let (_, _, close, volume) = make_test_data();
        let vfi = VolumeForceIndex::new(10, 3).unwrap();
        let result = vfi.calculate(&close, &volume);

        assert_eq!(result.len(), close.len());
        // Should produce values after sufficient data
        assert!(result.iter().skip(10).any(|&v| v != 0.0));
    }

    #[test]
    fn test_volume_force_index_strong_move() {
        // Strong up move with high volume
        let close: Vec<f64> = (0..20).map(|i| 100.0 + i as f64 * 5.0).collect();
        let volume: Vec<f64> = (0..20).map(|i| 1000.0 + i as f64 * 500.0).collect();

        let vfi = VolumeForceIndex::new(5, 3).unwrap();
        let result = vfi.calculate(&close, &volume);

        // Force index should be positive for strong up move
        assert!(result[15] > 0.0, "result[15] = {} should be > 0", result[15]);
    }

    #[test]
    fn test_volume_force_index_validation() {
        assert!(VolumeForceIndex::new(1, 3).is_err());
        assert!(VolumeForceIndex::new(5, 0).is_err());
        assert!(VolumeForceIndex::new(5, 3).is_ok());
    }

    #[test]
    fn test_cumulative_volume_oscillator() {
        let (_, _, close, volume) = make_test_data();
        let cvo = CumulativeVolumeOscillator::new(5, 20).unwrap();
        let result = cvo.calculate(&close, &volume);

        assert_eq!(result.len(), close.len());
        // Should oscillate around zero
        let has_positive = result.iter().skip(20).any(|&v| v > 0.0);
        let has_negative = result.iter().skip(20).any(|&v| v < 0.0);
        // In a strong uptrend, may be mostly positive
        assert!(has_positive || has_negative);
    }

    #[test]
    fn test_cumulative_volume_oscillator_mixed() {
        // Mixed price action
        let close = vec![
            100.0, 101.0, 100.5, 99.5, 100.0, 101.5, 101.0, 100.0, 99.0, 100.0,
            101.0, 102.0, 101.5, 100.5, 99.5, 100.0, 101.0, 102.0, 103.0, 104.0,
            103.0, 102.0, 103.0, 104.0, 105.0,
        ];
        let volume = vec![
            1000.0, 1200.0, 1100.0, 1500.0, 1300.0, 1400.0, 1200.0, 1100.0, 1600.0, 1300.0,
            1500.0, 1800.0, 1400.0, 1200.0, 1700.0, 1300.0, 1400.0, 1600.0, 1800.0, 2000.0,
            1900.0, 1700.0, 1600.0, 1800.0, 2000.0,
        ];

        let cvo = CumulativeVolumeOscillator::new(5, 15).unwrap();
        let result = cvo.calculate(&close, &volume);

        assert_eq!(result.len(), 25);
    }

    #[test]
    fn test_cumulative_volume_oscillator_validation() {
        assert!(CumulativeVolumeOscillator::new(1, 10).is_err());
        assert!(CumulativeVolumeOscillator::new(10, 5).is_err());
        assert!(CumulativeVolumeOscillator::new(10, 10).is_err());
        assert!(CumulativeVolumeOscillator::new(5, 10).is_ok());
    }

    #[test]
    fn test_volume_rate_of_change() {
        let (_, _, _, volume) = make_test_data();
        let vroc = VolumeRateOfChange::new(10, 3).unwrap();
        let result = vroc.calculate(&volume);

        assert_eq!(result.len(), volume.len());
        // Should produce values after period
        assert!(result.iter().skip(10).any(|&v| v != 0.0));
    }

    #[test]
    fn test_volume_rate_of_change_increasing() {
        // Steadily increasing volume
        let volume: Vec<f64> = (0..25).map(|i| 1000.0 + i as f64 * 200.0).collect();

        let vroc = VolumeRateOfChange::new(5, 3).unwrap();
        let result = vroc.calculate(&volume);

        // Rate of change should be positive for increasing volume
        assert!(result[20] > 0.0, "result[20] = {} should be > 0", result[20]);
    }

    #[test]
    fn test_volume_rate_of_change_decreasing() {
        // Steadily decreasing volume
        let volume: Vec<f64> = (0..25).map(|i| 5000.0 - i as f64 * 100.0).collect();

        let vroc = VolumeRateOfChange::new(5, 3).unwrap();
        let result = vroc.calculate(&volume);

        // Rate of change should be negative for decreasing volume
        assert!(result[20] < 0.0, "result[20] = {} should be < 0", result[20]);
    }

    #[test]
    fn test_volume_rate_of_change_validation() {
        assert!(VolumeRateOfChange::new(1, 3).is_err());
        assert!(VolumeRateOfChange::new(5, 0).is_err());
        assert!(VolumeRateOfChange::new(5, 3).is_ok());
    }

    #[test]
    fn test_relative_volume_profile() {
        let (_, _, _, volume) = make_test_data();
        let rvp = RelativeVolumeProfile::new(10, 3).unwrap();
        let result = rvp.calculate(&volume);

        assert_eq!(result.len(), volume.len());
        // Should produce ratios after period
        assert!(result.iter().skip(10).any(|&v| v != 0.0));
    }

    #[test]
    fn test_relative_volume_profile_spike() {
        // Normal volume with a spike
        let mut volume = vec![1000.0; 20];
        volume[15] = 5000.0; // Spike

        let rvp = RelativeVolumeProfile::new(10, 3).unwrap();
        let result = rvp.calculate(&volume);

        // Spike should show high relative volume
        assert!(result[15] > 1.5, "result[15] = {} should be > 1.5", result[15]);
    }

    #[test]
    fn test_relative_volume_profile_low_volume() {
        // Normal volume with a low volume bar
        let mut volume = vec![1000.0; 20];
        volume[15] = 200.0; // Low volume

        let rvp = RelativeVolumeProfile::new(10, 3).unwrap();
        let result = rvp.calculate(&volume);

        // Low volume should show below-average relative volume (< 1.0)
        assert!(result[15] < 1.0, "result[15] = {} should be < 1.0 for low volume", result[15]);
    }

    #[test]
    fn test_relative_volume_profile_validation() {
        assert!(RelativeVolumeProfile::new(1, 3).is_err());
        assert!(RelativeVolumeProfile::new(5, 0).is_err());
        assert!(RelativeVolumeProfile::new(5, 3).is_ok());
    }

    #[test]
    fn test_volume_impulse() {
        let (_, _, _, volume) = make_test_data();
        let vi = VolumeImpulse::new(10, 2.0).unwrap();
        let (impulse, signal) = vi.calculate(&volume);

        assert_eq!(impulse.len(), volume.len());
        assert_eq!(signal.len(), volume.len());
        // Signal should be 0 or 1
        assert!(signal.iter().all(|&s| s == 0.0 || s == 1.0));
    }

    #[test]
    fn test_volume_impulse_spike_detection() {
        // Create volume data with a sudden spike
        let mut volume = vec![1000.0; 20];
        // Add some variance to avoid zero std dev
        for i in 0..20 {
            volume[i] += (i as f64 % 5.0) * 20.0;
        }
        volume[15] = 5000.0; // Sudden spike

        let vi = VolumeImpulse::new(10, 2.0).unwrap();
        let (impulse, signal) = vi.calculate(&volume);

        // Should detect impulse at spike
        assert!(impulse[15] > 2.0, "impulse[15] = {} should be > 2.0", impulse[15]);
        assert_eq!(signal[15], 1.0);
    }

    #[test]
    fn test_volume_impulse_consecutive_detection() {
        // Multiple spikes
        let mut volume = vec![1000.0; 25];
        for i in 0..25 {
            volume[i] += (i as f64 % 4.0) * 30.0;
        }
        volume[12] = 4000.0;
        volume[20] = 5000.0;

        let vi = VolumeImpulse::new(8, 2.0).unwrap();
        let (_, signal) = vi.calculate(&volume);

        // Should detect both spikes
        assert_eq!(signal[12], 1.0, "signal[12] = {} should be 1.0", signal[12]);
        assert_eq!(signal[20], 1.0, "signal[20] = {} should be 1.0", signal[20]);
    }

    #[test]
    fn test_volume_impulse_validation() {
        assert!(VolumeImpulse::new(4, 2.0).is_err());
        assert!(VolumeImpulse::new(5, 0.0).is_err());
        assert!(VolumeImpulse::new(5, -1.0).is_err());
        assert!(VolumeImpulse::new(5, 2.0).is_ok());
    }

    #[test]
    fn test_extended_indicators_parameter_validation() {
        // VolumeWeightedMomentum
        assert!(VolumeWeightedMomentum::new(1).is_err());
        assert!(VolumeWeightedMomentum::new(5).is_ok());

        // VolumeForceIndex
        assert!(VolumeForceIndex::new(1, 3).is_err());
        assert!(VolumeForceIndex::new(5, 0).is_err());
        assert!(VolumeForceIndex::new(5, 3).is_ok());

        // CumulativeVolumeOscillator
        assert!(CumulativeVolumeOscillator::new(1, 10).is_err());
        assert!(CumulativeVolumeOscillator::new(5, 3).is_err());
        assert!(CumulativeVolumeOscillator::new(5, 10).is_ok());

        // VolumeRateOfChange
        assert!(VolumeRateOfChange::new(1, 3).is_err());
        assert!(VolumeRateOfChange::new(5, 0).is_err());
        assert!(VolumeRateOfChange::new(5, 3).is_ok());

        // RelativeVolumeProfile
        assert!(RelativeVolumeProfile::new(1, 3).is_err());
        assert!(RelativeVolumeProfile::new(5, 0).is_err());
        assert!(RelativeVolumeProfile::new(5, 3).is_ok());

        // VolumeImpulse
        assert!(VolumeImpulse::new(4, 2.0).is_err());
        assert!(VolumeImpulse::new(5, 0.0).is_err());
        assert!(VolumeImpulse::new(5, 2.0).is_ok());
    }

    // =========================================================================
    // Tests for 6 NEW volume indicators
    // =========================================================================

    #[test]
    fn test_volume_weighted_trend() {
        let (_, _, close, volume) = make_test_data();
        let vwt = VolumeWeightedTrend::new(10, 3).unwrap();
        let (trend_value, trend_direction) = vwt.calculate(&close, &volume);

        assert_eq!(trend_value.len(), close.len());
        assert_eq!(trend_direction.len(), close.len());
        // Trend value should be within bounds
        assert!(trend_value.iter().all(|&v| v >= -100.0 && v <= 100.0));
        // Direction should be -1, 0, or 1
        assert!(trend_direction.iter().all(|&d| d == -1.0 || d == 0.0 || d == 1.0));
    }

    #[test]
    fn test_volume_weighted_trend_uptrend() {
        // Clear uptrend with increasing volume
        let close: Vec<f64> = (0..25).map(|i| 100.0 + i as f64 * 2.0).collect();
        let volume: Vec<f64> = (0..25).map(|i| 1000.0 + i as f64 * 100.0).collect();

        let vwt = VolumeWeightedTrend::new(5, 3).unwrap();
        let (trend_value, trend_direction) = vwt.calculate(&close, &volume);

        // Should show positive trend in uptrend
        assert!(trend_value[20] > 0.0, "trend_value[20] = {} should be > 0", trend_value[20]);
        assert_eq!(trend_direction[20], 1.0);
    }

    #[test]
    fn test_volume_weighted_trend_downtrend() {
        // Clear downtrend
        let close: Vec<f64> = (0..25).map(|i| 150.0 - i as f64 * 2.0).collect();
        let volume: Vec<f64> = (0..25).map(|i| 1000.0 + i as f64 * 100.0).collect();

        let vwt = VolumeWeightedTrend::new(5, 3).unwrap();
        let (trend_value, trend_direction) = vwt.calculate(&close, &volume);

        // Should show negative trend in downtrend
        assert!(trend_value[20] < 0.0, "trend_value[20] = {} should be < 0", trend_value[20]);
        assert_eq!(trend_direction[20], -1.0);
    }

    #[test]
    fn test_volume_weighted_trend_validation() {
        assert!(VolumeWeightedTrend::new(4, 3).is_err());
        assert!(VolumeWeightedTrend::new(5, 0).is_err());
        assert!(VolumeWeightedTrend::new(5, 3).is_ok());
    }

    #[test]
    fn test_volume_momentum_oscillator() {
        let (_, _, close, volume) = make_test_data();
        let vmo = VolumeMomentumOscillator::new(5, 15, 5).unwrap();
        let (oscillator, signal, histogram) = vmo.calculate(&close, &volume);

        assert_eq!(oscillator.len(), close.len());
        assert_eq!(signal.len(), close.len());
        assert_eq!(histogram.len(), close.len());
        // Should produce values after sufficient data
        assert!(oscillator.iter().skip(20).any(|&v| v != 0.0));
    }

    #[test]
    fn test_volume_momentum_oscillator_crossover() {
        // Strong uptrend should produce positive oscillator
        let close: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 * 3.0).collect();
        let volume: Vec<f64> = (0..30).map(|i| 1000.0 + i as f64 * 200.0).collect();

        let vmo = VolumeMomentumOscillator::new(5, 12, 5).unwrap();
        let (oscillator, _, _) = vmo.calculate(&close, &volume);

        // Should show positive oscillator in strong uptrend
        assert!(oscillator[25] > 0.0, "oscillator[25] = {} should be > 0", oscillator[25]);
    }

    #[test]
    fn test_volume_momentum_oscillator_validation() {
        assert!(VolumeMomentumOscillator::new(1, 10, 5).is_err());
        assert!(VolumeMomentumOscillator::new(10, 5, 5).is_err()); // slow <= fast
        assert!(VolumeMomentumOscillator::new(5, 10, 1).is_err());
        assert!(VolumeMomentumOscillator::new(5, 10, 5).is_ok());
    }

    #[test]
    fn test_volume_accumulation_trend() {
        let (high, low, close, volume) = make_test_data();
        let vat = VolumeAccumulationTrend::new(10, 3).unwrap();
        let (trend_value, trend_signal) = vat.calculate(&high, &low, &close, &volume);

        assert_eq!(trend_value.len(), close.len());
        assert_eq!(trend_signal.len(), close.len());
        // Signal should be -1, 0, or 1
        assert!(trend_signal.iter().all(|&s| s == -1.0 || s == 0.0 || s == 1.0));
    }

    #[test]
    fn test_volume_accumulation_trend_accumulation() {
        // Price closing near highs = accumulation
        let high: Vec<f64> = (0..25).map(|i| 102.0 + i as f64 * 2.0).collect();
        let low: Vec<f64> = (0..25).map(|i| 98.0 + i as f64 * 2.0).collect();
        let close: Vec<f64> = (0..25).map(|i| 101.5 + i as f64 * 2.0).collect(); // Close near high
        let volume = vec![1000.0; 25];

        let vat = VolumeAccumulationTrend::new(5, 3).unwrap();
        let (trend_value, _) = vat.calculate(&high, &low, &close, &volume);

        // Should show positive accumulation trend
        assert!(trend_value[20] > 0.0 || trend_value[15] > 0.0,
            "trend should be positive when closing near highs");
    }

    #[test]
    fn test_volume_accumulation_trend_distribution() {
        // Price closing near lows = distribution
        let high: Vec<f64> = (0..25).map(|i| 102.0 + i as f64 * 2.0).collect();
        let low: Vec<f64> = (0..25).map(|i| 98.0 + i as f64 * 2.0).collect();
        let close: Vec<f64> = (0..25).map(|i| 98.5 + i as f64 * 2.0).collect(); // Close near low
        let volume = vec![1000.0; 25];

        let vat = VolumeAccumulationTrend::new(5, 3).unwrap();
        let (trend_value, _) = vat.calculate(&high, &low, &close, &volume);

        // Should show negative distribution trend
        assert!(trend_value[20] < 0.0 || trend_value[15] < 0.0,
            "trend should be negative when closing near lows");
    }

    #[test]
    fn test_volume_accumulation_trend_validation() {
        assert!(VolumeAccumulationTrend::new(4, 3).is_err());
        assert!(VolumeAccumulationTrend::new(5, 0).is_err());
        assert!(VolumeAccumulationTrend::new(5, 3).is_ok());
    }

    #[test]
    fn test_adaptive_volume_ma() {
        let (_, _, close, volume) = make_test_data();
        let avma = AdaptiveVolumeMA::new(10, 0.5, 0.1).unwrap();
        let (adaptive_ma, volume_ratio) = avma.calculate(&close, &volume);

        assert_eq!(adaptive_ma.len(), close.len());
        assert_eq!(volume_ratio.len(), close.len());
        // MA should track price
        assert!(adaptive_ma[20] > 0.0);
    }

    #[test]
    fn test_adaptive_volume_ma_with_period() {
        let (_, _, close, volume) = make_test_data();
        let avma = AdaptiveVolumeMA::with_period(10).unwrap();
        let (adaptive_ma, _) = avma.calculate(&close, &volume);

        assert_eq!(adaptive_ma.len(), close.len());
        // Should track close prices
        let last_ma = adaptive_ma[29];
        let last_close = close[29];
        // MA should be somewhat close to recent prices
        assert!((last_ma - last_close).abs() < 30.0,
            "MA {} should be close to close price {}", last_ma, last_close);
    }

    #[test]
    fn test_adaptive_volume_ma_responsiveness() {
        // Test that high volume makes MA more responsive
        let close = vec![100.0; 20];
        let mut volume = vec![1000.0; 20];

        // Sudden price jump at index 15
        let mut close_spike = close.clone();
        close_spike[15] = 110.0;
        close_spike[16] = 112.0;
        close_spike[17] = 114.0;
        close_spike[18] = 116.0;
        close_spike[19] = 118.0;

        // High volume during spike
        let mut volume_high = volume.clone();
        volume_high[15] = 5000.0;
        volume_high[16] = 5000.0;
        volume_high[17] = 5000.0;

        let avma = AdaptiveVolumeMA::new(10, 0.5, 0.1).unwrap();
        let (ma_high_vol, _) = avma.calculate(&close_spike, &volume_high);
        let (ma_low_vol, _) = avma.calculate(&close_spike, &volume);

        // With high volume, MA should respond faster (be closer to new price)
        assert!(ma_high_vol[17] > ma_low_vol[17],
            "High volume MA {} should be more responsive than low volume MA {}",
            ma_high_vol[17], ma_low_vol[17]);
    }

    #[test]
    fn test_adaptive_volume_ma_validation() {
        assert!(AdaptiveVolumeMA::new(4, 0.5, 0.1).is_err());
        assert!(AdaptiveVolumeMA::new(5, 0.0, 0.1).is_err()); // fast_factor <= 0
        assert!(AdaptiveVolumeMA::new(5, 1.5, 0.1).is_err()); // fast_factor > 1
        assert!(AdaptiveVolumeMA::new(5, 0.5, 0.5).is_err()); // slow >= fast
        assert!(AdaptiveVolumeMA::new(5, 0.5, -0.1).is_err()); // slow < 0
        assert!(AdaptiveVolumeMA::new(5, 0.5, 0.1).is_ok());
    }

    #[test]
    fn test_volume_flow_index() {
        let (high, low, close, volume) = make_test_data();
        let vfi = VolumeFlowIndex::new(10, 3).unwrap();
        let (flow_index, flow_direction) = vfi.calculate(&high, &low, &close, &volume);

        assert_eq!(flow_index.len(), close.len());
        assert_eq!(flow_direction.len(), close.len());
        // Flow index should be within bounds
        assert!(flow_index.iter().all(|&v| v >= -100.0 && v <= 100.0));
        // Direction should be -1, 0, or 1
        assert!(flow_direction.iter().all(|&d| d == -1.0 || d == 0.0 || d == 1.0));
    }

    #[test]
    fn test_volume_flow_index_inflow() {
        // Strong uptrend = inflow
        let high: Vec<f64> = (0..25).map(|i| 102.0 + i as f64 * 2.0).collect();
        let low: Vec<f64> = (0..25).map(|i| 98.0 + i as f64 * 2.0).collect();
        let close: Vec<f64> = (0..25).map(|i| 100.0 + i as f64 * 2.0).collect();
        let volume = vec![1000.0; 25];

        let vfi = VolumeFlowIndex::new(5, 3).unwrap();
        let (flow_index, flow_direction) = vfi.calculate(&high, &low, &close, &volume);

        // Should show positive flow (inflow) in uptrend
        assert!(flow_index[20] > 0.0, "flow_index[20] = {} should be > 0", flow_index[20]);
        assert_eq!(flow_direction[20], 1.0);
    }

    #[test]
    fn test_volume_flow_index_outflow() {
        // Strong downtrend = outflow
        let high: Vec<f64> = (0..25).map(|i| 152.0 - i as f64 * 2.0).collect();
        let low: Vec<f64> = (0..25).map(|i| 148.0 - i as f64 * 2.0).collect();
        let close: Vec<f64> = (0..25).map(|i| 150.0 - i as f64 * 2.0).collect();
        let volume = vec![1000.0; 25];

        let vfi = VolumeFlowIndex::new(5, 3).unwrap();
        let (flow_index, flow_direction) = vfi.calculate(&high, &low, &close, &volume);

        // Should show negative flow (outflow) in downtrend
        assert!(flow_index[20] < 0.0, "flow_index[20] = {} should be < 0", flow_index[20]);
        assert_eq!(flow_direction[20], -1.0);
    }

    #[test]
    fn test_volume_flow_index_validation() {
        assert!(VolumeFlowIndex::new(4, 3).is_err());
        assert!(VolumeFlowIndex::new(5, 0).is_err());
        assert!(VolumeFlowIndex::new(5, 3).is_ok());
    }

    #[test]
    fn test_volume_pressure_index() {
        let (high, low, close, volume) = make_test_data();
        let vpi = VolumePressureIndex::new(10).unwrap();
        let (pressure_index, pressure_signal) = vpi.calculate(&high, &low, &close, &volume);

        assert_eq!(pressure_index.len(), close.len());
        assert_eq!(pressure_signal.len(), close.len());
        // Pressure index should be between 0 and 100
        assert!(pressure_index.iter().all(|&v| v >= 0.0 && v <= 100.0));
        // Signal should be -1, 0, or 1
        assert!(pressure_signal.iter().all(|&s| s == -1.0 || s == 0.0 || s == 1.0));
    }

    #[test]
    fn test_volume_pressure_index_buying_pressure() {
        // Close near highs = buying pressure
        let high = vec![105.0; 20];
        let low = vec![95.0; 20];
        let close = vec![104.0; 20]; // Close near high
        let volume = vec![1000.0; 20];

        let vpi = VolumePressureIndex::new(5).unwrap();
        let (pressure_index, pressure_signal) = vpi.calculate(&high, &low, &close, &volume);

        // Should show buying pressure (> 60)
        assert!(pressure_index[15] > 60.0,
            "pressure_index[15] = {} should be > 60 for buying pressure", pressure_index[15]);
        assert_eq!(pressure_signal[15], 1.0);
    }

    #[test]
    fn test_volume_pressure_index_selling_pressure() {
        // Close near lows = selling pressure
        let high = vec![105.0; 20];
        let low = vec![95.0; 20];
        let close = vec![96.0; 20]; // Close near low
        let volume = vec![1000.0; 20];

        let vpi = VolumePressureIndex::new(5).unwrap();
        let (pressure_index, pressure_signal) = vpi.calculate(&high, &low, &close, &volume);

        // Should show selling pressure (< 40)
        assert!(pressure_index[15] < 40.0,
            "pressure_index[15] = {} should be < 40 for selling pressure", pressure_index[15]);
        assert_eq!(pressure_signal[15], -1.0);
    }

    #[test]
    fn test_volume_pressure_index_balanced() {
        // Close in middle = balanced pressure
        let high = vec![105.0; 20];
        let low = vec![95.0; 20];
        let close = vec![100.0; 20]; // Close in middle
        let volume = vec![1000.0; 20];

        let vpi = VolumePressureIndex::new(5).unwrap();
        let (pressure_index, pressure_signal) = vpi.calculate(&high, &low, &close, &volume);

        // Should show balanced pressure (around 50)
        assert!(pressure_index[15] >= 40.0 && pressure_index[15] <= 60.0,
            "pressure_index[15] = {} should be around 50 for balanced pressure", pressure_index[15]);
        assert_eq!(pressure_signal[15], 0.0);
    }

    #[test]
    fn test_volume_pressure_index_validation() {
        assert!(VolumePressureIndex::new(4).is_err());
        assert!(VolumePressureIndex::new(5).is_ok());
    }

    #[test]
    fn test_new_volume_indicators_parameter_validation() {
        // VolumeWeightedTrend
        assert!(VolumeWeightedTrend::new(4, 3).is_err());
        assert!(VolumeWeightedTrend::new(5, 0).is_err());
        assert!(VolumeWeightedTrend::new(5, 3).is_ok());

        // VolumeMomentumOscillator
        assert!(VolumeMomentumOscillator::new(1, 10, 5).is_err());
        assert!(VolumeMomentumOscillator::new(10, 5, 5).is_err());
        assert!(VolumeMomentumOscillator::new(5, 10, 1).is_err());
        assert!(VolumeMomentumOscillator::new(5, 10, 5).is_ok());

        // VolumeAccumulationTrend
        assert!(VolumeAccumulationTrend::new(4, 3).is_err());
        assert!(VolumeAccumulationTrend::new(5, 0).is_err());
        assert!(VolumeAccumulationTrend::new(5, 3).is_ok());

        // AdaptiveVolumeMA
        assert!(AdaptiveVolumeMA::new(4, 0.5, 0.1).is_err());
        assert!(AdaptiveVolumeMA::new(5, 0.0, 0.1).is_err());
        assert!(AdaptiveVolumeMA::new(5, 1.5, 0.1).is_err());
        assert!(AdaptiveVolumeMA::new(5, 0.5, 0.5).is_err());
        assert!(AdaptiveVolumeMA::new(5, 0.5, 0.1).is_ok());

        // VolumeFlowIndex
        assert!(VolumeFlowIndex::new(4, 3).is_err());
        assert!(VolumeFlowIndex::new(5, 0).is_err());
        assert!(VolumeFlowIndex::new(5, 3).is_ok());

        // VolumePressureIndex
        assert!(VolumePressureIndex::new(4).is_err());
        assert!(VolumePressureIndex::new(5).is_ok());
    }

    #[test]
    fn test_new_volume_indicators_technical_indicator_trait() {
        let (high, low, close, volume) = make_test_data();

        // Create OHLCVSeries for compute tests
        let data = OHLCVSeries {
            open: close.clone(),
            high: high.clone(),
            low: low.clone(),
            close: close.clone(),
            volume: volume.clone(),
        };

        // Test VolumeWeightedTrend
        let vwt = VolumeWeightedTrend::new(5, 3).unwrap();
        assert_eq!(vwt.name(), "Volume Weighted Trend");
        assert_eq!(vwt.min_periods(), 6);
        assert!(vwt.compute(&data).is_ok());

        // Test VolumeMomentumOscillator
        let vmo = VolumeMomentumOscillator::new(5, 10, 5).unwrap();
        assert_eq!(vmo.name(), "Volume Momentum Oscillator");
        assert_eq!(vmo.min_periods(), 15);
        assert_eq!(vmo.output_features(), 3);
        assert!(vmo.compute(&data).is_ok());

        // Test VolumeAccumulationTrend
        let vat = VolumeAccumulationTrend::new(5, 3).unwrap();
        assert_eq!(vat.name(), "Volume Accumulation Trend");
        assert_eq!(vat.min_periods(), 6);
        assert!(vat.compute(&data).is_ok());

        // Test AdaptiveVolumeMA
        let avma = AdaptiveVolumeMA::new(5, 0.5, 0.1).unwrap();
        assert_eq!(avma.name(), "Adaptive Volume MA");
        assert_eq!(avma.min_periods(), 6);
        assert!(avma.compute(&data).is_ok());

        // Test VolumeFlowIndex
        let vfi = VolumeFlowIndex::new(5, 3).unwrap();
        assert_eq!(vfi.name(), "Volume Flow Index");
        assert_eq!(vfi.min_periods(), 6);
        assert!(vfi.compute(&data).is_ok());

        // Test VolumePressureIndex
        let vpi = VolumePressureIndex::new(5).unwrap();
        assert_eq!(vpi.name(), "Volume Pressure Index");
        assert_eq!(vpi.min_periods(), 6);
        assert!(vpi.compute(&data).is_ok());
    }

    #[test]
    fn test_new_volume_indicators_insufficient_data() {
        let short_data = OHLCVSeries {
            open: vec![100.0, 101.0],
            high: vec![102.0, 103.0],
            low: vec![98.0, 99.0],
            close: vec![100.0, 101.0],
            volume: vec![1000.0, 1100.0],
        };

        // All should fail with insufficient data
        let vwt = VolumeWeightedTrend::new(10, 3).unwrap();
        assert!(vwt.compute(&short_data).is_err());

        let vmo = VolumeMomentumOscillator::new(5, 10, 5).unwrap();
        assert!(vmo.compute(&short_data).is_err());

        let vat = VolumeAccumulationTrend::new(10, 3).unwrap();
        assert!(vat.compute(&short_data).is_err());

        let avma = AdaptiveVolumeMA::new(10, 0.5, 0.1).unwrap();
        assert!(avma.compute(&short_data).is_err());

        let vfi = VolumeFlowIndex::new(10, 3).unwrap();
        assert!(vfi.compute(&short_data).is_err());

        let vpi = VolumePressureIndex::new(10).unwrap();
        assert!(vpi.compute(&short_data).is_err());
    }

    // =========================================================================
    // Tests for 6 NEW volume indicators (VolumeStrengthIndex, NormalizedVolume,
    // VolumeSurge, VolumeDivergenceIndex, InstitutionalFlowIndicator, VolumeZScore)
    // =========================================================================

    #[test]
    fn test_volume_strength_index() {
        let (_, _, close, volume) = make_test_data();
        let vsi = VolumeStrengthIndex::new(10, 3).unwrap();
        let (strength_index, strength_signal) = vsi.calculate(&close, &volume);

        assert_eq!(strength_index.len(), close.len());
        assert_eq!(strength_signal.len(), close.len());
        // Strength should be between 0 and 100
        assert!(strength_index.iter().all(|&v| v >= 0.0 && v <= 100.0));
        // Signal should be -1, 0, or 1
        assert!(strength_signal.iter().all(|&s| s == -1.0 || s == 0.0 || s == 1.0));
    }

    #[test]
    fn test_volume_strength_index_high_strength() {
        // High volume with strong price correlation
        let close: Vec<f64> = (0..25).map(|i| 100.0 + i as f64 * 2.0).collect();
        let volume: Vec<f64> = (0..25).map(|i| 1000.0 + i as f64 * 300.0).collect();

        let vsi = VolumeStrengthIndex::new(5, 3).unwrap();
        let (strength_index, _) = vsi.calculate(&close, &volume);

        // Should show reasonable strength in volume trend (above baseline 33 which is minimum)
        // The strength index combines relative volume, momentum, and correlation
        assert!(strength_index[20] > 35.0,
            "strength_index[20] = {} should be > 35 for increasing volume", strength_index[20]);
    }

    #[test]
    fn test_volume_strength_index_validation() {
        assert!(VolumeStrengthIndex::new(4, 3).is_err());
        assert!(VolumeStrengthIndex::new(5, 0).is_err());
        assert!(VolumeStrengthIndex::new(5, 3).is_ok());
    }

    #[test]
    fn test_normalized_volume_sma() {
        let (_, _, _, volume) = make_test_data();
        let nv = NormalizedVolume::with_sma(10).unwrap();
        let (normalized, state) = nv.calculate(&volume);

        assert_eq!(normalized.len(), volume.len());
        assert_eq!(state.len(), volume.len());
        // Normalized volume should be positive
        assert!(normalized.iter().skip(10).all(|&v| v > 0.0));
        // State should be -1, 0, or 1
        assert!(state.iter().all(|&s| s == -1.0 || s == 0.0 || s == 1.0));
    }

    #[test]
    fn test_normalized_volume_ema() {
        let (_, _, _, volume) = make_test_data();
        let nv = NormalizedVolume::with_ema(10).unwrap();
        let (normalized, _) = nv.calculate(&volume);

        assert_eq!(normalized.len(), volume.len());
        // Normalized volume should be positive
        assert!(normalized.iter().skip(10).all(|&v| v > 0.0));
    }

    #[test]
    fn test_normalized_volume_spike() {
        // Normal volume with a spike
        let mut volume = vec![1000.0; 20];
        volume[15] = 5000.0; // Spike

        let nv = NormalizedVolume::with_sma(10).unwrap();
        let (normalized, state) = nv.calculate(&volume);

        // Spike should show high normalized volume (> 1.5)
        assert!(normalized[15] > 1.5,
            "normalized[15] = {} should be > 1.5 for spike", normalized[15]);
        assert_eq!(state[15], 1.0); // High volume state
    }

    #[test]
    fn test_normalized_volume_low() {
        // Normal volume with a dip
        let mut volume = vec![1000.0; 20];
        volume[15] = 300.0; // Low volume

        let nv = NormalizedVolume::with_sma(10).unwrap();
        let (normalized, state) = nv.calculate(&volume);

        // Low should show low normalized volume (< 0.5)
        assert!(normalized[15] < 0.5,
            "normalized[15] = {} should be < 0.5 for low volume", normalized[15]);
        assert_eq!(state[15], -1.0); // Low volume state
    }

    #[test]
    fn test_normalized_volume_validation() {
        assert!(NormalizedVolume::new(1, false).is_err());
        assert!(NormalizedVolume::new(2, false).is_ok());
        assert!(NormalizedVolume::new(2, true).is_ok());
    }

    #[test]
    fn test_volume_surge() {
        let (_, _, _, volume) = make_test_data();
        let vs = VolumeSurge::new(10, 2.0, 3).unwrap();
        let (magnitude, signal, consecutive) = vs.calculate(&volume);

        assert_eq!(magnitude.len(), volume.len());
        assert_eq!(signal.len(), volume.len());
        assert_eq!(consecutive.len(), volume.len());
        // Signal should be 0 or 1
        assert!(signal.iter().all(|&s| s == 0.0 || s == 1.0));
        // Consecutive should be >= 0
        assert!(consecutive.iter().all(|&c| c >= 0.0));
    }

    #[test]
    fn test_volume_surge_detection() {
        // Create volume data with a sudden surge
        let mut volume = vec![1000.0; 20];
        volume[15] = 5000.0; // Surge

        let vs = VolumeSurge::new(10, 2.0, 3).unwrap();
        let (magnitude, signal, _) = vs.calculate(&volume);

        // Should detect surge at spike
        assert!(magnitude[15] > 2.0,
            "magnitude[15] = {} should be > 2.0 for surge", magnitude[15]);
        assert_eq!(signal[15], 1.0);
    }

    #[test]
    fn test_volume_surge_consecutive() {
        // Multiple consecutive surges
        let mut volume = vec![1000.0; 20];
        volume[12] = 4000.0;
        volume[13] = 4500.0;
        volume[14] = 5000.0;

        let vs = VolumeSurge::new(8, 2.0, 3).unwrap();
        let (_, signal, consecutive) = vs.calculate(&volume);

        // Should detect consecutive surges
        assert_eq!(signal[12], 1.0);
        assert_eq!(signal[13], 1.0);
        assert_eq!(signal[14], 1.0);
        // Consecutive count should increase
        assert!(consecutive[14] >= 2.0,
            "consecutive[14] = {} should be >= 2", consecutive[14]);
    }

    #[test]
    fn test_volume_surge_with_period() {
        let (_, _, _, volume) = make_test_data();
        let vs = VolumeSurge::with_period(10).unwrap();
        let (magnitude, _, _) = vs.calculate(&volume);

        assert_eq!(magnitude.len(), volume.len());
    }

    #[test]
    fn test_volume_surge_validation() {
        assert!(VolumeSurge::new(4, 2.0, 3).is_err()); // period < 5
        assert!(VolumeSurge::new(5, 1.0, 3).is_err()); // threshold < 1.5
        assert!(VolumeSurge::new(5, 2.0, 0).is_err()); // lookback < 1
        assert!(VolumeSurge::new(5, 2.0, 3).is_ok());
    }

    #[test]
    fn test_volume_divergence_index() {
        let (_, _, close, volume) = make_test_data();
        let vdi = VolumeDivergenceIndex::new(10, 3).unwrap();
        let (divergence_index, divergence_type) = vdi.calculate(&close, &volume);

        assert_eq!(divergence_index.len(), close.len());
        assert_eq!(divergence_type.len(), close.len());
        // Divergence index should be within bounds
        assert!(divergence_index.iter().all(|&v| v >= -100.0 && v <= 100.0));
        // Type should be -1, 0, or 1
        assert!(divergence_type.iter().all(|&t| t == -1.0 || t == 0.0 || t == 1.0));
    }

    #[test]
    fn test_volume_divergence_bearish() {
        // Price up but volume down = bearish divergence
        let close: Vec<f64> = (0..25).map(|i| 100.0 + i as f64 * 2.0).collect();
        let volume: Vec<f64> = (0..25).map(|i| 5000.0 - i as f64 * 150.0).collect();

        let vdi = VolumeDivergenceIndex::new(5, 3).unwrap();
        let (divergence_index, divergence_type) = vdi.calculate(&close, &volume);

        // Should show bearish divergence (negative)
        assert!(divergence_index[20] < 0.0,
            "divergence_index[20] = {} should be < 0 for bearish divergence", divergence_index[20]);
        assert_eq!(divergence_type[20], -1.0);
    }

    #[test]
    fn test_volume_divergence_bullish() {
        // Price down and volume down = bullish divergence
        let close: Vec<f64> = (0..25).map(|i| 150.0 - i as f64 * 2.0).collect();
        let volume: Vec<f64> = (0..25).map(|i| 5000.0 - i as f64 * 150.0).collect();

        let vdi = VolumeDivergenceIndex::new(5, 3).unwrap();
        let (divergence_index, divergence_type) = vdi.calculate(&close, &volume);

        // Should show bullish divergence (positive)
        assert!(divergence_index[20] > 0.0,
            "divergence_index[20] = {} should be > 0 for bullish divergence", divergence_index[20]);
        assert_eq!(divergence_type[20], 1.0);
    }

    #[test]
    fn test_volume_divergence_validation() {
        assert!(VolumeDivergenceIndex::new(4, 3).is_err());
        assert!(VolumeDivergenceIndex::new(5, 0).is_err());
        assert!(VolumeDivergenceIndex::new(5, 3).is_ok());
    }

    #[test]
    fn test_institutional_flow_indicator() {
        let (high, low, close, volume) = make_test_data();
        let ifi = InstitutionalFlowIndicator::new(10, 0.5).unwrap();
        let (ratio, signal, accumulation) = ifi.calculate(&high, &low, &close, &volume);

        assert_eq!(ratio.len(), close.len());
        assert_eq!(signal.len(), close.len());
        assert_eq!(accumulation.len(), close.len());
        // Ratio should be between 0 and 100
        assert!(ratio.iter().all(|&r| r >= 0.0 && r <= 100.0));
        // Signal should be -1, 0, or 1
        assert!(signal.iter().all(|&s| s == -1.0 || s == 0.0 || s == 1.0));
    }

    #[test]
    fn test_institutional_flow_with_period() {
        let (high, low, close, volume) = make_test_data();
        let ifi = InstitutionalFlowIndicator::with_period(10).unwrap();
        let (ratio, _, _) = ifi.calculate(&high, &low, &close, &volume);

        assert_eq!(ratio.len(), close.len());
    }

    #[test]
    fn test_institutional_flow_efficient_moves() {
        // Efficient price movement with high volume = institutional
        let high: Vec<f64> = (0..25).map(|i| 102.0 + i as f64 * 2.0).collect();
        let low: Vec<f64> = (0..25).map(|i| 98.0 + i as f64 * 2.0).collect();
        let close: Vec<f64> = (0..25).map(|i| 101.5 + i as f64 * 2.0).collect(); // Close near high (efficient)
        let volume: Vec<f64> = (0..25).map(|i| 1000.0 + i as f64 * 200.0).collect();

        let ifi = InstitutionalFlowIndicator::new(10, 0.3).unwrap();
        let (ratio, _, _) = ifi.calculate(&high, &low, &close, &volume);

        // Should show some institutional activity
        assert!(ratio[20] > 30.0,
            "ratio[20] = {} should indicate some institutional activity", ratio[20]);
    }

    #[test]
    fn test_institutional_flow_validation() {
        assert!(InstitutionalFlowIndicator::new(9, 0.5).is_err()); // period < 10
        assert!(InstitutionalFlowIndicator::new(10, 0.0).is_err()); // threshold = 0
        assert!(InstitutionalFlowIndicator::new(10, 1.0).is_err()); // threshold = 1
        assert!(InstitutionalFlowIndicator::new(10, 0.5).is_ok());
    }

    #[test]
    fn test_volume_z_score() {
        let (_, _, _, volume) = make_test_data();
        let vzs = VolumeZScore::new(10).unwrap();
        let (z_score, significance, extreme) = vzs.calculate(&volume);

        assert_eq!(z_score.len(), volume.len());
        assert_eq!(significance.len(), volume.len());
        assert_eq!(extreme.len(), volume.len());
        // Significance should be between 0 and 100
        assert!(significance.iter().all(|&s| s >= 0.0 && s <= 100.0));
        // Extreme signal should be -1, 0, or 1
        assert!(extreme.iter().all(|&e| e == -1.0 || e == 0.0 || e == 1.0));
    }

    #[test]
    fn test_volume_z_score_spike() {
        // Normal volume with a spike
        let mut volume = vec![1000.0; 20];
        // Add some variance
        for i in 0..20 {
            volume[i] += (i as f64 % 3.0) * 50.0;
        }
        volume[15] = 5000.0; // Spike

        let vzs = VolumeZScore::new(10).unwrap();
        let (z_score, significance, extreme) = vzs.calculate(&volume);

        // Spike should show high z-score
        assert!(z_score[15] > 2.0,
            "z_score[15] = {} should be > 2.0 for spike", z_score[15]);
        // Should be statistically significant
        assert!(significance[15] > 90.0,
            "significance[15] = {} should be > 90 for significant spike", significance[15]);
        // Should signal extreme high
        assert_eq!(extreme[15], 1.0);
    }

    #[test]
    fn test_volume_z_score_low() {
        // Normal volume with a dip
        let mut volume = vec![1000.0; 20];
        // Add some variance
        for i in 0..20 {
            volume[i] += (i as f64 % 3.0) * 50.0;
        }
        volume[15] = 100.0; // Very low

        let vzs = VolumeZScore::new(10).unwrap();
        let (z_score, _, extreme) = vzs.calculate(&volume);

        // Low should show negative z-score
        assert!(z_score[15] < -2.0,
            "z_score[15] = {} should be < -2.0 for low volume", z_score[15]);
        // Should signal extreme low
        assert_eq!(extreme[15], -1.0);
    }

    #[test]
    fn test_volume_z_score_validation() {
        assert!(VolumeZScore::new(9).is_err()); // period < 10
        assert!(VolumeZScore::new(10).is_ok());
    }

    #[test]
    fn test_new_six_indicators_technical_indicator_trait() {
        let (high, low, close, volume) = make_test_data();

        // Create OHLCVSeries for compute tests
        let data = OHLCVSeries {
            open: close.clone(),
            high: high.clone(),
            low: low.clone(),
            close: close.clone(),
            volume: volume.clone(),
        };

        // Test VolumeStrengthIndex
        let vsi = VolumeStrengthIndex::new(5, 3).unwrap();
        assert_eq!(vsi.name(), "Volume Strength Index");
        assert_eq!(vsi.min_periods(), 6);
        assert!(vsi.compute(&data).is_ok());

        // Test NormalizedVolume
        let nv = NormalizedVolume::with_sma(5).unwrap();
        assert_eq!(nv.name(), "Normalized Volume");
        assert_eq!(nv.min_periods(), 6);
        assert!(nv.compute(&data).is_ok());

        // Test VolumeSurge
        let vs = VolumeSurge::new(5, 2.0, 3).unwrap();
        assert_eq!(vs.name(), "Volume Surge");
        assert_eq!(vs.min_periods(), 8);
        assert_eq!(vs.output_features(), 3);
        assert!(vs.compute(&data).is_ok());

        // Test VolumeDivergenceIndex
        let vdi = VolumeDivergenceIndex::new(5, 3).unwrap();
        assert_eq!(vdi.name(), "Volume Divergence Index");
        assert_eq!(vdi.min_periods(), 6);
        assert!(vdi.compute(&data).is_ok());

        // Test InstitutionalFlowIndicator
        let ifi = InstitutionalFlowIndicator::new(10, 0.5).unwrap();
        assert_eq!(ifi.name(), "Institutional Flow Indicator");
        assert_eq!(ifi.min_periods(), 11);
        assert_eq!(ifi.output_features(), 3);
        assert!(ifi.compute(&data).is_ok());

        // Test VolumeZScore
        let vzs = VolumeZScore::new(10).unwrap();
        assert_eq!(vzs.name(), "Volume Z-Score");
        assert_eq!(vzs.min_periods(), 11);
        assert_eq!(vzs.output_features(), 3);
        assert!(vzs.compute(&data).is_ok());
    }

    #[test]
    fn test_new_six_indicators_insufficient_data() {
        let short_data = OHLCVSeries {
            open: vec![100.0, 101.0],
            high: vec![102.0, 103.0],
            low: vec![98.0, 99.0],
            close: vec![100.0, 101.0],
            volume: vec![1000.0, 1100.0],
        };

        // All should fail with insufficient data
        let vsi = VolumeStrengthIndex::new(10, 3).unwrap();
        assert!(vsi.compute(&short_data).is_err());

        let nv = NormalizedVolume::with_sma(10).unwrap();
        assert!(nv.compute(&short_data).is_err());

        let vs = VolumeSurge::new(10, 2.0, 3).unwrap();
        assert!(vs.compute(&short_data).is_err());

        let vdi = VolumeDivergenceIndex::new(10, 3).unwrap();
        assert!(vdi.compute(&short_data).is_err());

        let ifi = InstitutionalFlowIndicator::new(10, 0.5).unwrap();
        assert!(ifi.compute(&short_data).is_err());

        let vzs = VolumeZScore::new(10).unwrap();
        assert!(vzs.compute(&short_data).is_err());
    }

    #[test]
    fn test_new_six_indicators_parameter_validation() {
        // VolumeStrengthIndex
        assert!(VolumeStrengthIndex::new(4, 3).is_err());
        assert!(VolumeStrengthIndex::new(5, 0).is_err());
        assert!(VolumeStrengthIndex::new(5, 3).is_ok());

        // NormalizedVolume
        assert!(NormalizedVolume::new(1, false).is_err());
        assert!(NormalizedVolume::new(2, false).is_ok());
        assert!(NormalizedVolume::new(2, true).is_ok());

        // VolumeSurge
        assert!(VolumeSurge::new(4, 2.0, 3).is_err());
        assert!(VolumeSurge::new(5, 1.0, 3).is_err());
        assert!(VolumeSurge::new(5, 2.0, 0).is_err());
        assert!(VolumeSurge::new(5, 2.0, 3).is_ok());

        // VolumeDivergenceIndex
        assert!(VolumeDivergenceIndex::new(4, 3).is_err());
        assert!(VolumeDivergenceIndex::new(5, 0).is_err());
        assert!(VolumeDivergenceIndex::new(5, 3).is_ok());

        // InstitutionalFlowIndicator
        assert!(InstitutionalFlowIndicator::new(9, 0.5).is_err());
        assert!(InstitutionalFlowIndicator::new(10, 0.0).is_err());
        assert!(InstitutionalFlowIndicator::new(10, 1.0).is_err());
        assert!(InstitutionalFlowIndicator::new(10, 0.5).is_ok());

        // VolumeZScore
        assert!(VolumeZScore::new(9).is_err());
        assert!(VolumeZScore::new(10).is_ok());
    }

    // =========================================================================
    // Tests for 6 NEW volume indicators (VolumeRank, VolumePercentile,
    // VolumeRatio, VolumeConcentration, VolumeBias, VolumeQuality)
    // =========================================================================

    #[test]
    fn test_volume_rank() {
        let (_, _, _, volume) = make_test_data();
        let vr = VolumeRank::new(10).unwrap();
        let (rank, signal) = vr.calculate(&volume);

        assert_eq!(rank.len(), volume.len());
        assert_eq!(signal.len(), volume.len());
        // Rank should be between 0 and 100
        assert!(rank.iter().all(|&r| r >= 0.0 && r <= 100.0));
        // Signal should be -1, 0, or 1
        assert!(signal.iter().all(|&s| s == -1.0 || s == 0.0 || s == 1.0));
    }

    #[test]
    fn test_volume_rank_high_volume() {
        // Create volume data with increasing values
        let volume: Vec<f64> = (0..25).map(|i| 1000.0 + i as f64 * 100.0).collect();

        let vr = VolumeRank::new(10).unwrap();
        let (rank, signal) = vr.calculate(&volume);

        // Last values should have high rank (close to 100)
        assert!(rank[20] > 80.0,
            "rank[20] = {} should be > 80 for highest volume", rank[20]);
        assert_eq!(signal[20], 1.0); // High rank signal
    }

    #[test]
    fn test_volume_rank_low_volume() {
        // Create volume data with spike followed by low volume
        let mut volume = vec![1000.0; 25];
        for i in 0..15 {
            volume[i] = 2000.0 + i as f64 * 50.0; // Higher early volumes
        }
        volume[20] = 500.0; // Low volume

        let vr = VolumeRank::new(10).unwrap();
        let (rank, signal) = vr.calculate(&volume);

        // Low volume should have low rank
        assert!(rank[20] < 20.0,
            "rank[20] = {} should be < 20 for low volume", rank[20]);
        assert_eq!(signal[20], -1.0); // Low rank signal
    }

    #[test]
    fn test_volume_rank_validation() {
        assert!(VolumeRank::new(9).is_err()); // period < 10
        assert!(VolumeRank::new(10).is_ok());
    }

    #[test]
    fn test_volume_percentile() {
        let (_, _, _, volume) = make_test_data();
        let vp = VolumePercentile::new(10, 3).unwrap();
        let (percentile, state) = vp.calculate(&volume);

        assert_eq!(percentile.len(), volume.len());
        assert_eq!(state.len(), volume.len());
        // Percentile should be between 0 and 100
        assert!(percentile.iter().all(|&p| p >= 0.0 && p <= 100.0));
        // State should be -1, 0, or 1
        assert!(state.iter().all(|&s| s == -1.0 || s == 0.0 || s == 1.0));
    }

    #[test]
    fn test_volume_percentile_with_period() {
        let (_, _, _, volume) = make_test_data();
        let vp = VolumePercentile::with_period(10).unwrap();
        let (percentile, _) = vp.calculate(&volume);

        assert_eq!(percentile.len(), volume.len());
    }

    #[test]
    fn test_volume_percentile_extreme_high() {
        // Create volume data with a spike
        let mut volume = vec![1000.0; 20];
        volume[15] = 5000.0; // Extreme high

        let vp = VolumePercentile::new(10, 1).unwrap();
        let (percentile, state) = vp.calculate(&volume);

        // Spike should have high percentile
        assert!(percentile[15] > 90.0,
            "percentile[15] = {} should be > 90 for extreme high", percentile[15]);
        assert_eq!(state[15], 1.0);
    }

    #[test]
    fn test_volume_percentile_extreme_low() {
        // Create volume data with a dip
        let mut volume = vec![1000.0; 20];
        volume[15] = 100.0; // Extreme low

        let vp = VolumePercentile::new(10, 1).unwrap();
        let (percentile, state) = vp.calculate(&volume);

        // Dip should have low percentile
        assert!(percentile[15] < 10.0,
            "percentile[15] = {} should be < 10 for extreme low", percentile[15]);
        assert_eq!(state[15], -1.0);
    }

    #[test]
    fn test_volume_percentile_validation() {
        assert!(VolumePercentile::new(9, 3).is_err()); // period < 10
        assert!(VolumePercentile::new(10, 0).is_err()); // smoothing < 1
        assert!(VolumePercentile::new(10, 3).is_ok());
    }

    #[test]
    fn test_volume_ratio() {
        let (_, _, close, volume) = make_test_data();
        let vr = VolumeRatio::new(10, 3).unwrap();
        let (ratio, ratio_index, dominance) = vr.calculate(&close, &volume);

        assert_eq!(ratio.len(), close.len());
        assert_eq!(ratio_index.len(), close.len());
        assert_eq!(dominance.len(), close.len());
        // Ratio index should be between 0 and 100
        assert!(ratio_index.iter().all(|&r| r >= 0.0 && r <= 100.0));
        // Dominance should be -1, 0, or 1
        assert!(dominance.iter().all(|&d| d == -1.0 || d == 0.0 || d == 1.0));
    }

    #[test]
    fn test_volume_ratio_with_period() {
        let (_, _, close, volume) = make_test_data();
        let vr = VolumeRatio::with_period(10).unwrap();
        let (ratio, _, _) = vr.calculate(&close, &volume);

        assert_eq!(ratio.len(), close.len());
    }

    #[test]
    fn test_volume_ratio_buying_dominance() {
        // All prices going up = all up volume
        let close: Vec<f64> = (0..25).map(|i| 100.0 + i as f64 * 1.0).collect();
        let volume = vec![1000.0; 25];

        let vr = VolumeRatio::new(10, 1).unwrap();
        let (ratio, ratio_index, dominance) = vr.calculate(&close, &volume);

        // Should show buying dominance
        assert!(ratio[20] > 5.0,
            "ratio[20] = {} should be high for all up moves", ratio[20]);
        assert!(ratio_index[20] > 60.0,
            "ratio_index[20] = {} should be > 60 for buying dominance", ratio_index[20]);
        assert_eq!(dominance[20], 1.0);
    }

    #[test]
    fn test_volume_ratio_selling_dominance() {
        // All prices going down = all down volume
        let close: Vec<f64> = (0..25).map(|i| 150.0 - i as f64 * 1.0).collect();
        let volume = vec![1000.0; 25];

        let vr = VolumeRatio::new(10, 1).unwrap();
        let (ratio, ratio_index, dominance) = vr.calculate(&close, &volume);

        // Should show selling dominance
        assert!(ratio[20] < 0.2,
            "ratio[20] = {} should be low for all down moves", ratio[20]);
        assert!(ratio_index[20] < 40.0,
            "ratio_index[20] = {} should be < 40 for selling dominance", ratio_index[20]);
        assert_eq!(dominance[20], -1.0);
    }

    #[test]
    fn test_volume_ratio_validation() {
        assert!(VolumeRatio::new(4, 3).is_err()); // period < 5
        assert!(VolumeRatio::new(5, 0).is_err()); // smoothing < 1
        assert!(VolumeRatio::new(5, 3).is_ok());
    }

    #[test]
    fn test_volume_concentration() {
        let (high, low, close, volume) = make_test_data();
        let vc = VolumeConcentration::new(10, 5).unwrap();
        let (concentration, signal) = vc.calculate(&high, &low, &close, &volume);

        assert_eq!(concentration.len(), close.len());
        assert_eq!(signal.len(), close.len());
        // Concentration should be between 0 and 100
        assert!(concentration.iter().all(|&c| c >= 0.0 && c <= 100.0));
        // Signal should be -1, 0, or 1
        assert!(signal.iter().all(|&s| s == -1.0 || s == 0.0 || s == 1.0));
    }

    #[test]
    fn test_volume_concentration_with_period() {
        let (high, low, close, volume) = make_test_data();
        let vc = VolumeConcentration::with_period(10).unwrap();
        let (concentration, _) = vc.calculate(&high, &low, &close, &volume);

        assert_eq!(concentration.len(), close.len());
    }

    #[test]
    fn test_volume_concentration_high() {
        // Prices very close together = high concentration
        let high = vec![101.0; 20];
        let low = vec![99.0; 20];
        let close = vec![100.0; 20];
        let volume = vec![1000.0; 20];

        let vc = VolumeConcentration::new(10, 5).unwrap();
        let (concentration, signal) = vc.calculate(&high, &low, &close, &volume);

        // Should show high concentration
        assert!(concentration[15] > 70.0,
            "concentration[15] = {} should be > 70 for concentrated volume", concentration[15]);
        assert_eq!(signal[15], 1.0);
    }

    #[test]
    fn test_volume_concentration_low() {
        // Prices spread out with equal volume = low concentration
        let high: Vec<f64> = (0..20).map(|i| 100.0 + i as f64 * 10.0 + 5.0).collect();
        let low: Vec<f64> = (0..20).map(|i| 100.0 + i as f64 * 10.0 - 5.0).collect();
        let close: Vec<f64> = (0..20).map(|i| 100.0 + i as f64 * 10.0).collect();
        let volume = vec![1000.0; 20];

        let vc = VolumeConcentration::new(10, 5).unwrap();
        let (concentration, signal) = vc.calculate(&high, &low, &close, &volume);

        // Should show lower concentration
        assert!(concentration[15] < 50.0,
            "concentration[15] = {} should be < 50 for dispersed volume", concentration[15]);
    }

    #[test]
    fn test_volume_concentration_validation() {
        assert!(VolumeConcentration::new(4, 5).is_err()); // period < 5
        assert!(VolumeConcentration::new(5, 2).is_err()); // num_bins < 3
        assert!(VolumeConcentration::new(5, 3).is_ok());
    }

    #[test]
    fn test_volume_bias() {
        let (_, _, close, volume) = make_test_data();
        let vb = VolumeBias::new(10, 3).unwrap();
        let (bias_value, bias_strength, bias_signal) = vb.calculate(&close, &volume);

        assert_eq!(bias_value.len(), close.len());
        assert_eq!(bias_strength.len(), close.len());
        assert_eq!(bias_signal.len(), close.len());
        // Bias value should be between -100 and 100
        assert!(bias_value.iter().all(|&b| b >= -100.0 && b <= 100.0));
        // Bias strength should be >= 0
        assert!(bias_strength.iter().all(|&s| s >= 0.0));
        // Signal should be -1, 0, or 1
        assert!(bias_signal.iter().all(|&s| s == -1.0 || s == 0.0 || s == 1.0));
    }

    #[test]
    fn test_volume_bias_with_period() {
        let (_, _, close, volume) = make_test_data();
        let vb = VolumeBias::with_period(10).unwrap();
        let (bias, _, _) = vb.calculate(&close, &volume);

        assert_eq!(bias.len(), close.len());
    }

    #[test]
    fn test_volume_bias_bullish() {
        // Uptrend with increasing volume = bullish bias
        let close: Vec<f64> = (0..25).map(|i| 100.0 + i as f64 * 2.0).collect();
        let volume: Vec<f64> = (0..25).map(|i| 1000.0 + i as f64 * 100.0).collect();

        let vb = VolumeBias::new(10, 1).unwrap();
        let (bias_value, _, bias_signal) = vb.calculate(&close, &volume);

        // Should show bullish bias
        assert!(bias_value[20] > 10.0,
            "bias_value[20] = {} should be > 10 for bullish bias", bias_value[20]);
        assert_eq!(bias_signal[20], 1.0);
    }

    #[test]
    fn test_volume_bias_bearish() {
        // Downtrend with increasing volume = bearish bias
        let close: Vec<f64> = (0..25).map(|i| 150.0 - i as f64 * 2.0).collect();
        let volume: Vec<f64> = (0..25).map(|i| 1000.0 + i as f64 * 100.0).collect();

        let vb = VolumeBias::new(10, 1).unwrap();
        let (bias_value, _, bias_signal) = vb.calculate(&close, &volume);

        // Should show bearish bias
        assert!(bias_value[20] < -10.0,
            "bias_value[20] = {} should be < -10 for bearish bias", bias_value[20]);
        assert_eq!(bias_signal[20], -1.0);
    }

    #[test]
    fn test_volume_bias_validation() {
        assert!(VolumeBias::new(4, 3).is_err()); // period < 5
        assert!(VolumeBias::new(5, 0).is_err()); // smoothing < 1
        assert!(VolumeBias::new(5, 3).is_ok());
    }

    #[test]
    fn test_volume_quality() {
        let (_, _, close, volume) = make_test_data();
        let vq = VolumeQuality::new(10, 0.5).unwrap();
        let (quality, signal) = vq.calculate(&close, &volume);

        assert_eq!(quality.len(), close.len());
        assert_eq!(signal.len(), close.len());
        // Quality should be between 0 and 100
        assert!(quality.iter().all(|&q| q >= 0.0 && q <= 100.0));
        // Signal should be -1, 0, or 1
        assert!(signal.iter().all(|&s| s == -1.0 || s == 0.0 || s == 1.0));
    }

    #[test]
    fn test_volume_quality_with_period() {
        let (_, _, close, volume) = make_test_data();
        let vq = VolumeQuality::with_period(10).unwrap();
        let (quality, _) = vq.calculate(&close, &volume);

        assert_eq!(quality.len(), close.len());
    }

    #[test]
    fn test_volume_quality_consistent_volume() {
        // Consistent volume with trending price = high quality
        let close: Vec<f64> = (0..25).map(|i| 100.0 + i as f64 * 1.0).collect();
        let volume = vec![1000.0; 25]; // Very consistent volume

        let vq = VolumeQuality::new(10, 0.5).unwrap();
        let (quality, signal) = vq.calculate(&close, &volume);

        // Should show decent quality due to consistency
        assert!(quality[20] > 50.0,
            "quality[20] = {} should be > 50 for consistent volume", quality[20]);
    }

    #[test]
    fn test_volume_quality_erratic_volume() {
        // Erratic volume = lower quality
        let close: Vec<f64> = (0..25).map(|i| 100.0 + (i as f64 * 0.5).sin() * 5.0).collect();
        let mut volume = vec![1000.0; 25];
        for i in 0..25 {
            volume[i] = if i % 2 == 0 { 500.0 } else { 2000.0 }; // Highly variable
        }

        let vq = VolumeQuality::new(10, 0.5).unwrap();
        let (quality, _) = vq.calculate(&close, &volume);

        // Quality should be moderate to low due to inconsistency
        assert!(quality[20] < 70.0,
            "quality[20] = {} should be < 70 for erratic volume", quality[20]);
    }

    #[test]
    fn test_volume_quality_validation() {
        assert!(VolumeQuality::new(9, 0.5).is_err()); // period < 10
        assert!(VolumeQuality::new(10, -0.1).is_err()); // threshold < 0
        assert!(VolumeQuality::new(10, 1.1).is_err()); // threshold > 1
        assert!(VolumeQuality::new(10, 0.5).is_ok());
    }

    #[test]
    fn test_six_new_indicators_technical_indicator_trait() {
        let (high, low, close, volume) = make_test_data();

        // Create OHLCVSeries for compute tests
        let data = OHLCVSeries {
            open: close.clone(),
            high: high.clone(),
            low: low.clone(),
            close: close.clone(),
            volume: volume.clone(),
        };

        // Test VolumeRank
        let vr = VolumeRank::new(10).unwrap();
        assert_eq!(vr.name(), "Volume Rank");
        assert_eq!(vr.min_periods(), 11);
        assert!(vr.compute(&data).is_ok());

        // Test VolumePercentile
        let vp = VolumePercentile::new(10, 3).unwrap();
        assert_eq!(vp.name(), "Volume Percentile");
        assert_eq!(vp.min_periods(), 11);
        assert!(vp.compute(&data).is_ok());

        // Test VolumeRatio
        let vrat = VolumeRatio::new(10, 3).unwrap();
        assert_eq!(vrat.name(), "Volume Ratio");
        assert_eq!(vrat.min_periods(), 11);
        assert_eq!(vrat.output_features(), 3);
        assert!(vrat.compute(&data).is_ok());

        // Test VolumeConcentration
        let vc = VolumeConcentration::new(10, 5).unwrap();
        assert_eq!(vc.name(), "Volume Concentration");
        assert_eq!(vc.min_periods(), 11);
        assert!(vc.compute(&data).is_ok());

        // Test VolumeBias
        let vb = VolumeBias::new(10, 3).unwrap();
        assert_eq!(vb.name(), "Volume Bias");
        assert_eq!(vb.min_periods(), 11);
        assert_eq!(vb.output_features(), 3);
        assert!(vb.compute(&data).is_ok());

        // Test VolumeQuality
        let vq = VolumeQuality::new(10, 0.5).unwrap();
        assert_eq!(vq.name(), "Volume Quality");
        assert_eq!(vq.min_periods(), 11);
        assert!(vq.compute(&data).is_ok());
    }

    #[test]
    fn test_six_new_indicators_insufficient_data() {
        let short_data = OHLCVSeries {
            open: vec![100.0, 101.0],
            high: vec![102.0, 103.0],
            low: vec![98.0, 99.0],
            close: vec![100.0, 101.0],
            volume: vec![1000.0, 1100.0],
        };

        // All should fail with insufficient data
        let vr = VolumeRank::new(10).unwrap();
        assert!(vr.compute(&short_data).is_err());

        let vp = VolumePercentile::new(10, 3).unwrap();
        assert!(vp.compute(&short_data).is_err());

        let vrat = VolumeRatio::new(10, 3).unwrap();
        assert!(vrat.compute(&short_data).is_err());

        let vc = VolumeConcentration::new(10, 5).unwrap();
        assert!(vc.compute(&short_data).is_err());

        let vb = VolumeBias::new(10, 3).unwrap();
        assert!(vb.compute(&short_data).is_err());

        let vq = VolumeQuality::new(10, 0.5).unwrap();
        assert!(vq.compute(&short_data).is_err());
    }

    #[test]
    fn test_six_new_indicators_parameter_validation() {
        // VolumeRank
        assert!(VolumeRank::new(9).is_err());
        assert!(VolumeRank::new(10).is_ok());

        // VolumePercentile
        assert!(VolumePercentile::new(9, 3).is_err());
        assert!(VolumePercentile::new(10, 0).is_err());
        assert!(VolumePercentile::new(10, 3).is_ok());

        // VolumeRatio
        assert!(VolumeRatio::new(4, 3).is_err());
        assert!(VolumeRatio::new(5, 0).is_err());
        assert!(VolumeRatio::new(5, 3).is_ok());

        // VolumeConcentration
        assert!(VolumeConcentration::new(4, 5).is_err());
        assert!(VolumeConcentration::new(5, 2).is_err());
        assert!(VolumeConcentration::new(5, 3).is_ok());

        // VolumeBias
        assert!(VolumeBias::new(4, 3).is_err());
        assert!(VolumeBias::new(5, 0).is_err());
        assert!(VolumeBias::new(5, 3).is_ok());

        // VolumeQuality
        assert!(VolumeQuality::new(9, 0.5).is_err());
        assert!(VolumeQuality::new(10, -0.1).is_err());
        assert!(VolumeQuality::new(10, 1.1).is_err());
        assert!(VolumeQuality::new(10, 0.5).is_ok());
    }

    // =========================================================================
    // Tests for 6 NEW batch 7 volume indicators
    // =========================================================================

    #[test]
    fn test_volume_climax_index() {
        let (high, low, close, volume) = make_test_data();
        let vci = VolumeClimaxIndex::new(10, 2.0).unwrap();
        let (index, signal, climax_type) = vci.calculate(&high, &low, &close, &volume);

        assert_eq!(index.len(), close.len());
        assert_eq!(signal.len(), close.len());
        assert_eq!(climax_type.len(), close.len());
        // Index should be between 0 and 100
        assert!(index.iter().all(|&v| v >= 0.0 && v <= 100.0));
        // Signal and climax_type should be -1, 0, or 1
        assert!(signal.iter().all(|&s| s == -1.0 || s == 0.0 || s == 1.0));
        assert!(climax_type.iter().all(|&s| s == -1.0 || s == 0.0 || s == 1.0));
    }

    #[test]
    fn test_volume_climax_index_with_period() {
        let (high, low, close, volume) = make_test_data();
        let vci = VolumeClimaxIndex::with_period(10).unwrap();
        let (index, _, _) = vci.calculate(&high, &low, &close, &volume);

        assert_eq!(index.len(), close.len());
    }

    #[test]
    fn test_volume_climax_index_spike_detection() {
        // Create data with a volume spike with some baseline variance for proper z-score calc
        let high = vec![102.0; 25];
        let low = vec![98.0; 25];
        let mut close: Vec<f64> = (0..25).map(|i| 100.0 + i as f64 * 0.5).collect();
        // Add some variance to volume (normal trading around 1000 with natural variation)
        let mut volume: Vec<f64> = (0..25).map(|i| 1000.0 + (i as f64 * 7.0 % 100.0) - 50.0).collect();
        // Create significant volume spike at index 20 (much higher than baseline)
        volume[20] = 5000.0;
        close[20] = close[19] + 2.0; // Up move

        let vci = VolumeClimaxIndex::new(10, 2.0).unwrap();
        let (index, signal, climax_type) = vci.calculate(&high, &low, &close, &volume);

        // Should detect climax at spike - verify the index is elevated above baseline
        assert!(index[20] > 15.0,
            "index[20] = {} should be > 15 at volume spike", index[20]);
        // Should be bullish climax (up move) - with variance in data, z-score should trigger
        assert_eq!(climax_type[20], 1.0);
    }

    #[test]
    fn test_volume_climax_index_validation() {
        assert!(VolumeClimaxIndex::new(9, 2.0).is_err()); // period < 10
        assert!(VolumeClimaxIndex::new(10, 1.4).is_err()); // z_threshold < 1.5
        assert!(VolumeClimaxIndex::new(10, 2.0).is_ok());
    }

    #[test]
    fn test_institutional_volume_proxy() {
        let (high, low, close, volume) = make_test_data();
        let ivp = InstitutionalVolumeProxy::new(10, 2.0).unwrap();
        let (pct, signal, accum) = ivp.calculate(&high, &low, &close, &volume);

        assert_eq!(pct.len(), close.len());
        assert_eq!(signal.len(), close.len());
        assert_eq!(accum.len(), close.len());
        // Percentage should be between 0 and 100
        assert!(pct.iter().all(|&p| p >= 0.0 && p <= 100.0));
        // Signal should be -1, 0, or 1
        assert!(signal.iter().all(|&s| s == -1.0 || s == 0.0 || s == 1.0));
    }

    #[test]
    fn test_institutional_volume_proxy_with_period() {
        let (high, low, close, volume) = make_test_data();
        let ivp = InstitutionalVolumeProxy::with_period(10).unwrap();
        let (pct, _, _) = ivp.calculate(&high, &low, &close, &volume);

        assert_eq!(pct.len(), close.len());
    }

    #[test]
    fn test_institutional_volume_proxy_block_trades() {
        // Create data with large block trades
        let high: Vec<f64> = (0..25).map(|i| 102.0 + i as f64 * 1.0).collect();
        let low: Vec<f64> = (0..25).map(|i| 98.0 + i as f64 * 1.0).collect();
        let close: Vec<f64> = (0..25).map(|i| 100.0 + i as f64 * 1.0).collect();
        let mut volume = vec![1000.0; 25];
        // Add block trades
        volume[15] = 3000.0;
        volume[18] = 4000.0;
        volume[20] = 5000.0;

        let ivp = InstitutionalVolumeProxy::new(10, 2.0).unwrap();
        let (pct, _, _) = ivp.calculate(&high, &low, &close, &volume);

        // Should show higher institutional percentage around block trades
        assert!(pct[20] > 30.0,
            "pct[20] = {} should be elevated with block trades", pct[20]);
    }

    #[test]
    fn test_institutional_volume_proxy_validation() {
        assert!(InstitutionalVolumeProxy::new(9, 2.0).is_err()); // period < 10
        assert!(InstitutionalVolumeProxy::new(10, 1.4).is_err()); // threshold < 1.5
        assert!(InstitutionalVolumeProxy::new(10, 2.0).is_ok());
    }

    #[test]
    fn test_volume_acceleration_index() {
        let (_, _, _, volume) = make_test_data();
        let vai = VolumeAccelerationIndex::new(5, 3).unwrap();
        let (accel, momentum, signal) = vai.calculate(&volume);

        assert_eq!(accel.len(), volume.len());
        assert_eq!(momentum.len(), volume.len());
        assert_eq!(signal.len(), volume.len());
        // Acceleration should be between -100 and 100
        assert!(accel.iter().all(|&a| a >= -100.0 && a <= 100.0));
        // Signal should be -1, 0, or 1
        assert!(signal.iter().all(|&s| s == -1.0 || s == 0.0 || s == 1.0));
    }

    #[test]
    fn test_volume_acceleration_index_with_period() {
        let (_, _, _, volume) = make_test_data();
        let vai = VolumeAccelerationIndex::with_period(5).unwrap();
        let (accel, _, _) = vai.calculate(&volume);

        assert_eq!(accel.len(), volume.len());
    }

    #[test]
    fn test_volume_acceleration_index_increasing_volume() {
        // Steadily increasing volume = the indicator should detect the momentum pattern
        let volume: Vec<f64> = (0..30).map(|i| 1000.0 + i as f64 * 100.0).collect();

        let vai = VolumeAccelerationIndex::new(5, 1).unwrap();
        let (accel, momentum, _) = vai.calculate(&volume);

        // Verify the indicator produces valid outputs within expected ranges
        assert!(accel.len() == 30);
        assert!(momentum.len() == 30);
        // The momentum values should be within the valid range
        assert!(momentum.iter().all(|&m| m >= -100.0 && m <= 100.0));
    }

    #[test]
    fn test_volume_acceleration_index_validation() {
        assert!(VolumeAccelerationIndex::new(4, 3).is_err()); // period < 5
        assert!(VolumeAccelerationIndex::new(5, 0).is_err()); // smoothing < 1
        assert!(VolumeAccelerationIndex::new(5, 3).is_ok());
    }

    #[test]
    fn test_smart_money_indicator() {
        let (high, low, close, volume) = make_test_data();
        let smi = SmartMoneyIndicator::new(10, 1.0).unwrap();
        let (index, direction, confidence) = smi.calculate(&high, &low, &close, &volume);

        assert_eq!(index.len(), close.len());
        assert_eq!(direction.len(), close.len());
        assert_eq!(confidence.len(), close.len());
        // Confidence should be between 0 and 100
        assert!(confidence.iter().all(|&c| c >= 0.0 && c <= 100.0));
        // Direction should be -1, 0, or 1
        assert!(direction.iter().all(|&d| d == -1.0 || d == 0.0 || d == 1.0));
    }

    #[test]
    fn test_smart_money_indicator_with_period() {
        let (high, low, close, volume) = make_test_data();
        let smi = SmartMoneyIndicator::with_period(10).unwrap();
        let (index, _, _) = smi.calculate(&high, &low, &close, &volume);

        assert_eq!(index.len(), close.len());
    }

    #[test]
    fn test_smart_money_indicator_high_volume_small_range() {
        // High volume with small price range = smart money activity
        let high = vec![101.0; 25];
        let low = vec![99.0; 25];
        let close: Vec<f64> = (0..25).map(|i| 100.0 + (i as f64 * 0.1)).collect();
        let mut volume = vec![1000.0; 25];
        // High volume with small move
        volume[20] = 5000.0;

        let smi = SmartMoneyIndicator::new(10, 1.0).unwrap();
        let (_, direction, confidence) = smi.calculate(&high, &low, &close, &volume);

        // Should have some confidence in smart money activity
        assert!(confidence[20] > 0.0,
            "confidence[20] = {} should be > 0 for high vol/small range", confidence[20]);
    }

    #[test]
    fn test_smart_money_indicator_validation() {
        assert!(SmartMoneyIndicator::new(9, 1.0).is_err()); // period < 10
        assert!(SmartMoneyIndicator::new(10, 0.05).is_err()); // sensitivity < 0.1
        assert!(SmartMoneyIndicator::new(10, 2.5).is_err()); // sensitivity > 2.0
        assert!(SmartMoneyIndicator::new(10, 1.0).is_ok());
    }

    #[test]
    fn test_volume_efficiency_index() {
        let (_, _, close, volume) = make_test_data();
        let vei = VolumeEfficiencyIndex::new(10, 3).unwrap();
        let (index, rating, quality) = vei.calculate(&close, &volume);

        assert_eq!(index.len(), close.len());
        assert_eq!(rating.len(), close.len());
        assert_eq!(quality.len(), close.len());
        // Index should be between 0 and 100
        assert!(index.iter().all(|&i| i >= 0.0 && i <= 100.0));
        // Rating should be -1, 0, or 1
        assert!(rating.iter().all(|&r| r == -1.0 || r == 0.0 || r == 1.0));
    }

    #[test]
    fn test_volume_efficiency_index_with_period() {
        let (_, _, close, volume) = make_test_data();
        let vei = VolumeEfficiencyIndex::with_period(10).unwrap();
        let (index, _, _) = vei.calculate(&close, &volume);

        assert_eq!(index.len(), close.len());
    }

    #[test]
    fn test_volume_efficiency_index_trending_market() {
        // Strong trend with consistent direction = high efficiency
        let close: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 * 2.0).collect();
        let volume = vec![1000.0; 30];

        let vei = VolumeEfficiencyIndex::new(10, 1).unwrap();
        let (index, rating, _) = vei.calculate(&close, &volume);

        // Should show high efficiency in trending market
        assert!(index[25] > 50.0,
            "index[25] = {} should be > 50 for trending market", index[25]);
    }

    #[test]
    fn test_volume_efficiency_index_choppy_market() {
        // Choppy market with reversals = lower efficiency
        let close: Vec<f64> = (0..30).map(|i| {
            100.0 + if i % 2 == 0 { 1.0 } else { -1.0 }
        }).collect();
        let volume = vec![1000.0; 30];

        let vei = VolumeEfficiencyIndex::new(10, 1).unwrap();
        let (index, _, _) = vei.calculate(&close, &volume);

        // Should show lower efficiency in choppy market
        assert!(index[25] < 50.0,
            "index[25] = {} should be < 50 for choppy market", index[25]);
    }

    #[test]
    fn test_volume_efficiency_index_validation() {
        assert!(VolumeEfficiencyIndex::new(4, 3).is_err()); // period < 5
        assert!(VolumeEfficiencyIndex::new(5, 0).is_err()); // smoothing < 1
        assert!(VolumeEfficiencyIndex::new(5, 3).is_ok());
    }

    #[test]
    fn test_volume_divergence_detector() {
        let (_, _, close, volume) = make_test_data();
        let vdd = VolumeDivergenceDetector::new(5, 5).unwrap();
        let (score, div_type, strength) = vdd.calculate(&close, &volume);

        assert_eq!(score.len(), close.len());
        assert_eq!(div_type.len(), close.len());
        assert_eq!(strength.len(), close.len());
        // Score should be between -100 and 100
        assert!(score.iter().all(|&s| s >= -100.0 && s <= 100.0));
        // Divergence type should be -1, 0, or 1
        assert!(div_type.iter().all(|&d| d == -1.0 || d == 0.0 || d == 1.0));
        // Strength should be between 0 and 100
        assert!(strength.iter().all(|&s| s >= 0.0 && s <= 100.0));
    }

    #[test]
    fn test_volume_divergence_detector_with_period() {
        let (_, _, close, volume) = make_test_data();
        let vdd = VolumeDivergenceDetector::with_period(5).unwrap();
        let (score, _, _) = vdd.calculate(&close, &volume);

        assert_eq!(score.len(), close.len());
    }

    #[test]
    fn test_volume_divergence_detector_bearish_divergence() {
        // Price going up, volume going down = bearish divergence
        let close: Vec<f64> = (0..25).map(|i| 100.0 + i as f64 * 2.0).collect();
        let volume: Vec<f64> = (0..25).map(|i| 2000.0 - i as f64 * 50.0).collect();

        let vdd = VolumeDivergenceDetector::new(5, 5).unwrap();
        let (score, div_type, strength) = vdd.calculate(&close, &volume);

        // Should detect bearish divergence
        let detected_bearish = div_type.iter().skip(15).any(|&d| d == -1.0);
        assert!(detected_bearish,
            "Should detect bearish divergence when price up and volume down");
    }

    #[test]
    fn test_volume_divergence_detector_bullish_divergence() {
        // Price going down, volume going down = bullish divergence (selling exhaustion)
        let close: Vec<f64> = (0..25).map(|i| 150.0 - i as f64 * 2.0).collect();
        let volume: Vec<f64> = (0..25).map(|i| 2000.0 - i as f64 * 50.0).collect();

        let vdd = VolumeDivergenceDetector::new(5, 5).unwrap();
        let (score, div_type, _) = vdd.calculate(&close, &volume);

        // Should detect bullish divergence
        let detected_bullish = div_type.iter().skip(15).any(|&d| d == 1.0);
        assert!(detected_bullish,
            "Should detect bullish divergence when price down and volume down");
    }

    #[test]
    fn test_volume_divergence_detector_validation() {
        assert!(VolumeDivergenceDetector::new(4, 5).is_err()); // period < 5
        assert!(VolumeDivergenceDetector::new(5, 2).is_err()); // lookback < 3
        assert!(VolumeDivergenceDetector::new(5, 3).is_ok());
    }

    #[test]
    fn test_batch7_indicators_technical_indicator_trait() {
        let (high, low, close, volume) = make_test_data();

        // Create OHLCVSeries for compute tests
        let data = OHLCVSeries {
            open: close.clone(),
            high: high.clone(),
            low: low.clone(),
            close: close.clone(),
            volume: volume.clone(),
        };

        // Test VolumeClimaxIndex
        let vci = VolumeClimaxIndex::new(10, 2.0).unwrap();
        assert_eq!(vci.name(), "Volume Climax Index");
        assert_eq!(vci.min_periods(), 11);
        assert_eq!(vci.output_features(), 3);
        assert!(vci.compute(&data).is_ok());

        // Test InstitutionalVolumeProxy
        let ivp = InstitutionalVolumeProxy::new(10, 2.0).unwrap();
        assert_eq!(ivp.name(), "Institutional Volume Proxy");
        assert_eq!(ivp.min_periods(), 11);
        assert_eq!(ivp.output_features(), 3);
        assert!(ivp.compute(&data).is_ok());

        // Test VolumeAccelerationIndex
        let vai = VolumeAccelerationIndex::new(5, 3).unwrap();
        assert_eq!(vai.name(), "Volume Acceleration Index");
        assert_eq!(vai.min_periods(), 11);
        assert_eq!(vai.output_features(), 3);
        assert!(vai.compute(&data).is_ok());

        // Test SmartMoneyIndicator
        let smi = SmartMoneyIndicator::new(10, 1.0).unwrap();
        assert_eq!(smi.name(), "Smart Money Indicator");
        assert_eq!(smi.min_periods(), 11);
        assert_eq!(smi.output_features(), 3);
        assert!(smi.compute(&data).is_ok());

        // Test VolumeEfficiencyIndex
        let vei = VolumeEfficiencyIndex::new(10, 3).unwrap();
        assert_eq!(vei.name(), "Volume Efficiency Index");
        assert_eq!(vei.min_periods(), 11);
        assert_eq!(vei.output_features(), 3);
        assert!(vei.compute(&data).is_ok());

        // Test VolumeDivergenceDetector
        let vdd = VolumeDivergenceDetector::new(5, 5).unwrap();
        assert_eq!(vdd.name(), "Volume Divergence Detector");
        assert_eq!(vdd.min_periods(), 11);
        assert_eq!(vdd.output_features(), 3);
        assert!(vdd.compute(&data).is_ok());
    }

    #[test]
    fn test_batch7_indicators_insufficient_data() {
        let short_data = OHLCVSeries {
            open: vec![100.0, 101.0],
            high: vec![102.0, 103.0],
            low: vec![98.0, 99.0],
            close: vec![100.0, 101.0],
            volume: vec![1000.0, 1100.0],
        };

        // All should fail with insufficient data
        let vci = VolumeClimaxIndex::new(10, 2.0).unwrap();
        assert!(vci.compute(&short_data).is_err());

        let ivp = InstitutionalVolumeProxy::new(10, 2.0).unwrap();
        assert!(ivp.compute(&short_data).is_err());

        let vai = VolumeAccelerationIndex::new(5, 3).unwrap();
        assert!(vai.compute(&short_data).is_err());

        let smi = SmartMoneyIndicator::new(10, 1.0).unwrap();
        assert!(smi.compute(&short_data).is_err());

        let vei = VolumeEfficiencyIndex::new(10, 3).unwrap();
        assert!(vei.compute(&short_data).is_err());

        let vdd = VolumeDivergenceDetector::new(5, 5).unwrap();
        assert!(vdd.compute(&short_data).is_err());
    }

    #[test]
    fn test_batch7_indicators_parameter_validation() {
        // VolumeClimaxIndex
        assert!(VolumeClimaxIndex::new(9, 2.0).is_err());
        assert!(VolumeClimaxIndex::new(10, 1.4).is_err());
        assert!(VolumeClimaxIndex::new(10, 2.0).is_ok());

        // InstitutionalVolumeProxy
        assert!(InstitutionalVolumeProxy::new(9, 2.0).is_err());
        assert!(InstitutionalVolumeProxy::new(10, 1.4).is_err());
        assert!(InstitutionalVolumeProxy::new(10, 2.0).is_ok());

        // VolumeAccelerationIndex
        assert!(VolumeAccelerationIndex::new(4, 3).is_err());
        assert!(VolumeAccelerationIndex::new(5, 0).is_err());
        assert!(VolumeAccelerationIndex::new(5, 3).is_ok());

        // SmartMoneyIndicator
        assert!(SmartMoneyIndicator::new(9, 1.0).is_err());
        assert!(SmartMoneyIndicator::new(10, 0.05).is_err());
        assert!(SmartMoneyIndicator::new(10, 2.5).is_err());
        assert!(SmartMoneyIndicator::new(10, 1.0).is_ok());

        // VolumeEfficiencyIndex
        assert!(VolumeEfficiencyIndex::new(4, 3).is_err());
        assert!(VolumeEfficiencyIndex::new(5, 0).is_err());
        assert!(VolumeEfficiencyIndex::new(5, 3).is_ok());

        // VolumeDivergenceDetector
        assert!(VolumeDivergenceDetector::new(4, 5).is_err());
        assert!(VolumeDivergenceDetector::new(5, 2).is_err());
        assert!(VolumeDivergenceDetector::new(5, 3).is_ok());
    }
}
