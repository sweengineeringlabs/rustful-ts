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

        // Last value should show higher intensity
        assert!(intensity[14] > 50.0, "intensity[14] = {} should be > 50", intensity[14]);
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

        // Low volume should show low relative volume
        assert!(result[15] < 0.5, "result[15] = {} should be < 0.5", result[15]);
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
}
