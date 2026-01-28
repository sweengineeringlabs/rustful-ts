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
}
