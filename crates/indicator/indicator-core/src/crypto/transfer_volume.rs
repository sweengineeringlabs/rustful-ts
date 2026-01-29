//! Transfer Volume (IND-267)
//!
//! USD volume transferred on-chain indicator for cryptocurrency networks.
//! Measures the total economic value being transacted on the network.
//!
//! # Concept
//! Transfer volume in USD terms shows the actual economic activity on the network.
//! Unlike transaction count, it weights transactions by their value.
//! High transfer volume indicates significant capital movement and network usage.
//!
//! # Data Requirements
//! This indicator works best with actual on-chain transfer volume data.
//! When using OHLCV data, trading volume serves as a proxy.

use indicator_spi::{IndicatorError, IndicatorOutput, OHLCVSeries, Result, TechnicalIndicator};

/// Output from the Transfer Volume calculation.
#[derive(Debug, Clone)]
pub struct TransferVolumeOutput {
    /// Raw transfer volume in USD.
    pub volume: Vec<f64>,
    /// Moving average of transfer volume.
    pub ma: Vec<f64>,
    /// Transfer volume momentum (rate of change).
    pub momentum: Vec<f64>,
    /// Volume relative to market cap (velocity proxy).
    pub velocity: Vec<f64>,
    /// Normalized volume score (0-100).
    pub score: Vec<f64>,
}

/// Transfer volume classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolumeLevel {
    /// Extremely high transfer volume.
    Extreme,
    /// High transfer volume.
    High,
    /// Normal transfer volume.
    Normal,
    /// Low transfer volume.
    Low,
    /// Very low transfer volume.
    Minimal,
}

/// Transfer Volume (IND-267)
///
/// Measures USD value of on-chain transfers and related metrics.
///
/// # Calculation
/// ```text
/// MA = SMA(TransferVolume, ma_period)
/// Momentum = (Volume - Volume[momentum_period]) / Volume[momentum_period] * 100
/// Velocity = TransferVolume / MarketCap (if available)
/// Score = Percentile rank over lookback period
/// ```
///
/// # Example
/// ```
/// use indicator_core::crypto::TransferVolume;
///
/// let tv = TransferVolume::new(20, 10, 50).unwrap();
/// let volumes = vec![1e9, 1.2e9, 0.9e9, 1.5e9];
/// let output = tv.calculate(&volumes);
/// ```
#[derive(Debug, Clone)]
pub struct TransferVolume {
    /// Period for moving average calculation.
    ma_period: usize,
    /// Period for momentum calculation.
    momentum_period: usize,
    /// Lookback period for score normalization.
    lookback_period: usize,
}

impl TransferVolume {
    /// Create a new Transfer Volume indicator.
    ///
    /// # Arguments
    /// * `ma_period` - Period for moving average (minimum 5)
    /// * `momentum_period` - Period for momentum calculation (minimum 1)
    /// * `lookback_period` - Period for percentile ranking (minimum 20)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid.
    pub fn new(ma_period: usize, momentum_period: usize, lookback_period: usize) -> Result<Self> {
        if ma_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "ma_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if momentum_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        if lookback_period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback_period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        Ok(Self {
            ma_period,
            momentum_period,
            lookback_period,
        })
    }

    /// Create with default parameters (20, 10, 50).
    pub fn default_params() -> Result<Self> {
        Self::new(20, 10, 50)
    }

    /// Calculate transfer volume metrics from on-chain data.
    ///
    /// # Arguments
    /// * `volumes` - Slice of daily transfer volumes in USD
    ///
    /// # Returns
    /// TransferVolumeOutput containing all metrics.
    pub fn calculate(&self, volumes: &[f64]) -> TransferVolumeOutput {
        let n = volumes.len();
        let volume = volumes.to_vec();
        let mut ma = vec![0.0; n];
        let mut momentum = vec![0.0; n];
        let velocity = vec![0.0; n]; // Requires market cap data
        let mut score = vec![0.0; n];

        if n < self.ma_period {
            return TransferVolumeOutput {
                volume,
                ma,
                momentum,
                velocity,
                score,
            };
        }

        // Calculate moving average
        for i in (self.ma_period - 1)..n {
            let start = i + 1 - self.ma_period;
            ma[i] = volumes[start..=i].iter().sum::<f64>() / self.ma_period as f64;
        }

        // Calculate momentum
        for i in self.momentum_period..n {
            if volumes[i - self.momentum_period].abs() > 1e-10 {
                momentum[i] = (volumes[i] / volumes[i - self.momentum_period] - 1.0) * 100.0;
            }
        }

        // Calculate percentile score
        for i in (self.lookback_period - 1)..n {
            let start = i + 1 - self.lookback_period;
            let window = &volumes[start..=i];

            let current = volumes[i];
            let count_below = window.iter().filter(|&&x| x < current).count();
            score[i] = count_below as f64 / self.lookback_period as f64 * 100.0;
        }

        TransferVolumeOutput {
            volume,
            ma,
            momentum,
            velocity,
            score,
        }
    }

    /// Calculate with market cap to derive velocity.
    ///
    /// # Arguments
    /// * `volumes` - Daily transfer volumes in USD
    /// * `market_caps` - Daily market capitalizations
    pub fn calculate_with_market_cap(
        &self,
        volumes: &[f64],
        market_caps: &[f64],
    ) -> TransferVolumeOutput {
        let n = volumes.len().min(market_caps.len());
        let mut output = self.calculate(&volumes[..n]);

        // Calculate velocity (transfer volume / market cap)
        for i in 0..n {
            if market_caps[i] > 1e-10 {
                output.velocity[i] = volumes[i] / market_caps[i];
            }
        }

        output
    }

    /// Calculate using OHLCV trading volume as a proxy.
    pub fn calculate_proxy(&self, data: &OHLCVSeries) -> TransferVolumeOutput {
        let n = data.close.len();
        let mut volume = vec![0.0; n];
        let mut ma = vec![0.0; n];
        let mut momentum = vec![0.0; n];
        let mut velocity = vec![0.0; n];
        let mut score = vec![0.0; n];

        if n < self.ma_period {
            return TransferVolumeOutput {
                volume,
                ma,
                momentum,
                velocity,
                score,
            };
        }

        // Use dollar volume (trading volume * price) as proxy
        for i in 0..n {
            let typical_price = (data.high[i] + data.low[i] + data.close[i]) / 3.0;
            volume[i] = data.volume[i] * typical_price;
        }

        // Calculate moving average
        for i in (self.ma_period - 1)..n {
            let start = i + 1 - self.ma_period;
            ma[i] = volume[start..=i].iter().sum::<f64>() / self.ma_period as f64;
        }

        // Calculate momentum
        for i in self.momentum_period..n {
            if volume[i - self.momentum_period].abs() > 1e-10 {
                momentum[i] = (volume[i] / volume[i - self.momentum_period] - 1.0) * 100.0;
            }
        }

        // Calculate velocity proxy (normalized by rolling max)
        let lookback = self.lookback_period.min(n);
        for i in (lookback - 1)..n {
            let start = i + 1 - lookback;
            let max_vol = volume[start..=i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if max_vol > 1e-10 {
                velocity[i] = volume[i] / max_vol;
            }
        }

        // Calculate percentile score
        for i in (self.lookback_period - 1)..n {
            let start = i + 1 - self.lookback_period;
            let window = &volume[start..=i];

            let current = volume[i];
            let count_below = window.iter().filter(|&&x| x < current).count();
            score[i] = count_below as f64 / self.lookback_period as f64 * 100.0;
        }

        TransferVolumeOutput {
            volume,
            ma,
            momentum,
            velocity,
            score,
        }
    }

    /// Get volume level classification.
    pub fn interpret(&self, score: f64) -> VolumeLevel {
        if score >= 90.0 {
            VolumeLevel::Extreme
        } else if score >= 70.0 {
            VolumeLevel::High
        } else if score >= 30.0 {
            VolumeLevel::Normal
        } else if score >= 10.0 {
            VolumeLevel::Low
        } else {
            VolumeLevel::Minimal
        }
    }

    /// Get volume levels for all score values.
    pub fn volume_levels(&self, output: &TransferVolumeOutput) -> Vec<VolumeLevel> {
        output.score.iter().map(|&s| self.interpret(s)).collect()
    }

    /// Get the MA period.
    pub fn ma_period(&self) -> usize {
        self.ma_period
    }

    /// Get the momentum period.
    pub fn momentum_period(&self) -> usize {
        self.momentum_period
    }

    /// Get the lookback period.
    pub fn lookback_period(&self) -> usize {
        self.lookback_period
    }
}

impl Default for TransferVolume {
    fn default() -> Self {
        Self {
            ma_period: 20,
            momentum_period: 10,
            lookback_period: 50,
        }
    }
}

impl TechnicalIndicator for TransferVolume {
    fn name(&self) -> &str {
        "Transfer Volume"
    }

    fn min_periods(&self) -> usize {
        self.lookback_period.max(self.ma_period)
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let output = self.calculate_proxy(data);
        Ok(IndicatorOutput::triple(output.volume, output.ma, output.momentum))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> Vec<f64> {
        // Simulate transfer volumes with trend and variation
        (0..60)
            .map(|i| 1e9 + i as f64 * 1e7 + (i as f64 * 0.4).sin() * 1e8)
            .collect()
    }

    fn make_ohlcv_data() -> OHLCVSeries {
        let close: Vec<f64> = (0..60)
            .map(|i| 100.0 + i as f64 * 0.5 + (i as f64 * 0.3).sin() * 2.0)
            .collect();
        let n = close.len();
        let high: Vec<f64> = close.iter().map(|c| c + 2.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 2.0).collect();
        let open = close.clone();
        let volume: Vec<f64> = (0..n).map(|i| 1e6 + (i as f64 * 0.5).sin() * 5e5).collect();

        OHLCVSeries { open, high, low, close, volume }
    }

    #[test]
    fn test_transfer_volume_basic() {
        let data = make_test_data();
        let tv = TransferVolume::new(10, 5, 30).unwrap();
        let output = tv.calculate(&data);

        assert_eq!(output.volume.len(), data.len());
        assert_eq!(output.ma.len(), data.len());
        assert_eq!(output.momentum.len(), data.len());
        assert_eq!(output.score.len(), data.len());
    }

    #[test]
    fn test_transfer_volume_ma() {
        let data = vec![1e9; 30];
        let tv = TransferVolume::new(10, 5, 20).unwrap();
        let output = tv.calculate(&data);

        // MA of constant values should equal that constant
        for i in 9..30 {
            assert!((output.ma[i] - 1e9).abs() < 1.0);
        }
    }

    #[test]
    fn test_transfer_volume_momentum() {
        // Increasing values
        let data: Vec<f64> = (0..50).map(|i| 1e9 + i as f64 * 5e7).collect();
        let tv = TransferVolume::new(10, 10, 30).unwrap();
        let output = tv.calculate(&data);

        // Momentum should be positive for increasing data
        for i in 15..50 {
            assert!(output.momentum[i] > 0.0);
        }
    }

    #[test]
    fn test_transfer_volume_with_market_cap() {
        let volumes: Vec<f64> = (0..50).map(|i| 1e9 + i as f64 * 1e7).collect();
        let market_caps: Vec<f64> = (0..50).map(|_| 100e9).collect(); // $100B market cap

        let tv = TransferVolume::new(10, 5, 30).unwrap();
        let output = tv.calculate_with_market_cap(&volumes, &market_caps);

        // Velocity should be approximately 0.01 (1e9 / 100e9)
        assert!(output.velocity[0] > 0.005 && output.velocity[0] < 0.02);
    }

    #[test]
    fn test_transfer_volume_proxy() {
        let data = make_ohlcv_data();
        let tv = TransferVolume::new(10, 5, 30).unwrap();
        let output = tv.calculate_proxy(&data);

        assert_eq!(output.volume.len(), data.close.len());

        // Dollar volume should be positive
        for i in 0..output.volume.len() {
            assert!(output.volume[i] >= 0.0);
        }
    }

    #[test]
    fn test_transfer_volume_velocity_proxy() {
        let data = make_ohlcv_data();
        let tv = TransferVolume::new(10, 5, 30).unwrap();
        let output = tv.calculate_proxy(&data);

        // Velocity should be between 0 and 1 (normalized)
        for i in 29..output.velocity.len() {
            assert!(output.velocity[i] >= 0.0 && output.velocity[i] <= 1.0);
        }
    }

    #[test]
    fn test_transfer_volume_interpretation() {
        let tv = TransferVolume::default();

        assert_eq!(tv.interpret(95.0), VolumeLevel::Extreme);
        assert_eq!(tv.interpret(80.0), VolumeLevel::High);
        assert_eq!(tv.interpret(50.0), VolumeLevel::Normal);
        assert_eq!(tv.interpret(20.0), VolumeLevel::Low);
        assert_eq!(tv.interpret(5.0), VolumeLevel::Minimal);
    }

    #[test]
    fn test_transfer_volume_score_bounded() {
        let data = make_test_data();
        let tv = TransferVolume::new(10, 5, 30).unwrap();
        let output = tv.calculate(&data);

        for i in 29..data.len() {
            assert!(output.score[i] >= 0.0 && output.score[i] <= 100.0);
        }
    }

    #[test]
    fn test_transfer_volume_technical_indicator() {
        let data = make_ohlcv_data();
        let tv = TransferVolume::new(10, 5, 30).unwrap();

        assert_eq!(tv.name(), "Transfer Volume");
        assert_eq!(tv.min_periods(), 30);

        let output = tv.compute(&data).unwrap();
        assert!(output.values.contains_key("volume"));
        assert!(output.values.contains_key("ma"));
        assert!(output.values.contains_key("momentum"));
        assert!(output.values.contains_key("velocity"));
        assert!(output.values.contains_key("score"));
    }

    #[test]
    fn test_transfer_volume_validation() {
        assert!(TransferVolume::new(4, 5, 30).is_err());
        assert!(TransferVolume::new(10, 0, 30).is_err());
        assert!(TransferVolume::new(10, 5, 19).is_err());
    }

    #[test]
    fn test_transfer_volume_empty_input() {
        let tv = TransferVolume::default();
        let output = tv.calculate(&[]);

        assert!(output.volume.is_empty());
        assert!(output.ma.is_empty());
    }

    #[test]
    fn test_transfer_volume_default() {
        let tv = TransferVolume::default();
        assert_eq!(tv.ma_period(), 20);
        assert_eq!(tv.momentum_period(), 10);
        assert_eq!(tv.lookback_period(), 50);
    }

    #[test]
    fn test_volume_levels_iterator() {
        let data = make_test_data();
        let tv = TransferVolume::new(10, 5, 30).unwrap();
        let output = tv.calculate(&data);
        let levels = tv.volume_levels(&output);

        assert_eq!(levels.len(), data.len());
    }
}
