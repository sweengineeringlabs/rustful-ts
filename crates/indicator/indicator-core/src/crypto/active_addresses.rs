//! Active Addresses (IND-265)
//!
//! Daily active addresses indicator for cryptocurrency networks.
//! Measures network activity by tracking the number of unique addresses
//! that were active (sending or receiving) in a given period.
//!
//! # Concept
//! Active addresses is a fundamental on-chain metric that shows real network usage.
//! Rising active addresses typically indicates growing adoption and bullish sentiment.
//! Declining active addresses may signal decreasing interest or usage.
//!
//! # Data Requirements
//! This indicator requires on-chain data (active address counts) rather than just price data.
//! When used with OHLCV data, it provides a proxy based on volume patterns.

use indicator_spi::{IndicatorError, IndicatorOutput, OHLCVSeries, Result, TechnicalIndicator};

/// Output from the Active Addresses calculation.
#[derive(Debug, Clone)]
pub struct ActiveAddressesOutput {
    /// Raw active address values (or proxy).
    pub active_addresses: Vec<f64>,
    /// Moving average of active addresses.
    pub ma: Vec<f64>,
    /// Rate of change in active addresses.
    pub momentum: Vec<f64>,
    /// Normalized active address score (0-100).
    pub score: Vec<f64>,
}

/// Active address activity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivityLevel {
    /// Very high activity (top quartile).
    VeryHigh,
    /// Above average activity.
    High,
    /// Normal activity level.
    Normal,
    /// Below average activity.
    Low,
    /// Very low activity (bottom quartile).
    VeryLow,
}

/// Active Addresses (IND-265)
///
/// Measures network activity through unique active addresses.
/// Can use actual on-chain data or estimate from price/volume patterns.
///
/// # Calculation (with on-chain data)
/// ```text
/// MA = SMA(ActiveAddresses, ma_period)
/// Momentum = (ActiveAddresses - ActiveAddresses[momentum_period]) / ActiveAddresses[momentum_period]
/// Score = Percentile rank of current active addresses over lookback
/// ```
///
/// # Calculation (proxy from OHLCV)
/// ```text
/// ActivityProxy = Volume * |Returns| (volume-weighted price activity)
/// Normalized and smoothed to approximate address activity patterns
/// ```
///
/// # Example
/// ```
/// use indicator_core::crypto::ActiveAddresses;
///
/// let aa = ActiveAddresses::new(20, 10, 50).unwrap();
/// let active_counts = vec![100000.0, 105000.0, 98000.0, 110000.0];
/// let output = aa.calculate(&active_counts);
/// ```
#[derive(Debug, Clone)]
pub struct ActiveAddresses {
    /// Period for moving average calculation.
    ma_period: usize,
    /// Period for momentum calculation.
    momentum_period: usize,
    /// Lookback period for score normalization.
    lookback_period: usize,
}

impl ActiveAddresses {
    /// Create a new Active Addresses indicator.
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

    /// Calculate active addresses metrics from on-chain data.
    ///
    /// # Arguments
    /// * `active_counts` - Slice of daily active address counts
    ///
    /// # Returns
    /// ActiveAddressesOutput containing all metrics.
    pub fn calculate(&self, active_counts: &[f64]) -> ActiveAddressesOutput {
        let n = active_counts.len();
        let mut active_addresses = active_counts.to_vec();
        let mut ma = vec![0.0; n];
        let mut momentum = vec![0.0; n];
        let mut score = vec![0.0; n];

        if n < self.ma_period {
            return ActiveAddressesOutput {
                active_addresses,
                ma,
                momentum,
                score,
            };
        }

        // Calculate moving average
        for i in (self.ma_period - 1)..n {
            let start = i + 1 - self.ma_period;
            ma[i] = active_counts[start..=i].iter().sum::<f64>() / self.ma_period as f64;
        }

        // Calculate momentum
        for i in self.momentum_period..n {
            if active_counts[i - self.momentum_period].abs() > 1e-10 {
                momentum[i] = (active_counts[i] / active_counts[i - self.momentum_period] - 1.0) * 100.0;
            }
        }

        // Calculate percentile score
        for i in (self.lookback_period - 1)..n {
            let start = i + 1 - self.lookback_period;
            let window = &active_counts[start..=i];

            let current = active_counts[i];
            let count_below = window.iter().filter(|&&x| x < current).count();
            score[i] = count_below as f64 / self.lookback_period as f64 * 100.0;
        }

        ActiveAddressesOutput {
            active_addresses,
            ma,
            momentum,
            score,
        }
    }

    /// Calculate activity proxy from OHLCV data when on-chain data is unavailable.
    ///
    /// Uses volume and price volatility as a proxy for network activity.
    pub fn calculate_proxy(&self, close: &[f64], volume: &[f64]) -> ActiveAddressesOutput {
        let n = close.len().min(volume.len());
        let mut active_addresses = vec![0.0; n];
        let mut ma = vec![0.0; n];
        let mut momentum = vec![0.0; n];
        let mut score = vec![0.0; n];

        if n < self.ma_period {
            return ActiveAddressesOutput {
                active_addresses,
                ma,
                momentum,
                score,
            };
        }

        // Create activity proxy: volume * absolute returns
        for i in 1..n {
            let ret = (close[i] / close[i - 1] - 1.0).abs();
            active_addresses[i] = volume[i] * (1.0 + ret * 10.0); // Scale returns
        }

        // Normalize to reasonable scale
        let max_val = active_addresses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if max_val > 1e-10 {
            for i in 0..n {
                active_addresses[i] = active_addresses[i] / max_val * 100000.0;
            }
        }

        // Calculate moving average
        for i in (self.ma_period - 1)..n {
            let start = i + 1 - self.ma_period;
            ma[i] = active_addresses[start..=i].iter().sum::<f64>() / self.ma_period as f64;
        }

        // Calculate momentum
        for i in self.momentum_period..n {
            if active_addresses[i - self.momentum_period].abs() > 1e-10 {
                momentum[i] = (active_addresses[i] / active_addresses[i - self.momentum_period] - 1.0) * 100.0;
            }
        }

        // Calculate percentile score
        for i in (self.lookback_period - 1)..n {
            let start = i + 1 - self.lookback_period;
            let window = &active_addresses[start..=i];

            let current = active_addresses[i];
            let count_below = window.iter().filter(|&&x| x < current).count();
            score[i] = count_below as f64 / self.lookback_period as f64 * 100.0;
        }

        ActiveAddressesOutput {
            active_addresses,
            ma,
            momentum,
            score,
        }
    }

    /// Get activity level interpretation.
    pub fn interpret(&self, score: f64) -> ActivityLevel {
        if score >= 80.0 {
            ActivityLevel::VeryHigh
        } else if score >= 60.0 {
            ActivityLevel::High
        } else if score >= 40.0 {
            ActivityLevel::Normal
        } else if score >= 20.0 {
            ActivityLevel::Low
        } else {
            ActivityLevel::VeryLow
        }
    }

    /// Get activity levels for all score values.
    pub fn activity_levels(&self, output: &ActiveAddressesOutput) -> Vec<ActivityLevel> {
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

impl Default for ActiveAddresses {
    fn default() -> Self {
        Self {
            ma_period: 20,
            momentum_period: 10,
            lookback_period: 50,
        }
    }
}

impl TechnicalIndicator for ActiveAddresses {
    fn name(&self) -> &str {
        "Active Addresses"
    }

    fn min_periods(&self) -> usize {
        self.lookback_period.max(self.ma_period)
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let output = self.calculate_proxy(&data.close, &data.volume);
        Ok(IndicatorOutput::triple(output.active_addresses, output.ma, output.momentum))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> Vec<f64> {
        // Simulate active address counts with trend and variation
        (0..60)
            .map(|i| 100000.0 + i as f64 * 500.0 + (i as f64 * 0.3).sin() * 5000.0)
            .collect()
    }

    fn make_ohlcv_data() -> OHLCVSeries {
        let close: Vec<f64> = (0..60)
            .map(|i| 100.0 + i as f64 * 0.5 + (i as f64 * 0.3).sin() * 2.0)
            .collect();
        let n = close.len();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
        let open = close.clone();
        let volume: Vec<f64> = (0..n).map(|i| 1000.0 + (i as f64 * 0.5).sin() * 500.0).collect();

        OHLCVSeries { open, high, low, close, volume }
    }

    #[test]
    fn test_active_addresses_basic() {
        let data = make_test_data();
        let aa = ActiveAddresses::new(10, 5, 30).unwrap();
        let output = aa.calculate(&data);

        assert_eq!(output.active_addresses.len(), data.len());
        assert_eq!(output.ma.len(), data.len());
        assert_eq!(output.momentum.len(), data.len());
        assert_eq!(output.score.len(), data.len());
    }

    #[test]
    fn test_active_addresses_ma() {
        let data = vec![100.0; 30];
        let aa = ActiveAddresses::new(10, 5, 20).unwrap();
        let output = aa.calculate(&data);

        // MA of constant values should equal that constant
        for i in 9..30 {
            assert!((output.ma[i] - 100.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_active_addresses_momentum() {
        // Increasing values
        let data: Vec<f64> = (0..50).map(|i| 100000.0 + i as f64 * 1000.0).collect();
        let aa = ActiveAddresses::new(10, 10, 30).unwrap();
        let output = aa.calculate(&data);

        // Momentum should be positive for increasing data
        for i in 15..50 {
            assert!(output.momentum[i] > 0.0);
        }
    }

    #[test]
    fn test_active_addresses_score() {
        let data = make_test_data();
        let aa = ActiveAddresses::new(10, 5, 30).unwrap();
        let output = aa.calculate(&data);

        // Score should be bounded 0-100
        for i in 29..data.len() {
            assert!(output.score[i] >= 0.0 && output.score[i] <= 100.0);
        }
    }

    #[test]
    fn test_active_addresses_proxy() {
        let close: Vec<f64> = (0..60).map(|i| 100.0 + i as f64 * 0.5).collect();
        let volume: Vec<f64> = (0..60).map(|i| 1000.0 + (i % 5) as f64 * 200.0).collect();

        let aa = ActiveAddresses::new(10, 5, 30).unwrap();
        let output = aa.calculate_proxy(&close, &volume);

        assert_eq!(output.active_addresses.len(), close.len());

        // Proxy values should be non-negative
        for val in output.active_addresses.iter() {
            assert!(*val >= 0.0);
        }
    }

    #[test]
    fn test_active_addresses_interpretation() {
        let aa = ActiveAddresses::default();

        assert_eq!(aa.interpret(90.0), ActivityLevel::VeryHigh);
        assert_eq!(aa.interpret(70.0), ActivityLevel::High);
        assert_eq!(aa.interpret(50.0), ActivityLevel::Normal);
        assert_eq!(aa.interpret(30.0), ActivityLevel::Low);
        assert_eq!(aa.interpret(10.0), ActivityLevel::VeryLow);
    }

    #[test]
    fn test_active_addresses_technical_indicator() {
        let data = make_ohlcv_data();
        let aa = ActiveAddresses::new(10, 5, 30).unwrap();

        assert_eq!(aa.name(), "Active Addresses");
        assert_eq!(aa.min_periods(), 30);

        let output = aa.compute(&data).unwrap();
        assert!(output.values.contains_key("active_addresses"));
        assert!(output.values.contains_key("ma"));
        assert!(output.values.contains_key("momentum"));
        assert!(output.values.contains_key("score"));
    }

    #[test]
    fn test_active_addresses_validation() {
        assert!(ActiveAddresses::new(4, 5, 30).is_err());
        assert!(ActiveAddresses::new(10, 0, 30).is_err());
        assert!(ActiveAddresses::new(10, 5, 19).is_err());
    }

    #[test]
    fn test_active_addresses_empty_input() {
        let aa = ActiveAddresses::default();
        let output = aa.calculate(&[]);

        assert!(output.active_addresses.is_empty());
        assert!(output.ma.is_empty());
    }

    #[test]
    fn test_active_addresses_insufficient_data() {
        let aa = ActiveAddresses::new(20, 10, 50).unwrap();
        let data = vec![100000.0; 15];

        let output = aa.calculate(&data);

        // MA values before warmup should be zero
        for i in 0..15 {
            assert_eq!(output.ma[i], 0.0);
        }
    }

    #[test]
    fn test_active_addresses_default() {
        let aa = ActiveAddresses::default();
        assert_eq!(aa.ma_period(), 20);
        assert_eq!(aa.momentum_period(), 10);
        assert_eq!(aa.lookback_period(), 50);
    }
}
